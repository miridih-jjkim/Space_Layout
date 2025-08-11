#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Silhouette Mask Generator v4 (globals version, optimized)
- skin을 baseline으로 요소를 순차 합성하며, 변화한 픽셀만 태그별 색으로 칠한 마스크 생성
- 최종 단계에서 'content'(빨강) 영역을 HSV로 추출 → 컨투어의 바운딩박스를 빨강으로 꽉 채움
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageChops
import cv2
from tqdm import tqdm

# ===== 글로벌 설정 =====
CSV_PATH        = "template_info.csv"              # 입력 CSV
OUTPUT_DIR      = "/data/shared2/jjkim/margin_v4"  # 출력 디렉토리
TEMPLATE_TYPE   = "card_news"                      # 템플릿 타입 필터
VERBOSE_SKIP    = False                            # 스킵 사유 출력
LIMIT_PAGES     = None                             # 처리 페이지 제한 (None=전체)

# 태그 → 카테고리 규칙 / 카테고리 → 색상(RGB)
TAG_RULES = {
    "content": ["TEXT"],
    "media"  : ["PHOTO", "GIF", "VIDEO", "FrameItem"],
    "data"   : ["GRID", "Chart", "Barcode", "QRCode"],
}
TAG2COLOR = {
    "content": (255, 0, 0),      # 빨강
    "media"  : (0, 128, 255),    # 파랑
    "data"   : (0, 200, 0),      # 초록
    "etc"    : (160, 0, 255),    # 보라
}

# 빨간색 추출(HSV) 파라미터 (BGR→HSV 변환 후 inRange)
HSV_RED_RANGE_1 = (np.array([0, 70, 50]),   np.array([10, 255, 255]))
HSV_RED_RANGE_2 = (np.array([170, 70, 50]), np.array([180, 255, 255]))
MORPH_KERNEL    = np.ones((2, 2), np.uint8)
MORPH_ITERS     = 1
CV_FILL_THICK   = -1  # 꽉 채우기


# ===== 유틸 =====
def to_int(v, default=None):
    try:
        if pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default

def load_image(path, size, mode="RGB"):
    return Image.open(path).convert(mode).resize(size)

def classify_tag(raw_tag: str) -> str:
    t = str(raw_tag) if not pd.isna(raw_tag) else ""
    if any(k in t for k in TAG_RULES["content"]):
        return "content"
    if any(k in t for k in TAG_RULES["media"]):
        return "media"
    if any(k in t for k in TAG_RULES["data"]):
        return "data"
    return "etc"

def prepare_elements(df: pd.DataFrame) -> pd.DataFrame:
    """element만 추출하고 좌표/크기 수치화 + 결측 제거"""
    elements = df[df["image_type"] == "element"].copy()
    for col in ["img_width", "img_height", "left", "top"]:
        elements[col] = pd.to_numeric(elements[col], errors="coerce")
    elements = elements.dropna(subset=["img_width", "img_height", "left", "top"])
    return elements

def get_canvas_size(group_df):
    thumbs = group_df[group_df["image_type"] == "thumbnail"]
    if thumbs.empty:
        return None
    W = to_int(thumbs["img_width_resized"].iloc[0])
    H = to_int(thumbs["img_height_resized"].iloc[0])
    if not W or not H or W <= 0 or H <= 0:
        return None
    return W, H


# ===== 핵심 처리 =====
def render_mask(template_idx, page_num, group_df, output_dir):
    sample_name = f"{template_idx}_{page_num}"

    # 캔버스 크기 확보
    size = get_canvas_size(group_df)
    if not size:
        if VERBOSE_SKIP:
            print(f"[스킵] {sample_name}: 썸네일 사이즈 정보 없음")
        return False
    W, H = size

    # baseline = skin
    try:
        skin_path  = group_df[group_df["image_type"] == "skin"]["full_image_path"].iloc[0]
        _thumbpath = group_df[group_df["image_type"] == "thumbnail"]["full_image_path"].iloc[0]  # 미사용
        skin = load_image(skin_path, (W, H))
    except Exception as e:
        if VERBOSE_SKIP:
            print(f"[스킵] {sample_name}: 기본 이미지 로드 실패 ({e})")
        return False

    current = skin.copy()                    # 현재 합성 상태
    mask_img = Image.new("RGB", (W, H), (255, 255, 255))  # 초기 흰 배경
    mask_np  = np.array(mask_img, dtype=np.uint8)         # H x W x 3

    elements = prepare_elements(group_df)
    if elements.empty:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask_np).save(Path(output_dir) / f"{sample_name}.png")
        return True

    # 요소 루프: bbox 내부만 비교 → 변한 픽셀만 색칠
    for _, el in elements.iterrows():
        path = el["full_image_path"]
        if not isinstance(path, str) or not os.path.exists(path):
            if VERBOSE_SKIP:
                print(f"[스킵] 요소 없음: {path}")
            continue

        left   = max(0, to_int(el["left"], 0) or 0)
        top    = max(0, to_int(el["top"], 0) or 0)
        width  = max(1, to_int(el["img_width"], 1) or 1)
        height = max(1, to_int(el["img_height"], 1) or 1)

        # 경계 클리핑
        if left >= W or top >= H:
            continue
        width  = min(width,  W - left)
        height = min(height, H - top)
        if width <= 0 or height <= 0:
            continue

        # 요소 이미지 로드 (RGBA)
        try:
            el_img = load_image(path, (width, height), mode="RGBA")
        except Exception:
            if VERBOSE_SKIP:
                print(f"[스킵] 요소 이미지 불러오기 실패: {path}")
            continue

        # 다음 상태 합성
        next_img = current.copy()
        next_img.paste(el_img, (left, top), el_img)

        # 변경은 bbox 내부에만 존재 → crop 후 difference
        cur_crop  = current.crop((left, top, left + width, top + height))
        next_crop = next_img.crop((left, top, left + width, top + height))

        diff = ImageChops.difference(cur_crop, next_crop).convert("L")
        bin_small = diff.point(lambda p: 255 if p > 0 else 0)  # 변화=255

        # numpy 마스킹으로 색칠
        bin_np  = np.array(bin_small, dtype=np.uint8)   # h x w
        changed = bin_np > 0
        if changed.any():
            cat   = classify_tag(el.get("tag", ""))
            color = TAG2COLOR.get(cat, TAG2COLOR["etc"])
            region = mask_np[top:top+height, left:left+width]  # h x w x 3
            color_arr = np.zeros_like(region)
            color_arr[:] = color
            region[changed] = color_arr[changed]

        # 상태 업데이트
        current = next_img

    # ---- 빨간(TEXT=content) 영역 윤곽 추출 → 박스 채우기 ----
    # PIL(RGB) → BGR(np) → HSV
    bgr = mask_np[:, :, ::-1].copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower1, upper1 = HSV_RED_RANGE_1
    lower2, upper2 = HSV_RED_RANGE_2
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(m1, m2)

    red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=MORPH_ITERS)
    contours, _ = cv2.findContours(red_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 빨강(BGR)으로 꽉 채움
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255), thickness=CV_FILL_THICK)

    # BGR → RGB로 되돌려 저장
    final_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(final_rgb)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_img.save(Path(output_dir) / f"{sample_name}.png")
    return True


# ===== 실행 =====
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["template_type"] == TEMPLATE_TYPE].copy()
    grouped = df.groupby(["template_idx", "page_num"])

    saved = skipped = 0
    total = len(grouped)

    for i, ((template_idx, page_num), group_df) in enumerate(tqdm(grouped, desc="Rendering"), start=1):
        if LIMIT_PAGES and i > LIMIT_PAGES:
            break
        ok = render_mask(template_idx, page_num, group_df, OUTPUT_DIR)
        saved += int(ok)
        skipped += int(not ok)

    print(f"[DONE] 저장 {saved} / 스킵 {skipped} (총 {min(total, LIMIT_PAGES or total)})")
    print(f"[OUT] {OUTPUT_DIR}")
