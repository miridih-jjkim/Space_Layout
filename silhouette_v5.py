#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Silhouette Mask Generator v5 (globals version, optimized)
- skin을 baseline으로 요소를 순차 합성
- 변화 픽셀은 bbox 내부만 비교하여 태그별 색으로 실루엣 채움
- TEXT는 실루엣에 채우지 않고, 요소 영역 안에서 '타이트' 바운딩 박스를 찾아 빨간 테두리만 그림
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageChops
import cv2
from tqdm import tqdm

# ===================== 글로벌 설정 =====================
CSV_PATH        = "template_info.csv"               # 입력 CSV
OUTPUT_DIR      = "/data/shared2/jjkim/margin_v5"   # 출력 디렉토리
TEMPLATE_TYPE   = "card_news"                       # 필터할 템플릿 타입
VERBOSE_SKIP    = False                             # 스킵 사유 상세 출력
LIMIT_PAGES     = None                              # 처리 페이지 제한 (None = 전체)

# 타이트 박스/윤곽 관련
THICKNESS       = 4          # 최종 빨간 테두리 두께 (양수=테두리, -1=채우기)
TIGHT_PAD       = 0          # 타이트 박스 패딩(음수면 더 타이트)
MIN_TEXT_AREA   = 120        # 너무 작은 텍스트 조각 제거(픽셀 수)

# 태그 → 카테고리 규칙 / 카테고리 → 색상(RGB)
TAG_RULES = {
    "content": ["TEXT"],
    "media"  : ["PHOTO", "GIF", "VIDEO", "FrameItem"],
    "data"   : ["GRID", "Chart", "Barcode", "QRCode"],
    "decor"  : ["SHAPESVG"],
}
TAG2COLOR = {
    "content": (255, 0, 0),      # 빨강 (TEXT; 실루엣 채우지 않음)
    "media"  : (0, 128, 255),    # 파랑
    "data"   : (0, 200, 0),      # 초록
    "decor"  : (255, 255, 0),    # 노랑
    "etc"    : (160, 0, 255),    # 보라
}
# =====================================================

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
    for cat, keys in TAG_RULES.items():
        if any(k in t for k in keys):
            return cat
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

def _bbox_pad(x, y, w, h, W, H, pad):
    if not pad:
        return (x, y, w, h)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + 2 * pad)
    h = min(H - y, h + 2 * pad)
    return (x, y, w, h)

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
        skin_path = group_df[group_df["image_type"] == "skin"]["full_image_path"].iloc[0]
        _thumb    = group_df[group_df["image_type"] == "thumbnail"]["full_image_path"].iloc[0]  # 미사용
        skin = load_image(skin_path, (W, H))
    except Exception as e:
        if VERBOSE_SKIP:
            print(f"[스킵] {sample_name}: 기본 이미지 로드 실패 ({e})")
        return False

    # 상태/출력 초기화
    current = skin.copy()
    mask_np = np.full((H, W, 3), 255, np.uint8)  # 실루엣(흰 배경)
    text_change_mask = np.zeros((H, W), dtype=np.uint8)  # TEXT 변화 누적
    content_boxes = []  # (left, top, width, height)

    elements = prepare_elements(group_df)
    if elements.empty:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask_np).save(Path(output_dir) / f"{sample_name}.png")
        return True

    # 요소 루프: bbox 내부만 비교 → 변한 픽셀만 반영
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

        # 요소 로드
        try:
            el_img = load_image(path, (width, height), mode="RGBA")
        except Exception:
            if VERBOSE_SKIP:
                print(f"[스킵] 요소 이미지 불러오기 실패: {path}")
            continue

        # 다음 상태 합성
        next_img = current.copy()
        next_img.paste(el_img, (left, top), el_img)

        # 변경은 bbox 내부만 존재 → crop 후 difference
        cur_crop  = current.crop((left, top, left + width, top + height))
        next_crop = next_img.crop((left, top, left + width, top + height))
        diff = ImageChops.difference(cur_crop, next_crop).convert("L")

        bin_small = np.array(diff, dtype=np.uint8)           # h x w
        changed = bin_small > 0

        # 태그 분류
        cat = classify_tag(el.get("tag", ""))

        if changed.any():
            if cat == "content":
                # TEXT는 누적 마스크에만 기록 (실루엣은 채우지 않음)
                text_change_mask[top:top+height, left:left+width] |= (changed.astype(np.uint8) * 255)
                content_boxes.append((left, top, width, height))
            else:
                # 나머지는 실루엣 색칠 (bbox 영역만)
                color = np.array(TAG2COLOR.get(cat, TAG2COLOR["etc"]), dtype=np.uint8)
                region = mask_np[top:top+height, left:left+width]  # h x w x 3
                # 변경된 위치에만 색상 브로드캐스팅
                region[changed] = color

        current = next_img

    # ---------- content 요소 박스 내부에서 '타이트' 박스 계산 ----------
    vis_bgr = cv2.cvtColor(mask_np, cv2.COLOR_RGB2BGR)
    k_close = np.ones((2, 2), np.uint8)

    for (lx, ly, lw, lh) in content_boxes:
        x1, y1 = max(0, lx), max(0, ly)
        x2, y2 = min(W, lx + lw), min(H, ly + lh)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = text_change_mask[y1:y2, x1:x2]
        if crop.size == 0 or crop.max() == 0:
            continue

        # 내부 구멍 최소 보정 → 컨투어
        cleaned = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, k_close, iterations=1)
        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        # 요소 영역 안의 텍스트 픽셀 전체를 덮는 타이트 외접 사각형
        big = np.vstack(cnts)
        if cv2.contourArea(big) < MIN_TEXT_AREA:
            continue

        rx, ry, rw, rh = cv2.boundingRect(big)
        gx, gy, gw, gh = x1 + rx, y1 + ry, rw, rh
        gx, gy, gw, gh = _bbox_pad(gx, gy, gw, gh, W, H, TIGHT_PAD)

        # 빨간 테두리만(두께 THICKNESS)
        cv2.rectangle(vis_bgr, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), THICKNESS)

    # ---------- 저장 ----------
    out_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(out_rgb)
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
