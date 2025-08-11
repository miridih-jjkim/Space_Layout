#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Silhouette Mask Generator v2 (globals version)
- 각 페이지의 baseline(skin) 위에 element 이미지를 순차적으로 합성하며
  baseline 대비 차이 마스크를 누적 생성.
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image, ImageChops
from tqdm import tqdm

# ===== 글로벌 설정 =====
CSV_PATH        = "template_info.csv"  # 입력 CSV
OUTPUT_DIR      = "/data/shared2/jjkim/margin_v2"  # 출력 디렉토리
TEMPLATE_TYPE   = "card_news"  # 필터할 template_type
BASELINE_TYPE   = "skin"  # 'skin' 또는 'thumbnail'
BINARIZE_MODE   = "nonzero"  # 'nonzero' 또는 'threshold'
INCLUDE_TAGS    = None  # 예: {"TEXT", "PHOTO"}  / None이면 전체 포함
EXCLUDE_TAGS    = None  # 예: {"GRID", "Chart"} / None이면 제외 없음
VERBOSE_SKIP    = False  # 스킵 사유 출력 여부
LIMIT_PAGES     = None   # 최대 처리 페이지 수 (None = 전체)


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


def prepare_elements(df):
    elements = df[df["image_type"] == "element"].copy()
    for col in ["img_width", "img_height", "left", "top"]:
        elements[col] = pd.to_numeric(elements[col], errors="coerce")
    elements = elements.dropna(subset=["img_width", "img_height", "left", "top"])
    if INCLUDE_TAGS:
        elements = elements[elements["tag"].astype(str).isin(INCLUDE_TAGS)]
    if EXCLUDE_TAGS:
        elements = elements[~elements["tag"].astype(str).isin(EXCLUDE_TAGS)]
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


def get_baseline_path(group_df):
    if BASELINE_TYPE == "skin":
        rows = group_df[group_df["image_type"] == "skin"]
    else:
        rows = group_df[group_df["image_type"] == "thumbnail"]
    if rows.empty:
        return None
    return rows["full_image_path"].iloc[0]


# ===== 핵심 처리 =====
def render_mask(template_idx, page_num, group_df):
    sample_name = f"{template_idx}_{page_num}"
    size = get_canvas_size(group_df)
    if not size:
        if VERBOSE_SKIP:
            print(f"[스킵] {sample_name}: 썸네일 사이즈 정보 없음")
        return False
    W, H = size

    baseline_path = get_baseline_path(group_df)
    if not baseline_path or not os.path.exists(baseline_path):
        if VERBOSE_SKIP:
            print(f"[스킵] {sample_name}: baseline '{BASELINE_TYPE}' 경로 없음")
        return False

    try:
        base_img = load_image(baseline_path, (W, H), mode="RGB")
    except Exception as e:
        if VERBOSE_SKIP:
            print(f"[스킵] {sample_name}: baseline 로드 실패 ({e})")
        return False

    current = base_img.copy()
    mask = Image.new("L", (W, H), color=255)

    elements = prepare_elements(group_df)
    if elements.empty:
        mask.convert("RGB").save(Path(OUTPUT_DIR) / f"{sample_name}.png")
        return True

    for _, el in elements.iterrows():
        path = el["full_image_path"]
        if not isinstance(path, str) or not os.path.exists(path):
            if VERBOSE_SKIP:
                print(f"[스킵] 요소 없음: {path}")
            continue

        left = max(0, to_int(el["left"], 0) or 0)
        top = max(0, to_int(el["top"], 0) or 0)
        width = max(1, to_int(el["img_width"], 1) or 1)
        height = max(1, to_int(el["img_height"], 1) or 1)

        if left >= W or top >= H:
            continue
        width = min(width, W - left)
        height = min(height, H - top)
        if width <= 0 or height <= 0:
            continue

        try:
            element_img = load_image(path, (width, height), mode="RGBA")
        except:
            if VERBOSE_SKIP:
                print(f"[스킵] 요소 이미지 불러오기 실패: {path}")
            continue

        next_img = current.copy()
        next_img.paste(element_img, (left, top), element_img)
        diff = ImageChops.difference(base_img, next_img).convert("L")

        if BINARIZE_MODE == "threshold":
            binary = diff.point(lambda p: 0 if p >= 3 else 255)
        else:
            binary = diff.point(lambda p: 0 if p > 0 else 255)

        mask = ImageChops.darker(mask, binary)
        current = next_img

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    mask.convert("RGB").save(Path(OUTPUT_DIR) / f"{sample_name}.png")
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
        ok = render_mask(template_idx, page_num, group_df)
        saved += int(ok)
        skipped += int(not ok)

    print(f"[DONE] 저장 {saved} / 스킵 {skipped} (총 {min(total, LIMIT_PAGES or total)})")
    print(f"[OUT] {OUTPUT_DIR}")
