#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Silhouette Generator v1 (refactored)
- CSV에서 템플릿/페이지별 요소 박스를 읽어 실루엣 이미지를 생성합니다.
- 기본은 모든 element 박스를 회색(136)으로 꽉 채워 그립니다.
- 옵션으로 템플릿 타입/태그 필터를 적용할 수 있습니다.

사용 예)
python silhouette_v1.py \
  --csv template_info.csv \
  --output /data/shared2/jjkim/margin_data \
  --type card_news \
  --include-tags TEXT,PHOTO

필드 기대치)
- image_type: ['thumbnail', 'element']
- thumbnail 행: img_width_resized, img_height_resized
- element 행: left_new, top_new, img_width_new, img_height_new, tag
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Tuple, Optional, Set

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


# ---------------------------
# 유틸 함수
# ---------------------------
def safe_int(v: object, default: Optional[int] = None) -> Optional[int]:
    """숫자형/문자형 혼합 컬럼을 안전하게 int로 변환. 실패 시 default 반환."""
    try:
        if pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def parse_tag_set(arg: Optional[str]) -> Optional[Set[str]]:
    """콤마로 구분된 태그 문자열 -> 집합으로 변환."""
    if not arg:
        return None
    return {s.strip() for s in arg.split(",") if s.strip()}


# ---------------------------
# 핵심 로직
# ---------------------------
def draw_silhouette_for_group(
    group_df: pd.DataFrame,
    fill_bgr: Tuple[int, int, int] = (136, 136, 136),
) -> Optional[np.ndarray]:
    """
    단일 (template_idx, page_num) 그룹에 대해 실루엣 캔버스를 생성.
    썸네일 사이즈를 기반으로 흰색 캔버스를 만들고 요소 박스를 그린다.
    실패(썸네일 없음/사이즈 이상) 시 None 반환.
    """
    # 썸네일 한 줄 확보
    thumbs = group_df[group_df["image_type"] == "thumbnail"]
    if thumbs.empty:
        return None

    W = safe_int(thumbs["img_width_resized"].iloc[0])
    H = safe_int(thumbs["img_height_resized"].iloc[0])
    if not W or not H or W <= 0 or H <= 0:
        return None

    # 흰색 캔버스 (H x W x 3)
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    # 요소 박스만 추출
    elements = group_df[group_df["image_type"] == "element"].copy()
    if elements.empty:
        return canvas  # 요소가 없어도 빈 캔버스 저장

    # 좌표/크기 안전 캐스팅 + 경계 보정
    for _, el in elements.iterrows():
        left = safe_int(el.get("left_new"))
        top = safe_int(el.get("top_new"))
        width = safe_int(el.get("img_width_new"))
        height = safe_int(el.get("img_height_new"))
        if None in (left, top, width, height):
            continue
        if width <= 0 or height <= 0:
            continue

        x1, y1 = max(0, left), max(0, top)
        x2, y2 = min(W, left + width), min(H, top + height)
        if x2 <= x1 or y2 <= y1:
            continue

        # 꽉 채우기(-1)로 회색 사각형
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill_bgr, thickness=-1)

    return canvas


def filter_elements_by_tags(
    df: pd.DataFrame,
    include_tags: Optional[Set[str]] = None,
    exclude_tags: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    태그 포함/제외 필터를 element 행에만 적용.
    - include_tags: 지정 시 해당 태그만 남김
    - exclude_tags: 지정 시 해당 태그 제거
    """
    if include_tags is None and exclude_tags is None:
        return df

    df_ = df.copy()
    is_element = df_["image_type"] == "element"
    tags = df_.loc[is_element, "tag"].astype(str).fillna("")

    if include_tags:
        mask_inc = tags.isin(include_tags)
    else:
        mask_inc = pd.Series(True, index=tags.index)

    if exclude_tags:
        mask_exc = ~tags.isin(exclude_tags)
    else:
        mask_exc = pd.Series(True, index=tags.index)

    keep_mask = ~is_element | (mask_inc & mask_exc)
    return df_.loc[keep_mask]


# ---------------------------
# 엔트리포인트
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Silhouette Generator v1 (refactored)")
    ap.add_argument("--csv", required=True, help="입력 CSV 경로 (template_info.csv)")
    ap.add_argument("--output", required=True, help="출력 디렉토리")
    ap.add_argument(
        "--type",
        default="card_news",
        help="template_type 필터 (예: card_news, presentation)",
    )
    ap.add_argument(
        "--include-tags",
        default=None,
        help="포함할 태그(콤마 구분, 예: TEXT,PHOTO). 지정 시 해당 태그의 element만 그림",
    )
    ap.add_argument(
        "--exclude-tags",
        default=None,
        help="제외할 태그(콤마 구분, 예: GRID,Chart)",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    include_tags = parse_tag_set(args.include_tags)
    exclude_tags = parse_tag_set(args.exclude_tags)

    # CSV 로드 (mixed type warning 방지용 옵션 유지)
    df = pd.read_csv(csv_path, low_memory=False)

    # 타깃 템플릿 타입 필터
    df_target = df[df["template_type"] == args.type].copy()
    if df_target.empty:
        print(f"[WARN] template_type='{args.type}' 데이터가 없습니다.")
        return

    # (template_idx, page_num) 단위 그룹핑
    grouped = list(df_target.groupby(["template_idx", "page_num"]))

    saved = 0
    skipped = 0

    for (template_idx, page_num), gdf in tqdm(grouped, desc="Rendering", total=len(grouped)):
        # 태그 필터링 적용
        gdf = filter_elements_by_tags(gdf, include_tags, exclude_tags)

        img = draw_silhouette_for_group(gdf)
        if img is None:
            skipped += 1
            continue

        out_path = out_dir / f"{template_idx}_{page_num}.png"
        ok = cv2.imwrite(str(out_path), img)
        if ok:
            saved += 1
        else:
            skipped += 1

    print(f"[DONE] 저장: {saved}개, 건너뜀: {skipped}개, 출력경로: {out_dir}")


if __name__ == "__main__":
    main()
