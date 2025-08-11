#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare Silhouette Versions (v1~v5) with DreamSim
- 버전별 실루엣(v1~v5) 임베딩 → 코사인 유사도 비교 → 5행 그리드 저장
- 글로벌 변수로 경로/배치/Top-K 등 설정
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import torch
import torch.nn.functional as F
from dreamsim import dreamsim
import matplotlib.pyplot as plt

# ============ 글로벌 설정 ============
CSV_PATH        = "template_info.csv"            # 입력 CSV
TEMPLATE_TYPE   = "card_news"                    # 템플릿 타입 필터
IMG_COL         = "full_image_path"              # 원본 썸네일 경로 컬럼

# 실루엣 경로들
PATH_V1 = "/data/shared2/jjkim/margin_total"
PATH_V2 = "/data/shared2/jjkim/margin_v2"
PATH_V3 = "/data/shared2/jjkim/margin_v3"
PATH_V4 = "/data/shared2/jjkim/margin_v4"
PATH_V5 = "/data/shared2/jjkim/margin_v5"

SAVE_DIR        = "grid_results_v12345"          # 그리드 저장 디렉토리
BATCH_SIZE      = 256                             # 임베딩 배치
MAX_QUERIES     = 50                              # 랜덤 샘플 수
TOP_K           = 5                               # 각 버전 Top-K
MAX_WORKERS     = 8                               # 경로 수집 병렬 워커
RANDOM_SEED     = 2025                            # 재현성
# ====================================


# ============ DreamSim 모델 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = dreamsim(pretrained=True, device=device)
model = model.to(device).eval()
torch.set_grad_enabled(False)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ============ 경로 수집 유틸 ============
def exists_png(p: str) -> bool:
    try:
        return os.path.exists(p) and p.lower().endswith(".png")
    except Exception:
        return False

def _make_task(name: str, p1: str, p2: str, p3: str, p4: str, p5: str):
    return (name, p1, p2, p3, p4, p5)

def _check_exist(task: Tuple[str, str, str, str, str, str]):
    name, p1, p2, p3, p4, p5 = task
    return (
        name,
        p1 if exists_png(p1) else None,
        p2 if exists_png(p2) else None,
        p3 if exists_png(p3) else None,
        p4 if exists_png(p4) else None,
        p5 if exists_png(p5) else None,
    )

def collect_version_paths(names: List[str]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    tasks = []
    for name in names:
        tasks.append(_make_task(
            name,
            os.path.join(PATH_V1, f"{name}.png"),
            os.path.join(PATH_V2, f"{name}.png"),
            os.path.join(PATH_V3, f"{name}.png"),
            os.path.join(PATH_V4, f"{name}.png"),
            os.path.join(PATH_V5, f"{name}.png"),
        ))

    s2v1, s2v2, s2v3, s2v4, s2v5 = {}, {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, multiprocessing.cpu_count())) as ex:
        for name, p1, p2, p3, p4, p5 in tqdm(ex.map(_check_exist, tasks), total=len(tasks), desc="Checking silhouette paths"):
            if p1: s2v1[name] = p1
            if p2: s2v2[name] = p2
            if p3: s2v3[name] = p3
            if p4: s2v4[name] = p4
            if p5: s2v5[name] = p5
    return s2v1, s2v2, s2v3, s2v4, s2v5


# ============ 임베딩 ============
def embed_paths(name2path: Dict[str, str], batch_size: int = BATCH_SIZE) -> Dict[str, torch.Tensor]:
    """
    name->image path 딕셔너리 입력 → DreamSim 임베딩 (L2 정규화) 딕셔너리 반환
    """
    names = list(name2path.keys())
    feats: Dict[str, torch.Tensor] = {}

    batch_imgs: List[torch.Tensor] = []
    batch_names: List[str] = []

    pbar = tqdm(names, desc="Embedding images", total=len(names))
    for n in pbar:
        p = name2path[n]
        try:
            img = Image.open(p).convert("RGB")
            tensor = preprocess(img).unsqueeze(0)  # DreamSim 권장 전처리
        except Exception as e:
            print(f"[스킵] {n}: 로드/전처리 실패 - {e}")
            continue

        batch_imgs.append(tensor)
        batch_names.append(n)

        if len(batch_imgs) == batch_size:
            _flush_batch(batch_imgs, batch_names, feats)

    if batch_imgs:
        _flush_batch(batch_imgs, batch_names, feats)

    return feats

def _flush_batch(batch_imgs: List[torch.Tensor], batch_names: List[str], feats_out: Dict[str, torch.Tensor]):
    batch = torch.cat(batch_imgs).to(device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        vec = model.embed(batch)
    vec = F.normalize(vec, p=2, dim=1)
    for n, f in zip(batch_names, vec):
        feats_out[n] = f.detach().cpu()
    batch_imgs.clear()
    batch_names.clear()


# ============ 유사도 ============
def cosine_topk(feats: torch.Tensor, idx: int, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    정규화된 행렬 feats [N, D]에서 쿼리 idx에 대한 코사인 유사도 Top-K
    (정규화되어 있으므로 dot = cosine).
    """
    q = feats[idx:idx+1]                 # [1, D]
    sims = feats @ q.T                   # [N, 1]
    sims = sims.squeeze(1)               # [N]
    sims[idx] = -1.0                     # 자기 자신 제외
    vals, inds = torch.topk(sims, top_k)
    return inds, sims  # indices, full similarity vector


# ============ 그리드 시각화 ============
def concat_orig_and_silhouette(name: str, version: str, sample2orig: Dict[str, str],
                               path_v1: str, path_v2: str, path_v3: str, path_v4: str, path_v5: str) -> torch.Tensor:
    from torchvision.transforms import ToTensor, Resize, Compose
    t = Compose([Resize((224, 224)), ToTensor()])

    orig_path = sample2orig[name]
    try:
        orig_t = t(Image.open(orig_path).convert("RGB"))
    except Exception as e:
        raise RuntimeError(f"[로드 실패] 원본: {orig_path} ({e})")

    sil_map = {
        "v1": os.path.join(path_v1, f"{name}.png"),
        "v2": os.path.join(path_v2, f"{name}.png"),
        "v3": os.path.join(path_v3, f"{name}.png"),
        "v4": os.path.join(path_v4, f"{name}.png"),
        "v5": os.path.join(path_v5, f"{name}.png"),
    }
    sil_path = sil_map[version]
    try:
        sil_t = t(Image.open(sil_path).convert("RGB"))
    except Exception as e:
        raise RuntimeError(f"[로드 실패] 실루엣({version}): {sil_path} ({e})")

    return torch.cat([orig_t, sil_t], dim=2)  # [C, H, 2W]

def plot_grid(query: str,
              topk_ids: Dict[str, torch.Tensor],
              sims: Dict[str, torch.Tensor],
              sample_names: List[str],
              sample2orig: Dict[str, str],
              path_v1: str, path_v2: str, path_v3: str, path_v4: str, path_v5: str,
              save_dir: str = SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    def build_row(version: str):
        ids = topk_ids[version].tolist()
        names = [sample_names[i] for i in ids]
        tensors = [concat_orig_and_silhouette(n, version, sample2orig, path_v1, path_v2, path_v3, path_v4, path_v5)
                   for n in [query] + names]
        labels = ["Query"] + [f"{sims[version][sample_names.index(n)]:.3f}" for n in names]
        return tensors, labels

    rows = {}
    labels = {}
    for v in ["v1", "v2", "v3", "v4", "v5"]:
        rows[v], labels[v] = build_row(v)

    ncols = len(rows["v1"])
    fig, axs = plt.subplots(5, ncols, figsize=(2.8 * ncols, 12))
    for i, v in enumerate(["v1", "v2", "v3", "v4", "v5"]):
        for j, (img, lbl) in enumerate(zip(rows[v], labels[v])):
            axs[i, j].imshow(img.permute(1, 2, 0))
            axs[i, j].set_title(lbl, fontsize=10, color="black")
            axs[i, j].axis("off")
        axs[i, 0].set_ylabel(v.upper(), fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{query}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ 저장 완료: {out_path}")


# ============ 메인 ============
def main():
    random.seed(RANDOM_SEED)

    # 1) CSV 로드 및 템플릿/썸네일 필터
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["template_type"] == TEMPLATE_TYPE].copy()
    thumbs = df[df["image_type"] == "thumbnail"]

    # 샘플 이름 ↔ 원본 경로 매핑
    sample2orig = {
        f"{i}_{j}": g[IMG_COL].iloc[0]
        for (i, j), g in thumbs.groupby(["template_idx", "page_num"])
    }
    all_names = list(sample2orig.keys())
    if not all_names:
        print("[중단] 썸네일이 없습니다.")
        return

    # 2) 버전별 PNG 경로 존재 확인 (병렬)
    s2v1, s2v2, s2v3, s2v4, s2v5 = collect_version_paths(all_names)

    # 3) 공통 교집합 계산 (모든 버전에 존재 + 원본 존재)
    common = set(sample2orig) & set(s2v1) & set(s2v2) & set(s2v3) & set(s2v4) & set(s2v5)
    names = sorted(common)
    if not names:
        print("[중단] 모든 버전에 공통으로 존재하는 샘플이 없습니다.")
        return
    print(f"[INFO] 공통 샘플 수: {len(names)}")

    # 4) 공통만 임베딩 (버전별)
    emb_v1 = embed_paths({n: s2v1[n] for n in names}, BATCH_SIZE)
    emb_v2 = embed_paths({n: s2v2[n] for n in names}, BATCH_SIZE)
    emb_v3 = embed_paths({n: s2v3[n] for n in names}, BATCH_SIZE)
    emb_v4 = embed_paths({n: s2v4[n] for n in names}, BATCH_SIZE)
    emb_v5 = embed_paths({n: s2v5[n] for n in names}, BATCH_SIZE)

    # 5) 텐서 정렬
    feats_v1 = torch.stack([emb_v1[n] for n in names])
    feats_v2 = torch.stack([emb_v2[n] for n in names])
    feats_v3 = torch.stack([emb_v3[n] for n in names])
    feats_v4 = torch.stack([emb_v4[n] for n in names])
    feats_v5 = torch.stack([emb_v5[n] for n in names])

    # 6) 질의 샘플 선택
    num_queries = min(MAX_QUERIES, len(names))
    queries = random.sample(names, num_queries)

    # 7) 질의별 버전 Top-K 계산 및 그리드 저장
    for q in tqdm(queries, desc="Query comparisons"):
        idx = names.index(q)

        inds_v1, sims_v1 = cosine_topk(feats_v1, idx, TOP_K)
        inds_v2, sims_v2 = cosine_topk(feats_v2, idx, TOP_K)
        inds_v3, sims_v3 = cosine_topk(feats_v3, idx, TOP_K)
        inds_v4, sims_v4 = cosine_topk(feats_v4, idx, TOP_K)
        inds_v5, sims_v5 = cosine_topk(feats_v5, idx, TOP_K)

        plot_grid(
            query=q,
            topk_ids={"v1": inds_v1, "v2": inds_v2, "v3": inds_v3, "v4": inds_v4, "v5": inds_v5},
            sims={"v1": sims_v1, "v2": sims_v2, "v3": sims_v3, "v4": sims_v4, "v5": sims_v5},
            sample_names=names,
            sample2orig=sample2orig,
            path_v1=PATH_V1, path_v2=PATH_V2, path_v3=PATH_V3, path_v4=PATH_V4, path_v5=PATH_V5,
            save_dir=SAVE_DIR,
        )

if __name__ == "__main__":
    main()
