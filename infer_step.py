"""infer_step.py — 단일 스텝 / 파일 단위 추론.

실행 예시:
  python infer_step.py --model_path experiments/kfold/best_ResNetTCN.pt \
                       --model_name ResNetTCN \
                       --input data/sample_step.npy
"""
from __future__ import annotations

import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch

from config import CFG
from channel_groups import build_branch_idx, get_foot_accel_idx
from features import batch_extract
from models import MODEL_REGISTRY
from utils import log, move_bi


CLASS_NAMES = ["flat", "grass", "gravel", "slope_down", "slope_up", "stair"]  # 실제 레이블 순서로 교체


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,  help="저장된 .pt 파일 경로")
    p.add_argument("--model_name", required=True,  help="모델 키 (예: ResNetTCN, M7)")
    p.add_argument("--input",      required=True,  help="입력 numpy (.npy) 파일 경로 (N,C,T) or (C,T)")
    p.add_argument("--topk",       type=int, default=3)
    return p.parse_args()


@torch.no_grad()
def infer(model, X: np.ndarray, branch_idx, branch_ch, device, is_hybrid=False):
    """(N, C, T) → (N,) predicted class indices + (N, n_cls) probabilities."""
    if X.ndim == 2:
        X = X[np.newaxis]  # (C,T) → (1,C,T)

    # 44-feat (hybrid 모델만)
    feat44 = None
    if is_hybrid:
        foot_idx = get_foot_accel_idx(branch_idx)
        feat44   = torch.from_numpy(
            batch_extract(X, foot_idx, CFG.sample_rate)
        ).to(device)

    # 브랜치 분기
    bi = {
        nm: torch.from_numpy(X[:, idxs, :].astype(np.float32)).to(device)
        for nm, idxs in branch_idx.items()
    }

    model.eval()
    with torch.amp.autocast("cuda", enabled=(device != "cpu")):
        logits = model(bi, feat44) if is_hybrid else model(bi)

    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return preds, probs


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 로드
    if args.model_name not in MODEL_REGISTRY:
        print(f"Unknown model: {args.model_name}. Available: {list(MODEL_REGISTRY)}")
        sys.exit(1)

    branch_idx, branch_ch = build_branch_idx()
    model     = MODEL_REGISTRY[args.model_name](branch_ch).to(device)
    is_hybrid = getattr(model, "IS_HYBRID", False)

    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    log(f"모델 로드: {args.model_path}  ({args.model_name})")

    # 입력 로드
    X = np.load(args.input)   # (N,C,T) or (C,T)
    log(f"입력 shape: {X.shape}")

    preds, probs = infer(model, X, branch_idx, branch_ch, device, is_hybrid)

    for i, (pred, prob) in enumerate(zip(preds, probs)):
        topk_idx  = prob.argsort()[::-1][:args.topk]
        topk_info = "  ".join(
            f"{CLASS_NAMES[j] if j < len(CLASS_NAMES) else j}={prob[j]:.3f}"
            for j in topk_idx
        )
        log(f"[{i:04d}] pred={CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else pred}  | {topk_info}")


if __name__ == "__main__":
    main()