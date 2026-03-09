"""train_kfold.py — 5-Fold 교차 검증 (M2/M4/M6/ResNet1D/CNNTCN/ResNetTCN/M7).

실행 예시:
  python train_kfold.py
  python train_kfold.py --models M4,ResNet1D,ResNetTCN
  python train_kfold.py --models M7 --epochs 80
"""
from __future__ import annotations

import sys, gc, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
import torch

from config import CFG, apply_overrides, print_config
from datasets import H5Data, make_branch_loaders, make_hierarchical_loaders
from channel_groups import build_branch_idx, get_foot_accel_idx
from features import batch_extract, N_FEATURES
from models import get_model_factories, MODEL_REGISTRY
from train_common import fit_model
from train_common import save_report, save_cm, save_history
from train_common import seed_everything, log, ensure_dir, save_json, Timer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",     type=str, default="M2,M4,M6,ResNet1D,CNNTCN,ResNetTCN,M7")
    p.add_argument("--n_subjects", type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch",      type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--early_stop", type=int, default=None)
    p.add_argument("--no-balanced", action="store_true")
    p.add_argument("--no-tta",      action="store_true")
    p.add_argument("--no-focal",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    apply_overrides(
        n_subjects = args.n_subjects,
        epochs       = args.epochs,
        batch        = args.batch,
        seed         = args.seed,
        
        balanced = not args.no_balanced or None,
        tta      = not args.no_tta      or None,
        focal    = not args.no_focal    or None,
    )
    seed_everything(CFG.seed)
    print_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

    models = get_model_factories([m.strip() for m in args.models.split(",") if m.strip()])
    log(f"models: {[n for n,_ in models]}")

    # ── 데이터 로드 ──────────────────────────
    h5       = H5Data(CFG.h5_path)
    le       = LabelEncoder()
    y        = le.fit_transform(h5.y_raw).astype(np.int64)
    classes  = list(le.classes_)
    branch_idx, branch_ch = build_branch_idx(h5.channels)
    foot_idx = get_foot_accel_idx(branch_idx)
    groups   = h5.subj_id

    if args.n_subjects:
        keep = np.unique(groups)[:args.n_subjects]
        mask = np.isin(groups, keep)
        X_all, y_all, groups = h5.X[mask], y[mask], groups[mask]
        log(f"피험자 {len(keep)}명  N={mask.sum()}")
    else:
        X_all, y_all = h5.X, y
        log(f"피험자 {len(np.unique(groups))}명  N={len(y_all)}")

    # ── 출력 디렉토리 ─────────────────────────
    out = Path(CFG.result_kfold)
    ensure_dir(out)
    ensure_dir(out.parent / "tables")

    # ── K-Fold ───────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=CFG.kfold, shuffle=True, random_state=CFG.seed)

    all_preds:  dict[str, list] = {}
    all_hist:   dict[str, list] = {}
    all_labels: list[int]       = []
    timer = Timer()

    for fi, (tr_idx, te_idx) in enumerate(sgkf.split(np.zeros(len(y_all)), y_all, groups), 1):
        log(f"\n{'='*60}")
        log(f"  Fold {fi}/{CFG.kfold}  tr={len(tr_idx)}  te={len(te_idx)}")

        # feat 추출 (hybrid 모델 있으면)
        need_feat = any(getattr(MODEL_REGISTRY[n], "IS_HYBRID", False)
                        or (hasattr(MODEL_REGISTRY[n], "__call__") and
                            getattr(MODEL_REGISTRY[n](branch_ch), "IS_HYBRID", False))
                        for n, _ in models)
        feat_tr = feat_te = None
        if need_feat:
            log(f"  {N_FEATURES}-feat 추출 중...")
            feat_tr = batch_extract(X_all[tr_idx], foot_idx, CFG.sample_rate)
            feat_te = batch_extract(X_all[te_idx], foot_idx, CFG.sample_rate)

        for mname, mfn in models:
            log(f"── {mname}")
            model     = mfn(branch_ch)
            is_hybrid = getattr(model, "IS_HYBRID", False)

            if is_hybrid:
                tr_loader, te_loader = make_hierarchical_loaders(
                    X_all[tr_idx], feat_tr, y_all[tr_idx],
                    X_all[te_idx], feat_te, y_all[te_idx],
                    branch_idx, batch=CFG.batch, balanced=CFG.use_balanced_sampler,
                )
            else:
                tr_loader, te_loader = make_branch_loaders(
                    X_all[tr_idx], y_all[tr_idx],
                    X_all[te_idx], y_all[te_idx],
                    branch_idx, batch=CFG.batch, balanced=CFG.use_balanced_sampler,
                )

            preds, labels, hist = fit_model(model, tr_loader, te_loader, device, fi, mname)

            tag = mname
            all_preds.setdefault(tag, []).extend(preds.tolist())
            all_hist.setdefault(tag, []).append(hist)
            del model, tr_loader, te_loader
            gc.collect()

        if fi == 1:
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(labels.tolist())

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 최종 평가 & 저장 ─────────────────────
    label_arr = np.array(all_labels)
    results   = {}
    for tag, preds in all_preds.items():
        pred_arr      = np.array(preds)
        acc, f1       = save_report(pred_arr, label_arr, classes, tag, out)
        results[tag]  = (acc, f1)
        save_cm(pred_arr, label_arr, classes, tag, out)
        log(f"  {tag:<20} Acc={acc:.4f}  F1={f1:.4f}")

    save_history(all_hist, out)
    save_summary_table(results, out.parent / "tables", "kfold_summary.csv")

    elapsed = timer.elapsed_str()
    save_json({
        "experiment": "kfold",
        "models": [n for n, _ in models],
        "total_time": elapsed,
        "results": {t: {"acc": round(a,4), "f1": round(f,4)} for t,(a,f) in results.items()},
    }, out / "summary_kfold.json")

    log(f"\n★ K-Fold 완료  총 소요: {elapsed}")
    h5.close()


if __name__ == "__main__":
    main()