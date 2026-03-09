"""train_loso.py — Leave-One-Subject-Out 교차 검증.

실행 예시:
  python train_loso.py
  python train_loso.py --models M6,ResNet1D,ResNetTCN
"""
from __future__ import annotations

import sys, gc, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch

from config import CFG, apply_overrides, print_config
from datasets import H5Data, make_branch_loaders, make_hierarchical_loaders
from channel_groups import build_branch_idx, get_foot_accel_idx
from features import batch_extract
from models import get_model_factories, MODEL_REGISTRY
from train_common import fit_model
from eval_utils import (save_report, save_cm, save_history,
                        save_summary_table, save_per_subject_heatmap)
from utils import seed_everything, log, ensure_dir, save_json, Timer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",     type=str, default=",".join(CFG.models_loso))
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
        num_subjects = args.n_subjects,
        epochs       = args.epochs,
        batch        = args.batch,
        seed         = args.seed,
        early_stop   = args.early_stop,
        use_balanced = not args.no_balanced or None,
        use_tta      = not args.no_tta      or None,
        use_focal    = not args.no_focal    or None,
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
    else:
        X_all, y_all = h5.X, y

    unique_subjects = np.unique(groups)
    log(f"LOSO 피험자 수: {len(unique_subjects)}")

    # ── 출력 디렉토리 ─────────────────────────
    out = Path(CFG.result_loso)
    ensure_dir(out, Path(CFG.result_tables))

    # ── LOSO ────────────────────────────────
    need_feat = any(getattr(MODEL_REGISTRY[n], "IS_HYBRID", False)
                    or getattr(MODEL_REGISTRY[n](branch_ch), "IS_HYBRID", False)
                    for n, _ in models)

    all_preds:  dict[str, list] = {}
    all_hist:   dict[str, list] = {}
    all_labels: list[int]       = []
    per_subj:   dict[int, dict] = {}
    timer = Timer()

    for si, sid in enumerate(unique_subjects, 1):
        tr_idx = np.where(groups != sid)[0]
        te_idx = np.where(groups == sid)[0]
        log(f"\n{'='*60}")
        log(f"  LOSO [{si}/{len(unique_subjects)}]  Subject={sid}  "
            f"tr={len(tr_idx)}  te={len(te_idx)}")

        # 44-feat
        feat44_tr = feat44_te = None
        if need_feat:
            feat44_tr = batch_extract(X_all[tr_idx], foot_idx, CFG.sample_rate)
            feat44_te = batch_extract(X_all[te_idx], foot_idx, CFG.sample_rate)

        fold_preds: dict[str, np.ndarray] = {}

        for mname, mfn in models:
            log(f"── {mname}")
            model     = mfn(branch_ch)
            is_hybrid = getattr(model, "IS_HYBRID", False)

            if is_hybrid:
                tr_loader, te_loader = make_hierarchical_loaders(
                    X_all[tr_idx], feat44_tr, y_all[tr_idx],
                    X_all[te_idx], feat44_te, y_all[te_idx],
                    branch_idx, batch=CFG.batch, balanced=CFG.use_balanced,
                )
            else:
                tr_loader, te_loader = make_branch_loaders(
                    X_all[tr_idx], y_all[tr_idx],
                    X_all[te_idx], y_all[te_idx],
                    branch_idx, batch=CFG.batch, balanced=CFG.use_balanced,
                )

            preds, labels, hist = fit_model(model, tr_loader, te_loader, device, si, mname)
            all_preds.setdefault(mname, []).extend(preds.tolist())
            all_hist.setdefault(mname,  []).append(hist)
            fold_preds[mname] = preds
            del model, tr_loader, te_loader
            gc.collect()

        all_labels.extend(labels.tolist())
        per_subj[int(sid)] = {
            t: round(float(accuracy_score(labels, p)), 4)
            for t, p in fold_preds.items()
        }
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 최종 평가 & 저장 ─────────────────────
    label_arr = np.array(all_labels)
    results   = {}
    for tag, preds in all_preds.items():
        pred_arr     = np.array(preds)
        acc, f1      = save_report(pred_arr, label_arr, classes, f"LOSO_{tag}", out)
        results[tag] = (acc, f1)
        save_cm(pred_arr, label_arr, classes, f"LOSO_{tag}", out)
        log(f"  {tag:<20} Acc={acc:.4f}  F1={f1:.4f}")

    save_history(all_hist, out)
    save_summary_table(results, Path(CFG.result_tables), "loso_summary.csv")
    save_per_subject_heatmap(per_subj, results, out)

    elapsed = timer.elapsed_str()
    save_json({
        "experiment": "loso",
        "models": [n for n, _ in models],
        "total_time": elapsed,
        "results": {t: {"acc": round(a,4), "f1": round(f,4)} for t,(a,f) in results.items()},
    }, out / "summary_loso.json")

    log(f"\n★ LOSO 완료  총 소요: {elapsed}")
    h5.close()


if __name__ == "__main__":
    main()