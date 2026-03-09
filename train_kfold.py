"""train_kfold.py — 5-Fold 교차 검증 (M2/M4/M6/ResNet1D/CNNTCN/ResNetTCN/M7).

실행 예시:
  python train_kfold.py
  python train_kfold.py --models M4,ResNet1D,ResNetTCN
  python train_kfold.py --models M7 --epochs 80
"""
from __future__ import annotations

import sys, gc, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
import torch

from config import CFG, apply_overrides, print_config
from datasets import make_branch_loaders, make_hierarchical_loaders
from train_common import (
    H5Data,
    fit_model, save_report, save_cm, save_history, save_summary_table,
    seed_everything, log, ensure_dir, save_json, Timer,
)
from channel_groups import build_branch_idx, get_foot_accel_idx
from features import batch_extract, N_FEATURES
from models import get_model_factories, MODEL_REGISTRY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",      type=str, default="M2,M4,M6,ResNet1D,CNNTCN,ResNetTCN,M7")
    p.add_argument("--n_subjects",  type=int, default=None)
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--batch",       type=int, default=None)
    p.add_argument("--seed",        type=int, default=None)
    p.add_argument("--early_stop",  type=int, default=None)
    p.add_argument("--no-balanced", action="store_true")
    p.add_argument("--no-tta",      action="store_true")
    p.add_argument("--no-focal",    action="store_true")
    return p.parse_args()


def _is_hybrid(mname: str, mfn, branch_ch: dict) -> bool:
    """모델이 IS_HYBRID 인지 확인."""
    # 팩토리 함수에 IS_HYBRID 플래그가 있는 경우
    if getattr(mfn, "IS_HYBRID", False):
        return True
    # 인스턴스를 생성해서 확인
    try:
        m = mfn(branch_ch)
        return getattr(m, "IS_HYBRID", False)
    except Exception:
        return False


def main():
    args = parse_args()
    apply_overrides(
        n_subjects = args.n_subjects,
        epochs     = args.epochs,
        batch      = args.batch,
        seed       = args.seed,
        balanced   = (not args.no_balanced) or None,
        tta        = (not args.no_tta)      or None,
        focal      = (not args.no_focal)    or None,
    )
    seed_everything(CFG.seed)
    print_config()

    import config as _cfg
    log(f"device={_cfg.DEVICE}  amp_dtype={_cfg.AMP_DTYPE}")

    model_list = get_model_factories([m.strip() for m in args.models.split(",") if m.strip()])
    log(f"models: {[n for n, _ in model_list]}")

    # ── 데이터 로드 ──────────────────────────────────────────────
    h5     = H5Data(CFG.h5_path)
    le     = LabelEncoder()
    y      = le.fit_transform(h5.y_raw).astype(np.int64)
    groups = h5.subj_id

    branch_idx, branch_ch = build_branch_idx(h5.channels)
    foot_idx = get_foot_accel_idx(h5.channels)   # ← list[str] 전달

    if args.n_subjects:
        keep  = np.unique(groups)[:args.n_subjects]
        mask  = np.isin(groups, keep)
        X_all = h5.X[mask]
        y_all = y[mask]
        groups = groups[mask]
        log(f"피험자 {len(keep)}명  N={mask.sum()}")
    else:
        X_all = h5.X
        y_all = y
        log(f"피험자 {len(np.unique(groups))}명  N={len(y_all)}")

    # ── 출력 디렉토리 ────────────────────────────────────────────
    out = ensure_dir(CFG.result_kfold)
    ensure_dir(out.parent / "tables")

    # hybrid 모델 포함 여부 사전 확인
    need_feat = any(_is_hybrid(n, fn, branch_ch) for n, fn in model_list)
    if need_feat:
        log(f"Hybrid 모델 포함 → fold마다 {N_FEATURES}-feat 추출 예정")

    # ── K-Fold ───────────────────────────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=CFG.kfold, shuffle=True, random_state=CFG.seed)

    all_preds:  dict[str, list] = {}
    all_labels: list[int]       = []
    all_hist:   dict[str, list] = {}

    with Timer() as total_timer:
        for fi, (tr_idx, te_idx) in enumerate(
            sgkf.split(np.zeros(len(y_all)), y_all, groups), 1
        ):
            log(f"\n{'='*60}")
            log(f"  Fold {fi}/{CFG.kfold}  tr={len(tr_idx)}  te={len(te_idx)}")

            # feat 추출 (hybrid 모델이 있을 때만)
            feat_tr = feat_te = None
            if need_feat:
                log(f"  [{fi}/{CFG.kfold}] {N_FEATURES}-feat 추출 시작 ...")
                with Timer() as ft:
                    feat_tr = batch_extract(X_all[tr_idx], foot_idx, CFG.sample_rate)
                    feat_te = batch_extract(X_all[te_idx], foot_idx, CFG.sample_rate)
                log(f"  [{fi}/{CFG.kfold}] feat 추출 완료  "
                    f"tr={feat_tr.shape}  te={feat_te.shape}  ({ft})")

            # ── 모델별 학습 ─────────────────────────────────────
            for mname, mfn in model_list:
                log(f"\n── [{fi}/{CFG.kfold}] {mname}")

                model      = mfn(branch_ch)
                is_hybrid  = getattr(model, "IS_HYBRID", False)
                model      = model.to(str(_cfg.DEVICE))

                if is_hybrid:
                    tr_loader, te_loader = make_hierarchical_loaders(
                        X_all[tr_idx], feat_tr, y_all[tr_idx],
                        X_all[te_idx], feat_te, y_all[te_idx],
                        branch_idx,
                        batch    = CFG.batch,
                        balanced = CFG.use_balanced_sampler,
                    )
                else:
                    tr_loader, te_loader = make_branch_loaders(
                        X_all[tr_idx], y_all[tr_idx],
                        X_all[te_idx], y_all[te_idx],
                        branch_idx,
                        batch    = CFG.batch,
                        balanced = CFG.use_balanced_sampler,
                    )

                preds, labels, hist = fit_model(
                    model, tr_loader, te_loader,
                    branch   = True,
                    tag      = f"[F{fi}][{mname}]",
                    use_mixup= True,
                )

                all_preds.setdefault(mname, []).extend(preds.tolist())
                all_hist.setdefault(mname, []).append(hist)
                all_labels.extend(labels.tolist())

                del model, tr_loader, te_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            gc.collect()

    # ── 최종 평가 & 저장 ─────────────────────────────────────────
    label_arr = np.array(all_labels[:len(list(all_preds.values())[0])])
    results   = []

    for mname, preds in all_preds.items():
        # fold 수 × te 크기가 all_labels 길이와 다를 수 있으므로 길이 맞춤
        pred_arr = np.array(preds)
        lbl_arr  = np.array(all_labels[:len(pred_arr)])
        acc, f1  = save_report(pred_arr, lbl_arr, le, mname, out)
        save_cm(pred_arr, lbl_arr, le, mname, out)
        results.append({"model": mname, "acc": round(acc, 4), "f1": round(f1, 4)})
        log(f"  {mname:<20} Acc={acc:.4f}  F1={f1:.4f}")

    save_history(all_hist, out)
    save_summary_table(results, out.parent / "tables")

    summary = {
        "experiment":  "kfold",
        "models":      [n for n, _ in model_list],
        "n_features":  N_FEATURES,
        "total_time":  str(total_timer),
        "results":     {r["model"]: {"acc": r["acc"], "f1": r["f1"]} for r in results},
    }
    save_json(summary, out / "summary_kfold.json")
    log(f"\n★ K-Fold 완료  총 소요: {total_timer}")

    h5.close()


if __name__ == "__main__":
    main()