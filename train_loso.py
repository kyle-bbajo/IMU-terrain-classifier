"""train_loso.py — Leave-One-Subject-Out 교차 검증.

실행 예시:
  python train_loso.py
  python train_loso.py --models M6,ResNet1D,ResNetTCN
"""
from __future__ import annotations

import sys, gc, argparse, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from wandb_init import wandb_start, wandb_log_fold, wandb_finish

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch

from config import CFG, apply_overrides, print_config
import config as _cfg
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
    p.add_argument("--models",        type=str, default="M6,ResNet1D,ResNetTCN,M7")
    p.add_argument("--n_subjects",    type=int, default=None)
    p.add_argument("--epochs",        type=int, default=None)
    p.add_argument("--batch",         type=int, default=None)
    p.add_argument("--seed",          type=int, default=None)
    p.add_argument("--early_stop",    type=int, default=None)
    p.add_argument("--no-balanced",   action="store_true")
    p.add_argument("--no-tta",        action="store_true")
    p.add_argument("--no-focal",      action="store_true")
    p.add_argument("--no-feat-cache", action="store_true")
    p.add_argument("--run_name",      type=str, default=None)
    p.add_argument("--no-wandb",      action="store_true")
    return p.parse_args()


def _is_hybrid(mname, mfn, branch_ch):
    if getattr(mfn, "IS_HYBRID", False):
        return True
    try:
        return getattr(mfn(branch_ch), "IS_HYBRID", False)
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
    log(f"device={_cfg.DEVICE}  amp_dtype={_cfg.AMP_DTYPE}")

    model_list = get_model_factories([m.strip() for m in args.models.split(",") if m.strip()])
    log(f"models: {[n for n, _ in model_list]}")

    # ── W&B 초기화 ──────────────────────────────────────────────
    if not args.no_wandb:
        wandb_start("loso", args, cfg_dict={
            "n_features":   N_FEATURES,
            "epochs":       CFG.epochs,
            "batch":        CFG.batch,
            "lr":           CFG.lr,
            "early_stop":   CFG.early_stop,
            "seed":         CFG.seed,
            "focal_loss":   CFG.use_focal_loss,
            "label_smooth": CFG.label_smooth,
            "use_tta":      CFG.use_tta,
        })

    # ── 데이터 로드 ──────────────────────────────────────────────
    h5     = H5Data(CFG.h5_path)
    le     = LabelEncoder()
    y      = le.fit_transform(h5.y_raw).astype(np.int64)
    groups = h5.subj_id

    branch_idx, branch_ch = build_branch_idx(h5.channels)
    foot_idx = get_foot_accel_idx(h5.channels)

    if args.n_subjects:
        keep   = np.unique(groups)[:args.n_subjects]
        mask   = np.isin(groups, keep)
        X_all  = h5.X[mask]
        y_all  = y[mask]
        groups = groups[mask]
    else:
        X_all = h5.X
        y_all = y

    unique_subjects = np.unique(groups)
    log(f"LOSO 피험자 수: {len(unique_subjects)}  N={len(y_all)}")

    # ── 출력 디렉토리 ────────────────────────────────────────────
    out = ensure_dir(CFG.result_loso)
    ensure_dir(out.parent / "tables")

    # ── feat 전체 1회 추출 ───────────────────────────────────────
    need_feat = any(_is_hybrid(n, fn, branch_ch) for n, fn in model_list)
    use_cache = need_feat and (not args.no_feat_cache)
    cache_dir = ensure_dir(
        CFG.repo_dir / "cache" / f"feat{N_FEATURES}_seed{CFG.seed}_loso"
    )

    feat_all = None
    if need_feat:
        cache_path = cache_dir / "all_feat.npy"
        if use_cache and cache_path.exists():
            log(f"[feat] ★ 캐시 히트 → 로드 중...")
            with Timer() as t:
                feat_all = np.load(cache_path)
            log(f"[feat] 로드 완료  shape={feat_all.shape}  ({t})")
        else:
            log(f"[feat] {N_FEATURES}-feat 추출 시작...")
            with Timer() as t:
                feat_all = batch_extract(X_all, foot_idx, CFG.sample_rate)
            if use_cache:
                np.save(cache_path, feat_all)
            log(f"[feat] 추출 완료  shape={feat_all.shape}  ({t})")

    # ── LOSO ────────────────────────────────────────────────────
    all_preds:  dict[str, list] = {}
    all_labels: list[int]       = []
    all_hist:   dict[str, list] = {}
    per_subj:   dict[int, dict] = {}

    with Timer() as total_timer:
        for si, sid in enumerate(unique_subjects, 1):
            tr_idx = np.where(groups != sid)[0]
            te_idx = np.where(groups == sid)[0]

            log(f"\n{'='*60}")
            log(f"  LOSO [{si}/{len(unique_subjects)}]  Subject={sid}  "
                f"tr={len(tr_idx)}  te={len(te_idx)}")

            feat_tr = feat_all[tr_idx] if feat_all is not None else None
            feat_te = feat_all[te_idx] if feat_all is not None else None

            fold_preds: dict[str, np.ndarray] = {}

            for mname, mfn in model_list:
                log(f"── [{si}] {mname}")
                model     = mfn(branch_ch)
                is_hybrid = getattr(model, "IS_HYBRID", False)
                model     = model.to(str(_cfg.DEVICE))

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
                    branch    = True,
                    tag       = f"[S{si}][{mname}]",
                    use_mixup = True,
                )

                all_preds.setdefault(mname, []).extend(preds.tolist())
                all_hist.setdefault(mname,  []).append(hist)
                fold_preds[mname] = preds

                del model, tr_loader, te_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_labels.extend(labels.tolist())
            per_subj[int(sid)] = {
                mname: round(float(accuracy_score(labels, p)), 4)
                for mname, p in fold_preds.items()
            }

            # subject 단위 W&B 로그
            wandb_log_fold(si, {
                mname: round(float(accuracy_score(labels, p)), 4)
                for mname, p in fold_preds.items()
            })

            gc.collect()

    # ── 최종 평가 & 저장 ─────────────────────────────────────────
    results = []
    for mname, preds in all_preds.items():
        pred_arr = np.array(preds)
        lbl_arr  = np.array(all_labels[:len(pred_arr)])
        acc, f1  = save_report(pred_arr, lbl_arr, le, f"LOSO_{mname}", out)
        save_cm(pred_arr, lbl_arr, le, f"LOSO_{mname}", out)
        results.append({"model": mname, "acc": round(acc, 4), "f1": round(f1, 4)})
        log(f"  {mname:<20} Acc={acc:.4f}  F1={f1:.4f}")

    save_history(all_hist, out)
    save_summary_table(results, out.parent / "tables")

    save_json({
        "experiment":  "loso",
        "models":      [n for n, _ in model_list],
        "n_features":  N_FEATURES,
        "total_time":  str(total_timer),
        "per_subject": per_subj,
        "results":     {r["model"]: {"acc": r["acc"], "f1": r["f1"]} for r in results},
    }, out / "summary_loso.json")

    log(f"\n★ LOSO 완료  총 소요: {total_timer}")

    if not args.no_wandb:
        wandb_finish(results=results)

    h5.close()


if __name__ == "__main__":
    main()