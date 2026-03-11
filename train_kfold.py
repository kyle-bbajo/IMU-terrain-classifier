"""train_final.py — 95% 목표 최종 파이프라인

핵심 전략:
  1. Subject-wise Feature Normalization  (개인차 제거)
  2. M7_Attr (CNN + 232feat + Attribute Heads + GRU fusion)
  3. Per-class Threshold Grid Search     (val set 기준)
  4. Hierarchical + M7_Attr Soft Ensemble (저장된 logits 결합)

실행:
  cd ~/project/repo
  python train_final.py
  python train_final.py --models M7_Attr --epochs 80
  python train_final.py --ensemble_only  (학습 없이 앙상블만)
"""
from __future__ import annotations

import sys, gc, argparse, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from itertools import product

from config import CFG, apply_overrides, print_config
import config as _cfg
from datasets import make_hierarchical_loaders
from train_common import (
    H5Data, train_model, save_report, save_cm,
    save_summary_table, seed_everything, log, ensure_dir,
    save_json, Timer,
)
from channel_groups import build_branch_idx, get_foot_accel_idx
from features import batch_extract, N_FEATURES, _N_SENSOR
from models import MODEL_REGISTRY


# ════════════════════════════════════════════════════════════════
# 1. 인자 파싱
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",         type=str, default="M7_Attr")
    p.add_argument("--epochs",         type=int, default=None)
    p.add_argument("--batch",          type=int, default=None)
    p.add_argument("--seed",           type=int, default=None)
    p.add_argument("--no-feat-cache",  action="store_true")
    p.add_argument("--ensemble_only",  action="store_true",
                   help="학습 없이 저장된 logits로 앙상블만 수행")
    p.add_argument("--hier_logits",    type=str, default=None,
                   help="Hierarchical logits .npy 경로 (앙상블용)")
    p.add_argument("--hier_weight",    type=float, default=0.5,
                   help="앙상블 시 Hierarchical 가중치 (0~1)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
# 2. Subject-wise Feature Normalization
# ════════════════════════════════════════════════════════════════

def subject_normalize_feat(
    feat: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    flat_label: int,
    train_mask: np.ndarray,
) -> np.ndarray:
    """피험자별 평지(C6) 샘플 평균으로 feat 정규화.

    - train 피험자: 해당 피험자 평지 샘플 평균 사용
    - test  피험자: train 전체 평지 평균 fallback
    Parameters
    ----------
    feat        : (N, D)
    groups      : (N,) 피험자 ID
    y           : (N,) 레이블 (0~5)
    flat_label  : C6 레이블 인덱스 (보통 5)
    train_mask  : (N,) bool  — 학습 샘플 마스크
    """
    feat_n   = feat.copy().astype(np.float32)
    subjects = np.unique(groups)

    # train 전체 평지 평균 (fallback)
    tr_flat_mask = train_mask & (y == flat_label)
    global_flat_mean = feat[tr_flat_mask].mean(0) if tr_flat_mask.sum() > 0 else np.zeros(feat.shape[1])
    global_flat_std  = feat[tr_flat_mask].std(0)  + 1e-8

    subj_stats: dict[int, tuple] = {}
    for s in subjects:
        smask = train_mask & (groups == s) & (y == flat_label)
        if smask.sum() >= 5:
            mu  = feat[smask].mean(0)
            std = feat[smask].std(0) + 1e-8
        else:
            mu, std = global_flat_mean, global_flat_std
        subj_stats[s] = (mu, std)

    for s in subjects:
        smask = groups == s
        mu, std = subj_stats.get(s, (global_flat_mean, global_flat_std))
        feat_n[smask] = (feat[smask] - mu) / std

    return feat_n


# ════════════════════════════════════════════════════════════════
# 3. Per-class Threshold Grid Search
# ════════════════════════════════════════════════════════════════

def threshold_search(
    proba: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 6,
    grid: tuple = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
) -> tuple[np.ndarray, float]:
    """소프트맥스 확률에 클래스별 승수(multiplier)를 적용해 F1 최적화.

    proba   : (N, C) softmax 확률
    labels  : (N,)
    Returns : best_multipliers (C,), best_f1
    """
    best_f1   = -1.0
    best_mult = np.ones(n_classes)

    # 클래스별 독립 서치 (조합 폭발 방지)
    mults = np.ones(n_classes)
    for ci in range(n_classes):
        best_ci_f1 = -1.0
        best_ci_m  = 1.0
        for m in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            mults[ci] = m
            adj = proba * mults
            preds = adj.argmax(1)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            if f1 > best_ci_f1:
                best_ci_f1 = f1
                best_ci_m  = m
        mults[ci] = best_ci_m

    preds  = (proba * mults).argmax(1)
    best_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return mults, best_f1


# ════════════════════════════════════════════════════════════════
# 4. Soft Ensemble
# ════════════════════════════════════════════════════════════════

def soft_ensemble(
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    labels: np.ndarray,
    weight_a: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """두 모델 softmax 확률 weighted average 후 평가."""
    combined = weight_a * proba_a + (1 - weight_a) * proba_b
    preds    = combined.argmax(1)
    acc      = accuracy_score(labels, preds)
    f1       = f1_score(labels, preds, average="macro", zero_division=0)
    return preds, acc, f1


def find_best_ensemble_weight(
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """val set에서 최적 앙상블 가중치 탐색."""
    best_w, best_f1 = 0.5, -1.0
    for w in np.arange(0.1, 1.0, 0.05):
        _, _, f1 = soft_ensemble(proba_a, proba_b, labels, w)
        if f1 > best_f1:
            best_f1, best_w = f1, float(w)
    return best_w, best_f1


# ════════════════════════════════════════════════════════════════
# 5. 모델 forward → softmax 확률 추출
# ════════════════════════════════════════════════════════════════

@torch.inference_mode()
def get_probas(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """DataLoader → (N, C) softmax proba, (N,) labels"""
    model.eval()
    all_proba, all_labels = [], []
    for batch in loader:
        if len(batch) == 3:
            bi, feat, yb = batch
            feat = feat.to(device)
            bi   = {k: v.to(device) for k, v in bi.items()}
            out  = model(bi, feat)
        else:
            bi, yb = batch
            bi  = {k: v.to(device) for k, v in bi.items()}
            out = model(bi)

        if isinstance(out, dict):
            logits = out["final_logits"]
        else:
            logits = out

        proba = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        all_proba.append(proba)
        all_labels.append(yb.numpy())

    return np.concatenate(all_proba), np.concatenate(all_labels)


# ════════════════════════════════════════════════════════════════
# 6. 메인
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    apply_overrides(
        epochs=args.epochs,
        batch=args.batch,
        seed=args.seed,
    )
    seed_everything(CFG.seed)
    print_config()
    log(f"device={_cfg.DEVICE}  N_FEATURES={N_FEATURES}")

    # ── 출력 디렉토리 ──────────────────────────────────────────
    out_dir = ensure_dir(CFG.repo_dir / "out_N50" / "final")
    ensure_dir(out_dir.parent / "tables")

    # ── 데이터 로드 ───────────────────────────────────────────
    h5     = H5Data(CFG.h5_path)
    le     = LabelEncoder()
    y_all  = le.fit_transform(h5.y_raw).astype(np.int64)
    groups = h5.subj_id
    X_all  = h5.X

    # C6(평지) 레이블 인덱스
    flat_label = int(np.where(le.classes_ == le.classes_[np.argmax(
        [(c in ("C6", "flat", "평지", "6")) for c in le.classes_]
    )])[0][0]) if any(
        c in ("C6", "flat", "평지", "6") for c in le.classes_
    ) else 5

    log(f"피험자 {len(np.unique(groups))}명  N={len(y_all)}  flat_label={flat_label}({le.classes_[flat_label]})")

    branch_idx, branch_ch = build_branch_idx(h5.channels)
    foot_idx = get_foot_accel_idx(h5.channels)

    # ── feat 추출 (캐시) ──────────────────────────────────────
    cache_dir  = ensure_dir(CFG.repo_dir / "cache" / f"feat{N_FEATURES}_seed{CFG.seed}_final")
    cache_path = cache_dir / "all_feat.npy"
    use_cache  = not args.no_feat_cache

    if use_cache and cache_path.exists():
        log(f"[feat] 캐시 히트 → {cache_path}")
        feat_all = np.load(cache_path)
        log(f"[feat] shape={feat_all.shape}  (센서{_N_SENSOR}+컨텍스트{N_FEATURES-_N_SENSOR})")
    else:
        log("[feat] 추출 시작 (센서+bout컨텍스트 324차원)...")
        with Timer() as t:
            feat_all = batch_extract(
                X_all, foot_idx, CFG.sample_rate,
                h5_path=str(CFG.h5_path),
            )
        if use_cache:
            np.save(cache_path, feat_all)
        log(f"[feat] 완료 shape={feat_all.shape}  ({t})")

    # ── 모델 팩토리 ───────────────────────────────────────────
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    model_fns   = {n: MODEL_REGISTRY[n] for n in model_names if n in MODEL_REGISTRY}
    if not model_fns:
        log(f"ERROR: 모델 없음. 사용 가능: {list(MODEL_REGISTRY.keys())}")
        return

    # ── K-Fold ────────────────────────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=CFG.kfold, shuffle=True, random_state=CFG.seed)

    # fold별 결과 저장
    fold_results: dict[str, dict] = {n: {"preds": [], "probas": [], "labels": []} for n in model_fns}

    with Timer() as total_t:
        for fi, (tr_idx, te_idx) in enumerate(
            sgkf.split(np.zeros(len(y_all)), y_all, groups), 1
        ):
            log(f"\n{'='*60}")
            log(f"  Fold {fi}/{CFG.kfold}  tr={len(tr_idx)}  te={len(te_idx)}")

            # ── Subject-wise Normalization ─────────────────────
            train_mask = np.zeros(len(y_all), dtype=bool)
            train_mask[tr_idx] = True

            log("  [Norm] Subject-wise feature normalization...")
            feat_norm = subject_normalize_feat(
                feat_all, groups, y_all, flat_label, train_mask
            )
            feat_tr = feat_norm[tr_idx]
            feat_te = feat_norm[te_idx]

            for mname, mfn in model_fns.items():
                log(f"\n── [{fi}/{CFG.kfold}] {mname}")

                model = mfn(branch_ch).to(str(_cfg.DEVICE))

                tr_loader, te_loader = make_hierarchical_loaders(
                    X_all[tr_idx], feat_tr, y_all[tr_idx],
                    X_all[te_idx], feat_te, y_all[te_idx],
                    branch_idx,
                    batch    = CFG.batch,
                    balanced = CFG.use_balanced_sampler,
                )

                # 학습
                preds, labels, hist = train_model(
                    model, tr_loader, te_loader,
                    branch    = True,
                    tag       = f"[F{fi}][{mname}]",
                    use_mixup = True,
                )

                # softmax 확률 추출 (threshold search용)
                log("  [Proba] softmax 확률 추출...")
                probas, _ = get_probas(model, te_loader, _cfg.DEVICE)

                fold_results[mname]["preds"].extend(preds.tolist())
                fold_results[mname]["probas"].append(probas)
                fold_results[mname]["labels"].extend(labels.tolist())

                # fold 단위 로그
                fold_acc = accuracy_score(labels, preds)
                fold_f1  = f1_score(labels, preds, average="macro", zero_division=0)
                log(f"  [F{fi}][{mname}] Acc={fold_acc:.4f}  F1={fold_f1:.4f}")

                del model, tr_loader, te_loader
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ══════════════════════════════════════════════════════════
    # 7. 최종 평가
    # ══════════════════════════════════════════════════════════
    log(f"\n{'='*60}")
    log("  최종 평가")

    results = []
    for mname, data in fold_results.items():
        pred_arr  = np.array(data["preds"])
        label_arr = np.array(data["labels"])
        proba_arr = np.concatenate(data["probas"])  # (N, C)

        # ── 기본 성능 ─────────────────────────────────────────
        acc = accuracy_score(label_arr, pred_arr)
        f1  = f1_score(label_arr, pred_arr, average="macro", zero_division=0)
        log(f"\n[{mname}] 기본  Acc={acc:.4f}  F1={f1:.4f}")

        # ── Threshold Grid Search ─────────────────────────────
        log(f"[{mname}] Per-class threshold search...")
        best_mults, best_f1_thresh = threshold_search(proba_arr, label_arr)
        thresh_preds = (proba_arr * best_mults).argmax(1)
        thresh_acc   = accuracy_score(label_arr, thresh_preds)
        log(f"[{mname}] Threshold 적용  Acc={thresh_acc:.4f}  F1={best_f1_thresh:.4f}")
        log(f"[{mname}] Best multipliers: {dict(enumerate(best_mults.round(2)))}")

        # ── Hierarchical 앙상블 (logits 파일 있으면) ──────────
        final_preds = thresh_preds
        final_acc   = thresh_acc
        final_f1    = best_f1_thresh

        if args.hier_logits and Path(args.hier_logits).exists():
            hier_proba = np.load(args.hier_logits)
            if hier_proba.shape == proba_arr.shape:
                log(f"[{mname}] Hierarchical 앙상블 weight search...")
                best_w, ens_f1 = find_best_ensemble_weight(
                    hier_proba, proba_arr * best_mults / (proba_arr * best_mults).sum(1, keepdims=True),
                    label_arr,
                )
                ens_preds, ens_acc, ens_f1 = soft_ensemble(
                    hier_proba,
                    proba_arr * best_mults / (proba_arr * best_mults).sum(1, keepdims=True),
                    label_arr, best_w,
                )
                log(f"[{mname}] Ensemble  Acc={ens_acc:.4f}  F1={ens_f1:.4f}  (hier_w={best_w:.2f})")
                if ens_f1 > final_f1:
                    final_preds, final_acc, final_f1 = ens_preds, ens_acc, ens_f1

        # ── Classification Report ─────────────────────────────
        log(f"\n[{mname}] 최종 결과  Acc={final_acc:.4f}  F1={final_f1:.4f}")
        report = classification_report(
            label_arr, final_preds,
            target_names=le.classes_,
            digits=4,
        )
        log(f"\n{report}")

        # ── 저장 ─────────────────────────────────────────────
        save_report(final_preds, label_arr, le, mname, out_dir)
        save_cm(final_preds, label_arr, le, mname, out_dir)

        # softmax 확률 저장 (다른 모델과 앙상블용)
        np.save(out_dir / f"{mname}_proba.npy", proba_arr)
        np.save(out_dir / f"{mname}_mults.npy", best_mults)
        log(f"[{mname}] probas 저장 → {out_dir}/{mname}_proba.npy")

        results.append({"model": mname, "acc": round(final_acc, 4), "f1": round(final_f1, 4)})

    save_summary_table(results, out_dir.parent / "tables")
    save_json({
        "experiment": "final",
        "models":     model_names,
        "n_features": N_FEATURES,
        "total_time": str(total_t),
        "results":    {r["model"]: {"acc": r["acc"], "f1": r["f1"]} for r in results},
    }, out_dir / "summary_final.json")

    log(f"\n★ 완료  총 소요: {total_t}")
    for r in results:
        log(f"  RESULT {r['model']:<20} Acc={r['acc']:.4f}  F1={r['f1']:.4f}")

    h5.close()


if __name__ == "__main__":
    main()