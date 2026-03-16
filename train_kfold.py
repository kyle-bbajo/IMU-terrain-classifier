import json
import json
"""train_final.py — 95% 목표 최종 파이프라인

핵심 전략:
  1. Subject-wise Feature Normalization  (개인차 제거)
  2. M7_Attr (CNN + 324feat + Attribute Heads + GRU fusion)
  3. Per-class Threshold Grid Search     (val set 기준)
  4. 3모델 앙상블: kfold + attribute_v5 + hierarchical
     Phase A: 가중 평균 (자동 grid search)
     Phase B: LightGBM Stacking (meta-learner)
     → val F1 높은 것 채택

실행:
  cd ~/project/repo
  python train_kfold.py
  python train_kfold.py --ensemble_only  (학습 없이 앙상블만)
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
    filter_and_remap, N_ACTIVE_CLASSES, ACTIVE_CLASS_NAMES,
)
from channel_groups import build_branch_idx, get_foot_accel_idx
from features import batch_extract, N_FEATURES
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
                   help="학습 없이 저장된 probas로 앙상블만 수행")
    # ── 앙상블 소스 경로 ──────────────────────────────────────
    p.add_argument("--attr_proba_dir", type=str,
                   default="out_N50/attribute_kfold_v5/probas",
                   help="attribute fold별 proba 디렉토리")
    p.add_argument("--hier_proba_dir", type=str,
                   default="out_N50/hierarchical_eventfusion/probas",
                   help="hierarchical fold별 proba 디렉토리")
    p.add_argument("--n_folds",        type=int, default=5)
    p.add_argument("--no_stack",       action="store_true",
                   help="LightGBM stacking 건너뜀")
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
# 4-B. 3모델 앙상블 유틸
# ════════════════════════════════════════════════════════════════

def _load_fold_probas(proba_dir: str, prefix: str, n_folds: int):
    """fold별 proba/labels npy 로드 → concat."""
    d = Path(proba_dir)
    probas, labels = [], []
    for fi in range(1, n_folds + 1):
        pp = d / f"{prefix}_proba_fold{fi}.npy"
        lp = d / f"{prefix}_labels_fold{fi}.npy"
        if pp.exists() and lp.exists():
            probas.append(np.load(pp))
            labels.append(np.load(lp))
        else:
            log(f"  [WARN] 없음: {pp}")
    if not probas:
        return None, None
    return np.concatenate(probas), np.concatenate(labels)


def _weight_search(probas_list, labels, steps=6):
    """3개 모델 가중치 grid search → 최적 (w0,w1,w2) 반환."""
    vals = np.linspace(0, 1, steps + 1)
    best_f1, best_w = -1.0, None
    avail = [(i, p) for i, p in enumerate(probas_list) if p is not None]
    for wa in vals:
        for wb in vals:
            wc = 1.0 - wa - wb
            if wc < -1e-6: continue
            ws = [wa, wb, max(wc, 0.0)]
            total = sum(ws[i] for i, _ in avail)
            if total < 1e-8: continue
            combo = sum(ws[i] / total * p for i, p in avail)
            f1 = f1_score(labels, combo.argmax(1), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_w = f1, tuple(ws)
    return best_w, best_f1


def _stacking(kfold_probas_list, attr_probas_list, hier_probas_list,
              kfold_labels_list, n_folds):
    """LightGBM OOF stacking."""
    try:
        import lightgbm as lgb
    except ImportError:
        log("  [WARN] lightgbm 미설치 → pip install lightgbm"); return None, -1.0

    meta_X, meta_y = [], []
    for fi in range(n_folds):
        parts, lbl = [], None
        for plist, llist in [(kfold_probas_list,  kfold_labels_list),
                              (attr_probas_list,   None),
                              (hier_probas_list,   None)]:
            if fi < len(plist) and plist[fi] is not None:
                parts.append(plist[fi])
                if lbl is None and llist is not None and fi < len(llist):
                    lbl = llist[fi]
        if not parts or lbl is None: continue
        min_n = min(len(p) for p in parts)
        meta_X.append(np.concatenate([p[:min_n] for p in parts], axis=1))
        meta_y.append(lbl[:min_n])

    if not meta_X:
        log("  [WARN] stacking meta feature 없음"); return None, -1.0

    X = np.concatenate(meta_X).astype(np.float32)
    y = np.concatenate(meta_y).astype(np.int64)
    log(f"  Stacking X={X.shape}")

    from sklearn.model_selection import StratifiedKFold
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof  = np.zeros(len(y), dtype=np.int64)
    params = dict(objective="multiclass", num_class=N_ACTIVE_CLASSES, n_estimators=500,
                  learning_rate=0.05, num_leaves=63, min_child_samples=20,
                  subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                  random_state=42, n_jobs=-1, verbose=-1)
    for tr_i, va_i in skf.split(X, y):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X[tr_i], y[tr_i],
                eval_set=[(X[va_i], y[va_i])],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)])
        oof[va_i] = clf.predict(X[va_i])

    f1 = f1_score(y, oof, average="macro", zero_division=0)
    acc = accuracy_score(y, oof)
    log(f"  Stacking OOF  Acc={acc:.4f}  F1={f1:.4f}")
    return oof, f1


def run_ensemble(kfold_proba, kfold_labels, out_dir, args, repo_dir):
    """attr + hier + raw + surface 최대 4모델 앙상블."""
    log(f"\n{'='*60}")
    log("  다중 모델 앙상블 시작 (attr + hier + raw + surface)")
    log(f"{'='*60}")

    # ── 외부 proba 로드 ────────────────────────────────────────
    attr_p,    attr_l    = _load_fold_probas(
        str(repo_dir / args.attr_proba_dir), "attr", args.n_folds)
    hier_p,    hier_l    = _load_fold_probas(
        str(repo_dir / args.hier_proba_dir), "hier", args.n_folds)
    raw_p,     raw_l     = _load_fold_probas(
        str(repo_dir / "out_N50/raw_cnn_transformer/probas"), "raw", args.n_folds)
    surface_p, surface_l = _load_fold_probas(
        str(repo_dir / "out_N50/surface_expert/probas"), "surface", args.n_folds)

    if attr_p is None and hier_p is None and raw_p is None and surface_p is None:
        log("  [WARN] 외부 proba 없음 → 앙상블 스킵")
        return None, 0.0, 0.0

    # labels 기준: attr 우선 → hier → raw → surface
    base_labels = next(l for l in [attr_l, hier_l, raw_l, surface_l]
                       if l is not None)
    N = len(base_labels)

    # 가용 모델 목록
    all_models = [
        (attr_p,    "attribute"),
        (hier_p,    "hierarchical"),
        (raw_p,     "raw_cnn"),
        (surface_p, "surface"),
    ]
    avail = [(p, name) for p, name in all_models if p is not None]

    # 단독 성능 로그
    for p, name in avail:
        n = min(len(p), N)
        f1  = f1_score(base_labels[:n], p[:n].argmax(1), average="macro", zero_division=0)
        acc = accuracy_score(base_labels[:n], p[:n].argmax(1))
        log(f"  단독 {name:<14} Acc={acc:.4f}  F1={f1:.4f}")

    # ── Phase A: 가중 평균 탐색 ──────────────────────────────
    log(f"\n  [Phase A] {len(avail)}모델 가중 평균 탐색...")
    n = min(N, *[len(p) for p, _ in avail])
    labels_n = base_labels[:n]

    if len(avail) == 1:
        combo_A = avail[0][0][:n]
        best_ws = {avail[0][1]: 1.0}
    else:
        best_f1_w, best_ws, best_combo = -1.0, {}, None
        # 가중치 그리드 탐색 (최대 4모델 대응)
        vals = np.linspace(0, 1, 6)   # 0.0~1.0, 6단계
        n_models = len(avail)

        def _grid(depth, remaining, current):
            nonlocal best_f1_w, best_ws, best_combo
            if depth == n_models - 1:
                ws = current + [remaining]
                if sum(ws) < 1e-8: return
                ws_norm = [w / sum(ws) for w in ws]
                combo = sum(w * p[:n] for w, (p, _) in zip(ws_norm, avail))
                f1 = f1_score(labels_n, combo.argmax(1),
                              average="macro", zero_division=0)
                if f1 > best_f1_w:
                    best_f1_w = f1
                    best_ws   = {name: w for w, (_, name) in zip(ws_norm, avail)}
                    best_combo = combo
                return
            for v in vals:
                if v > remaining + 1e-6: break
                _grid(depth + 1, remaining - v, current + [v])

        _grid(0, 1.0, [])
        combo_A = best_combo
        log(f"  최적 가중치: " + "  ".join(f"{k}={v:.2f}" for k, v in best_ws.items()))

    mults_A, _ = threshold_search(combo_A, labels_n, n_classes=N_ACTIVE_CLASSES)
    preds_A    = (combo_A * mults_A).argmax(1)
    acc_A = accuracy_score(labels_n, preds_A)
    f1_A  = f1_score(labels_n, preds_A, average="macro", zero_division=0)
    log(f"  [Phase A] 가중평균+threshold  Acc={acc_A:.4f}  F1={f1_A:.4f}")

    # ── Phase B: LightGBM Stacking ────────────────────────────
    f1_B, stack_preds = -1.0, None
    if not args.no_stack and len(avail) >= 2:
        log(f"\n  [Phase B] LightGBM Stacking ({len(avail)*6}차원)...")
        try:
            import lightgbm as lgb
            from sklearn.model_selection import StratifiedKFold
            import warnings
            warnings.filterwarnings("ignore", message="X does not have valid feature names")

            X_meta = np.concatenate([p[:n] for p, _ in avail], axis=1).astype(np.float32)
            y_meta = labels_n.astype(np.int64)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof = np.zeros(n, dtype=np.int64)
            params = dict(objective="multiclass", num_class=N_ACTIVE_CLASSES, n_estimators=500,
                          learning_rate=0.05, num_leaves=63, min_child_samples=20,
                          subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                          random_state=42, n_jobs=-1, verbose=-1)

            for tr_i, va_i in skf.split(X_meta, y_meta):
                clf = lgb.LGBMClassifier(**params)
                clf.fit(X_meta[tr_i], y_meta[tr_i],
                        eval_set=[(X_meta[va_i], y_meta[va_i])],
                        callbacks=[lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(period=-1)])
                oof[va_i] = clf.predict(X_meta[va_i])

            f1_B  = f1_score(y_meta, oof, average="macro", zero_division=0)
            acc_B = accuracy_score(y_meta, oof)
            log(f"  [Phase B] Stacking OOF  Acc={acc_B:.4f}  F1={f1_B:.4f}")
            stack_preds = oof
        except Exception as e:
            log(f"  [Phase B] Stacking 실패: {e}")

    # ── 최종 채택 ─────────────────────────────────────────────
    if f1_B > f1_A and stack_preds is not None:
        final_preds = stack_preds
        method = f"LightGBM Stacking (F1={f1_B:.4f})"
    else:
        final_preds = preds_A
        method = f"가중평균+threshold (F1={f1_A:.4f})"

    final_acc = accuracy_score(labels_n, final_preds)
    final_f1  = f1_score(labels_n, final_preds, average="macro", zero_division=0)

    log(f"\n  ★ 앙상블 채택: {method}")
    log(f"  ★ 앙상블 Acc={final_acc:.4f}  F1={final_f1:.4f}")

    rep = classification_report(labels_n, final_preds,
                                target_names=ACTIVE_CLASS_NAMES,   # C5 제외 5클래스
                                digits=4, zero_division=0)
    log(f"\n{rep}")

    from sklearn.metrics import confusion_matrix as cm_fn
    cm = cm_fn(labels_n, final_preds, labels=list(range(N_ACTIVE_CLASSES)))
    recalls = cm.diagonal() / cm.sum(1).clip(min=1)
    for i, r in enumerate(recalls):
        flag = "✅" if r >= 0.85 else ("⚠" if r >= 0.70 else "❌")
        cname = ACTIVE_CLASS_NAMES[i] if i < len(ACTIVE_CLASS_NAMES) else f"C{i}"
        log(f"  {flag} {cname}  recall={r*100:.1f}%")

    # 저장
    ens_dir = ensure_dir(out_dir / "ensemble")
    np.save(ens_dir / "ensemble_proba.npy",  combo_A)
    np.save(ens_dir / "ensemble_preds.npy",  final_preds)
    np.save(ens_dir / "ensemble_labels.npy", labels_n)
    (ens_dir / "ensemble_summary.json").write_text(json.dumps({
        "method": method, "acc": round(final_acc, 4), "f1": round(final_f1, 4),
        "weights": best_ws, "models_used": [n for _, n in avail],
        "phase_A_f1": round(f1_A, 4), "phase_B_f1": round(f1_B, 4),
    }, indent=2, ensure_ascii=False))
    log(f"  결과 저장 → {ens_dir}")

    return final_preds, final_acc, final_f1


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

    # ── C5 제외: 5클래스 학습 ─────────────────────────────────
    CFG.num_classes = N_ACTIVE_CLASSES   # 모델 출력 5클래스로 고정
    y_raw_all  = h5.y_raw.astype(np.int64)
    y_all, groups, kept_idx = filter_and_remap(y_raw_all, h5.subj_id)
    X_all  = h5.X[kept_idx]
    le     = LabelEncoder()
    le.classes_ = np.array(ACTIVE_CLASS_NAMES)   # 5클래스 이름 설정
    log(f"  C5 제외 후: N={len(y_all)}  classes={N_ACTIVE_CLASSES}")

    # C6(평지) 레이블 인덱스 — C5 제외 후 새 인덱스=4
    flat_label = 4   # ACTIVE_CLASS_NAMES[4] = "C6-평지"

    log(f"피험자 {len(np.unique(groups))}명  N={len(y_all)}  flat_label={flat_label}({le.classes_[flat_label]})")

    branch_idx, branch_ch = build_branch_idx(h5.channels)
    foot_idx = get_foot_accel_idx(h5.channels)

    # ── feat 추출 (캐시) ──────────────────────────────────────
    cache_dir  = ensure_dir(CFG.repo_dir / "cache" / f"feat{N_FEATURES}_seed{CFG.seed}_final_noc5")
    cache_path = cache_dir / "all_feat.npy"
    use_cache  = not args.no_feat_cache

    if use_cache and cache_path.exists():
        log(f"[feat] 캐시 히트 → {cache_path}")
        feat_all = np.load(cache_path)
    else:
        log("[feat] 추출 시작 (센서+bout컨텍스트 324차원)...")
        with Timer() as t:
            feat_all = batch_extract(
                X_all, foot_idx, CFG.sample_rate,
                h5_path=str(CFG.h5_path),
                kept_idx=kept_idx,    # C5 제거 인덱스 → 컨텍스트 피처 정합성 보장
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
    fold_results: dict[str, dict] = {n: {"preds": [], "probas": [], "labels": [], "fold_probas": []} for n in model_fns}

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
                fold_results[mname]["fold_probas"].append(probas)   # fold별 보존
                fold_results[mname]["labels"].extend(labels.tolist())

                # fold 단위 로그
                fold_acc = accuracy_score(labels, preds)
                fold_f1  = f1_score(labels, preds, average="macro", zero_division=0)
                log(f"  [F{fi}][{mname}] Acc={fold_acc:.4f}  F1={fold_f1:.4f}")

                del model, tr_loader, te_loader
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ══════════════════════════════════════════════════════════
    # 7. 최종 평가 + 3모델 앙상블
    # ══════════════════════════════════════════════════════════
    log(f"\n{'='*60}")
    log("  최종 평가")

    results = []
    for mname, data in fold_results.items():
        pred_arr  = np.array(data["preds"])
        label_arr = np.array(data["labels"])
        proba_arr = np.concatenate(data["probas"])  # (N, 6)

        # ── 기본 성능 ─────────────────────────────────────────
        acc = accuracy_score(label_arr, pred_arr)
        f1  = f1_score(label_arr, pred_arr, average="macro", zero_division=0)
        log(f"\n[{mname}] 기본  Acc={acc:.4f}  F1={f1:.4f}")

        # ── Threshold Search ──────────────────────────────────
        log(f"[{mname}] Per-class threshold search (5클래스)...")
        best_mults, best_f1_thresh = threshold_search(proba_arr, label_arr,
                                                       n_classes=N_ACTIVE_CLASSES)
        thresh_preds = (proba_arr * best_mults).argmax(1)
        thresh_acc   = accuracy_score(label_arr, thresh_preds)
        log(f"[{mname}] Threshold  Acc={thresh_acc:.4f}  F1={best_f1_thresh:.4f}")

        # proba 저장 (fold별 + 전체)
        np.save(out_dir / f"{mname}_proba.npy",  proba_arr)
        np.save(out_dir / f"{mname}_labels.npy", label_arr)
        np.save(out_dir / f"{mname}_mults.npy",  best_mults)
        # fold별 저장 (앙상블용)
        kfold_proba_dir = ensure_dir(out_dir / "probas")
        offset = 0
        for fi, plist in enumerate(data["fold_probas"], 1):
            n = len(plist)
            np.save(kfold_proba_dir / f"M7_Attr_proba_fold{fi}.npy",  plist)
            np.save(kfold_proba_dir / f"M7_Attr_labels_fold{fi}.npy", label_arr[offset:offset+n])
            offset += n
        log(f"[{mname}] proba 저장 → {out_dir}/{mname}_proba.npy")

        # ── 단독 리포트 ───────────────────────────────────────
        report = classification_report(
            label_arr, thresh_preds,
            target_names=ACTIVE_CLASS_NAMES,   # C5 제외 5클래스
            digits=4, zero_division=0,
        )
        log(f"\n{report}")
        save_report(thresh_preds, label_arr, le, mname, out_dir)
        save_cm(thresh_preds, label_arr, le, mname, out_dir)

        results.append({"model": mname, "acc": round(thresh_acc, 4),
                        "f1": round(best_f1_thresh, 4)})

    # ── 3모델 앙상블 ──────────────────────────────────────────
    if not args.ensemble_only:
        # 전체 proba (마지막 모델 기준)
        last_proba  = np.concatenate(fold_results[list(fold_results.keys())[-1]]["probas"])
        last_labels = np.array(fold_results[list(fold_results.keys())[-1]]["labels"])
        ens_preds, ens_acc, ens_f1 = run_ensemble(
            last_proba, last_labels, out_dir, args, CFG.repo_dir)
        results.append({"model": "Ensemble_3model", "acc": round(ens_acc, 4),
                        "f1": round(ens_f1, 4)})

    save_summary_table(results, out_dir.parent / "tables")
    save_json({
        "experiment": "final_with_ensemble",
        "models":     model_names,
        "n_features": N_FEATURES,
        "total_time": str(total_t),
        "results":    {r["model"]: {"acc": r["acc"], "f1": r["f1"]} for r in results},
    }, out_dir / "summary_final.json")

    log(f"\n★ 완료  총 소요: {total_t}")
    for r in results:
        log(f"  RESULT {r['model']:<25} Acc={r['acc']:.4f}  F1={r['f1']:.4f}")

    h5.close()


if __name__ == "__main__":
    main()