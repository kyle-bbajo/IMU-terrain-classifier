# -*- coding: utf-8 -*-
"""
train_ablation.py — C4/C5/C6 구분 능력 비교 실험 (3가지 조건)
═══════════════════════════════════════════════════════════════
목적:
  새 G-그룹 특징(충격 물성 12개) 추가 후
  정상/흙/잔디가 얼마나 잘 구분되는지 3가지 조건으로 비교

조건:
  Condition A (전체)     : C1~C6 6클래스 전체 학습
  Condition B (흙 제외)  : C4 제거 → 5클래스 (C1,C2,C3,C5,C6)
  Condition C (잔디 제외): C5 제거 → 5클래스 (C1,C2,C3,C4,C6)

분석 포인트:
  - A의 C4/C5/C6 개별 recall 확인
  - B에서 잔디(C5) recall 상승 → C4가 잔디에 끼치는 혼동 확인
  - C에서 흙(C4) recall 상승   → C5가 흙에 끼치는 혼동 확인

실행:
  python train_ablation.py
  python train_ablation.py --model M7_Attr --epochs 60 --kfold 5
  python train_ablation.py --model SurfaceExpert --surface_only
"""
from __future__ import annotations

import sys, gc, argparse, json, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, recall_score,
)

import config as _cfg
from config import CFG
from train_common import (
    H5Data, seed_everything, log, ensure_dir,
    save_json, Timer,
)
from channel_groups import build_branch_idx, get_foot_accel_idx
from features import batch_extract, N_FEATURES


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

ORIG_NAMES = [
    "C1-미끄러운", "C2-오르막", "C3-내리막",
    "C4-흙길",     "C5-잔디",   "C6-평지"
]

# 조건 정의: (이름, 제거할 원본 라벨 목록)
CONDITIONS = [
    ("A_전체",     []),          # 6클래스
    ("B_흙제외",   [3]),         # C4 제거 → 5클래스
    ("C_잔디제외", [4]),         # C5 제거 → 5클래스
]

DEVICE = _cfg.DEVICE


# ══════════════════════════════════════════════════════════════
# 1. 데이터 필터링 & 재매핑
# ══════════════════════════════════════════════════════════════

def filter_condition(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feat: np.ndarray,
    exclude_labels: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """
    exclude_labels 에 해당하는 클래스를 제거하고
    연속 정수 라벨로 재매핑.

    Returns: X_f, y_f, groups_f, feat_f, class_names, label_map
    """
    if not exclude_labels:
        # 제거 없음 — 그대로
        class_names = ORIG_NAMES
        label_map = {i: i for i in range(len(ORIG_NAMES))}
        return X, y, groups, feat, class_names, label_map

    keep_mask = ~np.isin(y, exclude_labels)
    X_f      = X[keep_mask]
    y_raw    = y[keep_mask]
    grp_f    = groups[keep_mask]
    feat_f   = feat[keep_mask]

    # 라벨 재매핑 (연속 정수)
    orig_labels = sorted(np.unique(y_raw).tolist())
    remap = {orig: new for new, orig in enumerate(orig_labels)}
    y_f   = np.array([remap[v] for v in y_raw], dtype=np.int64)

    class_names = [ORIG_NAMES[i] for i in orig_labels]
    return X_f, y_f, grp_f, feat_f, class_names, remap


# ══════════════════════════════════════════════════════════════
# 2. 간단한 MLP 분류기 (Surface Expert 스타일)
# ══════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))


class TerrainMLP(nn.Module):
    """특징 기반 지면 분류 MLP"""
    def __init__(self, in_dim: int, n_classes: int,
                 hidden: int = 512, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout),
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, n_classes),
        )
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════
# 3. Dataset & DataLoader
# ══════════════════════════════════════════════════════════════

class FeatDataset(Dataset):
    def __init__(self, feat: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(feat.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


def make_loader(feat, y, batch, balanced=True, shuffle=True, drop_last=True):
    ds = FeatDataset(feat, y)
    if balanced:
        cls, cnt = np.unique(y, return_counts=True)
        cw = np.zeros(len(cls), dtype=np.float64)
        cw[cls] = 1.0 / cnt.astype(np.float64)
        weights = torch.tensor(cw[y], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(y), replacement=True)
        return DataLoader(ds, batch_size=batch, sampler=sampler,
                          drop_last=drop_last, pin_memory=True, num_workers=0)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      drop_last=drop_last, pin_memory=True, num_workers=0)


# ══════════════════════════════════════════════════════════════
# 4. 단일 Fold 학습
# ══════════════════════════════════════════════════════════════

def run_fold(fi, tr_feat, tr_y, te_feat, te_y,
             n_classes, args) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model = TerrainMLP(tr_feat.shape[1], n_classes,
                       args.hidden, args.dropout).to(DEVICE)

    # 클래스 가중치
    cls, cnt = np.unique(tr_y, return_counts=True)
    cw = np.ones(n_classes, dtype=np.float32)
    for c, n in zip(cls, cnt):
        cw[int(c)] = len(tr_y) / (n_classes * n)
    cw_t = torch.tensor(cw).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr_dl = make_loader(tr_feat, tr_y, args.batch, balanced=True)
    te_dl = make_loader(te_feat, te_y, args.batch * 2,
                        balanced=False, shuffle=False, drop_last=False)

    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=max(1, len(tr_dl)),
        pct_start=0.2, anneal_strategy="cos",
    )

    best_f1, best_state, patience = -1., None, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            # Focal Loss
            log_p  = F.log_softmax(logits, 1)
            log_pt = log_p.gather(1, yb.unsqueeze(1)).squeeze(1)
            pt     = log_pt.exp()
            loss   = (-(1 - pt) ** 2 * log_pt * cw_t[yb]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step(); sch.step()

        # 평가
        model.eval(); yt, yp = [], []
        with torch.no_grad():
            for xb, yb in te_dl:
                p = model(xb.to(DEVICE)).argmax(1).cpu()
                yp.extend(p.tolist()); yt.extend(yb.tolist())
        f1 = f1_score(yt, yp, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                break

    model.load_state_dict(best_state)
    model.eval(); yt, yp, prob = [], [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            p = F.softmax(model(xb.to(DEVICE)), 1).cpu().numpy()
            prob.append(p)
            yp.extend(p.argmax(1).tolist())
            yt.extend(yb.tolist())

    yt_a = np.array(yt); yp_a = np.array(yp)
    prob_a = np.concatenate(prob)
    acc = accuracy_score(yt_a, yp_a)
    f1  = f1_score(yt_a, yp_a, average="macro", zero_division=0)
    log(f"  [F{fi}] acc={acc:.4f}  macro_f1={f1:.4f}  (best_f1={best_f1:.4f})")
    return acc, f1, yt_a, yp_a, prob_a


# ══════════════════════════════════════════════════════════════
# 5. 조건별 K-Fold 실험
# ══════════════════════════════════════════════════════════════

def run_condition(
    cond_name: str,
    X: np.ndarray, y: np.ndarray,
    groups: np.ndarray, feat: np.ndarray,
    exclude_labels: list[int],
    args,
    out_dir: Path,
) -> dict:
    log(f"\n{'='*64}")
    log(f"  조건: {cond_name}  (제거 클래스: {[ORIG_NAMES[i] for i in exclude_labels] or '없음'})")

    X_f, y_f, grp_f, feat_f, class_names, remap = filter_condition(
        X, y, groups, feat, exclude_labels
    )
    n_classes = len(class_names)
    log(f"  샘플: {len(y_f)}  클래스({n_classes}): {class_names}")

    # K-Fold
    sgkf = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True,
                                random_state=args.seed)
    all_yt, all_yp = [], []
    fold_results = []

    with Timer() as t:
        for fi, (tr_idx, te_idx) in enumerate(
            sgkf.split(np.zeros(len(y_f)), y_f, grp_f), 1
        ):
            log(f"\n  ── Fold {fi}/{args.kfold}  tr={len(tr_idx)}  te={len(te_idx)}")

            # Subject-wise 정규화 (train 기준 fit)
            scaler = StandardScaler()
            tr_feat = scaler.fit_transform(feat_f[tr_idx])
            te_feat = scaler.transform(feat_f[te_idx])

            acc, f1, yt, yp, _ = run_fold(
                fi, tr_feat, y_f[tr_idx],
                    te_feat, y_f[te_idx],
                n_classes, args,
            )
            all_yt.extend(yt.tolist())
            all_yp.extend(yp.tolist())
            fold_results.append({"fold": fi, "acc": round(acc, 4),
                                  "f1": round(f1, 4)})

            del scaler; gc.collect()

    # 전체 집계
    all_yt = np.array(all_yt); all_yp = np.array(all_yp)
    total_acc = accuracy_score(all_yt, all_yp)
    total_f1  = f1_score(all_yt, all_yp, average="macro", zero_division=0)
    recalls   = recall_score(all_yt, all_yp, average=None, zero_division=0)
    cm        = confusion_matrix(all_yt, all_yp)

    log(f"\n  ★ [{cond_name}] 최종  Acc={total_acc:.4f}  MacroF1={total_f1:.4f}  ({t})")
    log(f"\n{classification_report(all_yt, all_yp, target_names=class_names, digits=4, zero_division=0)}")

    # 혼동 행렬 저장
    _save_confusion_matrix(cm, class_names, cond_name, out_dir)

    result = {
        "condition":   cond_name,
        "n_classes":   n_classes,
        "n_samples":   int(len(y_f)),
        "class_names": class_names,
        "acc":         round(total_acc, 4),
        "macro_f1":    round(total_f1,  4),
        "per_class_recall": {
            name: round(float(r), 4)
            for name, r in zip(class_names, recalls)
        },
        "confusion_matrix": cm.tolist(),
        "folds": fold_results,
        "elapsed": str(t),
    }
    save_json(result, out_dir / f"result_{cond_name}.json")
    return result


# ══════════════════════════════════════════════════════════════
# 6. 결과 비교 출력
# ══════════════════════════════════════════════════════════════

def _save_confusion_matrix(cm, class_names, tag, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {tag}")
        plt.tight_layout()
        plt.savefig(out_dir / f"cm_{tag}.png", dpi=150)
        plt.close()
    except Exception as e:
        log(f"  [WARN] CM 저장 실패: {e}")


def print_comparison(results: list[dict]):
    log(f"\n{'='*64}")
    log("  📊 3가지 조건 비교 결과")
    log(f"{'='*64}")
    log(f"  {'조건':<16}  {'Acc':>7}  {'MacroF1':>9}  "
        f"{'C4-흙':>8}  {'C5-잔디':>9}  {'C6-평지':>9}")
    log(f"  {'-'*62}")

    for r in results:
        pr = r["per_class_recall"]
        c4 = f"{pr.get('C4-흙길',   pr.get('C4-흙', 0))*100:6.1f}%"
        c5 = f"{pr.get('C5-잔디',   0)*100:6.1f}%"
        c6 = f"{pr.get('C6-평지',   0)*100:6.1f}%"
        log(f"  {r['condition']:<16}  {r['acc']:>6.4f}  "
            f"{r['macro_f1']:>9.4f}  {c4:>8}  {c5:>9}  {c6:>9}")

    log(f"  {'-'*62}")
    log("")
    log("  분석 포인트:")
    log("  • A vs B: C4 제거 시 C5(잔디) recall 변화 → C4가 잔디에 끼친 혼동")
    log("  • A vs C: C5 제거 시 C4(흙) recall 변화   → C5가 흙에 끼친 혼동")
    log(f"{'='*64}")


# ══════════════════════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=80)
    p.add_argument("--batch",       type=int,   default=512)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--hidden",      type=int,   default=512)
    p.add_argument("--dropout",     type=float, default=0.25)
    p.add_argument("--early_stop",  type=int,   default=20)
    p.add_argument("--kfold",       type=int,   default=5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no_feat_cache", action="store_true")
    p.add_argument("--conditions",  type=str,
                   default="A,B,C",
                   help="실행할 조건 (A=전체, B=흙제외, C=잔디제외)")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    run_conds = set(args.conditions.upper().split(","))

    log("=" * 64)
    log("  train_ablation.py — C4/C5/C6 구분 능력 비교 실험 (v5)")
    log(f"  N_FEATURES={N_FEATURES}  hidden={args.hidden}")
    log(f"  epochs={args.epochs}  kfold={args.kfold}  lr={args.lr}")
    log(f"  실행 조건: {run_conds}")
    log("=" * 64)

    # ── 데이터 로드 ────────────────────────────────────────────
    h5     = H5Data(CFG.h5_path)
    le     = LabelEncoder()
    y_all  = le.fit_transform(h5.y_raw).astype(np.int64)
    groups = h5.subj_id
    X_all  = h5.X

    log(f"  데이터: N={len(y_all)}  피험자={len(np.unique(groups))}명")
    for i, name in enumerate(ORIG_NAMES):
        cnt = int((y_all == i).sum())
        log(f"    {name}: {cnt}개")

    # ── 특징 추출 ──────────────────────────────────────────────
    cache_dir  = ensure_dir(CFG.repo_dir / "cache" / f"feat_v5_seed{args.seed}")
    cache_path = cache_dir / "all_feat_v5.npy"

    if cache_path.exists() and not args.no_feat_cache:
        log(f"  ★ 특징 캐시 히트 → {cache_path}")
        with Timer() as t:
            feat_all = np.load(cache_path)
        log(f"  로드 완료  shape={feat_all.shape}  ({t})")
    else:
        log(f"  특징 추출 시작 (N_FEATURES={N_FEATURES})...")
        with Timer() as t:
            foot_idx = get_foot_accel_idx(h5.channels)
            feat_all = batch_extract(X_all, foot_idx, CFG.sample_rate,
                                     h5_path=str(CFG.h5_path))
        np.save(cache_path, feat_all)
        log(f"  추출 완료  shape={feat_all.shape}  ({t})  저장: {cache_path}")

    assert feat_all.shape == (len(y_all), N_FEATURES), \
        f"특징 shape 불일치: {feat_all.shape} vs ({len(y_all)}, {N_FEATURES})"

    # ── 출력 디렉토리 ──────────────────────────────────────────
    out = ensure_dir(CFG.repo_dir / "out_ablation")

    # ── 3가지 조건 순차 실행 ───────────────────────────────────
    all_results = []
    cond_map = {"A": CONDITIONS[0], "B": CONDITIONS[1], "C": CONDITIONS[2]}

    with Timer() as total_t:
        for cond_key, (cond_name, exclude) in cond_map.items():
            if cond_key not in run_conds:
                log(f"\n  조건 {cond_key} 스킵")
                continue
            result = run_condition(
                cond_name, X_all, y_all, groups, feat_all,
                exclude, args, out,
            )
            all_results.append(result)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── 최종 비교 출력 ─────────────────────────────────────────
    if len(all_results) > 1:
        print_comparison(all_results)

    save_json({
        "experiment":  "ablation_C4C5C6",
        "n_features":  N_FEATURES,
        "epochs":      args.epochs,
        "kfold":       args.kfold,
        "total_time":  str(total_t),
        "conditions":  all_results,
    }, out / "summary_ablation.json")

    log(f"\n★ 실험 완료  총 소요: {total_t}")
    log(f"  결과 저장: {out}")
    h5.close()


if __name__ == "__main__":
    main()
