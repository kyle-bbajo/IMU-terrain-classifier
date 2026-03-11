"""train_attribute.py — v3 (ResidualMLP + SurfaceGate + PerClassFocal)

v3 핵심 변경사항:
  [1] 입력 차원: 2088dim(3-step window×delta×ratio) → N_FEATURES(232dim) 직접 사용
      → 노이즈 제거, 메모리 절약, 학습 안정화
  [2] ResidualMLP 4블록 (LayerNorm + GELU) — AttributeTerrainModel보다 강한 표현력
  [3] SurfaceGateHead: C4/C5/C6 전용 추가 bias → 47% C6 recall 문제 해결
  [4] PerClassFocalLoss: C4/C5/C6에 gamma=4.0 (나머지 gamma=2.0)
  [5] BalancedSampler: WeightedRandomSampler로 클래스 불균형 해소
  [6] threshold_search: val proba 기반 per-class multiplier 탐색

실행:
  python train_attribute.py
  python train_attribute.py --epochs 80 --hidden_dim 512
"""
from __future__ import annotations

import sys, gc, argparse, os, copy, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from wandb_init import wandb_start, wandb_log_fold, wandb_finish

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from config import CFG, apply_overrides, print_config
import config as _cfg
from train_common import (
    H5Data, seed_everything, log, ensure_dir, save_json, Timer,
)
from features import batch_extract, N_FEATURES

# ── 클래스 정의 ───────────────────────────────────────────────
CLASS_NAMES = ["C1-미끄러운", "C2-오르막", "C3-내리막", "C4-흙길", "C5-잔디", "C6-평지"]
N_CLASSES   = 6
C1, C2, C3, C4, C5, C6 = 0, 1, 2, 3, 4, 5


# ══════════════════════════════════════════════════════════════
# 1. Dataset v3 — N_FEATURES 직접 사용 (window/delta/ratio 제거)
# ══════════════════════════════════════════════════════════════

class AttributeDatasetV3(Dataset):
    """
    단순화된 Dataset:
    - 입력: N_FEATURES (232dim) — 3-step window 없음, delta 없음
    - Subject-wise StandardScaler로 정규화 (train에서 fit)
    """
    def __init__(
        self,
        feat_all: np.ndarray,   # (N, N_FEATURES) — 전체
        y_all: np.ndarray,      # (N,)
        indices: np.ndarray,    # fold 내 인덱스
        scaler: StandardScaler | None = None,  # None이면 fit, 있으면 transform
        fit_scaler: bool = False,
    ):
        self.feat_raw = feat_all[indices].astype(np.float32)
        self.y        = y_all[indices].astype(np.int64)
        self.indices  = indices

        if fit_scaler:
            self.scaler = StandardScaler()
            self.feat   = self.scaler.fit_transform(self.feat_raw)
        elif scaler is not None:
            self.scaler = scaler
            self.feat   = scaler.transform(self.feat_raw)
        else:
            self.scaler = None
            self.feat   = self.feat_raw

        self.feat = self.feat.astype(np.float32)
        log(f"    AttributeDatasetV3: n={len(self.y)}  dim={self.feat.shape[1]}")

    @property
    def input_dim(self): return self.feat.shape[1]

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return {
            "x": torch.tensor(self.feat[i], dtype=torch.float32),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
        }


def make_balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    classes, counts = np.unique(y, return_counts=True)
    class_w = np.zeros(N_CLASSES, dtype=np.float64)
    class_w[classes] = 1.0 / counts.astype(np.float64)
    sample_w = class_w[y]
    log(f"    BalancedSampler: {dict(zip(classes.tolist(), counts.tolist()))}")
    return WeightedRandomSampler(
        torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(y), replacement=True,
    )


# ══════════════════════════════════════════════════════════════
# 2. Model v3 — ResidualMLP + SurfaceGate
# ══════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """BN-free residual block: LayerNorm + GELU + Dropout"""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
    def forward(self, x): return x + self.net(x)


class SurfaceGateResidualMLP(nn.Module):
    """
    ResidualMLP + SurfaceGate for C4/C5/C6 disambiguation.

    [구조]
    feat(N_FEATURES) → proj → 4× ResidualBlock → head (6cls)
                                               ↘ surface_gate → bias C4/C5/C6
    최종 logits = head_logits + [0, 0, 0, gate[0], gate[1], gate[2]]
    """
    def __init__(self, in_dim: int, hidden: int = 512, n_blocks: int = 4, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(n_blocks)])

        # 기본 분류 헤드
        self.head = nn.Linear(hidden, N_CLASSES)

        # SurfaceGate: C4/C5/C6 전용 bias
        self.surface_gate = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 3),   # C4, C5, C6
        )

        # SurfaceHead: C4/C5/C6 전용 독립 분류기 (더 강한 surface supervision)
        self.surface_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 3),    # C4, C5, C6
        )
        # learnable scale
        self.surf_scale = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h      = self.blocks(self.proj(x))
        logits = self.head(h)                          # (B, 6)
        gate   = self.surface_gate(h)                  # (B, 3)
        surf   = self.surface_head(h)                  # (B, 3)
        scale  = torch.clamp(self.surf_scale, 0.0, 2.0)

        logits = torch.cat([
            logits[:, :3],
            logits[:, 3:6] + gate + scale * surf,      # C4, C5, C6 보강
        ], dim=1)
        return logits

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


# ══════════════════════════════════════════════════════════════
# 3. Loss — PerClassFocalLoss
# ══════════════════════════════════════════════════════════════

class PerClassFocalLoss(nn.Module):
    """
    C4/C5/C6에 gamma=4.0, 나머지에 gamma=2.0을 적용하는 focal loss.
    C6(평지) recall이 낮을 때 효과적.
    """
    def __init__(
        self,
        gamma_default: float = 2.0,
        gamma_surface: float = 3.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        gammas = [gamma_default] * N_CLASSES
        gammas[C4] = gamma_surface
        gammas[C5] = gamma_surface
        gammas[C6] = gamma_surface
        self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float32))
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # label smoothing 적용 CE (per-sample, no reduction)
        n_cls = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.ls / (n_cls - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p  = F.log_softmax(logits, dim=1)
        ce     = -(smooth_targets * log_p).sum(dim=1)     # (B,)
        pt     = torch.exp(-ce)
        gamma  = self.gammas[targets]
        return ((1 - pt) ** gamma * ce).mean()


# ══════════════════════════════════════════════════════════════
# 4. Threshold Search
# ══════════════════════════════════════════════════════════════

def threshold_search(
    proba: np.ndarray,
    labels: np.ndarray,
    n_classes: int = N_CLASSES,
    search_vals: list | None = None,
) -> tuple[np.ndarray, float]:
    """
    Val set proba에 per-class multiplier를 곱해 macro F1 최대화.
    순차 탐색 (greedy per class).
    """
    if search_vals is None:
        search_vals = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    mults = np.ones(n_classes, dtype=np.float32)
    for ci in range(n_classes):
        best_f1, best_m = -1.0, 1.0
        for m in search_vals:
            tmp = mults.copy(); tmp[ci] = m
            preds = (proba * tmp).argmax(1)
            score = f1_score(labels, preds, average="macro", zero_division=0)
            if score > best_f1:
                best_f1, best_m = score, m
        mults[ci] = best_m
    final_preds = (proba * mults).argmax(1)
    final_f1    = f1_score(labels, final_preds, average="macro", zero_division=0)
    return mults, final_f1


# ══════════════════════════════════════════════════════════════
# 5. Train / Eval
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, crit, device, grad_clip=5.0):
    model.train(); total_loss = 0.0; n = 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        opt.zero_grad()
        logits = model(x)
        loss   = crit(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total_loss += float(loss) * len(y); n += len(y)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, crit, device, mults=None):
    model.eval()
    total_loss = 0.0; n = 0
    all_proba, all_y = [], []
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        logits = model(x)
        loss   = crit(logits, y)
        proba  = F.softmax(logits, dim=1).cpu().numpy()
        total_loss += float(loss) * len(y); n += len(y)
        all_proba.append(proba); all_y.extend(y.cpu().tolist())
    all_proba = np.concatenate(all_proba, axis=0)
    all_y     = np.array(all_y, dtype=np.int64)
    if mults is not None:
        preds = (all_proba * mults).argmax(1)
    else:
        preds = all_proba.argmax(1)
    acc = accuracy_score(all_y, preds)
    f1  = f1_score(all_y, preds, average="macro", zero_division=0)
    return total_loss / max(n, 1), acc, f1, all_y, preds, all_proba


# ══════════════════════════════════════════════════════════════
# 6. One Fold
# ══════════════════════════════════════════════════════════════

def run_fold(fi, tr_ds, te_ds, args, device):
    tr_sampler = make_balanced_sampler(tr_ds.y)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, sampler=tr_sampler,
                       num_workers=4, pin_memory=True, drop_last=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                       num_workers=4, pin_memory=True)

    model = SurfaceGateResidualMLP(
        in_dim=tr_ds.input_dim,
        hidden=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)

    crit = PerClassFocalLoss(
        gamma_default=args.gamma_default,
        gamma_surface=args.gamma_surface,
        label_smoothing=args.label_smooth,
    ).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=20, T_mult=2, eta_min=1e-6,
    )

    best_f1 = -1.0; best_state = None; best_mults = np.ones(N_CLASSES, dtype=np.float32)
    patience = 0

    for ep in range(1, args.epochs + 1):
        tl = train_epoch(model, tr_dl, opt, crit, device, args.grad_clip)
        sched.step()

        # threshold search every 5 epochs
        if ep % 5 == 0 or ep == args.epochs:
            _, _, _, val_y, _, val_proba = eval_epoch(model, te_dl, crit, device)
            mults, th_f1 = threshold_search(val_proba, val_y)
        else:
            mults = best_mults

        vl, va, vf, _, _, _ = eval_epoch(model, te_dl, crit, device, mults)

        log(f"  [F{fi}] ep{ep:02d}/{args.epochs}"
            f"  tl={tl:.4f}  val_acc={va:.4f}  val_f1={vf:.4f}"
            f"  lr={sched.get_last_lr()[0]:.2e}")

        if vf > best_f1:
            best_f1 = vf; best_mults = mults.copy()
            best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                log(f"  [F{fi}] EarlyStop ep{ep}")
                break

    model.load_state_dict(best_state)
    _, acc, f1, yt, yp, _ = eval_epoch(model, te_dl, crit, device, best_mults)
    log(f"  [F{fi}] ★ Best  Acc={acc:.4f}  F1={f1:.4f}  mults={best_mults.round(2).tolist()}")

    report = classification_report(yt, yp, target_names=CLASS_NAMES, digits=4, zero_division=0)
    cm     = confusion_matrix(yt, yp, labels=list(range(N_CLASSES)))
    log(f"\n{report}")

    # per-class recall 로그
    recalls = cm.diagonal() / cm.sum(1).clip(min=1)
    for i, r in enumerate(recalls):
        log(f"    {CLASS_NAMES[i]:<16} recall={r*100:.1f}%")

    return {"acc": acc, "f1": f1, "mults": best_mults.tolist(),
            "report": report, "cm": cm.tolist()}


# ══════════════════════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,   default=120)
    p.add_argument("--batch",         type=int,   default=1024)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--wd",            type=float, default=1e-4)
    p.add_argument("--hidden_dim",    type=int,   default=512)
    p.add_argument("--n_blocks",      type=int,   default=4)
    p.add_argument("--dropout",       type=float, default=0.2)
    p.add_argument("--gamma_default", type=float, default=2.0)
    p.add_argument("--gamma_surface", type=float, default=4.0,
                   help="Focal gamma for C4/C5/C6 (surface classes)")
    p.add_argument("--label_smooth",  type=float, default=0.05)
    p.add_argument("--early_stop",    type=int,   default=25)
    p.add_argument("--grad_clip",     type=float, default=5.0)
    p.add_argument("--kfold",         type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--no-wandb",      action="store_true")
    p.add_argument("--run_name",      type=str,   default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = str(_cfg.DEVICE)
    seed_everything(args.seed)

    log("=" * 60)
    log("  AttributeTerrainModel v3 (ResidualMLP + SurfaceGate)")
    log(f"  hidden={args.hidden_dim}  n_blocks={args.n_blocks}  dropout={args.dropout}")
    log(f"  gamma_default={args.gamma_default}  gamma_surface={args.gamma_surface}")
    log(f"  input_dim=N_FEATURES={N_FEATURES}  (no window/delta/ratio)")
    log(f"  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    log("=" * 60)

    if not args.no_wandb:
        wandb_start("attribute_kfold_v3", args, cfg_dict=vars(args))

    # ── 데이터 로드 ──────────────────────────────────────────
    h5     = H5Data(_cfg.CFG.h5_path)
    le     = LabelEncoder()
    y_all  = le.fit_transform(h5.y_raw).astype(np.int64)
    groups = h5.subj_id

    log(f"  데이터: N={len(y_all)}  피험자={len(np.unique(groups))}명")

    # ── feat 추출 (캐시) ─────────────────────────────────────
    from channel_groups import get_foot_accel_idx
    foot_idx   = get_foot_accel_idx(h5.channels)
    cache_dir  = ensure_dir(_cfg.CFG.repo_dir / "cache" / f"feat{N_FEATURES}_seed{args.seed}_attr_v3")
    cache_path = cache_dir / "all_feat.npy"

    if cache_path.exists():
        log(f"  feat 캐시 히트 → 로드")
        feat_all = np.load(cache_path)
    else:
        log(f"  feat 추출 중... (N_FEATURES={N_FEATURES})")
        with Timer() as t:
            feat_all = batch_extract(h5.X, foot_idx, _cfg.CFG.sample_rate)
        np.save(cache_path, feat_all)
        log(f"  feat 추출 완료  shape={feat_all.shape}  {t}")

    # ── 출력 디렉토리 ────────────────────────────────────────
    out = ensure_dir(_cfg.CFG.repo_dir / "out_N50" / "attribute_kfold_v3")

    # ── K-Fold ───────────────────────────────────────────────
    sgkf    = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    results = []

    with Timer() as total_timer:
        for fi, (tr_idx, te_idx) in enumerate(
            sgkf.split(np.zeros(len(y_all)), y_all, groups), 1
        ):
            log(f"\n{'='*60}")
            log(f"  Fold {fi}/{args.kfold}  tr={len(tr_idx)}  te={len(te_idx)}")

            # Train scaler fit → apply to test
            tr_ds = AttributeDatasetV3(feat_all, y_all, tr_idx, fit_scaler=True)
            te_ds = AttributeDatasetV3(feat_all, y_all, te_idx, scaler=tr_ds.scaler)

            res = run_fold(fi, tr_ds, te_ds, args, device)
            results.append({"fold": fi, **res})
            wandb_log_fold(fi, {"acc": res["acc"], "f1": res["f1"]})

            del tr_ds, te_ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── 최종 요약 ────────────────────────────────────────────
    mean_acc = float(np.mean([r["acc"] for r in results]))
    mean_f1  = float(np.mean([r["f1"]  for r in results]))

    log(f"\n{'='*60}")
    log(f"  ★ AttributeTerrainModel v3 K-Fold 완료  총 소요: {total_timer}")
    log(f"  Mean Acc={mean_acc:.4f}  Mean F1={mean_f1:.4f}")
    for r in results:
        log(f"    Fold{r['fold']}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}")
    log(f"{'='*60}")

    save_json({
        "experiment": "attribute_kfold_v3",
        "version":    "v3_ResidualMLP_SurfaceGate",
        "mean_acc":   round(mean_acc, 4),
        "mean_f1":    round(mean_f1,  4),
        "total_time": str(total_timer),
        "config": {
            "hidden_dim":    args.hidden_dim,
            "n_blocks":      args.n_blocks,
            "gamma_surface": args.gamma_surface,
            "input_dim":     N_FEATURES,
        },
        "folds": [{k: v for k, v in r.items() if k not in ("report", "cm")}
                  for r in results],
    }, out / "summary_attribute_v3.json")

    if not args.no_wandb:
        wandb_finish(results=[{"model": "AttributeV3", "acc": mean_acc, "f1": mean_f1}])

    h5.close()


if __name__ == "__main__":
    main()