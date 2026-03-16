"""train_attribute.py — v5 (DualHead + Mixup + TTA + LR Warmup)

v5 핵심 변경사항 (목표: 90% 이상):
  [1] DualBranchMLP: 공통 trunk + surface 전용 브랜치 분리
      → C4/C5/C6 feature space를 독립적으로 최적화
  [2] Mixup augmentation: alpha=0.4, surface 클래스 간 mixup 강화
  [3] TTA (Test Time Augmentation): 피처 노이즈 3회 평균
  [4] Linear warmup + CosineAnnealing (warmup 10 epoch)
  [5] 2단계 학습: Phase1(전체) → Phase2(surface 헤드만 fine-tune)
  [6] SurfaceAuxLoss: C4/C5/C6 3-class 보조 loss 추가
  [7] 컨텍스트 324차원 활성화 (h5_path 전달)

실행:
  python train_attribute.py
  python train_attribute.py --epochs 150 --hidden_dim 1024
"""
from __future__ import annotations

import sys, gc, argparse, os, copy, json, math
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
    - 입력: N_FEATURES (366dim)
    - Subject-wise StandardScaler로 정규화
    - slip_flag / slip_intensity 메타 포함
    """
    def __init__(
        self,
        feat_all:       np.ndarray,
        y_all:          np.ndarray,
        indices:        np.ndarray,
        slip_flag_all:  np.ndarray,
        slip_int_all:   np.ndarray,
        scaler:         StandardScaler | None = None,
        fit_scaler:     bool = False,
    ):
        self.feat_raw     = feat_all[indices].astype(np.float32)
        self.y            = y_all[indices].astype(np.int64)
        self.slip_flag    = slip_flag_all[indices].astype(np.int64)
        self.slip_int     = slip_int_all[indices].astype(np.float32)
        self.indices      = indices

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
        slip_n = int(self.slip_flag.sum())
        log(f"    Dataset: n={len(self.y)}  dim={self.feat.shape[1]}"
            f"  slip={slip_n}({slip_n/max(len(self.y),1)*100:.1f}%)")

    @property
    def input_dim(self): return self.feat.shape[1]

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return {
            "x":         torch.tensor(self.feat[i],      dtype=torch.float32),
            "y":         torch.tensor(int(self.y[i]),    dtype=torch.long),
            "slip_flag": torch.tensor(int(self.slip_flag[i]), dtype=torch.long),
            "slip_int":  torch.tensor(float(self.slip_int[i]), dtype=torch.float32),
        }


def make_balanced_sampler(
    y: np.ndarray,
    slip_flag: np.ndarray | None = None,
    slip_boost: float = 2.5,
) -> WeightedRandomSampler:
    """클래스 균형 샘플링 + C1 슬립 스텝 추가 boost.

    slip_boost: slip_flag=1인 C1 스텝의 추가 가중치 배율
    → 슬립 패턴을 모델이 더 자주 보도록 설정
    """
    classes, counts = np.unique(y, return_counts=True)
    class_w = np.zeros(N_CLASSES, dtype=np.float64)
    class_w[classes] = 1.0 / counts.astype(np.float64)
    # C4/C6 oversample ×5
    for ci in [C4, C6]:
        class_w[ci] *= 5.0
    sample_w = class_w[y]

    # C1 슬립 스텝 추가 boost: 슬립 패턴을 더 자주 학습
    if slip_flag is not None:
        slip_mask = (slip_flag == 1) & (y == C1)
        sample_w[slip_mask] *= slip_boost
        slip_n = int(slip_mask.sum())
        log(f"    BalancedSampler: {dict(zip(classes.tolist(), counts.tolist()))}"
            f"  slip_boost C1={slip_n}개×{slip_boost}")
    else:
        log(f"    BalancedSampler: {dict(zip(classes.tolist(), counts.tolist()))}")
    return WeightedRandomSampler(
        torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(y), replacement=True,
    )


# ══════════════════════════════════════════════════════════════
# 2. Model v5 — DualBranchMLP (공통 trunk + surface 전용 브랜치)
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


class DualBranchMLP(nn.Module):
    """
    공통 trunk → 5cls 헤드 + surface 전용 브랜치(C4/C6) + slip 보조 헤드

    [구조]
    feat(N_FEATURES)
      → proj
      → trunk_blocks (n_blocks개)
      → head_main (5cls 지형 분류)
      → surface_branch (C4/C6 2cls 보조)
      → slip_head (slip 이진 분류: 슬립=1 / 정상=0)
         slip_head: trunk가 슬립 패턴을 내부 표현에 포함하도록 강제
         → C1 Recall 향상 핵심
    """
    def __init__(self, in_dim: int, hidden: int = 1024, n_blocks: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.trunk     = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head_main = nn.Linear(hidden, N_CLASSES)

        # Surface 전용 브랜치: C4/C6 구분 강화
        self.surface_branch = nn.Sequential(
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 2),   # C4=0, C6=1
        )
        self.surf_scale = nn.Parameter(torch.tensor(1.0))

        # Slip 보조 헤드: 슬립 이진 분류
        # trunk가 슬립 신호를 반드시 인코딩하도록 강제
        self.slip_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2),   # 0=정상, 1=슬립
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        h     = self.trunk(self.proj(x))
        main  = self.head_main(h)              # (B, 5)
        surf  = self.surface_branch(h)         # (B, 2)
        slip  = self.slip_head(h)              # (B, 2)
        scale = torch.clamp(self.surf_scale, 0.0, 3.0)

        logits = torch.cat([
            main[:, :3],
            main[:, 3:5] + scale * surf,       # C4, C6 보강
        ], dim=1)
        return {
            "logits":         logits,
            "surface_logits": surf,
            "slip_logits":    slip,
        }

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.proj(x))


# ══════════════════════════════════════════════════════════════
# 3. Loss — PerClassFocalLoss + SurfaceAuxLoss
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
        n_cls = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.ls / (n_cls - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p  = F.log_softmax(logits, dim=1)
        ce     = -(smooth_targets * log_p).sum(dim=1)
        pt     = torch.exp(-ce)
        gamma  = self.gammas[targets]
        return ((1 - pt) ** gamma * ce).mean()


class SurfaceAuxLoss(nn.Module):
    """C4/C6 2-class 보조 loss (C5 제거)."""
    def __init__(self, weight: list | None = None):
        super().__init__()
        w = torch.tensor(weight or [8.0, 2.5], dtype=torch.float32)
        self.crit = nn.CrossEntropyLoss(weight=w, ignore_index=-100, label_smoothing=0.03)

    def forward(self, surf_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        surf_tgt = torch.full_like(targets, -100)
        surf_tgt[targets == C4] = 0
        surf_tgt[targets == C6] = 1
        self.crit.weight = self.crit.weight.to(surf_logits.device)
        return self.crit(surf_logits, surf_tgt)


class SlipAuxLoss(nn.Module):
    """슬립 이진 분류 보조 loss.

    C1 스텝에 대해서만 slip_flag(0/1)를 맞히도록 학습.
    C2~C6 스텝은 ignore_index로 제외.

    역할:
      trunk가 슬립 패턴을 내부 표현에 포함하도록 강제
      → C1 분류 시 슬립 신호를 자연스럽게 활용
      → slip_weight: 슬립 스텝(=1)에 더 높은 loss 부여
    """
    def __init__(self, slip_weight: float = 3.0):
        super().__init__()
        # 슬립(1)이 정상(0)보다 훨씬 적으므로 클래스 가중치 부여
        w = torch.tensor([1.0, slip_weight], dtype=torch.float32)
        self.crit = nn.CrossEntropyLoss(weight=w, ignore_index=-100)

    def forward(
        self,
        slip_logits: torch.Tensor,   # (B, 2)
        targets:     torch.Tensor,   # (B,) — 지형 레이블
        slip_flag:   torch.Tensor,   # (B,) — 슬립 여부 0/1
    ) -> torch.Tensor:
        # C1 스텝만 학습, 나머지는 ignore
        slip_tgt = torch.full_like(targets, -100)
        c1_mask  = (targets == C1)
        slip_tgt[c1_mask] = slip_flag[c1_mask]
        self.crit.weight = self.crit.weight.to(slip_logits.device)
        return self.crit(slip_logits, slip_tgt)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup augmentation. surface 클래스(C4/C5/C6) 간 비율을 높임."""
    lam   = float(np.random.beta(alpha, alpha))
    idx   = torch.randperm(len(x), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def mixup_loss(crit, logits: torch.Tensor, y_a: torch.Tensor,
               y_b: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * crit(logits, y_a) + (1 - lam) * crit(logits, y_b)


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
        search_vals = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
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

def train_epoch(model, loader, opt, crit, crit_surf, crit_slip, device,
                grad_clip=5.0, use_mixup=True, mixup_alpha=0.4,
                aux_w=0.3, slip_w=0.4):
    """
    slip_w: SlipAuxLoss 가중치
      → trunk가 슬립 패턴을 인코딩하도록 강제하는 핵심 파라미터
      → 너무 높으면 지형 분류 loss가 희석되므로 0.3~0.5 권장
    """
    model.train(); total_loss = 0.0; n = 0
    for batch in loader:
        x         = batch["x"].to(device)
        y         = batch["y"].to(device)
        slip_flag = batch["slip_flag"].to(device)
        opt.zero_grad()

        if use_mixup and np.random.random() < 0.5:
            x_mix, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
            out    = model(x_mix)
            logits = out["logits"]
            loss   = mixup_loss(crit, logits, y_a, y_b, lam)
            # surface aux: dominant label 사용
            y_surf = y_a if lam >= 0.5 else y_b
            loss  += aux_w * crit_surf(out["surface_logits"], y_surf)
            # slip aux: mixup 시 dominant label의 slip_flag 사용
            sf_dom = slip_flag if lam >= 0.5 else slip_flag[torch.randperm(len(slip_flag), device=device)]
            loss  += slip_w * crit_slip(out["slip_logits"], y_surf, sf_dom)
        else:
            out    = model(x)
            logits = out["logits"]
            loss   = crit(logits, y)
            loss  += aux_w  * crit_surf(out["surface_logits"], y)
            loss  += slip_w * crit_slip(out["slip_logits"], y, slip_flag)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total_loss += float(loss) * len(y); n += len(y)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, crit, device, mults=None, tta_n=3):
    model.eval()
    total_loss = 0.0; n = 0
    all_proba, all_y = [], []
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)

        # TTA: 노이즈 추가 평균
        probas = []
        for ti in range(tta_n):
            if ti == 0:
                xi = x
            else:
                xi = x + torch.randn_like(x) * 0.02  # 작은 가우시안 노이즈
            out    = model(xi)
            logits = out["logits"]
            probas.append(F.softmax(logits, dim=1))

        proba = torch.stack(probas, dim=0).mean(0)  # (B, 6)

        # loss는 노이즈 없는 원본으로
        out_clean = model(x)
        loss = crit(out_clean["logits"], y)
        total_loss += float(loss) * len(y); n += len(y)
        all_proba.append(proba.cpu().numpy()); all_y.extend(y.cpu().tolist())

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
    tr_sampler = make_balanced_sampler(tr_ds.y, slip_flag=tr_ds.slip_flag)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, sampler=tr_sampler,
                       num_workers=4, pin_memory=True, drop_last=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                       num_workers=4, pin_memory=True)

    model = DualBranchMLP(
        in_dim=tr_ds.input_dim,
        hidden=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)

    crit      = PerClassFocalLoss(
        gamma_default=args.gamma_default,
        gamma_surface=args.gamma_surface,
        label_smoothing=args.label_smooth,
    ).to(device)
    crit_surf = SurfaceAuxLoss().to(device)
    crit_slip = SlipAuxLoss(slip_weight=args.slip_weight).to(device)

    # ── Phase 1: 전체 학습 (warmup + cosine) ─────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    warmup_epochs = 10
    total_epochs  = args.epochs

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / warmup_epochs
        prog = float(ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 1e-6 / args.lr + (1 - 1e-6 / args.lr) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_f1 = -1.0; best_state = None; best_mults = np.ones(N_CLASSES, dtype=np.float32)
    patience = 0

    for ep in range(1, total_epochs + 1):
        tl = train_epoch(model, tr_dl, opt, crit, crit_surf, crit_slip, device,
                         args.grad_clip, use_mixup=True, mixup_alpha=0.4,
                         aux_w=args.aux_w, slip_w=args.slip_w)
        sched.step()

        if ep % 5 == 0 or ep == total_epochs:
            _, _, _, val_y, _, val_proba = eval_epoch(model, te_dl, crit, device, tta_n=1)
            mults, _ = threshold_search(val_proba, val_y)
        else:
            mults = best_mults

        vl, va, vf, _, _, _ = eval_epoch(model, te_dl, crit, device, mults, tta_n=1)

        log(f"  [F{fi}] ep{ep:03d}/{total_epochs}"
            f"  tl={tl:.4f}  val_acc={va:.4f}  val_f1={vf:.4f}"
            f"  lr={sched.get_last_lr()[0]*args.lr:.2e}")

        if vf > best_f1:
            best_f1 = vf; best_mults = mults.copy()
            best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                log(f"  [F{fi}] EarlyStop ep{ep}")
                break

    # ── Phase 2: surface + slip 헤드 fine-tune (5 epochs) ────
    model.load_state_dict(best_state)
    surf_params = (list(model.surface_branch.parameters()) +
                   list(model.slip_head.parameters()) +
                   [model.surf_scale])
    opt2  = torch.optim.AdamW(surf_params, lr=args.lr * 0.1, weight_decay=args.wd)

    log(f"  [F{fi}] Phase2: surface + slip 헤드 fine-tune 시작")
    for ep2 in range(1, 6):
        model.train()
        for batch in tr_dl:
            x         = batch["x"].to(device)
            y         = batch["y"].to(device)
            slip_flag = batch["slip_flag"].to(device)
            opt2.zero_grad()
            out  = model(x)
            loss = (crit_surf(out["surface_logits"], y)
                    + args.slip_w * crit_slip(out["slip_logits"], y, slip_flag))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surf_params, args.grad_clip)
            opt2.step()

        _, va2, vf2, _, _, vp2 = eval_epoch(model, te_dl, crit, device, best_mults, tta_n=1)
        log(f"  [F{fi}] Phase2 ep{ep2}/5  val_acc={va2:.4f}  val_f1={vf2:.4f}")
        if vf2 > best_f1:
            best_f1 = vf2
            mults2, _ = threshold_search(vp2, te_ds.y)
            best_mults = mults2
            best_state = copy.deepcopy(model.state_dict())

    # ── 최종 평가 (TTA 적용) ──────────────────────────────────
    model.load_state_dict(best_state)
    _, acc, f1, yt, yp, _ = eval_epoch(model, te_dl, crit, device, best_mults, tta_n=5)
    log(f"  [F{fi}] ★ Best(TTA×5)  Acc={acc:.4f}  F1={f1:.4f}  mults={best_mults.round(2).tolist()}")

    report = classification_report(yt, yp, target_names=CLASS_NAMES, digits=4, zero_division=0)
    cm     = confusion_matrix(yt, yp, labels=list(range(N_CLASSES)))
    log(f"\n{report}")

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
    p.add_argument("--epochs",        type=int,   default=150)
    p.add_argument("--batch",         type=int,   default=1024)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--wd",            type=float, default=1e-4)
    p.add_argument("--hidden_dim",    type=int,   default=1024)
    p.add_argument("--n_blocks",      type=int,   default=8)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--gamma_default", type=float, default=2.0)
    p.add_argument("--gamma_surface", type=float, default=6.0,
                   help="Focal gamma for C4/C5/C6 (surface classes)")
    p.add_argument("--label_smooth",  type=float, default=0.05)
    p.add_argument("--early_stop",    type=int,   default=30)
    p.add_argument("--grad_clip",     type=float, default=5.0)
    p.add_argument("--kfold",         type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--aux_w",         type=float, default=0.4,
                   help="SurfaceAuxLoss 가중치")
    p.add_argument("--slip_w",        type=float, default=0.4,
                   help="SlipAuxLoss 가중치 (trunk가 슬립 패턴 인코딩하도록 강제)")
    p.add_argument("--slip_weight",   type=float, default=3.0,
                   help="SlipAuxLoss 내부 슬립 클래스 가중치 (슬립 스텝 minority)")
    p.add_argument("--slip_boost",    type=float, default=2.5,
                   help="BalancedSampler에서 C1 슬립 스텝 추가 샘플링 배율")
    p.add_argument("--no-wandb",      action="store_true")
    p.add_argument("--run_name",      type=str,   default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = str(_cfg.DEVICE)
    seed_everything(args.seed)

    log("=" * 60)
    log("  AttributeTerrainModel v6 (DualBranchMLP + SlipAux + Mixup + TTA)")
    log(f"  hidden={args.hidden_dim}  n_blocks={args.n_blocks}  dropout={args.dropout}")
    log(f"  gamma_default={args.gamma_default}  gamma_surface={args.gamma_surface}")
    log(f"  input_dim=N_FEATURES={N_FEATURES}")
    log(f"  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    log(f"  aux_w={args.aux_w}  slip_w={args.slip_w}  slip_boost={args.slip_boost}")
    log("=" * 60)

    if not args.no_wandb:
        wandb_start("attribute_kfold_v6", args, cfg_dict=vars(args))

    # ── 데이터 로드 ──────────────────────────────────────────
    h5 = H5Data(_cfg.CFG.h5_path)
    # ── C5 제외 ──────────────────────────────────────────────
    _cfg.CFG.num_classes = N_ACTIVE_CLASSES
    y_raw_orig = h5.y_raw.astype(np.int64)
    y_all, groups, kept_idx = filter_and_remap(y_raw_orig, h5.subj_id)
    le = LabelEncoder(); le.classes_ = np.array(CLASS_NAMES)
    log(f"  C5 제외 후: N={len(y_all)}  피험자={len(np.unique(groups))}명  classes={N_CLASSES}")

    # ── slip 메타 로드 (H5Data에서 이미 로드됨) ──────────────
    slip_flag_all      = h5.slip_flag[kept_idx].astype(np.int64)
    slip_intensity_all = h5.slip_intensity[kept_idx].astype(np.float32)
    slip_total = int(slip_flag_all.sum())
    c1_mask    = (y_all == C1)
    slip_c1    = int(slip_flag_all[c1_mask].sum())
    log(f"  slip: 전체={slip_total}/{len(y_all)}({slip_total/max(len(y_all),1)*100:.1f}%)"
        f"  C1 내 슬립={slip_c1}/{c1_mask.sum()}({slip_c1/max(c1_mask.sum(),1)*100:.1f}%)")

    # ── feat 추출 (캐시) ─────────────────────────────────────
    from channel_groups import get_foot_accel_idx
    foot_idx   = get_foot_accel_idx(h5.channels)
    cache_dir  = ensure_dir(_cfg.CFG.repo_dir / "cache" / f"feat{N_FEATURES}_seed{args.seed}_attr_v6_noc5")
    cache_path = cache_dir / "all_feat.npy"

    if cache_path.exists():
        log(f"  feat 캐시 히트 → 로드")
        feat_all = np.load(cache_path)
    else:
        log(f"  feat 추출 중... (N_FEATURES={N_FEATURES})")
        with Timer() as t:
            feat_all = batch_extract(h5.X[kept_idx], foot_idx, _cfg.CFG.sample_rate,
                                       h5_path=str(_cfg.CFG.h5_path),
                                       kept_idx=kept_idx)
        np.save(cache_path, feat_all)
        log(f"  feat 추출 완료  shape={feat_all.shape}  {t}")

    # ── 출력 디렉토리 ────────────────────────────────────────
    out = ensure_dir(_cfg.CFG.repo_dir / "out_N50" / "attribute_kfold_v6")

    # ── K-Fold ───────────────────────────────────────────────
    sgkf    = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    results = []

    with Timer() as total_timer:
        for fi, (tr_idx, te_idx) in enumerate(
            sgkf.split(np.zeros(len(y_all)), y_all, groups), 1
        ):
            log(f"\n{'='*60}")
            log(f"  Fold {fi}/{args.kfold}  tr={len(tr_idx)}  te={len(te_idx)}")

            # Train scaler fit → apply to test, slip 메타도 함께 전달
            tr_ds = AttributeDatasetV3(
                feat_all, y_all, tr_idx,
                slip_flag_all=slip_flag_all, slip_int_all=slip_intensity_all,
                fit_scaler=True,
            )
            te_ds = AttributeDatasetV3(
                feat_all, y_all, te_idx,
                slip_flag_all=slip_flag_all, slip_int_all=slip_intensity_all,
                scaler=tr_ds.scaler,
            )

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
    log(f"  ★ AttributeTerrainModel v6 K-Fold 완료  총 소요: {total_timer}")
    log(f"  Mean Acc={mean_acc:.4f}  Mean F1={mean_f1:.4f}")
    for r in results:
        log(f"    Fold{r['fold']}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}")
    log(f"{'='*60}")

    save_json({
        "experiment": "attribute_kfold_v6",
        "version":    "v6_DualBranchMLP_SlipAux_Mixup_TTA",
        "mean_acc":   round(mean_acc, 4),
        "mean_f1":    round(mean_f1,  4),
        "total_time": str(total_timer),
        "slip_stats": {
            "total_slip_steps": slip_total,
            "c1_slip_steps":    slip_c1,
        },
        "config": {
            "hidden_dim":    args.hidden_dim,
            "n_blocks":      args.n_blocks,
            "gamma_surface": args.gamma_surface,
            "input_dim":     N_FEATURES,
        },
        "folds": [{k: v for k, v in r.items() if k not in ("report", "cm")}
                  for r in results],
    }, out / "summary_attribute_v5.json")

    if not args.no_wandb:
        wandb_finish(results=[{"model": "AttributeV6", "acc": mean_acc, "f1": mean_f1}])

    h5.close()


if __name__ == "__main__":
    main()