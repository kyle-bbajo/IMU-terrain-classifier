# -*- coding: utf-8 -*-
"""
train_hierarchical.py — v11.5.1 (Review-Patched)
═══════════════════════════════════════════════════════
v11.5 → v11.5.1 리뷰 반영 수정:

  [R1] majority_vote_smooth: 짝수 window → 홀수 자동 보정
  [R2] majority_vote_by_group: 시간순 정렬 가정 NOTE 주석 명시
  [R3] train_superfusion Phase1: step_i = -1 초기화 (빈 loader 안전)
  [R4] WithinSubjectTripletLoss: valid_triplets 에포크 통계 로깅
  [R5] SubjectNormalizer: offline/transductive 설정 docstring 보강
  [R6] train_superfusion Phase2: grad accumulation Phase1과 통일

v11.5 핵심:
  [A] Subject-Wise BioMech Normalization
  [B] Within-Subject Triplet Loss  (margin=1.0, hard mining)
  [C] GRL + KinematicCrossAttn + S1 전이 + F1 EarlyStopping

목표: 85~93%
═══════════════════════════════════════════════════════"""

from __future__ import annotations

import sys, time, json, gc, warnings, math, argparse
warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass, field

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    wandb = None
    _WANDB_OK = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mode as scipy_mode
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from channel_groups import build_branch_idx
from models import M6_BranchCBAMCrossAug
from train_common import (
    log, H5Data,
    fit_bsc_on_train,
    make_branch_dataset, make_loader,
    save_cm, clear_fold_cache,
)

DEVICE = config.DEVICE

# ═══════════════════════════════════════════════
# 클래스 상수
# ═══════════════════════════════════════════════

FLAT_CLASSES     = [0, 3, 4, 5]
S2_3CLS_MAP      = {0: 0, 5: 1, 3: 2, 4: 2}
S2_3CLS_MAP_INV  = {0: 0, 1: 5}
S3_BINARY_MAP    = {3: 0, 4: 1}
S3_BINARY_MAP_INV= {0: 3, 1: 4}
CLASS_NAMES_ALL  = {0: "C1-미끄러운", 1: "C2-오르막", 2: "C3-내리막",
                    3: "C4-흙길",     4: "C5-잔디",   5: "C6-평지"}

# ═══════════════════════════════════════════════
# 하이퍼파라미터
# ═══════════════════════════════════════════════

S1_EPOCHS         = 60
S1_LR             = 5e-5
S1_PATIENCE       = 15
S1_SOFT_THRESHOLD = 0.50

S3_FFT_BINS  = 64
S3_FFT_DIM   = 128
FOCAL_GAMMA  = 1.5

SF_EPOCHS    = 120
SF_FT_EPOCHS = 40
SF_LR        = 1e-4
SF_PATIENCE  = 20
SF_AUX_W3    = 0.30
SF_AUX_WFLAT = 0.20
SF_AUX_WBIN  = 0.40
SF_WCONS     = 0.10
SF_WADV      = 0.10
SF_WTRIPLET  = 0.30
TRIPLET_MARGIN = 1.0

TCN_SEQ_LEN  = 9
TCN_HIDDEN   = 128
TCN_EPOCHS   = 40
TCN_LR       = 5e-4
TCN_PATIENCE = 10

# backward compat
E2E_EPOCHS    = SF_EPOCHS
E2E_FT_EPOCHS = SF_FT_EPOCHS
E2E_LR        = SF_LR
E2E_AUX_W3   = SF_AUX_W3
E2E_AUX_WB   = SF_AUX_WBIN
E2E_PATIENCE  = SF_PATIENCE


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SuperFusion + TCN Terrain Classifier v11.5.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--s1_epochs",    type=int,   default=S1_EPOCHS)
    p.add_argument("--s1_lr",        type=float, default=S1_LR)
    p.add_argument("--sf_epochs",    type=int,   default=SF_EPOCHS)
    p.add_argument("--sf_ft_epochs", type=int,   default=SF_FT_EPOCHS)
    p.add_argument("--sf_lr",        type=float, default=SF_LR)
    p.add_argument("--sf_patience",  type=int,   default=SF_PATIENCE)
    p.add_argument("--focal_gamma",  type=float, default=FOCAL_GAMMA)
    p.add_argument("--tcn_seq_len",  type=int,   default=TCN_SEQ_LEN)
    p.add_argument("--tcn_epochs",   type=int,   default=TCN_EPOCHS)
    p.add_argument("--vote_window",  type=int,   default=5,
                   help="Majority vote window size (0=off). "
                        "홀수 권장 — 짝수 입력 시 +1 자동 보정 [R1]")
    p.add_argument("--n_subjects",   type=int,   default=None)
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--wandb_project", type=str,  default="imu-terrain")
    p.add_argument("--run_name",     type=str,   default=None)
    return p.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global S1_EPOCHS, S1_LR
    global SF_EPOCHS, SF_FT_EPOCHS, SF_LR, SF_PATIENCE, FOCAL_GAMMA
    global TCN_SEQ_LEN, TCN_EPOCHS
    global E2E_EPOCHS, E2E_FT_EPOCHS, E2E_LR, E2E_PATIENCE
    S1_EPOCHS     = args.s1_epochs
    S1_LR         = args.s1_lr
    SF_EPOCHS     = args.sf_epochs
    SF_FT_EPOCHS  = args.sf_ft_epochs
    SF_LR         = args.sf_lr
    SF_PATIENCE   = args.sf_patience
    FOCAL_GAMMA   = args.focal_gamma
    TCN_SEQ_LEN   = args.tcn_seq_len
    TCN_EPOCHS    = args.tcn_epochs
    E2E_EPOCHS    = SF_EPOCHS
    E2E_FT_EPOCHS = SF_FT_EPOCHS
    E2E_LR        = SF_LR
    E2E_PATIENCE  = SF_PATIENCE
    if args.n_subjects is not None:
        config.apply_overrides(n_subjects=args.n_subjects)


# ═══════════════════════════════════════════════
# 훈련 곡선 추적기
# ═══════════════════════════════════════════════

@dataclass
class CurveTracker:
    name:   str
    losses: list = field(default_factory=list)
    accs:   list = field(default_factory=list)

    def record(self, loss: float | None = None, acc: float | None = None):
        if loss is not None: self.losses.append(round(loss, 6))
        if acc  is not None: self.accs.append(round(acc,  6))

    def save(self, out_dir: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"curve_{self.name}.json").write_text(
            json.dumps({"loss": self.losses, "acc": self.accs}, indent=2))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if self.losses:
            axes[0].plot(self.losses, color="steelblue")
            axes[0].set_title(f"{self.name} — Loss")
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
            axes[0].grid(alpha=0.3)
        if self.accs:
            axes[1].plot(self.accs, color="tomato")
            axes[1].set_title(f"{self.name} — Accuracy")
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Acc")
            axes[1].grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"curve_{self.name}.png", dpi=120)
        plt.close(fig)


# ═══════════════════════════════════════════════
# 공통 헬퍼
# ═══════════════════════════════════════════════

def _to_device(bi: dict, bio_f: torch.Tensor | None = None,
               yb: torch.Tensor | None = None):
    bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
    if not config.USE_AMP:
        bi = {k: v.float() for k, v in bi.items()}
    out = [bi]
    if bio_f is not None:
        out.append(bio_f.to(DEVICE, non_blocking=True).float())
    if yb is not None:
        out.append(yb.to(DEVICE, non_blocking=True))
    return tuple(out) if len(out) > 1 else out[0]


def _clone_state(model: nn.Module) -> dict:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def _run_epoch(model, loader, loss_fn, opt, scaler, params,
               has_bio=True, forward_fn=None) -> float:
    model.train()
    total_loss = n = 0
    opt.zero_grad(set_to_none=True)
    step_i = -1  # [R3] 초기화
    for step_i, batch in enumerate(loader):
        if has_bio:
            bi, bio_f, yb = batch
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                out = forward_fn(bi, bio_f) if forward_fn else model(bi, bio_f)
        else:
            bi, yb = batch
            bi, yb = _to_device(bi, yb=yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                out = model(bi)
        with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
            loss = loss_fn(out, yb) / config.GRAD_ACCUM_STEPS
        if scaler: scaler.scale(loss).backward()
        else:      loss.backward()
        if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
            if scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                opt.step()
            opt.zero_grad(set_to_none=True)
        total_loss += loss.item() * config.GRAD_ACCUM_STEPS * len(yb)
        n += len(yb)

    if step_i >= 0 and (step_i + 1) % config.GRAD_ACCUM_STEPS != 0:
        if scaler:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
            opt.step()
        opt.zero_grad(set_to_none=True)
    return total_loss / max(n, 1)


# ═══════════════════════════════════════════════
# 후처리
# ═══════════════════════════════════════════════

def auto_class_weights(y_flat: np.ndarray, num_classes: int = 6) -> torch.Tensor:
    present = np.unique(y_flat)
    w = np.ones(num_classes, dtype=np.float32)
    if len(present) > 0:
        balanced = compute_class_weight("balanced", classes=present, y=y_flat)
        for c, wc in zip(present, balanced):
            w[int(c)] = float(wc)
    parts = "  ".join(f"{i}={w[i]:.3f}" for i in range(num_classes))
    log(f"    auto class_weights (balanced): {parts}")
    return torch.tensor(w, dtype=torch.float32)


def majority_vote_smooth(preds: np.ndarray, window: int = 5) -> np.ndarray:
    """보행 연속성 기반 majority vote.

    [R1] 짝수 window는 중심 해석이 모호하므로 +1 하여 홀수로 보정한다.
         CLI에서 --vote_window 4 를 넘겨도 내부에서 5로 처리됨.
    """
    if window <= 1:
        return preds.copy()
    # [R1] 짝수 → 홀수 자동 보정
    if window % 2 == 0:
        window += 1
    half     = window // 2
    smoothed = preds.copy()
    for i in range(half, len(preds) - half):
        smoothed[i] = scipy_mode(
            preds[i - half: i + half + 1], keepdims=False
        ).mode
    return smoothed


def majority_vote_by_group(preds: np.ndarray,
                           groups: np.ndarray,
                           window: int) -> np.ndarray:
    """Subject 경계를 존중하는 majority vote.

    [R2] NOTE (시간순 정렬 가정):
        이 함수는 preds/groups 배열 안에서 같은 subject의 샘플들이
        이미 시간순(연속순)으로 정렬되어 있다고 가정한다.
        _make_seq_ds()는 subject별 순회 후 순서대로 append하므로
        TCN 경로에서는 이 가정이 성립한다.
        단, 데이터셋 셔플이나 외부 재배열이 있는 경우에는
        vote smoothing이 시간축 prior를 잘못 적용할 수 있으므로 주의.
    """
    out = preds.copy()
    if window <= 1:
        return out
    for g in np.unique(groups):
        m = groups == g
        out[m] = majority_vote_smooth(preds[m], window=window)
    return out


# ═══════════════════════════════════════════════
# BioMechFeatures (44-dim)
# ═══════════════════════════════════════════════

class BioMechFeatures(nn.Module):
    """생체역학 충격 피처 추출기 v11.2 (44-dim).

    0~19:  Accel 기반 (피크/비율/HF/std/감쇠/진동/분산비/SpectralCentroid/Duration)
    20~37: 자이로 기반 Foot·Shank·Thigh LT/RT (분산·피크)
    32~37: 비대칭/전달 기반 (가속도·자이로 비대칭, Foot→Shank 전달비)
    38~43: 통계 기반 Kurtosis·Skewness·ZCR LT/RT
    """
    N_BIO = 44

    def __init__(self) -> None:
        super().__init__()
        self.foot_z  = config.FOOT_Z_ACCEL_IDX
        self.shank_z = config.SHANK_Z_ACCEL_IDX
        self.hf_bin  = int(30 * config.TS / config.SAMPLE_RATE)
        self.gyro_lt = [3, 4, 5]
        self.gyro_rt = [9, 10, 11]

    @torch.no_grad()
    def forward(self, bi: dict) -> torch.Tensor:
        foot_x  = bi["Foot"].float()
        shank_x = bi["Shank"].float()
        thigh_x = bi.get("Thigh")
        if thigh_x is not None:
            thigh_x = thigh_x.float()
        eps = 1e-6

        fz_lt = foot_x[:,  self.foot_z[0],  :]
        fz_rt = foot_x[:,  self.foot_z[1],  :]
        sz_lt = shank_x[:, self.shank_z[0], :]
        sz_rt = shank_x[:, self.shank_z[1], :]

        f_pk_lt = fz_lt.abs().max(dim=1).values
        f_pk_rt = fz_rt.abs().max(dim=1).values
        s_pk_lt = sz_lt.abs().max(dim=1).values
        s_pk_rt = sz_rt.abs().max(dim=1).values

        ratio_lt = torch.log1p(f_pk_lt / (s_pk_lt + 1e-4))
        ratio_rt = torch.log1p(f_pk_rt / (s_pk_rt + 1e-4))
        hf_lt    = self._hf_ratio(fz_lt)
        hf_rt    = self._hf_ratio(fz_rt)
        std_lt   = fz_lt.std(dim=1)
        std_rt   = fz_rt.std(dim=1)

        T_half   = fz_lt.shape[1] // 2
        decay_lt = (fz_lt[:, :T_half].abs().mean(dim=1) /
                    (fz_lt[:, T_half:].abs().mean(dim=1) + eps))
        decay_rt = (fz_rt[:, :T_half].abs().mean(dim=1) /
                    (fz_rt[:, T_half:].abs().mean(dim=1) + eps))
        vib_lt   = (sz_lt[:, 1:] - sz_lt[:, :-1]).abs().mean(dim=1)
        vib_rt   = (sz_rt[:, 1:] - sz_rt[:, :-1]).abs().mean(dim=1)

        var_ratio_lt = torch.log1p(fz_lt.var(dim=1) / (sz_lt.var(dim=1) + 1e-4))
        var_ratio_rt = torch.log1p(fz_rt.var(dim=1) / (sz_rt.var(dim=1) + 1e-4))

        sc_lt  = self._spectral_centroid(fz_lt)
        sc_rt  = self._spectral_centroid(fz_rt)
        dur_lt = self._impact_duration(fz_lt)
        dur_rt = self._impact_duration(fz_rt)

        fg_lt = foot_x[:, self.gyro_lt, :]
        fg_rt = foot_x[:, self.gyro_rt, :]
        fg_var_lt  = torch.log1p(fg_lt.var(dim=2).sum(dim=1))
        fg_var_rt  = torch.log1p(fg_rt.var(dim=2).sum(dim=1))
        fg_peak_lt = fg_lt.abs().amax(dim=(1, 2))
        fg_peak_rt = fg_rt.abs().amax(dim=(1, 2))

        sg_lt = shank_x[:, self.gyro_lt, :]
        sg_rt = shank_x[:, self.gyro_rt, :]
        sg_var_lt  = torch.log1p(sg_lt.var(dim=2).sum(dim=1))
        sg_var_rt  = torch.log1p(sg_rt.var(dim=2).sum(dim=1))
        sg_peak_lt = sg_lt.abs().amax(dim=(1, 2))
        sg_peak_rt = sg_rt.abs().amax(dim=(1, 2))

        if thigh_x is not None:
            tg_lt = thigh_x[:, self.gyro_lt, :]
            tg_rt = thigh_x[:, self.gyro_rt, :]
            tg_var_lt  = torch.log1p(tg_lt.var(dim=2).sum(dim=1))
            tg_var_rt  = torch.log1p(tg_rt.var(dim=2).sum(dim=1))
            tg_peak_lt = tg_lt.abs().amax(dim=(1, 2))
            tg_peak_rt = tg_rt.abs().amax(dim=(1, 2))
        else:
            z = torch.zeros(foot_x.shape[0], device=foot_x.device)
            tg_var_lt = tg_var_rt = tg_peak_lt = tg_peak_rt = z

        asym_acc_lt = (f_pk_lt - s_pk_lt).abs()
        asym_acc_rt = (f_pk_rt - s_pk_rt).abs()
        asym_gy_lt  = (fg_var_lt - sg_var_lt).abs()
        asym_gy_rt  = (fg_var_rt - sg_var_rt).abs()

        foot_rms_lt  = fz_lt.pow(2).mean(dim=1).sqrt()
        foot_rms_rt  = fz_rt.pow(2).mean(dim=1).sqrt()
        shank_rms_lt = sz_lt.pow(2).mean(dim=1).sqrt()
        shank_rms_rt = sz_rt.pow(2).mean(dim=1).sqrt()
        trans_lt = torch.log1p(shank_rms_lt / (foot_rms_lt + eps))
        trans_rt = torch.log1p(shank_rms_rt / (foot_rms_rt + eps))

        kurt_lt = self._kurtosis(fz_lt)
        kurt_rt = self._kurtosis(fz_rt)
        skew_lt = self._skewness(fz_lt)
        skew_rt = self._skewness(fz_rt)
        zcr_lt  = self._zcr(fz_lt)
        zcr_rt  = self._zcr(fz_rt)

        return torch.stack([
            f_pk_lt, f_pk_rt, s_pk_lt, s_pk_rt,
            ratio_lt, ratio_rt, hf_lt, hf_rt,
            std_lt, std_rt, decay_lt, decay_rt,
            vib_lt, vib_rt, var_ratio_lt, var_ratio_rt,
            sc_lt, sc_rt, dur_lt, dur_rt,
            fg_var_lt, fg_var_rt, fg_peak_lt, fg_peak_rt,
            sg_var_lt, sg_var_rt, sg_peak_lt, sg_peak_rt,
            tg_var_lt, tg_var_rt, tg_peak_lt, tg_peak_rt,
            asym_acc_lt, asym_acc_rt,
            asym_gy_lt,  asym_gy_rt,
            trans_lt, trans_rt,
            kurt_lt, kurt_rt,
            skew_lt, skew_rt,
            zcr_lt,  zcr_rt,
        ], dim=1)

    def _kurtosis(self, x):
        mu  = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        return ((x - mu) / std).pow(4).mean(dim=1).clamp(-10, 30)

    def _skewness(self, x):
        mu  = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        return ((x - mu) / std).pow(3).mean(dim=1).clamp(-10, 10)

    def _zcr(self, x):
        signs   = torch.sign(x)
        signs   = torch.where(signs == 0, torch.ones_like(signs), signs)
        crosses = (signs[:, 1:] * signs[:, :-1] < 0).float()
        return crosses.mean(dim=1)

    def _hf_ratio(self, x):
        fft_mag = torch.fft.rfft(x, dim=1).abs()
        total   = fft_mag.pow(2).sum(dim=1) + 1e-6
        hf      = fft_mag[:, self.hf_bin:].pow(2).sum(dim=1)
        return hf / total

    def _spectral_centroid(self, x):
        fft_mag  = torch.fft.rfft(x, dim=1).abs()
        n_bins   = fft_mag.shape[1]
        freqs    = torch.arange(n_bins, device=x.device, dtype=torch.float32)
        power    = fft_mag.pow(2)
        centroid = (freqs * power).sum(dim=1) / (power.sum(dim=1) + 1e-6)
        return centroid / n_bins

    def _impact_duration(self, x, threshold=0.3):
        pk    = x.abs().max(dim=1, keepdim=True).values
        above = (x.abs() >= pk * threshold).float()
        return above.mean(dim=1)


# ═══════════════════════════════════════════════
# SubjectNormalizer
# ═══════════════════════════════════════════════

class SubjectNormalizer:
    """Subject-Wise BioMech Feature Normalization.

    각 subject의 피처를 해당 subject의 mean/std로 z-score 정규화.
    개인차(체중/키/보행 습관)를 제거하고 순수 지형 신호만 남긴다.

    [R5] 평가 설정 (StratifiedGroupKFold):
        · StratifiedGroupKFold에서 val/test subject는 group-held-out이므로
          사실상 항상 unseen 상태다.
        · Train subject:        fold train set 통계로 fit_transform
        · Val/Test subject:     해당 subject의 전체 window 통계로 정규화
            → No label leakage (raw feature 통계만 사용, label 접근 없음)
            → Offline / subject-batch / transductive normalization
               (strict online causal inference와 다름 — 논문 보고 시 반드시 명시)
    """
    def __init__(self):
        self.stats: dict = {}

    def fit(self, bio_feats: np.ndarray, groups: np.ndarray) -> None:
        self.stats = {}
        for sbj in np.unique(groups):
            mask  = groups == sbj
            feats = bio_feats[mask]
            self.stats[sbj] = (
                feats.mean(axis=0),
                feats.std(axis=0).clip(min=1e-6),
            )

    def transform(self, bio_feats: np.ndarray,
                  groups: np.ndarray) -> np.ndarray:
        out = bio_feats.copy().astype(np.float32)
        for sbj in np.unique(groups):
            mask = groups == sbj
            if sbj in self.stats:
                mu, std = self.stats[sbj]
            else:
                # [R5] unseen test subject → offline transductive normalization
                feats = bio_feats[mask]
                mu    = feats.mean(axis=0)
                std   = feats.std(axis=0).clip(min=1e-6)
            out[mask] = (bio_feats[mask] - mu) / std
        return out

    def fit_transform(self, bio_feats: np.ndarray,
                      groups: np.ndarray) -> np.ndarray:
        self.fit(bio_feats, groups)
        return self.transform(bio_feats, groups)


# ═══════════════════════════════════════════════
# WithinSubjectTripletLoss
# ═══════════════════════════════════════════════

class WithinSubjectTripletLoss(nn.Module):
    """Within-Subject Hard Negative Triplet Loss.

    같은 subject 안에서만 triplet 구성 → 순수 지형 신호 학습.

    [R4] 배치 내 유효 triplet 수를 에포크 단위로 집계·로깅.
         배치가 작거나 subject 다양성이 낮으면 triplet이 자주 0이 됨.
         → subject-aware batch sampler 또는 memory bank 사용 권장.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin  = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")
        # [R4] 에포크 통계
        self._valid_count = 0
        self._step_count  = 0

    def reset_epoch_stats(self) -> None:
        self._valid_count = 0
        self._step_count  = 0

    def epoch_stats_str(self) -> str:
        avg = self._valid_count / max(self._step_count, 1)
        return (f"valid_triplets={self._valid_count}"
                f"  avg/step={avg:.1f}"
                f"  steps={self._step_count}")

    def forward(self, emb: torch.Tensor,
                labels: torch.Tensor,
                sbj: torch.Tensor) -> torch.Tensor:
        anchors, positives, negatives = [], [], []

        for s in sbj.unique():
            s_mask = sbj == s
            s_emb  = emb[s_mask]
            s_lbl  = labels[s_mask]
            if s_lbl.unique().shape[0] < 2:
                continue
            dist = torch.cdist(s_emb, s_emb, p=2)
            for i in range(s_emb.shape[0]):
                cls      = s_lbl[i]
                pos_mask = (s_lbl == cls)
                neg_mask = (s_lbl != cls)
                pos_mask[i] = False
                if pos_mask.sum() < 1 or neg_mask.sum() < 1:
                    continue
                hard_pos = s_emb[pos_mask][dist[i][pos_mask].argmax()]
                hard_neg = s_emb[neg_mask][dist[i][neg_mask].argmin()]
                anchors.append(s_emb[i])
                positives.append(hard_pos)
                negatives.append(hard_neg)

        # [R4] 통계 누적
        self._valid_count += len(anchors)
        self._step_count  += 1

        if len(anchors) == 0:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)

        return self.loss_fn(
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives),
        )


# ═══════════════════════════════════════════════
# BioMechHead
# ═══════════════════════════════════════════════

class BioMechHead(nn.Module):
    def __init__(self, in_dim: int = BioMechFeatures.N_BIO,
                 out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim), nn.ReLU(),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Khosla et al. NeurIPS 2020 — 하위 호환."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)
        features = F.normalize(features, dim=1)
        sim      = torch.matmul(features, features.T) / self.temperature
        eye      = torch.eye(B, dtype=torch.bool, device=features.device)
        labels   = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & ~eye
        log_prob = sim - torch.logsumexp(sim.masked_fill(eye, -1e9), dim=1, keepdim=True)
        n_pos    = pos_mask.sum(1).float().clamp(min=1)
        return -(log_prob * pos_mask).sum(1).div(n_pos).mean()


class FocalLoss(nn.Module):
    """Lin et al. ICCV 2017."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class ArcFaceLoss(nn.Module):
    """Deng et al. CVPR 2019 — 하위 호환."""
    def __init__(self, feat_dim, num_classes=2, s=32.0, m=0.5):
        super().__init__()
        self.s = s; self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        cosine  = F.linear(F.normalize(features, dim=1),
                           F.normalize(self.weight, dim=1))
        cosine  = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta   = torch.acos(cosine)
        one_hot = F.one_hot(labels, cosine.shape[1]).float()
        logit   = self.s * (one_hot * torch.cos(theta + self.m)
                            + (1 - one_hot) * cosine)
        return F.cross_entropy(logit, labels)


class FFTBranch(nn.Module):
    """주파수 도메인 Branch (Stage3 / SuperFusion 공용)."""
    def __init__(self, n_bins=129, out_dim=128):
        super().__init__()
        self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Linear(n_bins * 2, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim), nn.ReLU(),
        )

    def forward(self, foot_x):
        sig_lt = foot_x[:, 2, :]
        sig_rt = foot_x[:, 8, :]
        fft_lt = F.normalize(torch.fft.rfft(sig_lt, dim=1).abs().pow(2)[:, :self.n_bins], dim=1)
        fft_rt = F.normalize(torch.fft.rfft(sig_rt, dim=1).abs().pow(2)[:, :self.n_bins], dim=1)
        return self.net(torch.cat([fft_lt, fft_rt], dim=1))


# ─── Stage2 / Stage3 (하위 호환 — 메인 경로 미사용) ───────────

class Stage2Model(nn.Module):
    def __init__(self, backbone, feat_dim, bio_dim=64, num_classes=4):
        super().__init__()
        self.backbone  = backbone
        self.bio_head  = BioMechHead(BioMechFeatures.N_BIO, bio_dim)
        total          = feat_dim + bio_dim
        self.proj_head = nn.Sequential(nn.Linear(total, 256), nn.ReLU(), nn.Linear(256, 128))
        self.classifier= nn.Sequential(nn.Linear(total, 256), nn.ReLU(),
                                        nn.Dropout(config.DROPOUT_CLF), nn.Linear(256, num_classes))
    def _extract(self, bi, bio_feat):
        return torch.cat([self.backbone.extract(bi), self.bio_head(bio_feat)], dim=1)
    def forward_proj(self, bi, bio_feat):
        return F.normalize(self.proj_head(self._extract(bi, bio_feat)), dim=1)
    def forward(self, bi, bio_feat):
        return self.classifier(self._extract(bi, bio_feat))


class Stage3Model(nn.Module):
    def __init__(self, backbone, feat_dim, bio_dim=128, embed_dim=128, fft_dim=S3_FFT_DIM):
        super().__init__()
        self.backbone   = backbone
        self.bio_head   = BioMechHead(BioMechFeatures.N_BIO, bio_dim)
        self.fft_branch = FFTBranch(n_bins=129, out_dim=fft_dim)
        total = feat_dim + bio_dim + fft_dim
        self.embed = nn.Sequential(
            nn.Linear(total, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(config.DROPOUT_CLF), nn.Linear(256, embed_dim), nn.BatchNorm1d(embed_dim))
        self.classifier = nn.Linear(embed_dim, 2)
    def _fuse(self, bi, bio_feat):
        return torch.cat([self.backbone.extract(bi), self.bio_head(bio_feat),
                          self.fft_branch(bi["Foot"].float())], dim=1)
    def forward_embed(self, bi, bio_feat):
        return F.normalize(self.embed(self._fuse(bi, bio_feat)), dim=1)
    def forward(self, bi, bio_feat):
        return self.classifier(self.embed(self._fuse(bi, bio_feat)))


# ═══════════════════════════════════════════════
# SuperFusion Model
# ═══════════════════════════════════════════════

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        lam, = ctx.saved_tensors
        return -lam.item() * grad, None


class GRL(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x, lam=1.0):
        return GradientReversalFn.apply(x, lam)


class KinematicCrossAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads,
                                           batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)
    def forward(self, foot, shank):
        q   = shank.unsqueeze(1)
        kv  = foot.unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        return self.norm(shank + out.squeeze(1))


class SuperFusionModel(nn.Module):
    """v11.4 Kinematic Cross-Attention + Subject-Adversarial GRL."""
    def __init__(self, backbone, feat_dim, bio_dim=128, fft_dim=S3_FFT_DIM,
                 n_subjects=50):
        super().__init__()
        self.backbone   = backbone
        self.bio_head   = BioMechHead(BioMechFeatures.N_BIO, bio_dim)
        self.fft_branch = FFTBranch(n_bins=129, out_dim=fft_dim)
        self.foot_proj  = nn.Linear(12, feat_dim)
        self.cross_attn = KinematicCrossAttention(feat_dim, n_heads=4)
        total = feat_dim + bio_dim + fft_dim
        self.shared = nn.Sequential(
            nn.Linear(total, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.20),
        )
        self.head_6cls = nn.Linear(256, 6)
        self.head_3cls = nn.Linear(256, 3)
        self.head_flat = nn.Linear(256, 3)
        self.head_bin  = nn.Linear(256, 2)
        self.grl         = GRL()
        self.subject_clf = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_subjects),
        )
        self.n_subjects = n_subjects

    def _embed(self, bi, bio_f):
        cnn       = self.backbone.extract(bi)
        bio       = self.bio_head(bio_f)
        fft       = self.fft_branch(bi["Foot"].float())
        foot_proj = self.foot_proj(bi["Foot"].float().mean(dim=-1))
        cnn_att   = self.cross_attn(foot_proj, cnn)
        return self.shared(torch.cat([cnn_att, bio, fft], dim=1))

    def forward(self, bi, bio_f, lam=1.0):
        emb        = self._embed(bi, bio_f)
        subj_logit = self.subject_clf(self.grl(emb, lam))
        return (self.head_6cls(emb), self.head_3cls(emb),
                self.head_flat(emb), self.head_bin(emb),
                subj_logit, emb)

    def predict(self, bi, bio_f):
        return self.head_6cls(self._embed(bi, bio_f))

    def embed(self, bi, bio_f):
        return self._embed(bi, bio_f)


def consistency_kl_loss(l6, l3):
    p6     = torch.softmax(l6.float(), dim=1)
    p_flat = p6[:, 0] + p6[:, 3] + p6[:, 4] + p6[:, 5]
    p3_from6 = torch.stack([p_flat, p6[:, 1], p6[:, 2]], dim=1).clamp(1e-8, 1.0)
    p3_head  = torch.softmax(l3.float(), dim=1).clamp(1e-8, 1.0)
    return F.kl_div(p3_head.log(), p3_from6, reduction="batchmean")


# ═══════════════════════════════════════════════
# TCN Sequence Refiner
# ═══════════════════════════════════════════════

class _TCNBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn   = nn.BatchNorm1d(ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        return x + self.drop(self.act(self.bn(self.conv(x)[:, :, :x.shape[2]])))


class TCNRefiner(nn.Module):
    def __init__(self, in_dim=256, hidden=TCN_HIDDEN, num_classes=6):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.tcn  = nn.Sequential(
            _TCNBlock(hidden, 1), _TCNBlock(hidden, 2), _TCNBlock(hidden, 4))
        self.head = nn.Linear(hidden, num_classes)
    def forward(self, x):
        h = self.proj(x).transpose(1, 2)
        h = self.tcn(h).transpose(1, 2)
        return self.head(h)


# ═══════════════════════════════════════════════
# Dataset / DataLoader
# ═══════════════════════════════════════════════

class FlatBranchDataset(Dataset):
    """하위 호환."""
    def __init__(self, branch_ds, bio_extractor, flat_mask, y_flat):
        self.ds      = branch_ds
        self.bio     = bio_extractor
        self.indices = np.where(flat_mask)[0]
        self.y_flat  = y_flat
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        bi, _ = self.ds[int(self.indices[i])]
        with torch.no_grad():
            bio_f = self.bio({
                "Foot":  bi["Foot"].unsqueeze(0).float(),
                "Shank": bi["Shank"].unsqueeze(0).float(),
            }).squeeze(0)
        return bi, bio_f, int(self.y_flat[i])


def flat_collate(batch):
    bi_keys = batch[0][0].keys()
    bi  = {k: torch.stack([b[0][k] for b in batch]) for k in bi_keys}
    bio = torch.stack([b[1] for b in batch])
    y   = torch.tensor([b[2] for b in batch], dtype=torch.long)
    sbj = (torch.tensor([b[3] for b in batch], dtype=torch.long)
           if len(batch[0]) >= 4
           else torch.full((len(batch),), -1, dtype=torch.long))
    return bi, bio, y, sbj


def make_flat_loader(ds, shuffle, balanced=False):
    sampler     = None
    use_shuffle = shuffle
    if shuffle and balanced:
        classes, counts = np.unique(ds.y_flat, return_counts=True)
        sample_w = (1.0 / counts.astype(np.float64))[ds.y_flat]
        sampler  = WeightedRandomSampler(sample_w, len(ds.y_flat), replacement=True)
        use_shuffle = False
    return DataLoader(ds, batch_size=config.BATCH, shuffle=use_shuffle,
                      sampler=sampler, collate_fn=flat_collate,
                      drop_last=shuffle, pin_memory=config.USE_GPU)


class AllDataBioDataset(Dataset):
    """6cls + BioMech — SuperFusion 학습용."""
    def __init__(self, branch_ds, bio_extractor, y_all, groups=None,
                 bio_feats_norm=None):
        self.ds             = branch_ds
        self.bio            = bio_extractor
        self.y              = y_all
        self.bio_feats_norm = bio_feats_norm
        if groups is not None:
            unique_sbj = sorted(set(groups.tolist()))
            sbj_map    = {s: i for i, s in enumerate(unique_sbj)}
            self.sbj   = np.array([sbj_map[g] for g in groups], dtype=np.int64)
        else:
            self.sbj = np.full(len(y_all), -1, dtype=np.int64)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        bi, _ = self.ds[i]
        if self.bio_feats_norm is not None:
            bio_f = torch.from_numpy(self.bio_feats_norm[i]).float()
        else:
            with torch.no_grad():
                bio_f = self.bio({k: v.unsqueeze(0) for k, v in bi.items()}).squeeze(0)
        return bi, bio_f, int(self.y[i]), int(self.sbj[i])


def make_all_loader(ds, shuffle, balanced=False):
    sampler     = None
    use_shuffle = shuffle
    if shuffle and balanced:
        classes, counts = np.unique(ds.y, return_counts=True)
        sample_w = (1.0 / counts.astype(np.float64))[ds.y]
        sampler  = WeightedRandomSampler(sample_w, len(ds.y), replacement=True)
        use_shuffle = False
        log(f"      ★ E2E 균형 샘플링: {dict(zip(classes.tolist(), counts.tolist()))}")
    return DataLoader(ds, batch_size=config.BATCH, shuffle=use_shuffle,
                      sampler=sampler, collate_fn=flat_collate,
                      drop_last=shuffle, pin_memory=config.USE_GPU)


def _make_sch(opt, epochs, warmup=10, min_lr=config.MIN_LR, base_lr=None):
    base = base_lr or opt.param_groups[0]["lr"]
    def fn(ep):
        if ep < warmup: return float(ep + 1) / warmup
        prog = float(ep - warmup) / max(epochs - warmup, 1)
        cos  = 0.5 * (1.0 + math.cos(math.pi * prog))
        mf   = min_lr / base
        return mf + (1.0 - mf) * cos
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# ═══════════════════════════════════════════════
# Inner Val Split / Eval Helpers
# ═══════════════════════════════════════════════

def _inner_val_split(tr_idx, groups, val_ratio=0.15):
    tr_groups  = groups[tr_idx]
    unique_sbj = np.unique(tr_groups)
    n_val_sbj  = max(1, int(len(unique_sbj) * val_ratio))
    rng        = np.random.default_rng(config.SEED)
    val_sbj    = set(rng.choice(unique_sbj, n_val_sbj, replace=False).tolist())
    mask       = np.array([g not in val_sbj for g in tr_groups])
    log(f"    inner split: tr={mask.sum()}  val={(~mask).sum()}"
        f"  val_sbj={sorted(val_sbj)}")
    return tr_idx[mask], tr_idx[~mask]


def _eval_flat_dl(model, loader, crit):
    model.eval()
    vl_sum = va_c = va_n = 0
    with torch.inference_mode():
        for bi, bio_f, yb, _ in loader:
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model(bi, bio_f)
                loss   = crit(logits, yb)
            vl_sum += loss.item() * len(yb)
            va_c   += (logits.argmax(1) == yb).sum().item()
            va_n   += len(yb)
    return vl_sum / max(va_n, 1), va_c / max(va_n, 1)


# ═══════════════════════════════════════════════
# Stage1 — 3cls CE warmup
# ═══════════════════════════════════════════════

def _get_feat_dim(backbone):
    n_groups = len(backbone.names)
    n_extra  = (1 if getattr(backbone, "use_fft",          False) else 0) + \
               (1 if getattr(backbone, "use_foot_impact",  False) else 0) + \
               (1 if getattr(backbone, "use_shank_impact", False) else 0)
    return config.FEAT_DIM * (n_groups + n_extra)


class _S1Wrapper(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head     = head
    def forward(self, bi):
        return self.head(self.backbone.extract(bi))


def _y6_to_y3(y6):
    y3 = torch.zeros_like(y6)
    y3[y6 == 1] = 1
    y3[y6 == 2] = 2
    return y3


def train_stage1(backbone, tr_dl, val_dl, te_dl, tag="", curve_dir=None):
    feat_dim = _get_feat_dim(backbone)
    head     = nn.Linear(feat_dim, 3).to(DEVICE)
    model    = _S1Wrapper(backbone, head).to(DEVICE)
    params   = list(model.parameters())
    opt      = torch.optim.AdamW(params, lr=S1_LR, weight_decay=config.WEIGHT_DECAY)
    sch      = _make_sch(opt, S1_EPOCHS, warmup=10, base_lr=S1_LR)
    crit     = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler   = GradScaler(enabled=(config.USE_AMP and config.AMP_DTYPE == torch.float16))
    curve    = CurveTracker(f"S1_{tag.replace('[','').replace(']','')}")

    best_va = 0.0; best_state = None; patience = 0; t0 = time.time()
    log(f"  {tag} Stage1 3cls ({S1_EPOCHS}ep, LR={S1_LR:.0e})")

    for ep in range(1, S1_EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        step_i = -1
        for step_i, (bi, yb) in enumerate(tr_dl):
            bi, yb = _to_device(bi, yb=yb)
            yb3    = _y6_to_y3(yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                loss = crit(model(bi), yb3) / config.GRAD_ACCUM_STEPS
            if scaler: scaler.scale(loss).backward()
            else:      loss.backward()
            if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                    opt.step()
                opt.zero_grad(set_to_none=True)
        if step_i >= 0 and (step_i + 1) % config.GRAD_ACCUM_STEPS != 0:
            if scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                opt.step()
            opt.zero_grad(set_to_none=True)

        model.eval()
        va_c = va_n = 0
        with torch.inference_mode():
            for bi, yb in val_dl:
                bi, yb = _to_device(bi, yb=yb)
                yb3 = _y6_to_y3(yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model(bi)
                va_c += (logits.argmax(1) == yb3).sum().item()
                va_n += len(yb3)
        sch.step()
        va = va_c / max(va_n, 1)
        curve.record(acc=va)
        if va > best_va:
            best_va = va; best_state = _clone_state(model); patience = 0
        else:
            patience += 1
            if patience >= S1_PATIENCE:
                log(f"  {tag} S1 EarlyStop ep{ep}"); break
        if ep % 20 == 0:
            log(f"  {tag} S1 ep{ep:03d}/{S1_EPOCHS}"
                f"  val={va:.4f}  best={best_va:.4f}  ({time.time()-t0:.0f}s)")

    if best_state: model.load_state_dict(best_state); model.to(DEVICE)
    log(f"  {tag} S1 완료  best_val={best_va:.4f}")
    if curve_dir: curve.save(curve_dir)

    model.eval()
    ps, ls, probs = [], [], []
    with torch.inference_mode():
        for bi, yb in te_dl:
            bi, yb = _to_device(bi, yb=yb)
            yb3    = _y6_to_y3(yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model(bi)
            probs.append(torch.softmax(logits.float(), dim=1).cpu())
            ps.append(logits.argmax(1).cpu()); ls.append(yb3.cpu())
    s1_probs = torch.cat(probs).numpy()
    hard_p   = torch.cat(ps).numpy().astype(np.int64)
    s1_soft  = np.where(s1_probs[:, 0] >= S1_SOFT_THRESHOLD,
                        np.int64(0), hard_p).astype(np.int64)
    return s1_soft, torch.cat(ls).numpy(), s1_probs, model


# ═══════════════════════════════════════════════
# SuperFusion Training
# ═══════════════════════════════════════════════

def train_superfusion(backbone, tr_dl, val_dl, te_dl, tag="",
                      curve_dir=None, n_subjects=50):
    """SuperFusion v11.5.1.

    Phase1: 4-head multi-task + GRL + Triplet
    Phase2: 6cls fine-tune  [R6: grad accumulation Phase1과 통일]
    """
    feat_dim = _get_feat_dim(backbone)
    model    = SuperFusionModel(backbone, feat_dim, n_subjects=n_subjects).to(DEVICE)
    params   = list(model.parameters())

    all_y = np.asarray(tr_dl.dataset.y, dtype=np.int64)
    cls_w = auto_class_weights(all_y, num_classes=6).to(DEVICE)
    log(f"  {tag} class_weights: {[f'{w:.2f}' for w in cls_w.tolist()]}")

    opt    = torch.optim.AdamW(params, lr=SF_LR, weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, SF_EPOCHS, warmup=10, base_lr=SF_LR)
    scaler = GradScaler(enabled=(config.USE_AMP and config.AMP_DTYPE == torch.float16))
    curve  = CurveTracker(f"SF_{tag.replace('[','').replace(']','')}")

    crit_6    = FocalLoss(gamma=1.5, weight=cls_w)
    crit_3    = nn.CrossEntropyLoss(label_smoothing=0.05)
    crit_flat = nn.CrossEntropyLoss(label_smoothing=0.05)
    crit_bin  = nn.CrossEntropyLoss()
    crit_subj = nn.CrossEntropyLoss()
    crit_trip = WithinSubjectTripletLoss(margin=TRIPLET_MARGIN)
    flat3_map = {0: 0, 3: 1, 4: 1, 5: 2}

    best_va = 0.0; best_state = None; patience = 0; t0 = time.time()
    log(f"  {tag} SF Phase1 ({SF_EPOCHS}ep, LR={SF_LR:.0e})"
        f"  [R3] step_i init  [R4] triplet logging  [R6] grad_accum unified")

    for ep in range(1, SF_EPOCHS + 1):
        p   = ep / SF_EPOCHS
        lam = float(2.0 / (1.0 + math.exp(-10 * p)) - 1.0)

        model.train()
        opt.zero_grad(set_to_none=True)
        crit_trip.reset_epoch_stats()  # [R4]

        # [R3] step_i = -1 초기화 — 빈 loader에서도 안전
        step_i = -1
        for step_i, (bi, bio_f, yb, sbj) in enumerate(tr_dl):
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            sbj       = sbj.to(DEVICE)
            yb3       = _y6_to_y3(yb)
            flat_mask = ((yb == 0) | (yb == 3) | (yb == 4) | (yb == 5))
            c4c5_mask = ((yb == 3) | (yb == 4))

            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                l6, l3, lflat, lbin, l_subj, emb = model(bi, bio_f, lam=lam)
                loss = crit_6(l6, yb) + SF_AUX_W3 * crit_3(l3, yb3)

                if flat_mask.sum() > 2:
                    yb_flat3 = torch.tensor(
                        [flat3_map.get(int(c), 1) for c in yb[flat_mask].tolist()],
                        dtype=torch.long, device=DEVICE)
                    loss = loss + SF_AUX_WFLAT * crit_flat(lflat[flat_mask], yb_flat3)

                if c4c5_mask.sum() > 1:
                    yb_bin = (yb[c4c5_mask] == 4).long()
                    loss   = loss + SF_AUX_WBIN * crit_bin(lbin[c4c5_mask], yb_bin)

                loss = loss + SF_WCONS * consistency_kl_loss(l6, l3)
                if l_subj is not None and (sbj >= 0).all():
                    loss = loss + SF_WADV * crit_subj(l_subj, sbj)
                loss = loss + SF_WTRIPLET * crit_trip(emb.float(), yb, sbj)
                loss = loss / config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

        # [R3] 마지막 배치 (step_i >= 0 보장됨)
        if step_i >= 0 and (step_i + 1) % config.GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        sch.step()

        # [R4] 에포크별 triplet 통계 출력 + 낮을 때 경고
        if ep % 15 == 0 or ep == 1:
            trip_avg = crit_trip._valid_count / max(crit_trip._step_count, 1)
            log(f"  {tag} [R4-Triplet] ep{ep}: {crit_trip.epoch_stats_str()}")
            if trip_avg < 1.0:
                log(f"  {tag} [WARN] triplet 활성도 낮음: avg_valid_triplets/step={trip_avg:.2f}"
                    f" — subject-aware sampler 또는 memory bank 도입을 검토하세요.")

        # validation
        model.eval()
        va_ps, va_ls = [], []
        with torch.inference_mode():
            for bi, bio_f, yb, _ in val_dl:
                bi, bio_f, yb = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model.predict(bi, bio_f)
                va_ps.append(logits.argmax(1).cpu())
                va_ls.append(yb.cpu())
        va_p    = torch.cat(va_ps).numpy()
        va_l    = torch.cat(va_ls).numpy()
        va_acc  = accuracy_score(va_l, va_p)
        va_f1   = f1_score(va_l, va_p, average="macro", zero_division=0)
        va_score = 0.4 * va_acc + 0.6 * va_f1
        curve.record(acc=va_score)

        if va_score > best_va:
            best_va = va_score; best_state = _clone_state(model); patience = 0
        else:
            patience += 1
            if patience >= SF_PATIENCE and ep > 20:
                log(f"  {tag} SF EarlyStop ep{ep}  best={best_va:.4f}"); break

        if ep % 15 == 0 or ep == 1:
            log(f"  {tag} SF ep{ep:03d}/{SF_EPOCHS}"
                f"  acc={va_acc:.4f}  f1={va_f1:.4f}"
                f"  score={va_score:.4f}  best={best_va:.4f}"
                f"  lr={opt.param_groups[0]['lr']:.1e}  ({time.time()-t0:.0f}s)")
            if _WANDB_OK and wandb.run is not None:
                trip_avg = (crit_trip._valid_count /
                            max(crit_trip._step_count, 1))
                wandb.log({
                    f"{tag}/sf_p1_val_acc":   va_acc,
                    f"{tag}/sf_p1_val_f1":    va_f1,
                    f"{tag}/sf_p1_val_score": va_score,
                    f"{tag}/sf_p1_lr":        opt.param_groups[0]['lr'],
                    f"{tag}/triplet_valid_avg": trip_avg,
                }, step=ep)

    if best_state: model.load_state_dict(best_state); model.to(DEVICE)
    log(f"  {tag} SF Phase1 완료  best_val={best_va:.4f}")
    if curve_dir: curve.save(curve_dir)

    # ── Phase2: 6cls fine-tune ────────────────
    for p in model.head_3cls.parameters(): p.requires_grad = False
    for p in model.head_flat.parameters(): p.requires_grad = False
    for p in model.head_bin.parameters():  p.requires_grad = False
    params2  = [p for p in model.parameters() if p.requires_grad]
    opt2     = torch.optim.AdamW(params2, lr=SF_LR * 0.5,
                                  weight_decay=config.WEIGHT_DECAY)
    crit_ft  = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.03)
    best_va2 = 0.0; best_st2 = None; pat2 = 0

    log(f"  {tag} SF Phase2 ({SF_FT_EPOCHS}ep, LR={SF_LR*0.5:.0e})"
        f"  grad_accum={config.GRAD_ACCUM_STEPS}  [R6: Phase1/2 정책 통일]")
    for ep in range(1, SF_FT_EPOCHS + 1):
        model.train()
        opt2.zero_grad(set_to_none=True)
        # [R6] Phase2도 grad accumulation 적용
        step_i = -1
        for step_i, (bi, bio_f, yb, _) in enumerate(tr_dl):
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                loss = crit_ft(model.predict(bi, bio_f), yb) / config.GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(params2, config.GRAD_CLIP_NORM)
                scaler.step(opt2); scaler.update()
                opt2.zero_grad(set_to_none=True)
        if step_i >= 0 and (step_i + 1) % config.GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(params2, config.GRAD_CLIP_NORM)
            scaler.step(opt2); scaler.update()
            opt2.zero_grad(set_to_none=True)

        model.eval()
        va_p2, va_l2 = [], []
        with torch.inference_mode():
            for bi, bio_f, yb, _ in val_dl:
                bi, bio_f, yb = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model.predict(bi, bio_f)
                va_p2.append(logits.argmax(1).cpu()); va_l2.append(yb.cpu())
        vp2 = torch.cat(va_p2).numpy(); vl2 = torch.cat(va_l2).numpy()
        va2 = 0.4 * accuracy_score(vl2, vp2) + 0.6 * f1_score(vl2, vp2,
                                                                  average="macro",
                                                                  zero_division=0)
        if va2 > best_va2:
            best_va2 = va2; best_st2 = _clone_state(model); pat2 = 0
        else:
            pat2 += 1
            if pat2 >= 12: break

    if best_st2: model.load_state_dict(best_st2); model.to(DEVICE)
    for p in model.parameters(): p.requires_grad = True
    log(f"  {tag} SF Phase2 완료  best_val={best_va2:.4f}")

    def _predict_dl(dl):
        model.eval(); ps, ls = [], []
        with torch.inference_mode():
            for bi, bio_f, yb, _ in dl:
                bi, bio_f, yb = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model.predict(bi, bio_f)
                ps.append(logits.argmax(1).cpu()); ls.append(yb.cpu())
        return torch.cat(ps).numpy(), torch.cat(ls).numpy()

    va_preds, va_labels = _predict_dl(val_dl)
    te_preds, te_labels = _predict_dl(te_dl)
    return va_preds, va_labels, te_preds, te_labels, model


# ═══════════════════════════════════════════════
# TCN Refiner
# ═══════════════════════════════════════════════

def train_tcn_refiner(sf_model, tr_all_ds, va_all_ds, te_all_ds,
                      tr_groups, va_groups, te_groups,
                      sf_va_preds, sf_va_labels,
                      sf_te_preds, sf_te_labels,
                      vote_window=0, tag=""):
    """TCN Sequence Refiner — subject-aware.

    Returns:
        tcn_preds            : (N_te_center,)  TCN test 예측
        tcn_labels           : (N_te_center,)  TCN test 정답 (center-valid subset)
        te_center_groups     : (N_te_center,)  test center 샘플의 subject 그룹
        sf_test_preds_center : (N_te_center,)  SF test 예측 (center-valid subset)
        sf_test_labels_center: (N_te_center,)  SF test 정답 (center-valid subset)
        best_val_tcn_score   : float  val 기준 TCN 최고 score (0.4×acc + 0.6×f1 + vote)
        best_val_sf_score    : float  val 기준 SF  최고 score (동일 subset + vote)
    """
    def _extract_embs(ds, model):
        dl = DataLoader(ds, batch_size=config.BATCH, shuffle=False,
                        collate_fn=flat_collate, pin_memory=config.USE_GPU)
        embs, lbls = [], []
        model.eval()
        with torch.inference_mode():
            for bi, bio_f, yb, _ in dl:
                bi, bio_f, _ = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    embs.append(sf_model.embed(bi, bio_f).cpu())
                lbls.append(yb)
        return torch.cat(embs), torch.cat(lbls)

    log(f"  {tag} TCN: embedding 추출...")
    tr_emb, tr_lbl = _extract_embs(tr_all_ds, sf_model)
    va_emb, va_lbl = _extract_embs(va_all_ds, sf_model)
    te_emb, te_lbl = _extract_embs(te_all_ds, sf_model)

    def _make_seq_ds(emb, lbl, grp) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Subject-aware stride=1 sliding window.

        [R2] NOTE: 각 subject의 샘플은 emb/grp 배열에서 시간순으로 정렬되어 있다고 가정.
        subject 순회 후 순서대로 append하므로 TCN 경로에서는 이 가정이 성립.
        """
        seqs, tgts, flat_idxs, seq_groups = [], [], [], []
        half = TCN_SEQ_LEN // 2
        for sbj in np.unique(grp):
            idx = np.where(grp == sbj)[0]
            se  = emb[idx]; sl = lbl[idx]; N = len(idx)
            if N < TCN_SEQ_LEN:
                pad_r = TCN_SEQ_LEN - N
                seqs.append(F.pad(se.T, (0, pad_r)).T)
                tgts.append(int(sl[N // 2]))
                flat_idxs.append(int(idx[N // 2]))
                seq_groups.append(sbj)
            else:
                for i in range(N - TCN_SEQ_LEN + 1):
                    c = i + half
                    seqs.append(se[i: i + TCN_SEQ_LEN])
                    tgts.append(int(sl[c]))
                    flat_idxs.append(int(idx[c]))
                    seq_groups.append(sbj)
        return (torch.stack(seqs),
                torch.tensor(tgts, dtype=torch.long),
                np.array(flat_idxs,  dtype=np.int64),
                np.array(seq_groups))

    log(f"  {tag} TCN: sequence 생성...")
    tr_seq, tr_tgt, tr_cidx, tr_cgrp = _make_seq_ds(tr_emb, tr_lbl, tr_groups)
    va_seq, va_tgt, va_cidx, va_cgrp = _make_seq_ds(va_emb, va_lbl, va_groups)
    te_seq, te_tgt, te_cidx, te_cgrp = _make_seq_ds(te_emb, te_lbl, te_groups)
    log(f"  {tag} TCN: tr={len(tr_seq)}  va={len(va_seq)}  te={len(te_seq)}")

    # [Edge case] sequence가 비어 있으면 TCN 학습 불가 → SF 결과 그대로 반환
    if len(tr_seq) == 0 or len(va_seq) == 0 or len(te_seq) == 0:
        log(f"  {tag} [WARN] TCN sequence 부족 (tr={len(tr_seq)} va={len(va_seq)}"
            f" te={len(te_seq)}) — SF 결과를 그대로 사용합니다.")
        sf_te_preds_sub  = sf_te_preds[te_cidx]
        sf_te_labels_sub = sf_te_labels[te_cidx]
        # best_val_tcn=-inf, best_val_sf_sub=+inf → use_tcn=False 보장
        # 로그에서 "SF fallback" 으로 올바르게 표시됨
        return (sf_te_preds_sub, sf_te_labels_sub, te_cgrp,
                sf_te_preds_sub, sf_te_labels_sub,
                float("-inf"), float("+inf"))

    sf_va_preds_sub  = sf_va_preds[va_cidx]
    sf_va_labels_sub = sf_va_labels[va_cidx]
    sf_te_preds_sub  = sf_te_preds[te_cidx]
    sf_te_labels_sub = sf_te_labels[te_cidx]

    tcn       = TCNRefiner(in_dim=256, hidden=TCN_HIDDEN, num_classes=6).to(DEVICE)
    cls_w_tcn = auto_class_weights(tr_tgt.numpy(), num_classes=6).to(DEVICE)
    crit      = nn.CrossEntropyLoss(weight=cls_w_tcn)
    opt       = torch.optim.AdamW(tcn.parameters(), lr=TCN_LR, weight_decay=1e-4)
    best_va   = 0.0; best_st = None; pat = 0; half = TCN_SEQ_LEN // 2

    # drop_last: 샘플이 batch_size보다 적으면 한 배치도 안 생기므로 동적으로 결정
    _tcn_bs   = min(512, len(tr_seq))
    tr_loader = DataLoader(list(zip(tr_seq, tr_tgt)),
                           batch_size=_tcn_bs, shuffle=True,
                           drop_last=(len(tr_seq) >= 512))
    va_loader = DataLoader(list(zip(va_seq, va_tgt)),
                           batch_size=512, shuffle=False)

    log(f"  {tag} TCN 학습 ({TCN_EPOCHS}ep, LR={TCN_LR:.0e})")
    for ep in range(1, TCN_EPOCHS + 1):
        tcn.train()
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE).float(); yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(tcn(xb)[:, half, :], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tcn.parameters(), 1.0)
            opt.step()

        tcn.eval()
        vp, vl = [], []
        with torch.inference_mode():
            for xb, yb in va_loader:
                xb = xb.to(DEVICE).float()
                vp.append(tcn(xb)[:, half, :].argmax(1).cpu())
                vl.append(yb)
        vp = torch.cat(vp).numpy(); vl = torch.cat(vl).numpy()
        va = 0.4 * accuracy_score(vl, vp) + 0.6 * f1_score(vl, vp,
                                                               average="macro",
                                                               zero_division=0)
        if va > best_va:
            best_va = va; best_st = {k: v.clone() for k, v in tcn.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= TCN_PATIENCE: break
        if ep % 10 == 0:
            log(f"  {tag} TCN ep{ep:02d}/{TCN_EPOCHS}  score={va:.4f}  best={best_va:.4f}")

    if best_st: tcn.load_state_dict(best_st)
    log(f"  {tag} TCN 완료  best_val={best_va:.4f}")

    def _apply_vote(preds, grps):
        return majority_vote_by_group(preds, grps, window=vote_window) \
               if vote_window > 0 else preds.copy()

    sf_va_voted = _apply_vote(sf_va_preds_sub, va_cgrp)
    best_val_sf_sub = (0.4 * accuracy_score(sf_va_labels_sub, sf_va_voted) +
                       0.6 * f1_score(sf_va_labels_sub, sf_va_voted,
                                       average="macro", zero_division=0))

    tcn.eval()
    tcn_vp = []
    with torch.inference_mode():
        for xb, _ in va_loader:
            xb = xb.to(DEVICE).float()
            tcn_vp.append(tcn(xb)[:, half, :].argmax(1).cpu())
    tcn_vp_voted = _apply_vote(torch.cat(tcn_vp).numpy(), va_cgrp)
    best_val_tcn = (0.4 * accuracy_score(va_tgt.numpy(), tcn_vp_voted) +
                    0.6 * f1_score(va_tgt.numpy(), tcn_vp_voted,
                                    average="macro", zero_division=0))

    log(f"  {tag} val(+vote,w={vote_window})"
        f"  SF={best_val_sf_sub:.4f}  TCN={best_val_tcn:.4f}"
        f"  → {'TCN' if best_val_tcn >= best_val_sf_sub else 'SF'} 선택")

    tcn.eval()
    te_loader = DataLoader(list(zip(te_seq, te_tgt)), batch_size=512, shuffle=False)
    te_preds  = []
    with torch.inference_mode():
        for xb, _ in te_loader:
            te_preds.append(tcn(xb.to(DEVICE).float())[:, half, :].argmax(1).cpu())
    tcn_preds  = torch.cat(te_preds).numpy()
    tcn_labels = te_tgt.numpy()
    log(f"  {tag} TCN test Acc={accuracy_score(tcn_labels, tcn_preds):.4f}")

    return (tcn_preds, tcn_labels, te_cgrp,
            sf_te_preds_sub, sf_te_labels_sub,
            float(best_val_tcn), float(best_val_sf_sub))


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    apply_args(args)

    use_wandb = args.wandb and _WANDB_OK
    if args.wandb and not _WANDB_OK:
        log("  [W&B] wandb 미설치 — pip install wandb 후 재실행.")
    if use_wandb:
        import subprocess
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent).decode().strip()
        except Exception:
            git_hash = "unknown"
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"v11.5.1-N{getattr(args,'n_subjects','?')}",
            config={
                "version":      "v11.5.1",
                "git_commit":   git_hash,
                "sf_epochs":    SF_EPOCHS,
                "sf_ft_epochs": SF_FT_EPOCHS,
                "sf_lr":        SF_LR,
                "sf_patience":  SF_PATIENCE,
                "focal_gamma":  FOCAL_GAMMA,
                "tcn_seq_len":  TCN_SEQ_LEN,
                "tcn_epochs":   TCN_EPOCHS,
                "n_bio":        BioMechFeatures.N_BIO,
                "vote_window":  args.vote_window,
            },
        )
        log(f"  [W&B] project={args.wandb_project}  run={wandb.run.name}")

    config.print_config()
    log(
        f"  ★ v11.5.1  SuperFusion + TCN (리뷰 6항목 반영)\n"
        f"  [R1] vote_window 짝수 → 홀수 자동 보정\n"
        f"  [R2] majority_vote 시간순 가정 NOTE 명시\n"
        f"  [R3] SF Phase1 step_i=-1 초기화\n"
        f"  [R4] Triplet valid_count 에포크 로깅\n"
        f"  [R5] SubjectNorm offline/transductive 설정 명시\n"
        f"  [R6] SF Phase2 grad accumulation Phase1과 통일\n"
    )

    out       = config.RESULT_KFOLD / "hierarchical"
    curve_dir = out / "curves"
    out.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)

    h5data        = H5Data(config.H5_PATH)
    le            = LabelEncoder()
    y             = le.fit_transform(h5data.y_raw).astype(np.int64)
    groups        = h5data.subj_id
    branch_idx, branch_ch = build_branch_idx(h5data.channels)
    bio_extractor = BioMechFeatures()

    log(f"  클래스: {le.classes_.tolist()}"
        f"  피험자: {len(np.unique(groups))}명"
        f"  샘플: {len(y)}")

    sgkf = StratifiedGroupKFold(
        n_splits=config.KFOLD, shuffle=True, random_state=config.SEED)

    all_preds:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    fold_meta:  list[dict]       = []
    t_total = time.time()

    for fi, (tr_idx, te_idx) in enumerate(
        sgkf.split(np.zeros(len(y)), y, groups=groups), 1
    ):
        t_fold = time.time()
        te_s   = sorted(set(groups[te_idx].tolist()))
        log(f"\n{'='*60}")
        log(f"  Fold {fi}/{config.KFOLD}"
            f"  tr={len(tr_idx)}  te={len(te_idx)}"
            f"  test_sbj={te_s}")
        log(f"{'='*60}")

        inner_tr_idx, inner_va_idx = _inner_val_split(tr_idx, groups)
        bsc    = fit_bsc_on_train(h5data, inner_tr_idx)
        tr_ds  = make_branch_dataset(h5data, y, inner_tr_idx, bsc,
                                     branch_idx, fold_tag=f"HC{fi}",  split="train")
        val_ds = make_branch_dataset(h5data, y, inner_va_idx, bsc,
                                     branch_idx, fold_tag=f"HC{fi}v", split="val")
        te_ds  = make_branch_dataset(h5data, y, te_idx, bsc,
                                     branch_idx, fold_tag=f"HC{fi}",  split="test")
        tr_dl  = make_loader(tr_ds,  True,  branch=True)
        val_dl = make_loader(val_ds, False, branch=True)
        te_dl  = make_loader(te_ds,  False, branch=True)

        tag         = f"[F{fi}]"
        backbone_s1 = M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        s1_preds, s1_labels, s1_probs, s1_model = train_stage1(
            backbone_s1, tr_dl, val_dl, te_dl, tag, curve_dir=curve_dir)
        s1_acc = accuracy_score(s1_labels, s1_preds)
        log(f"  {tag} Stage1 Acc={s1_acc:.4f}")

        backbone_sf = M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        backbone_sf.load_state_dict(s1_model.backbone.state_dict())
        log(f"  {tag} ★ S1→SF backbone 전이 완료")

        tr_groups_local = groups[inner_tr_idx]
        va_groups_local = groups[inner_va_idx]
        te_groups_local = groups[te_idx]

        # BioMech 피처 사전 추출 + Subject 정규화
        log(f"  {tag} BioMech 피처 추출 + Subject 정규화 [R5: offline/transductive]")
        bio_extractor.eval()

        def _extract_all_bio(ds_inner):
            feats = []
            for i in range(len(ds_inner)):
                bi_s, _ = ds_inner[i]
                with torch.no_grad():
                    feats.append(
                        bio_extractor({k: v.unsqueeze(0) for k, v in bi_s.items()})
                        .squeeze(0).cpu().numpy()
                    )
            return np.stack(feats).astype(np.float32)

        tr_bio_raw = _extract_all_bio(tr_ds)
        va_bio_raw = _extract_all_bio(val_ds)
        te_bio_raw = _extract_all_bio(te_ds)

        subj_norm   = SubjectNormalizer()
        tr_bio_norm = subj_norm.fit_transform(tr_bio_raw, tr_groups_local)
        va_bio_norm = subj_norm.transform(va_bio_raw, va_groups_local)
        te_bio_norm = subj_norm.transform(te_bio_raw, te_groups_local)
        log(f"  {tag} Subject 정규화 완료 "
            f"[offline/transductive — label leakage 없음]")

        tr_all_ds = AllDataBioDataset(tr_ds,  bio_extractor, y[inner_tr_idx],
                                      groups=tr_groups_local, bio_feats_norm=tr_bio_norm)
        va_all_ds = AllDataBioDataset(val_ds, bio_extractor, y[inner_va_idx],
                                      groups=va_groups_local, bio_feats_norm=va_bio_norm)
        te_all_ds = AllDataBioDataset(te_ds,  bio_extractor, y[te_idx],
                                      groups=te_groups_local, bio_feats_norm=te_bio_norm)
        tr_sf_dl  = make_all_loader(tr_all_ds, True,  balanced=True)
        va_sf_dl  = make_all_loader(va_all_ds, False)
        te_sf_dl  = make_all_loader(te_all_ds, False)

        n_fold_subjects = len(np.unique(tr_groups_local))
        (sf_val_preds, sf_val_labels,
         sf_test_preds, sf_test_labels,
         sf_model) = train_superfusion(
            backbone_sf, tr_sf_dl, va_sf_dl, te_sf_dl,
            tag=f"{tag}[SF]", curve_dir=curve_dir,
            n_subjects=n_fold_subjects)

        sf_acc = accuracy_score(sf_test_labels, sf_test_preds)
        sf_f1  = f1_score(sf_test_labels, sf_test_preds, average="macro", zero_division=0)
        log(f"  {tag} SF (before TCN)  Acc={sf_acc:.4f}  F1={sf_f1:.4f}")

        (tcn_preds, tcn_labels, te_cgrp,
         sf_preds_sub, sf_labels_sub,
         best_val_tcn, best_val_sf_sub) = train_tcn_refiner(
            sf_model,
            tr_all_ds, va_all_ds, te_all_ds,
            tr_groups=tr_groups_local, va_groups=va_groups_local,
            te_groups=te_groups_local,
            sf_va_preds=sf_val_preds, sf_va_labels=sf_val_labels,
            sf_te_preds=sf_test_preds, sf_te_labels=sf_test_labels,
            vote_window=args.vote_window, tag=f"{tag}[TCN]")

        use_tcn = (best_val_tcn >= best_val_sf_sub)

        def _vote(p, g):
            return (majority_vote_by_group(p, g, window=args.vote_window)
                    if args.vote_window > 0 else p.copy())

        sf_vote  = _vote(sf_preds_sub, te_cgrp)
        tcn_vote = _vote(tcn_preds,    te_cgrp)

        final_preds, final_labels = (tcn_vote, tcn_labels) if use_tcn \
                                     else (sf_vote, sf_labels_sub)
        acc = accuracy_score(final_labels, final_preds)
        f1  = f1_score(final_labels, final_preds, average="macro", zero_division=0)

        acc_sf  = accuracy_score(sf_labels_sub, sf_vote)
        f1_sf   = f1_score(sf_labels_sub, sf_vote,  average="macro", zero_division=0)
        acc_tcn = accuracy_score(tcn_labels, tcn_vote)
        f1_tcn  = f1_score(tcn_labels, tcn_vote, average="macro", zero_division=0)
        log(f"  {tag} [ref] SF  Acc={acc_sf:.4f}  F1={f1_sf:.4f}")
        log(f"  {tag} [ref] TCN Acc={acc_tcn:.4f}  F1={f1_tcn:.4f}")
        log(f"  {tag} ★ 최종 ({'TCN' if use_tcn else 'SF'})"
            f"  Acc={acc:.4f}  F1={f1:.4f}")

        all_preds.append(final_preds)
        all_labels.append(final_labels)
        fold_meta.append({
            "fold":          fi,
            "test_subjects": te_s,
            "s1_acc":        round(s1_acc, 4),
            "sf_acc":        round(acc_sf, 4),  "sf_f1":  round(f1_sf, 4),
            "tcn_acc":       round(acc_tcn, 4), "tcn_f1": round(f1_tcn, 4),
            "final_acc":     round(acc, 4),     "final_f1": round(f1, 4),
            "used_tcn":      use_tcn,
            "fold_time_min": round((time.time() - t_fold) / 60, 1),
        })

        if _WANDB_OK and wandb.run is not None:
            wandb.log({f"fold{fi}/{k}": v for k, v in fold_meta[-1].items()
                       if isinstance(v, (int, float))})

        del backbone_s1, backbone_sf, s1_model, sf_model
        del tr_all_ds, va_all_ds, te_all_ds
        del tr_sf_dl, va_sf_dl, te_sf_dl
        del tr_ds, val_ds, te_ds
        gc.collect()
        if config.USE_GPU: torch.cuda.empty_cache()
        clear_fold_cache(f"HC{fi}"); clear_fold_cache(f"HC{fi}v")

    # ── 전체 결과 ─────────────────────────────────
    preds_all  = np.concatenate(all_preds)
    labels_all = np.concatenate(all_labels)
    acc_all    = accuracy_score(labels_all, preds_all)
    f1_all     = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    # 고정 6클래스 축: center-valid subset에 일부 클래스가 없어도
    # cm shape / recalls 길이 / classification_report / save_cm 축이
    # fold·run 간 항상 동일하게 유지됨 → 실험 간 비교 가능
    all_class_ids = np.arange(6)
    cm      = confusion_matrix(labels_all, preds_all, labels=all_class_ids)
    recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    total_min  = (time.time() - t_total) / 60

    print(f"\n{'='*60}")
    print(f"  ★ v11.5.1 SuperFusion+TCN  {config.KFOLD}-Fold")
    print(f"  총 소요: {total_min:.1f}분")
    print(f"  Acc={acc_all:.4f}  MacroF1={f1_all:.4f}")
    print(f"{'='*60}")
    for i, r in enumerate(recalls):
        print(f"    {CLASS_NAMES_ALL.get(i, f'C{i+1}'):<14} {r*100:.1f}%")
    rep = classification_report(
        labels_all, preds_all,
        labels=all_class_ids,
        target_names=[CLASS_NAMES_ALL[i] for i in all_class_ids],
        digits=4,
        zero_division=0,
    )
    (out / "report_v1151.txt").write_text(
        f"v11.5.1 SuperFusion+TCN\nAcc={acc_all:.4f}  F1={f1_all:.4f}\n\n{rep}")

    le_out = LabelEncoder()
    le_out.fit(all_class_ids)   # 항상 0~5 고정 → save_cm 축 일관성 보장
    save_cm(preds_all, labels_all, le_out, "SuperFusion_TCN_v1151_KFold", out)

    summary = {
        "experiment":  "hierarchical_kfold_v1151",
        "version":     "v11.5.1",
        "method":      ("SuperFusion + TCN + SubjectNorm(offline/transductive)"
                        " + WithinSubjectTriplet + GRL + CrossAttn"),
        "normalization_note": (
            "Test subject BioMech normalization uses the test subject's own "
            "unlabeled window statistics (no label leakage). "
            "This is an offline/subject-batch/transductive setting, "
            "NOT strict online causal inference. "
            "Please state this explicitly when reporting results."),
        "n_bio":         BioMechFeatures.N_BIO,
        "total_minutes": round(total_min, 1),
        "overall":  {"acc": round(acc_all, 4), "f1": round(f1_all, 4)},
        "per_class_recall": {
            CLASS_NAMES_ALL.get(i, f"C{i+1}"): round(float(r), 4)
            for i, r in enumerate(recalls)
        },
        "fold_meta": fold_meta,
    }
    path_json = out / "summary_v1151.json"
    path_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  ✅ {path_json}")

    if _WANDB_OK and wandb.run is not None:
        wandb.summary.update({
            "overall_acc": acc_all, "overall_f1": f1_all,
            **{f"recall_{CLASS_NAMES_ALL.get(i, f'C{i+1}')}": round(float(r), 4)
               for i, r in enumerate(recalls)}
        })
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=labels_all.tolist(), preds=preds_all.tolist(),
                class_names=[CLASS_NAMES_ALL.get(i, f"C{i+1}") for i in range(6)]),
            "overall_acc": acc_all, "overall_f1": f1_all,
        })
        wandb.finish()
        log("  [W&B] 로깅 완료")

    h5data.close()


if __name__ == "__main__":
    main()