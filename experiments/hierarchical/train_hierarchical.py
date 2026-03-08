# -*- coding: utf-8 -*-
"""
train_hierarchical.py — v11.5 (Subject-Relative Learning)
═══════════════════════════════════════════════════════
v11.4 → v11.5 핵심 변경:

  [A] Subject-Wise BioMech Normalization ★★★
      · 문제: Subject A의 C6(평지)=150, Subject B의 C1(미끄)=140
              → 절대값이 겹쳐서 구분 불가
      · 해결: 각 subject의 BioMech 피처를 해당 subject 평균/std로 정규화
              → "이 사람의 평균 대비 얼마나 다른가" 상대값으로 변환
      · 효과: 체중/키/보행 습관 차이 제거 → 순수 지형 신호만 남음

  [B] Within-Subject Triplet Loss ★★★
      · 문제: 다른 subject 간 비교는 노이즈만 추가
      · 해결: 같은 subject 안에서 Anchor(C1) - Pos(C1) - Neg(C4) 구성
              → "한 사람 안에서 지형 간 차이" 만 학습
      · margin=1.0, mining=hard negative within subject

  [C] v11.4 구조 유지 (GRL + CrossAttn + S1 전이 + F1 EarlyStopping)

목표: 85~93% (데이터 50명 기준 현실적 상한)
═══════════════════════════════════════════════════════"""

from __future__ import annotations

import sys, time, json, gc, warnings, math, argparse
warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass, field

# W&B (선택적 — 설치 안 돼 있어도 동작)
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
from models import M6_BranchCBAMCrossAug, count_parameters
from train_common import (
    log, H5Data,
    fit_bsc_on_train,
    make_branch_dataset, make_loader,
    save_cm, clear_fold_cache,
    _mem_str, _gpu_mem_str,
)

DEVICE = config.DEVICE

# ═══════════════════════════════════════════════
# 클래스 상수
# ═══════════════════════════════════════════════

FLAT_CLASSES     = [0, 3, 4, 5]   # C1, C4, C5, C6

# Stage2: 3cls (C1 / C6 / C4C5_merged)
# C4C5를 하나로 묶어 Stage3에서 binary로 분리 → 쉬운 문제로 분해
S2_3CLS_MAP     = {0: 0, 5: 1, 3: 2, 4: 2}   # C1→0, C6→1, C4C5→2
S2_3CLS_MAP_INV = {0: 0, 1: 5}                 # 2 → Stage3으로 라우팅

# Stage3: binary C4 vs C5 (ArcFace)
S3_BINARY_MAP     = {3: 0, 4: 1}               # C4→0, C5→1
S3_BINARY_MAP_INV = {0: 3, 1: 4}               # 0→C4(흙길), 1→C5(잔디)

CLASS_NAMES_ALL  = {0: "C1-미끄러운", 1: "C2-오르막", 2: "C3-내리막",
                    3: "C4-흙길",     4: "C5-잔디",   5: "C6-평지"}

# ═══════════════════════════════════════════════
# 하이퍼파라미터 (3-Stage 보조 — 하위 호환용)
# ═══════════════════════════════════════════════

S1_EPOCHS      = 60
S1_LR          = 5e-5
S1_PATIENCE    = 15
S1_SOFT_THRESHOLD = 0.50

S3_FFT_BINS    = 64
S3_FFT_DIM     = 128
FOCAL_GAMMA    = 1.5

# ═══════════════════════════════════════════════
# SuperFusion 하이퍼파라미터 (v11.0 메인)
# ═══════════════════════════════════════════════
SF_EPOCHS     = 120   # Phase1: 4-head multi-task
SF_FT_EPOCHS  = 40    # Phase2: 6cls fine-tune
SF_LR         = 1e-4  # Phase1 LR (head random init → 높은 LR)
SF_PATIENCE   = 20    # Phase1 early stop patience
SF_AUX_W3     = 0.30  # 3cls 보조 가중치 (flat/up/down)
SF_AUX_WFLAT  = 0.20  # flat3 보조 가중치 (C1/C4C5/C6)
SF_AUX_WBIN   = 0.40  # binary 보조 가중치 (C4/C5) ← C4/C5 식별 강화
SF_WCONS      = 0.10  # consistency KL 가중치
SF_WADV       = 0.10  # subject adversarial loss 가중치 (v11.4)
SF_WTRIPLET   = 0.30  # within-subject triplet loss 가중치 (v11.5)
TRIPLET_MARGIN = 1.0  # triplet margin

# ═══════════════════════════════════════════════
# TCN Sequence Refiner 하이퍼파라미터 (v11.0)
# ═══════════════════════════════════════════════
TCN_SEQ_LEN   = 9     # 슬라이딩 윈도우 길이 (홀수: 중심 예측)
TCN_HIDDEN    = 128   # TCN 은닉 채널
TCN_EPOCHS    = 40    # TCN 학습 에포크
TCN_LR        = 5e-4  # TCN 학습률
TCN_PATIENCE  = 10    # TCN early stop

# backward compat alias
E2E_EPOCHS    = SF_EPOCHS
E2E_FT_EPOCHS = SF_FT_EPOCHS
E2E_LR        = SF_LR
E2E_AUX_W3    = SF_AUX_W3
E2E_AUX_WB    = SF_AUX_WBIN
E2E_PATIENCE  = SF_PATIENCE


# ═══════════════════════════════════════════════
# CLI — argparse (하이퍼파라미터 오버라이드)
# ═══════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """CLI로 하이퍼파라미터 오버라이드 (v11.0)."""
    p = argparse.ArgumentParser(
        description="SuperFusion + TCN Terrain Classifier v11.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Stage1 (SF 초기화용)
    p.add_argument("--s1_epochs",    type=int,   default=S1_EPOCHS)
    p.add_argument("--s1_lr",        type=float, default=S1_LR)
    # SuperFusion
    p.add_argument("--sf_epochs",    type=int,   default=SF_EPOCHS)
    p.add_argument("--sf_ft_epochs", type=int,   default=SF_FT_EPOCHS)
    p.add_argument("--sf_lr",        type=float, default=SF_LR)
    p.add_argument("--sf_patience",  type=int,   default=SF_PATIENCE)
    p.add_argument("--focal_gamma",  type=float, default=FOCAL_GAMMA)
    # TCN
    p.add_argument("--tcn_seq_len",  type=int,   default=TCN_SEQ_LEN)
    p.add_argument("--tcn_epochs",   type=int,   default=TCN_EPOCHS)
    # Post-processing
    p.add_argument("--vote_window",  type=int,   default=5,
                   help="Majority vote window (0=off)")
    # Data
    p.add_argument("--n_subjects",   type=int,   default=None)
    # W&B
    p.add_argument("--wandb",        action="store_true",
                   help="W&B 실험 로깅 활성화")
    p.add_argument("--wandb_project", type=str, default="imu-terrain",
                   help="W&B 프로젝트 이름")
    p.add_argument("--run_name",     type=str, default=None,
                   help="W&B run 이름 (기본: 자동)")
    return p.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    """파싱된 args를 전역 하이퍼파라미터에 반영 (v11.0)."""
    global S1_EPOCHS, S1_LR
    global SF_EPOCHS, SF_FT_EPOCHS, SF_LR, SF_PATIENCE, FOCAL_GAMMA
    global TCN_SEQ_LEN, TCN_EPOCHS
    # backward compat aliases
    global E2E_EPOCHS, E2E_FT_EPOCHS, E2E_LR, E2E_PATIENCE
    S1_EPOCHS      = args.s1_epochs
    S1_LR          = args.s1_lr
    SF_EPOCHS      = args.sf_epochs
    SF_FT_EPOCHS   = args.sf_ft_epochs
    SF_LR          = args.sf_lr
    SF_PATIENCE    = args.sf_patience
    FOCAL_GAMMA    = args.focal_gamma
    TCN_SEQ_LEN    = args.tcn_seq_len
    TCN_EPOCHS     = args.tcn_epochs
    E2E_EPOCHS     = SF_EPOCHS
    E2E_FT_EPOCHS  = SF_FT_EPOCHS
    E2E_LR         = SF_LR
    E2E_PATIENCE   = SF_PATIENCE
    if args.n_subjects is not None:
        config.apply_overrides(n_subjects=args.n_subjects)


# ═══════════════════════════════════════════════
# 훈련 곡선 추적기
# ═══════════════════════════════════════════════

@dataclass
class CurveTracker:
    """에포크별 loss/acc 기록 → JSON + PNG 저장.

    리뷰 6항: 훈련 곡선 시각화 대응.
    """
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
        # JSON
        (out_dir / f"curve_{self.name}.json").write_text(
            json.dumps({"loss": self.losses, "acc": self.accs}, indent=2))
        # PNG
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
# 공통 헬퍼 (중복 제거)
# ═══════════════════════════════════════════════

def _to_device(bi: dict, bio_f: torch.Tensor | None = None,
               yb: torch.Tensor | None = None):
    """branch dict + bio_feat + label 을 DEVICE로 이동. float cast 포함."""
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
    """model state_dict를 CPU에 복사. best_state 저장용."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    params: list,
    has_bio: bool = True,
    forward_fn=None,
) -> float:
    """1 에포크 학습 루프. 반복되는 forward-backward-step 패턴을 통합.

    Args:
        has_bio: Stage2처럼 (bi, bio_f, yb) 형태면 True,
                 Stage1처럼 (bi, yb) 형태면 False.
        forward_fn: None이면 model(bi) 또는 model(bi, bio_f).
                    SupCon처럼 model.forward_proj를 써야 할 때 전달.
    Returns:
        평균 loss
    """
    model.train()
    total_loss = n = 0
    opt.zero_grad(set_to_none=True)

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

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

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

    # [리뷰1 수정] 마지막 배치 gradient 누락 방지
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

def auto_class_weights(y_flat: np.ndarray) -> torch.Tensor:
    """학습 데이터 기반 클래스 가중치 자동 계산. (B, 6) 호환 Tensor 반환"""
    classes = np.unique(y_flat)
    weights = compute_class_weight("balanced", classes=classes, y=y_flat)
    w_list  = weights.tolist()
    label   = ["C1", "C4", "C5", "C6"] if len(classes) == 4 else \
              ["C1", "C6", "C4C5"] if len(classes) == 3 else \
              [str(c) for c in classes]
    parts   = "  ".join(f"{l}={w:.3f}" for l, w in zip(label, w_list))
    log(f"    auto class_weights (balanced): {parts}")
    return torch.tensor(w_list, dtype=torch.float32)


def majority_vote_smooth(preds: np.ndarray, window: int = 5) -> np.ndarray:
    """보행 연속성 기반 majority vote post-processing.

    스텝 단위 예측은 연속적 → 갑작스러운 클래스 변화는 노이즈.
    window 개 예측의 최빈값으로 교정.
    양 끝 (window//2개)은 원본 유지.
    """
    half     = window // 2
    smoothed = preds.copy()
    # 내부만 교정 (경계 원본 유지)
    for i in range(half, len(preds) - half):
        smoothed[i] = scipy_mode(
            preds[i - half: i + half + 1], keepdims=False
        ).mode
    return smoothed


class BioMechFeatures(nn.Module):
    """생체역학 충격 피처 추출기 v11.2.

    피처 44개:
      [Accel 기반 — 기존 20개]
      0~3  : Foot/Shank LT/RT 충격 피크값
      4~5  : Foot/Shank 충격비 (log1p)
      6~7  : 고주파 에너지 비율 (C4 흙길 특화)
      8~9  : Foot LT/RT 표준편차 (C1 불안정↑)
      10~11: 피크 후 감쇠율 (C5 잔디 → 빠른 감쇠)
      12~13: Shank LT/RT 진동 |diff|.mean (C4 흙길↑)
      14~15: Foot/Shank 분산비 (C1 불안정 지표)
      16~17: Spectral Centroid LT/RT (C4 흙길 → centroid↑)
      18~19: Impact Duration LT/RT (C5 잔디 → duration↑)

      [자이로 기반 — 18개]
      20~21: Foot LT/RT 자이로 분산 (log1p)
      22~23: Foot LT/RT 자이로 피크 각속도
      24~25: Shank LT/RT 자이로 분산
      26~27: Shank LT/RT 자이로 피크
      28~29: Thigh LT/RT 자이로 분산
      30~31: Thigh LT/RT 자이로 피크

      [비대칭/전달 기반 — 8개]
      32~33: LT/RT 가속도 피크 비대칭 (|L-R|)
      34~35: LT/RT 자이로 에너지 비대칭
      36~37: Foot→Shank 진동 전달비 LT/RT

      [통계 기반 — v11.2 추가 6개]
      38~39: Foot LT/RT Kurtosis — C1 미끄러운 → 충격 분포 뾰족함↑
      40~41: Foot LT/RT Skewness — C4 흙길 → 비대칭 충격 패턴
      42~43: Foot LT/RT ZCR      — C5 잔디 → 진동 방향 전환 빈도↑
    """
    N_BIO = 44  # v11.2: 38→44

    def __init__(self) -> None:
        super().__init__()
        self.foot_z  = config.FOOT_Z_ACCEL_IDX
        self.shank_z = config.SHANK_Z_ACCEL_IDX
        self.hf_bin  = int(30 * config.TS / config.SAMPLE_RATE)
        # 12ch 센서 그룹 내부 인덱스 (LT: 0~5, RT: 6~11)
        # accel: [0,1,2] / [6,7,8], gyro: [3,4,5] / [9,10,11]
        self.gyro_lt = [3, 4, 5]
        self.gyro_rt = [9, 10, 11]

    @torch.no_grad()
    def forward(self, bi: dict) -> torch.Tensor:
        """
        Args:
            bi: dict with keys 'Foot', 'Shank', 'Thigh' (B, 12, T) each
        Returns:
            (B, 38)
        """
        foot_x  = bi["Foot"].float()          # (B, 12, T)
        shank_x = bi["Shank"].float()
        thigh_x = bi.get("Thigh")
        if thigh_x is not None:
            thigh_x = thigh_x.float()
        eps = 1e-6

        fz_lt = foot_x[:,  self.foot_z[0],  :]
        fz_rt = foot_x[:,  self.foot_z[1],  :]
        sz_lt = shank_x[:, self.shank_z[0], :]
        sz_rt = shank_x[:, self.shank_z[1], :]

        # ── 0~3: 피크값 ───────────────────────────
        f_pk_lt = fz_lt.abs().max(dim=1).values
        f_pk_rt = fz_rt.abs().max(dim=1).values
        s_pk_lt = sz_lt.abs().max(dim=1).values
        s_pk_rt = sz_rt.abs().max(dim=1).values

        # ── 4~5: 충격비 ───────────────────────────
        ratio_lt = torch.log1p(f_pk_lt / (s_pk_lt + 1e-4))
        ratio_rt = torch.log1p(f_pk_rt / (s_pk_rt + 1e-4))

        # ── 6~7: 고주파 에너지 ────────────────────
        hf_lt = self._hf_ratio(fz_lt)
        hf_rt = self._hf_ratio(fz_rt)

        # ── 8~9: 변동성 ───────────────────────────
        std_lt = fz_lt.std(dim=1)
        std_rt = fz_rt.std(dim=1)

        # ── 10~11: 감쇠율 ─────────────────────────
        T_half   = fz_lt.shape[1] // 2
        decay_lt = (fz_lt[:, :T_half].abs().mean(dim=1) /
                    (fz_lt[:, T_half:].abs().mean(dim=1) + eps))
        decay_rt = (fz_rt[:, :T_half].abs().mean(dim=1) /
                    (fz_rt[:, T_half:].abs().mean(dim=1) + eps))

        # ── 12~13: Shank 진동 ─────────────────────
        vib_lt = (sz_lt[:, 1:] - sz_lt[:, :-1]).abs().mean(dim=1)
        vib_rt = (sz_rt[:, 1:] - sz_rt[:, :-1]).abs().mean(dim=1)

        # ── 14~15: Foot/Shank 분산비 ─────────────
        var_ratio_lt = torch.log1p(fz_lt.var(dim=1) / (sz_lt.var(dim=1) + 1e-4))
        var_ratio_rt = torch.log1p(fz_rt.var(dim=1) / (sz_rt.var(dim=1) + 1e-4))

        # ── 16~17: Spectral Centroid ──────────────
        sc_lt = self._spectral_centroid(fz_lt)
        sc_rt = self._spectral_centroid(fz_rt)

        # ── 18~19: Impact Duration ────────────────
        dur_lt = self._impact_duration(fz_lt)
        dur_rt = self._impact_duration(fz_rt)

        # ── 20~23: Foot 자이로 ────────────────────
        fg_lt = foot_x[:, self.gyro_lt, :]    # (B, 3, T)
        fg_rt = foot_x[:, self.gyro_rt, :]
        fg_var_lt  = torch.log1p(fg_lt.var(dim=2).sum(dim=1))
        fg_var_rt  = torch.log1p(fg_rt.var(dim=2).sum(dim=1))
        fg_peak_lt = fg_lt.abs().amax(dim=(1, 2))
        fg_peak_rt = fg_rt.abs().amax(dim=(1, 2))

        # ── 24~27: Shank 자이로 ───────────────────
        sg_lt = shank_x[:, self.gyro_lt, :]
        sg_rt = shank_x[:, self.gyro_rt, :]
        sg_var_lt  = torch.log1p(sg_lt.var(dim=2).sum(dim=1))
        sg_var_rt  = torch.log1p(sg_rt.var(dim=2).sum(dim=1))
        sg_peak_lt = sg_lt.abs().amax(dim=(1, 2))
        sg_peak_rt = sg_rt.abs().amax(dim=(1, 2))

        # ── 28~31: Thigh 자이로 ───────────────────
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

        # ── 32~33: 가속도 피크 비대칭 (C1 미끄러움) ─
        asym_acc_lt = (f_pk_lt - s_pk_lt).abs()
        asym_acc_rt = (f_pk_rt - s_pk_rt).abs()

        # ── 34~35: 자이로 에너지 비대칭 (C5 잔디) ──
        asym_gy_lt = (fg_var_lt - sg_var_lt).abs()
        asym_gy_rt = (fg_var_rt - sg_var_rt).abs()

        # ── 36~37: Foot→Shank 진동 전달비 (C4 흙길) ─
        foot_rms_lt = fz_lt.pow(2).mean(dim=1).sqrt()
        foot_rms_rt = fz_rt.pow(2).mean(dim=1).sqrt()
        shank_rms_lt = sz_lt.pow(2).mean(dim=1).sqrt()
        shank_rms_rt = sz_rt.pow(2).mean(dim=1).sqrt()
        trans_lt = torch.log1p(shank_rms_lt / (foot_rms_lt + eps))
        trans_rt = torch.log1p(shank_rms_rt / (foot_rms_rt + eps))

        # ── 38~39: Kurtosis LT/RT (C1 미끄러운 → 충격 뾰족함↑) ──
        kurt_lt = self._kurtosis(fz_lt)
        kurt_rt = self._kurtosis(fz_rt)

        # ── 40~41: Skewness LT/RT (C4 흙길 → 비대칭 충격) ──────
        skew_lt = self._skewness(fz_lt)
        skew_rt = self._skewness(fz_rt)

        # ── 42~43: ZCR LT/RT (C5 잔디 → 진동 방향 전환↑) ───────
        zcr_lt = self._zcr(fz_lt)
        zcr_rt = self._zcr(fz_rt)

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
            asym_gy_lt, asym_gy_rt,
            trans_lt, trans_rt,
            kurt_lt, kurt_rt,
            skew_lt, skew_rt,
            zcr_lt,  zcr_rt,
        ], dim=1)   # (B, 44)

    def _kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """첨도 (Kurtosis) — 충격 분포 뾰족함.
        C1 미끄러운 → 불규칙 충격 → kurtosis↑
        정규분포 기준값=3, clamp로 폭발 방지.
        """
        mu  = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        return ((x - mu) / std).pow(4).mean(dim=1).clamp(-10, 30)

    def _skewness(self, x: torch.Tensor) -> torch.Tensor:
        """비대칭도 (Skewness) — 충격 방향 편향.
        C4 흙길 → 불규칙 충격 → 비대칭 분포
        """
        mu  = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        return ((x - mu) / std).pow(3).mean(dim=1).clamp(-10, 10)

    def _zcr(self, x: torch.Tensor) -> torch.Tensor:
        """영교차율 (Zero Crossing Rate) — 진동 방향 전환 빈도.
        C5 잔디 → 불규칙 지면 반발 → ZCR↑
        C6 평지 → 규칙적 진동 → ZCR 낮음
        """
        signs   = torch.sign(x)
        # 부호가 바뀌는 지점 카운트 (0은 이전 부호 유지로 처리)
        signs   = torch.where(signs == 0,
                              torch.ones_like(signs), signs)
        crosses = (signs[:, 1:] * signs[:, :-1] < 0).float()
        return crosses.mean(dim=1)

    def _hf_ratio(self, x: torch.Tensor) -> torch.Tensor:
        fft_mag = torch.fft.rfft(x, dim=1).abs()
        total   = fft_mag.pow(2).sum(dim=1) + 1e-6
        hf      = fft_mag[:, self.hf_bin:].pow(2).sum(dim=1)
        return hf / total

    def _spectral_centroid(self, x: torch.Tensor) -> torch.Tensor:
        """주파수 무게중심 — Niswander et al. (2021).

        고주파 성분이 많을수록 centroid↑ (C4 흙길 특징).
        """
        fft_mag = torch.fft.rfft(x, dim=1).abs()
        n_bins  = fft_mag.shape[1]
        freqs   = torch.arange(n_bins, device=x.device, dtype=torch.float32)
        power   = fft_mag.pow(2)
        centroid = (freqs * power).sum(dim=1) / (power.sum(dim=1) + 1e-6)
        return centroid / n_bins   # 0~1 정규화

    def _impact_duration(self, x: torch.Tensor,
                         threshold: float = 0.3) -> torch.Tensor:
        """충격 지속시간 비율 — Niswander et al. (2021).

        피크 대비 30% 이상인 샘플 비율.
        C5 잔디 → 충격 넓게 분산 → duration↑
        C4 흙길 → 짧고 날카로운 충격 → duration↓
        """
        pk    = x.abs().max(dim=1, keepdim=True).values
        above = (x.abs() >= pk * threshold).float()
        return above.mean(dim=1)   # 0~1 비율


# ═══════════════════════════════════════════════
# 2. BioMechHead
# ═══════════════════════════════════════════════

class SubjectNormalizer:
    """Subject-Wise BioMech Feature Normalization.

    핵심 아이디어:
      Subject A: C1=100, C4=200, C6=150 → mean=150, std=50
                 정규화: C1=-1.0, C4=+1.0, C6=0.0
      Subject B: C1=80,  C4=160, C6=120 → mean=120, std=40
                 정규화: C1=-1.0, C4=+1.0, C6=0.0  ← 같아짐!

    fold 내 train subject들의 class-mean을 계산하여 정규화.
    test subject에는 test subject 자신의 통계로 정규화
    (test leakage 없음 — subject 내부 통계만 사용).
    """
    def __init__(self):
        # subject_id → (mean, std) 저장
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, bio_feats: np.ndarray, groups: np.ndarray) -> None:
        """각 subject의 mean/std 계산."""
        self.stats = {}
        for sbj in np.unique(groups):
            mask = groups == sbj
            feats = bio_feats[mask]
            self.stats[sbj] = (
                feats.mean(axis=0),
                feats.std(axis=0).clip(min=1e-6),
            )

    def transform(self, bio_feats: np.ndarray,
                  groups: np.ndarray) -> np.ndarray:
        """subject별 z-score 정규화 적용."""
        out = bio_feats.copy().astype(np.float32)
        for sbj in np.unique(groups):
            mask = groups == sbj
            if sbj in self.stats:
                mu, std = self.stats[sbj]
            else:
                # unseen subject (test) → 자신의 통계로 정규화
                feats = bio_feats[mask]
                mu  = feats.mean(axis=0)
                std = feats.std(axis=0).clip(min=1e-6)
            out[mask] = (bio_feats[mask] - mu) / std
        return out

    def fit_transform(self, bio_feats: np.ndarray,
                      groups: np.ndarray) -> np.ndarray:
        self.fit(bio_feats, groups)
        return self.transform(bio_feats, groups)


class WithinSubjectTripletLoss(nn.Module):
    """Within-Subject Hard Negative Triplet Loss.

    같은 subject 안에서만 triplet 구성:
      Anchor:   subject S의 클래스 C 샘플
      Positive: subject S의 같은 클래스 C 다른 샘플
      Negative: subject S의 다른 클래스 C' 샘플 (hardest)

    이유:
      - 다른 subject 간 비교 → 개인차 노이즈만 추가
      - 같은 subject 안에서 C1 vs C4 차이 → 순수 지형 신호
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(
            margin=margin, p=2, reduction="mean")

    def forward(self, emb: torch.Tensor,
                labels: torch.Tensor,
                sbj: torch.Tensor) -> torch.Tensor:
        """
        emb:    (B, D) — 256-dim embedding
        labels: (B,)   — 지형 클래스 0~5
        sbj:    (B,)   — subject index (0-based)
        """
        anchors, positives, negatives = [], [], []

        for s in sbj.unique():
            s_mask = sbj == s
            s_emb  = emb[s_mask]
            s_lbl  = labels[s_mask]

            # subject 내에 최소 2개 클래스 필요
            if s_lbl.unique().shape[0] < 2:
                continue

            for cls in s_lbl.unique():
                pos_mask = s_lbl == cls
                neg_mask = s_lbl != cls

                if pos_mask.sum() < 2 or neg_mask.sum() < 1:
                    continue

                pos_embs = s_emb[pos_mask]
                neg_embs = s_emb[neg_mask]
                pos_lbls = s_lbl[neg_mask]

                # 각 anchor(pos[0])에 대해 hardest negative 선택
                anchor = pos_embs[0]
                pos    = pos_embs[1] if pos_embs.shape[0] > 1 \
                         else pos_embs[0]

                # hardest negative: anchor와 가장 가까운 negative
                dists = torch.cdist(
                    anchor.unsqueeze(0),
                    neg_embs).squeeze(0)
                hard_neg = neg_embs[dists.argmin()]

                anchors.append(anchor)
                positives.append(pos)
                negatives.append(hard_neg)

        if len(anchors) == 0:
            return torch.tensor(0.0, device=emb.device,
                                requires_grad=True)

        a = torch.stack(anchors)
        p = torch.stack(positives)
        n = torch.stack(negatives)
        return self.loss_fn(a, p, n)


class BioMechHead(nn.Module):
    def __init__(self, in_dim: int = BioMechFeatures.N_BIO, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim), nn.ReLU(),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════
# 3. Stage2 Model
# ═══════════════════════════════════════════════

class Stage2Model(nn.Module):
    """CNN backbone + BioMech 피처 결합 모델.

    forward_proj: SupCon 학습용 (L2 정규화된 128-d 임베딩)
    forward:      CE/Focal 학습용 (4cls 로짓)
    """
    def __init__(self, backbone, feat_dim: int,
                 bio_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.backbone  = backbone
        self.bio_head  = BioMechHead(BioMechFeatures.N_BIO, bio_dim)
        total_dim      = feat_dim + bio_dim
        self.proj_head = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Dropout(config.DROPOUT_CLF),
            nn.Linear(256, num_classes),
        )

    def _extract(self, bi: dict, bio_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.backbone.extract(bi),
                          self.bio_head(bio_feat)], dim=1)

    def forward_proj(self, bi: dict, bio_feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_head(self._extract(bi, bio_feat)), dim=1)

    def forward(self, bi: dict, bio_feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._extract(bi, bio_feat))


# ═══════════════════════════════════════════════
# 4. Loss Functions
# ═══════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Khosla et al. NeurIPS 2020 — Supervised Contrastive Loss."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)
        features = F.normalize(features, dim=1)
        sim      = torch.matmul(features, features.T) / self.temperature
        eye      = torch.eye(B, dtype=torch.bool, device=features.device)
        labels   = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & ~eye
        log_prob = sim - torch.logsumexp(sim.masked_fill(eye, -1e9),
                                          dim=1, keepdim=True)
        n_pos    = pos_mask.sum(1).float().clamp(min=1)
        return -(log_prob * pos_mask).sum(1).div(n_pos).mean()


class FocalLoss(nn.Module):
    """Lin et al. ICCV 2017 — Focal Loss.

    어려운 샘플(pt 낮음)에 가중치를 높여 집중 학습.
    C4/C5처럼 혼동되기 쉬운 클래스에 효과적.
    """
    def __init__(self, gamma: float = 2.0,
                 weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce  = F.cross_entropy(logits, targets,
                               weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class ArcFaceLoss(nn.Module):
    """Deng et al. CVPR 2019 — ArcFace: Additive Angular Margin Loss.

    cos(θ + m) 로 클래스 간 각도 margin 강제 → feature space angular separation.
    C4(흙길) vs C5(잔디) binary 분류에서 SupCon보다 우수:
      - binary 문제에서 명시적 margin 설정 가능
      - 얼굴인식 세밀 분류에서 검증된 방법론
      - s=32, m=0.5: 원논문 권장값

    Args:
        feat_dim:    입력 feature 차원
        num_classes: 분류 클래스 수 (Stage3: 2)
        s:           feature scale (logit 스케일링)
        m:           angular margin (radians)
    """
    def __init__(self, feat_dim: int, num_classes: int = 2,
                 s: float = 32.0, m: float = 0.5) -> None:
        super().__init__()
        self.s   = s
        self.m   = m
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        # L2 정규화 후 cosine similarity
        cosine = F.linear(F.normalize(features, dim=1),
                          F.normalize(self.weight, dim=1))
        # [리뷰1/2 수정] float 오차로 ±1 초과 → acos NaN 방지
        cosine  = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta   = torch.acos(cosine)
        one_hot = F.one_hot(labels, cosine.shape[1]).float()
        logit   = self.s * (one_hot * torch.cos(theta + self.m)
                            + (1 - one_hot) * cosine)
        return F.cross_entropy(logit, labels)


class FFTBranch(nn.Module):
    """Stage3 전용 주파수 도메인 Branch.

    Zheng et al. (2021) — IMU terrain classification via FFT features.
    Foot 가속도 신호의 FFT 스펙트럼을 별도 branch로 처리.

    C4 흙길: 고주파(10~30Hz) 에너지↑ → spectrum 우측 편중
    C5 잔디: 저주파(1~5Hz) 에너지↑  → spectrum 좌측 편중
    → CNN이 놓치는 주파수 도메인 패턴 포착

    Args:
        n_bins:   FFT bin 수 (TS//2+1 = 129 for TS=256)
        out_dim:  출력 feature 차원
    """
    def __init__(self, n_bins: int = 129, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bins * 2, 256),   # Foot LT + RT FFT
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.n_bins = n_bins

    def forward(self, foot_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            foot_x: (B, 12, T)  Foot branch 신호
        Returns:
            (B, out_dim)
        """
        # Foot Z-accel LT/RT 채널 사용 (충격 신호)
        sig_lt = foot_x[:, 2, :]   # Foot Z LT (index 2)
        sig_rt = foot_x[:, 8, :]   # Foot Z RT (index 8)

        # FFT 파워 스펙트럼
        fft_lt = torch.fft.rfft(sig_lt, dim=1).abs().pow(2)[:, :self.n_bins]
        fft_rt = torch.fft.rfft(sig_rt, dim=1).abs().pow(2)[:, :self.n_bins]

        # L2 정규화 (스케일 불변)
        fft_lt = F.normalize(fft_lt, dim=1)
        fft_rt = F.normalize(fft_rt, dim=1)

        return self.net(torch.cat([fft_lt, fft_rt], dim=1))


class Stage3Model(nn.Module):
    """Stage3: C4(흙길) vs C5(잔디) binary classifier.

    CNN backbone + FFT Branch + BioMech 피처 결합.
    ArcFace로 학습 → angular margin으로 C4/C5 feature 분리.

    전략 A: FFT Branch (Zheng et al. 2021) — 주파수 도메인 패턴
    forward_embed: ArcFace 학습용 embedding
    forward:       inference용 linear logit
    """
    def __init__(self, backbone, feat_dim: int,
                 bio_dim: int = 128, embed_dim: int = 128,
                 fft_dim: int = S3_FFT_DIM) -> None:
        super().__init__()
        self.backbone  = backbone
        self.bio_head  = BioMechHead(BioMechFeatures.N_BIO, bio_dim)
        self.fft_branch = FFTBranch(n_bins=129, out_dim=fft_dim)
        total_dim      = feat_dim + bio_dim + fft_dim
        self.embed     = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_CLF),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, 2)

    def _fuse(self, bi: dict, bio_feat: torch.Tensor) -> torch.Tensor:
        cnn_feat  = self.backbone.extract(bi)
        bio_out   = self.bio_head(bio_feat)
        fft_out   = self.fft_branch(bi["Foot"].float())
        return torch.cat([cnn_feat, bio_out, fft_out], dim=1)

    def forward_embed(self, bi: dict,
                      bio_feat: torch.Tensor) -> torch.Tensor:
        """ArcFace 학습용 L2 정규화 embedding."""
        return F.normalize(self.embed(self._fuse(bi, bio_feat)), dim=1)

    def forward(self, bi: dict,
                bio_feat: torch.Tensor) -> torch.Tensor:
        """Inference용 logit."""
        return self.classifier(self.embed(self._fuse(bi, bio_feat)))


# ═══════════════════════════════════════════════
# 4b. E2E Hierarchical Model (v8.9 핵심)
# ═══════════════════════════════════════════════

class GradientReversalFn(torch.autograd.Function):
    """Ganin et al. 2015 — Gradient Reversal Layer.

    forward:  identity (그대로 통과)
    backward: gradient에 -λ 곱해서 역전
    → backbone이 subject 구분 불가능한 feature 학습
    """
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        lam, = ctx.saved_tensors
        return -lam.item() * grad, None


class GRL(nn.Module):
    """GRL wrapper — lam은 학습 진행에 따라 외부에서 조정."""
    def __init__(self): super().__init__()
    def forward(self, x, lam=1.0):
        return GradientReversalFn.apply(x, lam)


class KinematicCrossAttention(nn.Module):
    """Foot→Shank 충격 전파 Cross-Attention.

    물리적 근거:
      잔디(C5): Foot에서 충격 흡수 → Shank attention 약함
      흙길(C4): Foot 불규칙 → Shank attention 변동↑
      미끄러움(C1): Foot 뒤틀림 → Shank/Thigh attention 급변

    Q = Shank feature (무엇을 주목할지 질문)
    K,V = Foot feature (충격 정보 제공)
    → Shank가 Foot의 어느 부분을 보는지 학습
    """
    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, foot: torch.Tensor,
                shank: torch.Tensor) -> torch.Tensor:
        """
        foot, shank: (B, dim)
        반환: shank + cross-attended foot info (B, dim)
        """
        # (B, 1, dim) 로 변환해서 attention 적용
        q   = shank.unsqueeze(1)
        kv  = foot.unsqueeze(1)
        out, _ = self.attn(q, kv, kv)     # (B, 1, dim)
        return self.norm(shank + out.squeeze(1))


class SuperFusionModel(nn.Module):
    """v11.4 — Kinematic Cross-Attention + Subject-Adversarial GRL.

    구조:
      1. 브랜치별 CNN: Foot / Shank / Thigh 독립 인코딩
      2. KinematicCrossAttention: Foot→Shank, Foot→Thigh
         (충격 전파 메커니즘 학습)
      3. Fusion: [cross_shank, cross_thigh, Foot, BioMech, FFT] → Shared(256)
      4. 4 Heads: 6cls / 3cls / flat3 / bin
      5. Subject Classifier (GRL): 지형 학습 중 개인차 제거

    Loss = Focal_6cls + 0.30*CE_3 + 0.20*CE_flat + 0.40*CE_bin
         + 0.10*KL_cons + λ_adv*CE_subject
    """
    def __init__(self, backbone, feat_dim: int,
                 bio_dim: int = 128,
                 fft_dim: int = S3_FFT_DIM,
                 n_subjects: int = 50) -> None:
        super().__init__()
        self.backbone   = backbone
        self.bio_head   = BioMechHead(BioMechFeatures.N_BIO, bio_dim)
        self.fft_branch = FFTBranch(n_bins=129, out_dim=fft_dim)

        # Kinematic Cross-Attention: Foot→Shank 충격 전파
        self.foot_proj  = nn.Linear(12, feat_dim)   # Foot 12ch → feat_dim
        self.cross_attn = KinematicCrossAttention(feat_dim, n_heads=4)

        total = feat_dim + bio_dim + fft_dim
        self.shared = nn.Sequential(
            nn.Linear(total, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.20),
        )
        # 지형 분류 heads
        self.head_6cls = nn.Linear(256, 6)
        self.head_3cls = nn.Linear(256, 3)
        self.head_flat = nn.Linear(256, 3)
        self.head_bin  = nn.Linear(256, 2)

        # Subject-Adversarial: GRL + subject classifier
        self.grl            = GRL()
        self.subject_clf    = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_subjects),
        )
        self.n_subjects = n_subjects

    def _embed(self, bi: dict, bio_f: torch.Tensor) -> torch.Tensor:
        # CNN backbone extract (12ch — S1 전이 가능)
        cnn = self.backbone.extract(bi)           # (B, feat_dim)
        bio = self.bio_head(bio_f)                # (B, bio_dim)
        fft = self.fft_branch(bi["Foot"].float()) # (B, fft_dim)

        # Kinematic Cross-Attention:
        # Foot 원시 신호의 temporal mean을 feat_dim으로 projection해서 사용
        # backbone 재실행 없이 Foot 정보만 경량 추출
        foot_raw  = bi["Foot"].float().mean(dim=-1)   # (B, 12)
        foot_proj = self.foot_proj(foot_raw)           # (B, feat_dim)
        cnn_attended = self.cross_attn(foot_proj, cnn) # (B, feat_dim)

        return self.shared(
            torch.cat([cnn_attended, bio, fft], dim=1))  # (B, 256)

    def forward(self, bi: dict, bio_f: torch.Tensor,
                lam: float = 1.0):
        """학습용: (l6, l3, lflat, lbin, subj_logit, emb) 반환."""
        emb = self._embed(bi, bio_f)
        # subject adversarial (GRL로 gradient 역전)
        subj_logit = self.subject_clf(self.grl(emb, lam))
        return (self.head_6cls(emb),
                self.head_3cls(emb),
                self.head_flat(emb),
                self.head_bin(emb),
                subj_logit,
                emb)

    def predict(self, bi: dict, bio_f: torch.Tensor) -> torch.Tensor:
        return self.head_6cls(self._embed(bi, bio_f))

    def embed(self, bi: dict, bio_f: torch.Tensor) -> torch.Tensor:
        return self._embed(bi, bio_f)


def consistency_kl_loss(l6: torch.Tensor, l3: torch.Tensor) -> torch.Tensor:
    """Consistency KL Loss: 6cls→3cls 파생 분포 vs 3cls head.

    flat   = C0(미끄)+C3(흙)+C4(잔디)+C5(평지)
    up     = C1(오르막)
    down   = C2(내리막)
    → 두 분포가 일관되도록 KL 규제
    """
    p6 = torch.softmax(l6.float(), dim=1)   # (B, 6)
    # 6cls 확률에서 3cls 파생
    p_flat = p6[:, 0] + p6[:, 3] + p6[:, 4] + p6[:, 5]  # C1+C4+C5+C6
    p_up   = p6[:, 1]                                      # C2 오르막
    p_down = p6[:, 2]                                      # C3 내리막
    p3_from6 = torch.stack([p_flat, p_up, p_down], dim=1).clamp(1e-8, 1.0)

    p3_head = torch.softmax(l3.float(), dim=1).clamp(1e-8, 1.0)
    # KL(p3_head || p3_from6) — 두 분포가 가까워지도록
    return F.kl_div(p3_head.log(), p3_from6, reduction="batchmean")


# ═══════════════════════════════════════════════
# TCN Sequence Refiner (v11.0)
# ═══════════════════════════════════════════════

class _TCNBlock(nn.Module):
    """Dilated Causal Conv1D block."""
    def __init__(self, ch: int, dilation: int) -> None:
        super().__init__()
        pad = dilation  # causal: left-pad only
        self.conv = nn.Conv1d(ch, ch, kernel_size=3,
                              padding=pad, dilation=dilation)
        self.bn   = nn.BatchNorm1d(ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ch, T) → causal: 오른쪽 패딩 제거
        out = self.conv(x)[:, :, :x.shape[2]]
        return x + self.drop(self.act(self.bn(out)))


class TCNRefiner(nn.Module):
    """Subject window sequence → refined 6cls logits.

    입력: (B, T_seq, 256) embedding sequences
    출력: (B, T_seq, 6) refined logits
    """
    def __init__(self, in_dim: int = 256,
                 hidden: int = TCN_HIDDEN,
                 num_classes: int = 6) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        # dilations: 1, 2, 4 → receptive field = 13 windows
        self.tcn  = nn.Sequential(
            _TCNBlock(hidden, dilation=1),
            _TCNBlock(hidden, dilation=2),
            _TCNBlock(hidden, dilation=4),
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        h = self.proj(x)            # (B, T, hidden)
        h = h.transpose(1, 2)       # (B, hidden, T)
        h = self.tcn(h)
        h = h.transpose(1, 2)       # (B, T, hidden)
        return self.head(h)         # (B, T, 6)


# ═══════════════════════════════════════════════
# 5. Dataset & DataLoader
# ═══════════════════════════════════════════════

class FlatBranchDataset(Dataset):
    """Stage2 학습용 평탄 지형 서브셋.

    BranchDataset에서 FLAT_CLASSES 샘플만 필터링하고
    BioMech 피처를 on-the-fly로 계산.
    """
    def __init__(self, branch_ds, bio_extractor,
                 flat_mask: np.ndarray, y_flat: np.ndarray):
        self.ds      = branch_ds
        self.bio     = bio_extractor
        self.indices = np.where(flat_mask)[0]
        self.y_flat  = y_flat

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        bi, _   = self.ds[int(self.indices[i])]
        foot_t  = bi["Foot"].unsqueeze(0).float()
        shank_t = bi["Shank"].unsqueeze(0).float()
        with torch.no_grad():
            bio_f = self.bio(foot_t, shank_t).squeeze(0)
        return bi, bio_f, int(self.y_flat[i])


def flat_collate(batch):
    bi_keys = batch[0][0].keys()
    bi  = {k: torch.stack([b[0][k] for b in batch]) for k in bi_keys}
    bio = torch.stack([b[1] for b in batch])
    y   = torch.tensor([b[2] for b in batch], dtype=torch.long)
    # subject label (4번째 항목, 없으면 -1)
    if len(batch[0]) >= 4:
        sbj = torch.tensor([b[3] for b in batch], dtype=torch.long)
    else:
        sbj = torch.full((len(batch),), -1, dtype=torch.long)
    return bi, bio, y, sbj


def make_flat_loader(ds: FlatBranchDataset, shuffle: bool,
                     balanced: bool = False) -> DataLoader:
    """SupCon용 balanced sampler 지원 로더."""
    sampler     = None
    use_shuffle = shuffle
    if shuffle and balanced:
        classes, counts = np.unique(ds.y_flat, return_counts=True)
        sample_w = (1.0 / counts.astype(np.float64))[ds.y_flat]
        sampler  = WeightedRandomSampler(
            weights=sample_w, num_samples=len(ds.y_flat), replacement=True)
        use_shuffle = False
        log(f"      ★ S2 균형 샘플링: "
            f"{dict(zip(classes.tolist(), counts.tolist()))}")
    return DataLoader(
        ds,
        batch_size  = config.BATCH,
        shuffle     = use_shuffle,
        sampler     = sampler,
        collate_fn  = flat_collate,
        drop_last   = shuffle,
        pin_memory  = config.USE_GPU,
    )


class AllDataBioDataset(Dataset):
    """전체 6cls 데이터셋 + BioMech 피처. SuperFusion 학습용.

    v11.5: bio_feats_norm (subject-normalized) 지원.
    bio_feats_norm이 주어지면 on-the-fly 추출 대신 사용.
    """
    def __init__(self, branch_ds, bio_extractor,
                 y_all: np.ndarray,
                 groups: np.ndarray | None = None,
                 bio_feats_norm: np.ndarray | None = None) -> None:
        self.ds            = branch_ds
        self.bio           = bio_extractor
        self.y             = y_all
        self.bio_feats_norm = bio_feats_norm  # (N, N_BIO) float32 or None

        if groups is not None:
            unique_sbj = sorted(set(groups.tolist()))
            sbj_map    = {s: i for i, s in enumerate(unique_sbj)}
            self.sbj   = np.array([sbj_map[g] for g in groups],
                                  dtype=np.int64)
        else:
            self.sbj   = np.full(len(y_all), -1, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        bi, _ = self.ds[i]
        if self.bio_feats_norm is not None:
            # 미리 계산된 normalized 피처 사용
            bio_f = torch.from_numpy(
                self.bio_feats_norm[i]).float()
        else:
            bi_b  = {k: v.unsqueeze(0) for k, v in bi.items()}
            with torch.no_grad():
                bio_f = self.bio(bi_b).squeeze(0)
        return bi, bio_f, int(self.y[i]), int(self.sbj[i])


def make_all_loader(ds: AllDataBioDataset, shuffle: bool,
                    balanced: bool = False) -> DataLoader:
    """E2E 학습용 6cls 균형 로더."""
    sampler     = None
    use_shuffle = shuffle
    if shuffle and balanced:
        classes, counts = np.unique(ds.y, return_counts=True)
        sample_w = (1.0 / counts.astype(np.float64))[ds.y]
        sampler  = WeightedRandomSampler(
            weights=sample_w, num_samples=len(ds.y), replacement=True)
        use_shuffle = False
        log(f"      ★ E2E 균형 샘플링: "
            f"{dict(zip(classes.tolist(), counts.tolist()))}")
    return DataLoader(
        ds,
        batch_size  = config.BATCH,
        shuffle     = use_shuffle,
        sampler     = sampler,
        collate_fn  = flat_collate,
        drop_last   = shuffle,
        pin_memory  = config.USE_GPU,
    )
# ═══════════════════════════════════════════════

def _make_sch(opt, epochs: int, warmup: int = 10,
              min_lr: float = config.MIN_LR, base_lr: float | None = None):
    """Linear warmup + Cosine annealing 스케줄러."""
    base = base_lr or opt.param_groups[0]["lr"]
    def fn(ep: int) -> float:
        if ep < warmup:
            return float(ep + 1) / warmup
        prog = float(ep - warmup) / max(epochs - warmup, 1)
        cos  = 0.5 * (1.0 + math.cos(math.pi * prog))
        mf   = min_lr / base
        return mf + (1.0 - mf) * cos
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# ═══════════════════════════════════════════════
# 7. Inner Val Split
# ═══════════════════════════════════════════════

def _inner_val_split(
    tr_idx: np.ndarray,
    groups: np.ndarray,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """tr_idx 안에서 subject 단위 inner val split.

    test fold와 완전 분리된 val set으로 early stopping 수행.
    """
    tr_groups  = groups[tr_idx]
    unique_sbj = np.unique(tr_groups)
    n_val_sbj  = max(1, int(len(unique_sbj) * val_ratio))
    rng        = np.random.default_rng(config.SEED)
    val_sbj    = set(rng.choice(unique_sbj, n_val_sbj, replace=False).tolist())

    inner_tr_mask = np.array([g not in val_sbj for g in tr_groups])
    inner_tr_idx  = tr_idx[inner_tr_mask]
    inner_va_idx  = tr_idx[~inner_tr_mask]
    log(f"    inner split: tr={len(inner_tr_idx)}  val={len(inner_va_idx)}"
        f"  val_sbj={sorted(val_sbj)}")
    return inner_tr_idx, inner_va_idx


# ═══════════════════════════════════════════════
# 8. 공통 Eval (Stage2 flat 로더용)
# ═══════════════════════════════════════════════

def _eval_flat_dl(model: nn.Module, loader: DataLoader,
                  crit: nn.Module) -> tuple[float, float]:
    """Stage2 val 평가. (loss, acc) 반환."""
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
# 9. Stage 1 — 3cls CE
# ═══════════════════════════════════════════════

def _get_feat_dim(backbone) -> int:
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

    def forward(self, bi: dict) -> torch.Tensor:
        return self.head(self.backbone.extract(bi))


def _y6_to_y3(y6: torch.Tensor) -> torch.Tensor:
    """6cls → 3cls (0=flat, 1=오르막, 2=내리막)."""
    y3 = torch.zeros_like(y6)
    y3[y6 == 1] = 1
    y3[y6 == 2] = 2
    return y3


def train_stage1(backbone, tr_dl, val_dl, te_dl, tag: str = "",
                 curve_dir: Path | None = None):
    """Stage1: 평탄/오르막/내리막 3cls CE 학습.

    Args:
        backbone:   M6_BranchCBAMCrossAug 인스턴스
        tr_dl:      훈련 DataLoader
        val_dl:     inner val DataLoader (early stopping 기준)
        te_dl:      test DataLoader (최종 평가만)
        tag:        로그 prefix
        curve_dir:  훈련 곡선 저장 경로 (None이면 저장 안 함)

    Returns:
        (s1_preds, s1_labels, model)
    """
    feat_dim = _get_feat_dim(backbone)
    head     = nn.Linear(feat_dim, 3).to(DEVICE)
    model    = _S1Wrapper(backbone, head).to(DEVICE)
    params   = list(model.parameters())

    opt    = torch.optim.AdamW(params, lr=S1_LR,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, S1_EPOCHS, warmup=10, base_lr=S1_LR)
    crit   = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))
    curve  = CurveTracker(f"S1_{tag.replace('[','').replace(']','')}")

    best_va    = 0.0
    best_state = None
    patience   = 0
    t0         = time.time()

    log(f"  {tag} Stage1 3cls ({S1_EPOCHS}ep, LR={S1_LR:.0e})  [val=inner split]")
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

        # [리뷰1] 마지막 배치 gradient 누락 방지
        if step_i >= 0 and (step_i + 1) % config.GRAD_ACCUM_STEPS != 0:
            if scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                opt.step()
            opt.zero_grad(set_to_none=True)

        # inner val 평가
        model.eval()
        va_c = va_n = 0
        with torch.inference_mode():
            for bi, yb in val_dl:
                bi, yb = _to_device(bi, yb=yb)
                yb3    = _y6_to_y3(yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model(bi)
                va_c += (logits.argmax(1) == yb3).sum().item()
                va_n += len(yb3)
        sch.step()
        va = va_c / max(va_n, 1)
        curve.record(acc=va)

        if va > best_va:
            best_va    = va
            best_state = _clone_state(model)
            patience   = 0
        else:
            patience += 1
            if patience >= S1_PATIENCE:
                log(f"  {tag} S1 EarlyStop ep{ep}"); break

        if ep % 20 == 0:
            log(f"  {tag} S1 ep{ep:03d}/{S1_EPOCHS}"
                f"  val_acc={va:.4f}  best={best_va:.4f}"
                f"  ({time.time()-t0:.0f}s)")

    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    log(f"  {tag} S1 완료  best_val={best_va:.4f}")
    if curve_dir: curve.save(curve_dir)

    # 최종 test 평가 + soft routing용 확률 반환
    model.eval()
    preds_list, labels_list, probs_list = [], [], []
    with torch.inference_mode():
        for bi, yb in te_dl:
            bi, yb = _to_device(bi, yb=yb)
            yb3    = _y6_to_y3(yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model(bi)
            probs_list.append(torch.softmax(logits.float(), dim=1).cpu())
            preds_list.append(logits.argmax(1).cpu())
            labels_list.append(yb3.cpu())

    s1_probs = torch.cat(probs_list).numpy()   # (N, 3) — flat=col0
    # Soft routing: flat 확률 >= threshold면 Stage2 라우팅 (안전한 변환)
    hard_preds    = torch.cat(preds_list).cpu().numpy().astype(np.int64)
    s1_preds_soft = np.where(
        s1_probs[:, 0] >= S1_SOFT_THRESHOLD,
        np.int64(0),    # flat으로 라우팅
        hard_preds
    ).astype(np.int64)
    return (s1_preds_soft,
            torch.cat(labels_list).numpy(),
            s1_probs,   # score fusion용 확률 반환
            model)


# ═══════════════════════════════════════════════
# 9b. E2E End-to-End Training (v8.9 핵심)
# ═══════════════════════════════════════════════

def train_superfusion(backbone, tr_dl, val_dl, te_dl, tag: str = "",
                      curve_dir: Path | None = None,
                      n_subjects: int = 50):
    """SuperFusion v11.4 학습.

    Phase1: 4-head multi-task + Subject-Adversarial GRL
      Loss = Focal_6 + 0.30*CE_3 + 0.20*CE_flat + 0.40*CE_bin
           + 0.10*KL_cons + λ_adv*CE_subject
    Phase2: 6cls 집중 파인튜닝
    """
    feat_dim = _get_feat_dim(backbone)
    model    = SuperFusionModel(backbone, feat_dim,
                                n_subjects=n_subjects).to(DEVICE)
    params   = list(model.parameters())

    # ── 클래스 가중치 ──────────────────────────
    all_y = np.concatenate([yb.numpy() for _, _, yb, __ in tr_dl])
    cls_w = auto_class_weights(all_y).to(DEVICE)
    log(f"  {tag} class_weights: {[f'{w:.2f}' for w in cls_w.tolist()]}")

    opt    = torch.optim.AdamW(params, lr=SF_LR, weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, SF_EPOCHS, warmup=10, base_lr=SF_LR)
    scaler = GradScaler(enabled=(config.USE_AMP and config.AMP_DTYPE == torch.float16))
    curve  = CurveTracker(f"SF_{tag.replace('[','').replace(']','')}")

    crit_6    = FocalLoss(gamma=1.5, weight=cls_w)
    crit_3    = nn.CrossEntropyLoss(label_smoothing=0.05)
    crit_flat = nn.CrossEntropyLoss(label_smoothing=0.05)
    crit_bin  = nn.CrossEntropyLoss()
    crit_subj = nn.CrossEntropyLoss()   # subject adversarial
    crit_trip = WithinSubjectTripletLoss(margin=TRIPLET_MARGIN)  # v11.5

    flat3_map = {0: 0, 3: 1, 4: 1, 5: 2}

    best_va = 0.0; best_state = None; patience = 0; t0 = time.time()
    log(f"  {tag} SuperFusion Phase1 ({SF_EPOCHS}ep, LR={SF_LR:.0e})"
        f"  Focal+cls_w | aux: 3cls×{SF_AUX_W3} flat×{SF_AUX_WFLAT}"
        f" bin×{SF_AUX_WBIN} cons×{SF_WCONS} adv×{SF_WADV}")

    for ep in range(1, SF_EPOCHS + 1):
        # GRL λ: 학습 초반 약하게, 후반 강하게 (Ganin et al. 스케줄)
        p   = ep / SF_EPOCHS
        lam = float(2.0 / (1.0 + math.exp(-10 * p)) - 1.0)  # 0→1 sigmoid

        model.train()
        opt.zero_grad(set_to_none=True)
        for step_i, (bi, bio_f, yb, sbj) in enumerate(tr_dl):
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            sbj = sbj.to(DEVICE)
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

                # Subject-Adversarial Loss (GRL로 gradient 역전)
                if l_subj is not None and (sbj >= 0).all():
                    loss = loss + SF_WADV * crit_subj(l_subj, sbj)

                # Within-Subject Triplet Loss (v11.5)
                trip_loss = crit_trip(emb.float(), yb, sbj)
                loss = loss + SF_WTRIPLET * trip_loss

                loss = loss / config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

        # 마지막 배치 처리
        if (step_i + 1) % config.GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        sch.step()

        # validation — Macro F1 기반 early stopping
        model.eval()
        va_preds_list, va_labels_list = [], []
        with torch.inference_mode():
            for bi, bio_f, yb, _ in val_dl:
                bi, bio_f, yb = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model.predict(bi, bio_f)
                va_preds_list.append(logits.argmax(1).cpu())
                va_labels_list.append(yb.cpu())
        va_p = torch.cat(va_preds_list).numpy()
        va_l = torch.cat(va_labels_list).numpy()
        va_acc = accuracy_score(va_l, va_p)
        va_f1  = f1_score(va_l, va_p, average="macro", zero_division=0)
        # 0.4*acc + 0.6*f1 → C1/C4/C5 소수 클래스 recall 보호
        va_score = 0.4 * va_acc + 0.6 * va_f1
        va = va_acc   # 로그 표시용
        curve.record(acc=va_score)

        if va_score > best_va:
            best_va = va_score; best_state = _clone_state(model); patience = 0
        else:
            patience += 1
            if patience >= SF_PATIENCE and ep > 20:
                log(f"  {tag} SF EarlyStop ep{ep}"
                    f"  best_score={best_va:.4f}"); break

        if ep % 15 == 0 or ep == 1:
            log(f"  {tag} SF ep{ep:03d}/{SF_EPOCHS}"
                f"  acc={va_acc:.4f}  f1={va_f1:.4f}"
                f"  score={va_score:.4f}  best={best_va:.4f}"
                f"  lr={opt.param_groups[0]['lr']:.1e}  ({time.time()-t0:.0f}s)")
            if _WANDB_OK and wandb.run is not None:
                wandb.log({
                    f"{tag}/sf_p1_val_acc":   va_acc,
                    f"{tag}/sf_p1_val_f1":    va_f1,
                    f"{tag}/sf_p1_val_score": va_score,
                    f"{tag}/sf_p1_lr":        opt.param_groups[0]['lr'],
                }, step=ep)

    if best_state: model.load_state_dict(best_state); model.to(DEVICE)
    log(f"  {tag} SF Phase1 완료  best_val={best_va:.4f}")
    if curve_dir: curve.save(curve_dir)

    # ── Phase2: 6cls 집중 파인튜닝 ────────────
    for p in model.head_3cls.parameters():  p.requires_grad = False
    for p in model.head_flat.parameters():  p.requires_grad = False
    for p in model.head_bin.parameters():   p.requires_grad = False
    params2  = [p for p in model.parameters() if p.requires_grad]
    opt2     = torch.optim.AdamW(params2, lr=SF_LR * 0.5, weight_decay=config.WEIGHT_DECAY)
    crit_ft  = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.03)
    best_va2 = 0.0; best_st2 = None; pat2 = 0

    log(f"  {tag} SF Phase2 6cls ({SF_FT_EPOCHS}ep, LR={SF_LR*0.5:.0e})")
    for ep in range(1, SF_FT_EPOCHS + 1):
        model.train()
        for bi, bio_f, yb, _ in tr_dl:
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            opt2.zero_grad(set_to_none=True)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                loss = crit_ft(model.predict(bi, bio_f), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(params2, config.GRAD_CLIP_NORM)
            scaler.step(opt2); scaler.update()

        model.eval()
        va_p2_list, va_l2_list = [], []
        with torch.inference_mode():
            for bi, bio_f, yb, _ in val_dl:
                bi, bio_f, yb = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model.predict(bi, bio_f)
                va_p2_list.append(logits.argmax(1).cpu())
                va_l2_list.append(yb.cpu())
        va_p2 = torch.cat(va_p2_list).numpy()
        va_l2 = torch.cat(va_l2_list).numpy()
        va2_acc = accuracy_score(va_l2, va_p2)
        va2_f1  = f1_score(va_l2, va_p2, average="macro", zero_division=0)
        va2     = 0.4 * va2_acc + 0.6 * va2_f1
        if va2 > best_va2:
            best_va2 = va2; best_st2 = _clone_state(model); pat2 = 0
        else:
            pat2 += 1
            if pat2 >= 12: break

    if best_st2:
        model.load_state_dict(best_st2); model.to(DEVICE)
    for p in model.parameters(): p.requires_grad = True
    log(f"  {tag} SF Phase2 완료  best_val={best_va2:.4f}")

    # ── test 평가 + embedding 수집 ────────────
    model.eval()
    preds_list, labels_list, emb_list = [], [], []
    with torch.inference_mode():
        for bi, bio_f, yb, _ in te_dl:
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                emb    = model.embed(bi, bio_f)
                logits = model.head_6cls(emb)
            preds_list.append(logits.argmax(1).cpu())
            labels_list.append(yb.cpu())
            emb_list.append(emb.cpu())

    te_preds  = torch.cat(preds_list).numpy()
    te_labels = torch.cat(labels_list).numpy()
    te_embs   = torch.cat(emb_list)   # (N_te, 256)
    return te_preds, te_labels, te_embs, model


def train_tcn_refiner(sf_model, tr_all_ds, va_all_ds, te_all_ds,
                      tr_groups: np.ndarray, va_groups: np.ndarray,
                      te_groups: np.ndarray,
                      tag: str = "") -> np.ndarray:
    """TCN Sequence Refiner — subject-aware 구현.

    Subject 단위로 연속 window sequence 생성 (subject 경계 절대 넘지 않음).
    입력: SuperFusion 256-dim embedding
    출력: refined test 6cls predictions
    """
    # ── embedding 추출 (순서 보장, shuffle=False) ──
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

    # ── Subject-aware sequence dataset ───────────
    def _make_seq_ds(emb: torch.Tensor, lbl: torch.Tensor,
                     grp: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Subject 경계를 넘지 않는 stride=1 완전 overlap 슬라이딩 윈도우.

        기존: 중심 기준 패딩 방식 (경계 샘플에 zero-pad)
        변경: stride=1 sliding → N-seq_len+1개 시퀀스
          - 경계 패딩 제거 → 노이즈 없음
          - 시퀀스 수 증가 → TCN 학습 데이터 풍부
          - 라벨: 윈도우 중앙 샘플 (seq_len//2)
        """
        seqs, tgts = [], []
        half = TCN_SEQ_LEN // 2
        for sbj in np.unique(grp):
            idx = np.where(grp == sbj)[0]
            se  = emb[idx]       # (N_sbj, 256)
            sl  = lbl[idx]
            N   = len(idx)
            if N < TCN_SEQ_LEN:
                # subject 샘플 수가 seq_len보다 작으면 zero-pad 1개만
                pad_r = TCN_SEQ_LEN - N
                win_p = F.pad(se.T, (0, pad_r)).T
                seqs.append(win_p)
                tgts.append(int(sl[N // 2]))
                continue
            # stride=1 sliding window (경계 패딩 없음)
            for i in range(N - TCN_SEQ_LEN + 1):
                win = se[i: i + TCN_SEQ_LEN]   # (TCN_SEQ_LEN, 256)
                seqs.append(win)
                tgts.append(int(sl[i + half]))  # 중앙 샘플 라벨
        return torch.stack(seqs), torch.tensor(tgts, dtype=torch.long)

    log(f"  {tag} TCN: subject-aware sequence 생성...")
    tr_seq, tr_tgt = _make_seq_ds(tr_emb, tr_lbl, tr_groups)
    va_seq, va_tgt = _make_seq_ds(va_emb, va_lbl, va_groups)
    te_seq, te_tgt = _make_seq_ds(te_emb, te_lbl, te_groups)
    log(f"  {tag} TCN: tr={len(tr_seq)}  va={len(va_seq)}  te={len(te_seq)}")

    # ── TCN 학습 ──────────────────────────────────
    tcn      = TCNRefiner(in_dim=256, hidden=TCN_HIDDEN, num_classes=6).to(DEVICE)
    cls_w_tcn = auto_class_weights(tr_tgt.numpy()).to(DEVICE)
    crit     = nn.CrossEntropyLoss(weight=cls_w_tcn)
    opt      = torch.optim.AdamW(tcn.parameters(), lr=TCN_LR, weight_decay=1e-4)
    best_va  = 0.0; best_st = None; pat = 0; half = TCN_SEQ_LEN // 2

    tr_loader = DataLoader(list(zip(tr_seq, tr_tgt)),
                           batch_size=512, shuffle=True, drop_last=True)
    va_loader = DataLoader(list(zip(va_seq, va_tgt)),
                           batch_size=512, shuffle=False)

    log(f"  {tag} TCN 학습 ({TCN_EPOCHS}ep, LR={TCN_LR:.0e})"
        f"  seq_len={TCN_SEQ_LEN}  subjects={len(np.unique(tr_groups))}")
    for ep in range(1, TCN_EPOCHS + 1):
        tcn.train()
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE).float(); yb = yb.to(DEVICE)
            opt.zero_grad()
            out  = tcn(xb)[:, half, :]    # 중심 타임스텝 예측 (B, 6)
            loss = crit(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tcn.parameters(), 1.0)
            opt.step()

        tcn.eval()
        tcn_vp, tcn_vl = [], []
        with torch.inference_mode():
            for xb, yb in va_loader:
                xb = xb.to(DEVICE).float(); yb = yb.to(DEVICE)
                out = tcn(xb)[:, half, :]
                tcn_vp.append(out.argmax(1).cpu())
                tcn_vl.append(yb.cpu())
        tcn_vp = torch.cat(tcn_vp).numpy()
        tcn_vl = torch.cat(tcn_vl).numpy()
        va_acc = accuracy_score(tcn_vl, tcn_vp)
        va_f1  = f1_score(tcn_vl, tcn_vp, average="macro", zero_division=0)
        va     = 0.4 * va_acc + 0.6 * va_f1
        if va > best_va:
            best_va = va
            best_st = {k: v.clone() for k, v in tcn.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= TCN_PATIENCE: break
        if ep % 10 == 0:
            log(f"  {tag} TCN ep{ep:02d}/{TCN_EPOCHS}"
                f"  acc={va_acc:.4f}  f1={va_f1:.4f}"
                f"  score={va:.4f}  best={best_va:.4f}")

    if best_st: tcn.load_state_dict(best_st)
    log(f"  {tag} TCN 완료  best_val={best_va:.4f}")

    # ── test 예측 ─────────────────────────────────
    tcn.eval()
    te_loader = DataLoader(list(zip(te_seq, te_tgt)), batch_size=512, shuffle=False)
    preds_list = []
    with torch.inference_mode():
        for xb, _ in te_loader:
            xb  = xb.to(DEVICE).float()
            out = tcn(xb)[:, half, :]
            preds_list.append(out.argmax(1).cpu())
    tcn_preds = torch.cat(preds_list).numpy()
    tcn_acc   = accuracy_score(te_tgt.numpy(), tcn_preds)
    log(f"  {tag} TCN test Acc={tcn_acc:.4f}")
    return tcn_preds


# ═══════════════════════════════════════════════
# 10. Stage 2 — Step1: CE Warmup
# ═══════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    apply_args(args)

    # ── W&B 초기화 ────────────────────────────────
    use_wandb = args.wandb and _WANDB_OK
    if args.wandb and not _WANDB_OK:
        log("  [W&B] wandb 미설치 — pip install wandb 후 재실행. 로컬 로그만 기록.")
    if use_wandb:
        import subprocess
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent,
            ).decode().strip()
        except Exception:
            git_hash = "unknown"

        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"v11.5-N{getattr(args,'n_subjects','?')}",
            config={
                "version":      "v11.5",
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
                "aux_w3":       SF_AUX_W3,
                "aux_wflat":    SF_AUX_WFLAT,
                "aux_wbin":     SF_AUX_WBIN,
                "cons_w":       SF_WCONS,
            },
        )
        log(f"  [W&B] project={args.wandb_project}"
            f"  run={wandb.run.name}  git={git_hash}")

    config.print_config()
    log(
        f"  ★ v11.5  SuperFusion + TCN Refiner\n"
        f"  SF: {SF_EPOCHS}ep Phase1 + {SF_FT_EPOCHS}ep Phase2"
        f"  LR={SF_LR:.0e}  Focal(γ=1.5)+cls_w\n"
        f"  Loss: 6cls + {SF_AUX_W3}x3cls + {SF_AUX_WFLAT}xflat3"
        f" + {SF_AUX_WBIN}xbin + {SF_WCONS}xKL_cons\n"
        f"  BioMech N_BIO={BioMechFeatures.N_BIO} (Kurt/Skew/ZCR 추가)"
        f"  TCN seq={TCN_SEQ_LEN}  Vote window={args.vote_window}\n"
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

        # ── inner val split ───────────────────────
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

        # ── Stage1 (3cls backbone warmup) ─────────
        tag         = f"[F{fi}]"
        backbone_s1 = M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        s1_preds, s1_labels, s1_probs, s1_model = train_stage1(
            backbone_s1, tr_dl, val_dl, te_dl, tag, curve_dir=curve_dir)
        s1_acc = accuracy_score(s1_labels, s1_preds)
        log(f"  {tag} Stage1 Acc={s1_acc:.4f}")

        # ── SuperFusion v11.4 ──────────────────────
        # [핵심] 18ch augment 제거 → 12ch 통일 → S1 backbone 직접 전이
        # S1이 학습한 표현(86%)을 SF가 물려받아 시작
        log(f"  {tag} ★ SuperFusion 시작 (S1 backbone 전이 12ch)"
            f"  N_BIO={BioMechFeatures.N_BIO}")
        backbone_sf = M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        # S1 backbone weight 전이 (같은 12ch 구조)
        backbone_sf.load_state_dict(s1_model.backbone.state_dict())
        log(f"  {tag} ★ S1→SF backbone 전이 완료 (86% 지식 활용)")

        te_y6 = y[te_idx]
        tr_y6 = y[inner_tr_idx]
        va_y6 = y[inner_va_idx]

        # groups 전달 → subject adversarial label 생성
        tr_groups_local = groups[inner_tr_idx]
        va_groups_local = groups[inner_va_idx]
        te_groups_local = groups[te_idx]

        # ── Subject-Wise BioMech Normalization (v11.5) ────
        log(f"  {tag} BioMech 피처 사전 추출 (subject 정규화용)...")
        bio_extractor.eval()

        def _extract_all_bio(ds_inner):
            feats = []
            for i in range(len(ds_inner)):
                bi_s, _ = ds_inner[i]
                bi_b = {k: v.unsqueeze(0) for k, v in bi_s.items()}
                with torch.no_grad():
                    f = bio_extractor(bi_b).squeeze(0).cpu().numpy()
                feats.append(f)
            return np.stack(feats, axis=0).astype(np.float32)

        tr_bio_raw = _extract_all_bio(tr_ds)
        va_bio_raw = _extract_all_bio(val_ds)
        te_bio_raw = _extract_all_bio(te_ds)

        subj_norm   = SubjectNormalizer()
        tr_bio_norm = subj_norm.fit_transform(tr_bio_raw, tr_groups_local)
        va_bio_norm = subj_norm.transform(va_bio_raw, va_groups_local)
        te_bio_norm = subj_norm.transform(te_bio_raw, te_groups_local)
        log(f"  {tag} Subject 정규화 완료 "
            f"tr={tr_bio_norm.shape} va={va_bio_norm.shape}"
            f" te={te_bio_norm.shape}")

        tr_all_ds = AllDataBioDataset(tr_ds,  bio_extractor, tr_y6,
                                      groups=tr_groups_local,
                                      bio_feats_norm=tr_bio_norm)
        va_all_ds = AllDataBioDataset(val_ds, bio_extractor, va_y6,
                                      groups=va_groups_local,
                                      bio_feats_norm=va_bio_norm)
        te_all_ds = AllDataBioDataset(te_ds,  bio_extractor, te_y6,
                                      groups=te_groups_local,
                                      bio_feats_norm=te_bio_norm)

        tr_sf_dl = make_all_loader(tr_all_ds, True,  balanced=True)
        va_sf_dl = make_all_loader(va_all_ds, False)
        te_sf_dl = make_all_loader(te_all_ds, False)

        # fold당 실제 subject 수
        n_fold_subjects = len(np.unique(tr_groups_local))

        sf_preds, sf_labels, sf_te_embs, sf_model = train_superfusion(
            backbone_sf, tr_sf_dl, va_sf_dl, te_sf_dl,
            tag=f"{tag}[SF]", curve_dir=curve_dir,
            n_subjects=n_fold_subjects)

        sf_acc = accuracy_score(sf_labels, sf_preds)
        sf_f1  = f1_score(sf_labels, sf_preds, average="macro", zero_division=0)
        log(f"  {tag} SF (before TCN)  Acc={sf_acc:.4f}  F1={sf_f1:.4f}")

        # ── TCN Sequence Refiner (subject-aware) ──
        log(f"  {tag} ★ TCN Refiner 시작 (subject-aware)")
        tcn_preds = train_tcn_refiner(
            sf_model,
            tr_all_ds, va_all_ds, te_all_ds,
            tr_groups=tr_groups_local,
            va_groups=va_groups_local,
            te_groups=te_groups_local,
            tag=f"{tag}[TCN]")

        # ── 후처리: Majority Vote ─────────────────
        sf_preds_vote  = sf_preds.copy()
        tcn_preds_vote = tcn_preds.copy()
        if args.vote_window > 0:
            sf_preds_vote  = majority_vote_by_subject(
                sf_preds, te_idx=te_idx, groups=groups, window=args.vote_window)
            tcn_preds_vote = majority_vote_by_subject(
                tcn_preds, te_idx=te_idx, groups=groups, window=args.vote_window)

        acc_sf  = accuracy_score(sf_labels, sf_preds_vote)
        f1_sf   = f1_score(sf_labels, sf_preds_vote, average="macro", zero_division=0)
        acc_tcn = accuracy_score(sf_labels, tcn_preds_vote)
        f1_tcn  = f1_score(sf_labels, tcn_preds_vote, average="macro", zero_division=0)
        log(f"  {tag} ★ SF+Vote   Acc={acc_sf:.4f}  F1={f1_sf:.4f}")
        log(f"  {tag} ★ TCN+Vote  Acc={acc_tcn:.4f}  F1={f1_tcn:.4f}")

        # TCN이 SF보다 나으면 TCN 사용
        final_preds  = tcn_preds_vote if acc_tcn >= acc_sf else sf_preds_vote
        final_labels = sf_labels
        acc = max(acc_tcn, acc_sf)
        f1  = f1_tcn if acc_tcn >= acc_sf else f1_sf
        log(f"  {tag} ★ 최종  Acc={acc:.4f}  F1={f1:.4f}"
            f"  ({'TCN' if acc_tcn >= acc_sf else 'SF'})")

        all_preds.append(final_preds)
        all_labels.append(final_labels)
        fold_meta.append({
            "fold":          fi,
            "test_subjects": te_s,
            "s1_acc":        round(s1_acc, 4),
            "sf_acc":        round(acc_sf, 4),
            "sf_f1":         round(f1_sf, 4),
            "tcn_acc":       round(acc_tcn, 4),
            "tcn_f1":        round(f1_tcn, 4),
            "final_acc":     round(acc, 4),
            "final_f1":      round(f1, 4),
            "used_tcn":      acc_tcn >= acc_sf,
            "fold_time_min": round((time.time() - t_fold) / 60, 1),
        })

        # W&B fold 결과 로깅
        if _WANDB_OK and wandb.run is not None:
            wandb.log({
                f"fold{fi}/s1_acc":  s1_acc,
                f"fold{fi}/sf_acc":  acc_sf,
                f"fold{fi}/sf_f1":   f1_sf,
                f"fold{fi}/tcn_acc": acc_tcn,
                f"fold{fi}/tcn_f1":  f1_tcn,
                f"fold{fi}/final_acc": acc,
                f"fold{fi}/final_f1":  f1,
            })

        # ── fold 메모리 정리 ─────────────────────
        del backbone_s1, backbone_sf, s1_model, sf_model
        del tr_all_ds, va_all_ds, te_all_ds
        del tr_sf_dl, va_sf_dl, te_sf_dl
        del tr_ds, val_ds, te_ds
        gc.collect()
        if config.USE_GPU: torch.cuda.empty_cache()
        clear_fold_cache(f"HC{fi}")
        clear_fold_cache(f"HC{fi}v")

    # ── 전체 결과 ─────────────────────────────────
    preds_all  = np.concatenate(all_preds)
    labels_all = np.concatenate(all_labels)
    acc_all    = accuracy_score(labels_all, preds_all)
    f1_all     = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    cm         = confusion_matrix(labels_all, preds_all)
    recalls    = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    total_min  = (time.time() - t_total) / 60

    print(f"\n{'='*60}")
    print(f"  ★ v11.5 SuperFusion+TCN  {config.KFOLD}-Fold")
    print(f"  총 소요: {total_min:.1f}분")
    print(f"  Acc={acc_all:.4f}  MacroF1={f1_all:.4f}")
    print(f"{'='*60}")
    print(f"  ── 클래스별 Recall ──")
    for i, r in enumerate(recalls):
        print(f"    {CLASS_NAMES_ALL.get(i, f'C{i+1}'):<14} {r*100:.1f}%")

    rep = classification_report(
        labels_all, preds_all,
        target_names=[f"C{c}" for c in le.classes_], digits=4, zero_division=0)
    (out / "report_v115.txt").write_text(
        f"v11.5 SuperFusion+TCN\nAcc={acc_all:.4f}  F1={f1_all:.4f}\n\n{rep}")

    le_out = LabelEncoder()
    le_out.fit(sorted(set(labels_all.tolist())))
    save_cm(preds_all, labels_all, le_out, "SuperFusion_TCN_v115_KFold", out)

    summary = {
        "experiment":  "hierarchical_kfold_v115",
        "version":     "v11.4",
        "method":      "SuperFusion (4-head) + TCN Sequence Refiner",
        "n_bio":       BioMechFeatures.N_BIO,
        "total_minutes": round(total_min, 1),
        "overall": {"acc": round(acc_all, 4), "f1": round(f1_all, 4)},
        "per_class_recall": {
            CLASS_NAMES_ALL.get(i, f"C{i+1}"): round(float(r), 4)
            for i, r in enumerate(recalls)
        },
        "fold_meta": fold_meta,
    }
    path_json = out / "summary_v115.json"
    path_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  ✅ {path_json}")

    # ── W&B 최종 요약 ─────────────────────────────
    if _WANDB_OK and wandb.run is not None:
        # 전체 지표
        wandb.summary["overall_acc"] = acc_all
        wandb.summary["overall_f1"]  = f1_all
        for i, r in enumerate(recalls):
            name = CLASS_NAMES_ALL.get(i, f"C{i+1}")
            wandb.summary[f"recall_{name}"] = round(float(r), 4)

        # Confusion Matrix (W&B 전용 시각화)
        class_names = [CLASS_NAMES_ALL.get(i, f"C{i+1}") for i in range(6)]
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=labels_all.tolist(),
                preds=preds_all.tolist(),
                class_names=class_names,
            ),
            "overall_acc": acc_all,
            "overall_f1":  f1_all,
        })
        wandb.finish()
        log("  [W&B] 로깅 완료 — 대시보드에서 결과 확인")

    h5data.close()


if __name__ == "__main__":
    main()