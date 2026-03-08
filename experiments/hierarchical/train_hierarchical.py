"""
train_hierarchical.py — Hierarchical v9.0 (3-Stage + ArcFace + C5Fix)
═══════════════════════════════════════════════════════
v8.9 → v9.0 변경:
  [A] Bug Fix: S3_CE_FT_EP 40→20 (주석과 값 불일치 수정)
  [B] FOCAL_GAMMA 2.0→1.5 (S2 수렴 안정화)
  [C] S3_PATIENCE 20→25 (C5 충분히 학습)
  [D] S2_FOCAL_PAT 25→30 (Focal finetune 조기종료 완화)
  [E] Stage3 CE 마무리: plain CE → Focal(γ=1.0, C4=0.8, C5=1.4)
  [F] Stage3 ArcFace val: plain acc → balanced acc (C4/C5 편향 방지)
  [G] 결합: Hard Routing (Score Fusion 제거 — 확률 미보정 역효과 확인)
논문 베이스:
  [1] Ordóñez & Roggen (2016) IEEE TNNLS — Hierarchical HAR
  [2] Khosla et al. (2020) NeurIPS — Supervised Contrastive
  [3] Niswander et al. (2021) — 발 IMU 충격 생체역학
  [4] Lin et al. (2017) ICCV — Focal Loss
  [5] Deng et al. (2019) CVPR — ArcFace
  [6] Zheng et al. (2021) — IMU terrain FFT classification
  [7] Zhang et al. (2018) ICLR — mixup (미적용 — ArcFace와 충돌)

학습 파이프라인:
  Stage1: 3cls CE (평탄/오르막/내리막)
          Soft Routing: flat 확률 ≥ 0.35 → Stage2 라우팅 (hard argmax 대비 flat recall↑)
  Stage2: 3cls [C1 / C6 / C4C5_merged]
          Step1 CE-warmup  (60ep)   — random init backbone 기본 표현 학습
          Step2 SupCon     (120ep)  — Khosla 2020, balanced sampler
          Step3 FocalLoss  (100ep)  — gamma=2.0, 소수 클래스 집중
          Step4 CE-마무리  (50ep)
  Stage3: binary [C4 vs C5], S2 backbone 전이학습 → ArcFace + FFT Branch
          ArcFace (120ep, s=32, m=0.5) — angular margin, Mixup 미사용
          CE 마무리 (20ep)
  결합: Hard Routing (Soft Stage1 → Stage2 → Stage3)
  후처리: Majority Vote (window=5)

v8.8 → v8.9 변경:
  [A] 3-Stage 구조: Stage2 4cls → 3cls(C1/C6/C4C5) + Stage3 binary(C4/C5)
  [B] ArcFace + FFT Branch (Stage3 전용)
  [C] BioMech N_BIO 16→20 (Spectral Centroid, Impact Duration 추가)
  [D] S1 Soft Routing (threshold=0.35)
  [E] grad accum 마지막 step 누락 수정 (Stage1/공통 _run_epoch)
  [F] ArcFace cosine clamp(-1+1e-7, 1-1e-7) — NaN 방지
  [G] BioMech ratio/var_ratio log1p — 값 폭발 방지
  [H] Stage3 S2 backbone 전이 — random init 과적합 방지
  [I] Stage3 Mixup 제거 — ArcFace 구면 기하와 충돌
  [J] Stage3 oracle 평가 구현 (model 반환)
  [K] CE Warmup 유지 — random init → SupCon 직행 시 oracle 39% 확인
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys, time, json, gc, warnings, math, argparse
warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass, field

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
# 하이퍼파라미터
# ═══════════════════════════════════════════════

S1_EPOCHS      = 60     # Stage1 에포크
S1_LR          = 5e-5   # Stage1 학습률

S2_WARMUP_EP   = 80     # Step1 CE 워밍업
                        # S1 backbone 전이 → random init 아님 → 40ep 충분
S2_PRETRAIN_EP = 100    # Step2 SupCon (미사용 — oracle 40% 실험 확인)
S2_FOCAL_EP    = 100    # Step3 FocalLoss finetune (전체 파라미터)
S2_FINETUNE_EP = 50     # Step4 CE 마무리 (backbone partial unfreeze)

S2_LR          = 1e-4   # Stage2 기본 학습률
S2_FOCAL_LR    = 3e-5   # Step3 Focal LR — 3e-5에서 낮춤 (repr drift 방지)
S2_FOCAL_PAT   = 30     # Step3 early stop patience  ← 25→30
S1_PATIENCE    = 15     # Stage1 전용 patience (config.EARLY_STOP=7과 분리)

FOCAL_GAMMA    = 1.5    # Focal Loss gamma  ← 2.0→1.5 (S2 수렴 안정화)
TEMPERATURE    = 0.10   # SupCon temperature (미사용)

# Stage3 (C4 vs C5 binary, ArcFace)
S3_EPOCHS      = 120    # Stage3 ArcFace 에포크 (2-phase)
S3_CE_FT_EP    = 20     # Stage3 CE 마무리 20ep (ArcFace margin 구조 보존)  ← 버그수정 40→20
S3_LR          = 5e-5   # Stage3 학습률
S3_ARCFACE_S   = 32.0   # ArcFace scale
S3_ARCFACE_M   = 0.50   # ArcFace margin
S3_PATIENCE    = 25     # Stage3 early stop patience  ← 20→25 (C5 충분히 학습)
S3_C5_WEIGHT   = 1.4    # Stage3 CE 마무리 C5 upweight (C5 recall 개선)

S3_FFT_BINS    = 64     # FFT 스펙트럼 bin 수
S3_FFT_DIM     = 128    # FFT branch 출력 차원
S3_MIXUP_ALPHA = 0.3    # Mixup alpha (미사용 — ArcFace 충돌 확인)

S1_SOFT_THRESHOLD = 0.35  # Soft routing: flat 확률 ≥ threshold → Stage2 라우팅
S2_WEIGHTS        = None  # None → auto_class_weights() 사용


# ═══════════════════════════════════════════════
# CLI — argparse (하이퍼파라미터 오버라이드)
# ═══════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """CLI로 하이퍼파라미터 오버라이드.

    기본값은 위 상수와 동일. 실험별로 변경 가능.

    Examples:
        python train_hierarchical.py
        python train_hierarchical.py --s1_epochs 80 --temperature 0.05
        python train_hierarchical.py --s2_lr 5e-5 --focal_gamma 1.5 --no_vote
    """
    p = argparse.ArgumentParser(
        description="Hierarchical SupCon Terrain Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Stage1
    p.add_argument("--s1_epochs",    type=int,   default=S1_EPOCHS)
    p.add_argument("--s1_lr",        type=float, default=S1_LR)
    # Stage2
    p.add_argument("--s2_warmup",    type=int,   default=S2_WARMUP_EP)
    p.add_argument("--s2_pretrain",  type=int,   default=S2_PRETRAIN_EP)
    p.add_argument("--s2_focal",     type=int,   default=S2_FOCAL_EP)
    p.add_argument("--s2_finetune",  type=int,   default=S2_FINETUNE_EP)
    p.add_argument("--s2_lr",        type=float, default=S2_LR)
    p.add_argument("--s2_focal_lr",  type=float, default=S2_FOCAL_LR)
    p.add_argument("--s2_focal_pat", type=int,   default=S2_FOCAL_PAT)
    p.add_argument("--temperature",  type=float, default=TEMPERATURE)
    p.add_argument("--focal_gamma",  type=float, default=FOCAL_GAMMA)
    # Post-processing
    p.add_argument("--vote_window",  type=int,   default=5,
                   help="Majority vote window size (0 = 비활성)")
    # Stage3 ArcFace
    p.add_argument("--s3_epochs",    type=int,   default=S3_EPOCHS)
    p.add_argument("--arcface_s",    type=float, default=S3_ARCFACE_S)
    p.add_argument("--arcface_m",    type=float, default=S3_ARCFACE_M)
    return p.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    """파싱된 args를 전역 하이퍼파라미터에 반영."""
    global S1_EPOCHS, S1_LR
    global S2_WARMUP_EP, S2_PRETRAIN_EP, S2_FOCAL_EP, S2_FINETUNE_EP
    global S2_LR, S2_FOCAL_LR, S2_FOCAL_PAT, TEMPERATURE, FOCAL_GAMMA
    global S3_EPOCHS, S3_ARCFACE_S, S3_ARCFACE_M
    S1_EPOCHS      = args.s1_epochs
    S1_LR          = args.s1_lr
    S2_WARMUP_EP   = args.s2_warmup
    S2_PRETRAIN_EP = args.s2_pretrain
    S2_FOCAL_EP    = args.s2_focal
    S2_FINETUNE_EP = args.s2_finetune
    S2_LR          = args.s2_lr
    S2_FOCAL_LR    = args.s2_focal_lr
    S2_FOCAL_PAT   = args.s2_focal_pat
    TEMPERATURE    = args.temperature
    FOCAL_GAMMA    = args.focal_gamma
    S3_EPOCHS      = args.s3_epochs
    S3_ARCFACE_S   = args.arcface_s
    S3_ARCFACE_M   = args.arcface_m


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

def auto_class_weights(y_flat: np.ndarray) -> list[float]:
    """학습 데이터 기반 클래스 가중치 자동 계산.

    classes를 y_flat에서 동적으로 추론 → 3cls/4cls 모두 호환.
    """
    classes = np.unique(y_flat)
    weights = compute_class_weight("balanced", classes=classes, y=y_flat)
    w_list  = weights.tolist()
    label   = ["C1", "C4", "C5", "C6"] if len(classes) == 4 else \
              ["C1", "C6", "C4C5"] if len(classes) == 3 else \
              [str(c) for c in classes]
    parts   = "  ".join(f"{l}={w:.3f}" for l, w in zip(label, w_list))
    log(f"    auto class_weights (balanced): {parts}")
    return w_list


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


def combine_predictions_3stage(
    s1_preds: np.ndarray,
    s2_preds: np.ndarray,
    s3_preds: np.ndarray,
) -> np.ndarray:
    """3단계 Hard Routing 결합 (벡터화).

    Stage1: 0=flat → Stage2,  1=C2 오르막,  2=C3 내리막
    Stage2: 0=C1,  1=C6,  2=C4C5 → Stage3
    Stage3: 0=C4,  1=C5

    Args:
        s1_preds: (N_te,)     Stage1 예측
        s2_preds: (N_flat,)   Stage2 예측 (s1==0 샘플만)
        s3_preds: (N_c4c5,)   Stage3 예측 (s2==2 샘플만)
    """
    final    = s1_preds.copy().astype(np.int64)
    flat_idx = np.where(s1_preds == 0)[0]

    # S2 결과: 0→C1(0), 1→C6(5), 2→placeholder(-1)
    s2_full = np.where(s2_preds == 0, 0,
              np.where(s2_preds == 1, 5, -1)).astype(np.int64)

    # S3 결과: 0→C4(3), 1→C5(4), s2==2 위치에 삽입
    c4c5_in_s2 = (s2_preds == 2)
    s2_full[c4c5_in_s2] = np.where(s3_preds == 0, 3, 4)

    final[flat_idx] = s2_full
    return final


def majority_vote_by_subject(preds: np.ndarray,
                              te_idx: np.ndarray,
                              groups: np.ndarray,
                              window: int = 5) -> np.ndarray:
    """Subject-aware majority vote post-processing.

    fold 전체에 걸지 않고 subject 단위로 적용:
    다른 subject 샘플끼리 vote되는 평가 artifact 방지.
    """
    out       = preds.copy()
    te_groups = groups[te_idx]
    for sbj in np.unique(te_groups):
        idx_local        = np.where(te_groups == sbj)[0]
        out[idx_local]   = majority_vote_smooth(out[idx_local], window=window)
    return out


def combine_predictions_score_fusion(
    s1_probs: np.ndarray,
    s2_probs: np.ndarray,
    s3_probs: np.ndarray,
    s1_preds: np.ndarray,
) -> np.ndarray:
    """Score Fusion — hard label routing 대신 확률 곱으로 최종 6cls 결정.

    Hard routing은 S1이 flat을 argmax로 안 줘도 P(flat)이 있으면
    downstream 정보가 완전히 버려지는 문제 → score fusion으로 해결.

    최종 6cls 확률 계산:
      C2(오르막) = P_s1(up)
      C3(내리막) = P_s1(down)
      C1(미끄러운) = P_s1(flat) × P_s2(C1)
      C6(평지)     = P_s1(flat) × P_s2(C6)
      C4(흙길)     = P_s1(flat) × P_s2(C4C5) × P_s3(C4)
      C5(잔디)     = P_s1(flat) × P_s2(C4C5) × P_s3(C5)

    Args:
        s1_probs: (N, 3)      P(flat), P(up), P(down)
        s2_probs: (N_flat, 3) P(C1), P(C6), P(C4C5) — s1_preds==0 샘플만
        s3_probs: (N_c4c5, 2) P(C4), P(C5) — s2에서 C4C5 라우팅된 샘플만
        s1_preds: (N,)        soft routing 결과 (슬롯 인덱스 추적용)
    """
    N       = len(s1_probs)
    score6  = np.zeros((N, 6), dtype=np.float64)

    p_flat = s1_probs[:, 0]   # (N,)
    p_up   = s1_probs[:, 1]   # (N,)
    p_down = s1_probs[:, 2]   # (N,)

    score6[:, 1] = p_up    # C2 오르막
    score6[:, 2] = p_down  # C3 내리막

    # S2 확률 삽입 (s1_preds==0 위치)
    flat_idx = np.where(s1_preds == 0)[0]
    if len(flat_idx) > 0 and len(s2_probs) > 0:
        n_s2 = min(len(flat_idx), len(s2_probs))
        for i, fi in enumerate(flat_idx[:n_s2]):
            pf         = p_flat[fi]
            score6[fi, 0] = pf * s2_probs[i, 0]   # C1
            score6[fi, 5] = pf * s2_probs[i, 1]   # C6
            # C4C5 → S3
            p_c4c5    = pf * s2_probs[i, 2]
            score6[fi, 3] = p_c4c5 * 0.5          # placeholder C4
            score6[fi, 4] = p_c4c5 * 0.5          # placeholder C5

        # S3 확률 삽입 (s2에서 C4C5 → 2 라우팅된 샘플)
        if len(s3_probs) > 0:
            c4c5_in_flat = np.where(
                np.argmax(s2_probs[:n_s2], axis=1) == 2)[0]
            for j, ci in enumerate(c4c5_in_flat):
                if j >= len(s3_probs): break
                fi         = flat_idx[ci]
                p_c4c5    = score6[fi, 3] + score6[fi, 4]   # 기존 합
                score6[fi, 3] = p_c4c5 * s3_probs[j, 0]    # C4
                score6[fi, 4] = p_c4c5 * s3_probs[j, 1]    # C5

    return score6.argmax(axis=1).astype(np.int64)


# ═══════════════════════════════════════════════
# 1. Biomechanical Feature Extractor (N_BIO=20)
# ═══════════════════════════════════════════════

class BioMechFeatures(nn.Module):
    """생체역학 충격 피처 추출기.

    스텝 분절 신호에서 Heel-Strike 기반 생체역학 지표 추출.
    Niswander et al. (2021) 충격 분석 기반.

    피처 20개:
      0~3  : Foot/Shank LT/RT 충격 피크값
      4~5  : Foot/Shank 충격비 (지면 흡수량 지표)
      6~7  : step-domain 상위 주파수 에너지 비율
      8~9  : Foot LT/RT 신호 표준편차 (C1 미끄러운 → 변동성↑)
      10~11: Foot LT/RT 피크 후 감쇠율 (C5 잔디 → 빠른 감쇠)
      12~13: Shank LT/RT 진동 |diff|.mean (C4 흙길 → 진동↑)
      14~15: Foot/Shank 분산비 LT/RT (C1 불안정 지표)
      16~17: Spectral Centroid LT/RT — Niswander2021
             C4 흙길 → 고주파↑ → centroid↑
             C5 잔디 → 부드러운 흡수 → centroid↓
      18~19: Impact Duration LT/RT — Niswander2021
             C5 잔디 → 충격 넓게 분산 → duration↑
             C4 흙길 → 짧고 날카로운 충격 → duration↓
    """
    N_BIO = 20

    def __init__(self) -> None:
        super().__init__()
        self.foot_z  = config.FOOT_Z_ACCEL_IDX
        self.shank_z = config.SHANK_Z_ACCEL_IDX
        self.sr      = config.SAMPLE_RATE
        self.hf_bin  = int(30 * config.TS / config.SAMPLE_RATE)  # 30Hz 이상 고주파 (C4 흙길 특화)

    @torch.no_grad()
    def forward(self, foot_x: torch.Tensor, shank_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            foot_x:  (B, 12, T)
            shank_x: (B, 12, T)
        Returns:
            (B, N_BIO)
        """
        foot_x  = foot_x.float()
        shank_x = shank_x.float()
        eps     = 1e-6

        fz_lt = foot_x[:,  self.foot_z[0],  :]
        fz_rt = foot_x[:,  self.foot_z[1],  :]
        sz_lt = shank_x[:, self.shank_z[0], :]
        sz_rt = shank_x[:, self.shank_z[1], :]

        # 0~3: 피크값
        f_pk_lt = fz_lt.abs().max(dim=1).values
        f_pk_rt = fz_rt.abs().max(dim=1).values
        s_pk_lt = sz_lt.abs().max(dim=1).values
        s_pk_rt = sz_rt.abs().max(dim=1).values

        # 4~5: Foot/Shank 충격비 (log1p: 분모 0 근접 시 값 폭발 방지 — 리뷰2)
        ratio_lt = torch.log1p(f_pk_lt / (s_pk_lt + 1e-4))
        ratio_rt = torch.log1p(f_pk_rt / (s_pk_rt + 1e-4))

        # 6~7: step-domain 상대 고주파 에너지
        hf_lt = self._hf_ratio(fz_lt)
        hf_rt = self._hf_ratio(fz_rt)

        # 8~9: 변동성
        std_lt = fz_lt.std(dim=1)
        std_rt = fz_rt.std(dim=1)

        # 10~11: 피크 후 감쇠율
        T_half   = fz_lt.shape[1] // 2
        decay_lt = (fz_lt[:, :T_half].abs().mean(dim=1) /
                    (fz_lt[:, T_half:].abs().mean(dim=1) + eps))
        decay_rt = (fz_rt[:, :T_half].abs().mean(dim=1) /
                    (fz_rt[:, T_half:].abs().mean(dim=1) + eps))

        # 12~13: Shank 진동
        vib_lt = (sz_lt[:, 1:] - sz_lt[:, :-1]).abs().mean(dim=1)
        vib_rt = (sz_rt[:, 1:] - sz_rt[:, :-1]).abs().mean(dim=1)

        # 14~15: Foot/Shank 분산비 (log1p: 폭발 방지 — 리뷰2)
        var_ratio_lt = torch.log1p(fz_lt.var(dim=1) / (sz_lt.var(dim=1) + 1e-4))
        var_ratio_rt = torch.log1p(fz_rt.var(dim=1) / (sz_rt.var(dim=1) + 1e-4))

        # 16~17: Spectral Centroid (Niswander 2021)
        sc_lt = self._spectral_centroid(fz_lt)
        sc_rt = self._spectral_centroid(fz_rt)

        # 18~19: Impact Duration (Niswander 2021)
        dur_lt = self._impact_duration(fz_lt)
        dur_rt = self._impact_duration(fz_rt)

        return torch.stack([
            f_pk_lt, f_pk_rt, s_pk_lt, s_pk_rt,
            ratio_lt, ratio_rt, hf_lt, hf_rt,
            std_lt, std_rt, decay_lt, decay_rt,
            vib_lt, vib_rt, var_ratio_lt, var_ratio_rt,
            sc_lt, sc_rt, dur_lt, dur_rt,
        ], dim=1)   # (B, 20)

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

class E2EHierModel(nn.Module):
    """End-to-end 6cls + 계층 보조 손실 단일 모델.

    3-Stage 하드 라우팅의 컴파운딩 오류 제거:
      S1 오류 → S2 못 봄, S2 오류 → S3 못 봄 문제 근본 해결.

    구조: backbone + BioMech + FFT → shared → 3 heads
      head_6cls: 주 출력 (직접 6cls 최적화)
      head_3cls: 보조 (flat/오르막/내리막, S1 역할)
      head_bin:  보조 (C4/C5 binary, S3 역할)

    Loss = CE_6cls + W3*CE_3cls + WB*CE_bin(C4C5 샘플만)
    """
    def __init__(self, backbone, feat_dim: int,
                 bio_dim: int = 128,
                 fft_dim: int = S3_FFT_DIM) -> None:
        super().__init__()
        self.backbone   = backbone
        self.bio_head   = BioMechHead(BioMechFeatures.N_BIO, bio_dim)
        self.fft_branch = FFTBranch(n_bins=129, out_dim=fft_dim)
        total = feat_dim + bio_dim + fft_dim
        self.shared = nn.Sequential(
            nn.Linear(total, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_6cls = nn.Linear(256, 6)   # 주: 6cls
        self.head_3cls = nn.Linear(256, 3)   # 보조: flat/up/down
        self.head_bin  = nn.Linear(256, 2)   # 보조: C4/C5 binary

    def _fuse(self, bi: dict, bio_f: torch.Tensor) -> torch.Tensor:
        cnn = self.backbone.extract(bi)
        bio = self.bio_head(bio_f)
        fft = self.fft_branch(bi["Foot"].float())
        return self.shared(torch.cat([cnn, bio, fft], dim=1))

    def forward(self, bi: dict,
                bio_f: torch.Tensor):
        """Multi-task forward — 학습용. (l6, l3, lbin) 반환."""
        h = self._fuse(bi, bio_f)
        return self.head_6cls(h), self.head_3cls(h), self.head_bin(h)

    def predict(self, bi: dict,
                bio_f: torch.Tensor) -> torch.Tensor:
        """Inference용: 6cls logit만 반환."""
        return self.head_6cls(self._fuse(bi, bio_f))


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
    return bi, bio, y


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
    """전체 6cls 데이터셋 + BioMech 피처. E2E 학습용.

    FlatBranchDataset과 동일 구조이나 클래스 필터링 없이
    전체 샘플에 BioMech 피처를 on-the-fly로 계산.
    """
    def __init__(self, branch_ds, bio_extractor,
                 y_all: np.ndarray) -> None:
        self.ds  = branch_ds
        self.bio = bio_extractor
        self.y   = y_all

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        bi, _   = self.ds[i]
        foot_t  = bi["Foot"].unsqueeze(0).float()
        shank_t = bi["Shank"].unsqueeze(0).float()
        with torch.no_grad():
            bio_f = self.bio(foot_t, shank_t).squeeze(0)
        return bi, bio_f, int(self.y[i])


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
        for bi, bio_f, yb in loader:
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

def train_e2e(backbone, tr_dl, val_dl, te_dl, tag: str = "",
              curve_dir: Path | None = None):
    """End-to-end 6cls 학습 + 계층 보조 손실.

    3-Stage 컴파운딩 라우팅 오류 제거.
    S1 backbone 전이 → multi-task fine-tuning:
      Phase1: 6cls + 3cls보조 + binary보조 (E2E_EPOCHS ep)
      Phase2: 6cls 집중 파인튜닝 (E2E_FT_EPOCHS ep)

    Returns:
        (preds, labels) — 6cls 예측 및 정답
    """
    feat_dim = _get_feat_dim(backbone)
    model    = E2EHierModel(backbone, feat_dim).to(DEVICE)
    params   = list(model.parameters())

    opt    = torch.optim.AdamW(params, lr=E2E_LR,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, E2E_EPOCHS, warmup=15, base_lr=E2E_LR)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))
    curve  = CurveTracker(f"E2E_{tag.replace('[','').replace(']','')}")

    crit_6 = nn.CrossEntropyLoss(label_smoothing=0.05)
    crit_3 = nn.CrossEntropyLoss(label_smoothing=0.05)
    crit_b = nn.CrossEntropyLoss()

    best_va    = 0.0
    best_state = None
    patience   = 0
    t0         = time.time()

    log(f"  {tag} E2E Phase1 ({E2E_EPOCHS}ep, LR={E2E_LR:.0e})"
        f"  aux: 3cls×{E2E_AUX_W3}  bin×{E2E_AUX_WB}")

    for ep in range(1, E2E_EPOCHS + 1):
        model.train()
        step_i = -1
        opt.zero_grad(set_to_none=True)
        for step_i, (bi, bio_f, yb) in enumerate(tr_dl):
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            yb3       = _y6_to_y3(yb)
            c4c5_mask = (yb == 3) | (yb == 4)   # C4/C5 in 0-indexed

            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                l6, l3, lbin = model(bi, bio_f)
                loss_6 = crit_6(l6, yb)
                loss_3 = crit_3(l3, yb3)
                if c4c5_mask.sum() > 1:
                    yb_bin = (yb[c4c5_mask] == 4).long()   # C5→1, C4→0
                    loss_b = crit_b(lbin[c4c5_mask], yb_bin)
                    loss   = (loss_6 + E2E_AUX_W3 * loss_3 +
                              E2E_AUX_WB * loss_b) / config.GRAD_ACCUM_STEPS
                else:
                    loss = (loss_6 + E2E_AUX_W3 * loss_3) / config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

        # 마지막 배치 grad 누락 방지
        if step_i >= 0 and (step_i + 1) % config.GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        sch.step()

        # val
        model.eval()
        va_c = va_n = 0
        with torch.inference_mode():
            for bi, bio_f, yb in val_dl:
                bi, bio_f, yb = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model.predict(bi, bio_f)
                va_c += (logits.argmax(1) == yb).sum().item()
                va_n += len(yb)
        va = va_c / max(va_n, 1)
        curve.record(acc=va)

        if va > best_va:
            best_va    = va
            best_state = _clone_state(model)
            patience   = 0
        else:
            patience += 1
            if patience >= E2E_PATIENCE and ep > 30:
                log(f"  {tag} E2E EarlyStop ep{ep}"); break

        if ep % 30 == 0:
            log(f"  {tag} E2E ep{ep:03d}/{E2E_EPOCHS}"
                f"  val={va:.4f}  best={best_va:.4f}"
                f"  ({time.time()-t0:.0f}s)")

    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    log(f"  {tag} E2E Phase1 완료  best_val={best_va:.4f}")
    if curve_dir: curve.save(curve_dir)

    # Phase 2: 6cls CE 집중 파인튜닝 (aux heads 동결)
    for p in model.head_3cls.parameters(): p.requires_grad = False
    for p in model.head_bin.parameters():  p.requires_grad = False
    params2  = list(filter(lambda p: p.requires_grad, model.parameters()))
    opt2     = torch.optim.AdamW(params2, lr=E2E_LR * 0.2,
                                  weight_decay=config.WEIGHT_DECAY)
    crit_ft  = nn.CrossEntropyLoss(label_smoothing=0.03)
    best_va2    = 0.0
    best_state2 = None
    patience2   = 0

    log(f"  {tag} E2E Phase2 6cls 파인튜닝 ({E2E_FT_EPOCHS}ep)")
    for ep in range(1, E2E_FT_EPOCHS + 1):
        model.train()
        for bi, bio_f, yb in tr_dl:
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            opt2.zero_grad(set_to_none=True)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                loss = crit_ft(model.predict(bi, bio_f), yb)
            scaler.scale(loss).backward()
            scaler.step(opt2); scaler.update()

        model.eval()
        va_c = va_n = 0
        with torch.inference_mode():
            for bi, bio_f, yb in val_dl:
                bi, bio_f, yb = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model.predict(bi, bio_f)
                va_c += (logits.argmax(1) == yb).sum().item()
                va_n += len(yb)
        va2 = va_c / max(va_n, 1)
        if va2 > best_va2:
            best_va2    = va2
            best_state2 = _clone_state(model)
            patience2   = 0
        else:
            patience2 += 1
            if patience2 >= 10:
                break

    if best_state2:
        model.load_state_dict(best_state2); model.to(DEVICE)
    for p in model.parameters(): p.requires_grad = True
    log(f"  {tag} E2E Phase2 완료  best_val={best_va2:.4f}")

    # test 평가
    model.eval()
    preds_list, labels_list = [], []
    with torch.inference_mode():
        for bi, bio_f, yb in te_dl:
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model.predict(bi, bio_f)
            preds_list.append(logits.argmax(1).cpu())
            labels_list.append(yb.cpu())

    return (torch.cat(preds_list).numpy(),
            torch.cat(labels_list).numpy())


# ═══════════════════════════════════════════════
# 10. Stage 2 — Step1: CE Warmup
# ═══════════════════════════════════════════════

def warmup_stage2_ce(model: Stage2Model, tr_dl, val_dl,
                     tag: str = "", weights: list | None = None,
                     curve_dir: Path | None = None):
    """Step1: CE 워밍업. 전체 파라미터 학습."""
    w      = weights or [1.0, 1.0, 1.0, 1.0]
    weight = torch.tensor(w, dtype=torch.float32).to(DEVICE)
    crit   = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    params = list(model.parameters())
    opt    = torch.optim.AdamW(params, lr=S2_LR,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, S2_WARMUP_EP, warmup=5, base_lr=S2_LR)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))
    curve  = CurveTracker(f"S2W_{tag.replace('[','').replace(']','')}")

    best_va    = 0.0
    best_state = None
    patience   = 0
    t0         = time.time()

    log(f"  {tag} S2 Step1 CE Warmup ({S2_WARMUP_EP}ep)  [val=inner split]")
    for ep in range(1, S2_WARMUP_EP + 1):
        _run_epoch(model, tr_dl, crit, opt, scaler, params)
        sch.step()
        _, va = _eval_flat_dl(model, val_dl, crit)

        if va > best_va:
            best_va    = va
            best_state = _clone_state(model)
            patience   = 0
        else:
            patience += 1
            if patience >= 20:
                log(f"  {tag} S2 Warmup EarlyStop ep{ep}"); break

        curve.record(acc=va)
        if ep % 10 == 0:
            log(f"  {tag} S2W ep{ep:03d}/{S2_WARMUP_EP}"
                f"  val={va:.4f}  best={best_va:.4f}"
                f"  ({time.time()-t0:.0f}s)")

    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    log(f"  {tag} S2 Warmup 완료  best_val={best_va:.4f}")
    if curve_dir: curve.save(curve_dir)


# ═══════════════════════════════════════════════
# 11. Stage 2 — Step2: SupCon Pretrain
# ═══════════════════════════════════════════════

def pretrain_stage2(model: Stage2Model, tr_dl, tag: str = ""):
    """Step2: SupCon (balanced sampler 사용).

    proj_head + backbone + bio_head 학습.
    classifier는 학습 제외 (표현 학습에 집중).
    """
    supcon = SupConLoss(temperature=TEMPERATURE)
    params = (list(model.backbone.parameters()) +
              list(model.bio_head.parameters()) +
              list(model.proj_head.parameters()))
    opt    = torch.optim.AdamW(params, lr=S2_LR * 0.5,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, S2_PRETRAIN_EP, warmup=5, base_lr=S2_LR * 0.5)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))

    best_loss = float("inf")
    t0        = time.time()

    log(f"  {tag} S2 Step2 SupCon ({S2_PRETRAIN_EP}ep, T={TEMPERATURE})"
        f"  [balanced sampler]")
    for ep in range(1, S2_PRETRAIN_EP + 1):
        avg = _run_epoch(
            model, tr_dl,
            loss_fn    = lambda proj, yb: supcon(proj, yb),
            opt        = opt,
            scaler     = scaler,
            params     = params,
            forward_fn = model.forward_proj,
        )
        sch.step()
        if avg < best_loss: best_loss = avg
        if ep % 30 == 0:
            log(f"  {tag} S2P2 ep{ep:03d}/{S2_PRETRAIN_EP}"
                f"  loss={avg:.4f}  best={best_loss:.4f}"
                f"  lr={opt.param_groups[0]['lr']:.1e}"
                f"  ({time.time()-t0:.0f}s)")
    log(f"  {tag} S2 SupCon 완료  best_loss={best_loss:.4f}")


# ═══════════════════════════════════════════════
# 12. Stage 2 — Step3: Focal Loss Finetune
# ═══════════════════════════════════════════════

def focal_finetune_stage2(model: Stage2Model, tr_dl, val_dl,
                          tag: str = "", weights: list | None = None):
    """Step3: Focal Loss Finetune.

    SupCon으로 feature space 분리 후,
    FocalLoss로 C4/C5 어려운 샘플 집중 학습.
    backbone fc 레이어만 부분 해제 → 세밀한 representation tuning.
    """
    w      = weights or [1.0, 1.0, 1.0, 1.0]
    weight = torch.tensor(w, dtype=torch.float32).to(DEVICE)
    crit   = FocalLoss(gamma=FOCAL_GAMMA, weight=weight)
    log(f"    Focal Loss(gamma={FOCAL_GAMMA}, weights=[{', '.join(f'{x:.3f}' for x in w)}])")

    # [리뷰2] backbone 동결 제거: CE Warmup 표현을 Focal로 전체 파인튜닝
    for p in model.parameters(): p.requires_grad = True
    for p in model.proj_head.parameters(): p.requires_grad = False  # proj_head만 제외

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    opt    = torch.optim.AdamW(params, lr=S2_FOCAL_LR,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, S2_FOCAL_EP, warmup=5, base_lr=S2_FOCAL_LR)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))

    val_crit   = nn.CrossEntropyLoss(weight=weight)
    best_vl    = float("inf")
    best_state = None
    patience   = 0
    t0         = time.time()

    log(f"  {tag} S2 Step3 FocalFinetune ({S2_FOCAL_EP}ep)  [val=inner split]")
    for ep in range(1, S2_FOCAL_EP + 1):
        _run_epoch(model, tr_dl, crit, opt, scaler, params)
        sch.step()
        vl, va = _eval_flat_dl(model, val_dl, val_crit)

        if ep % 25 == 0:
            log(f"  {tag} S2P3F ep{ep:03d}/{S2_FOCAL_EP}"
                f"  val={vl:.4f}/{va:.4f}"
                f"  lr={opt.param_groups[0]['lr']:.1e}"
                f"  ({time.time()-t0:.0f}s)")

        if vl < best_vl:
            best_vl    = vl
            best_state = _clone_state(model)
            patience   = 0
        else:
            patience += 1
            if patience >= S2_FOCAL_PAT:
                log(f"  {tag} S2 FocalFinetune EarlyStop ep{ep}"); break

    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    # backbone 다시 활성화
    for p in model.backbone.parameters(): p.requires_grad = True
    log(f"  {tag} S2 FocalFinetune 완료  best_val_loss={best_vl:.4f}")


# ═══════════════════════════════════════════════
# 13. Stage 2 — Step4: CE Finetune
# ═══════════════════════════════════════════════

def finetune_stage2(model: Stage2Model, tr_dl, val_dl, te_dl,
                    tag: str = "", weights: list | None = None,
                    curve_dir: Path | None = None):
    """Step4: CE 마무리.

    backbone 완전 고정 대신 상위 fc/head 레이어만 partial unfreeze.
    Focal로 전체 파인튜닝 후 최종 조정 → 과도한 frozen이 성능 손실.
    """
    # backbone 전체 동결
    for p in model.backbone.parameters():   p.requires_grad = False
    for p in model.proj_head.parameters():  p.requires_grad = False
    # backbone 상위 fc/head 레이어만 선택적 해제 (표현 미세 조정)
    for name, p in model.backbone.named_parameters():
        if any(k in name for k in ("fc", "clf", "out", "head", "proj", "linear")):
            p.requires_grad = True

    w      = weights or [1.0, 1.0, 1.0, 1.0]
    weight = torch.tensor(w, dtype=torch.float32).to(DEVICE)
    crit   = nn.CrossEntropyLoss(weight=weight,
                                  label_smoothing=config.LABEL_SMOOTH)
    log(f"    CE 마무리(auto_weights=[{', '.join(f'{x:.3f}' for x in w)}]) + backbone fc partial unfreeze")

    params = (list(model.classifier.parameters()) +
              list(model.bio_head.parameters()) +
              [p for p in model.backbone.parameters() if p.requires_grad])
    opt    = torch.optim.AdamW(params, lr=S2_LR * 0.5,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, S2_FINETUNE_EP, warmup=5, base_lr=S2_LR * 0.5)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))
    curve  = CurveTracker(f"S2FT_{tag.replace('[','').replace(']','')}")

    best_vl    = float("inf")
    best_state = None
    patience   = 0
    t0         = time.time()

    log(f"  {tag} S2 Step4 CE-마무리 ({S2_FINETUNE_EP}ep)  [val=inner split]")
    for ep in range(1, S2_FINETUNE_EP + 1):
        _run_epoch(model, tr_dl, crit, opt, scaler, params)
        sch.step()
        vl, va = _eval_flat_dl(model, val_dl, crit)

        if ep % 15 == 0:
            log(f"  {tag} S2P4 ep{ep:03d}/{S2_FINETUNE_EP}"
                f"  val={vl:.4f}/{va:.4f}"
                f"  lr={opt.param_groups[0]['lr']:.1e}"
                f"  ({time.time()-t0:.0f}s)")

        if vl < best_vl:
            best_vl    = vl
            best_state = _clone_state(model)
            patience   = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP:
                log(f"  {tag} S2 Finetune EarlyStop ep{ep}"); break

        curve.record(loss=vl, acc=va)

    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    for p in model.backbone.parameters(): p.requires_grad = True
    if curve_dir: curve.save(curve_dir)

    # 최종 test 평가 + score fusion용 probs 반환
    model.eval()
    preds_list, labels_list, probs_list = [], [], []
    with torch.inference_mode():
        for bi, bio_f, yb in te_dl:
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model(bi, bio_f)
            probs_list.append(torch.softmax(logits.float(), dim=1).cpu())
            preds_list.append(logits.argmax(1).cpu())
            labels_list.append(yb.cpu())

    s2_probs = torch.cat(probs_list).numpy() if probs_list else np.zeros((0, 3))
    return (torch.cat(preds_list).numpy(),
            torch.cat(labels_list).numpy(),
            s2_probs)


# ═══════════════════════════════════════════════
# 13. Stage 3 — C4 vs C5 Binary (ArcFace)
# ═══════════════════════════════════════════════

def mixup_batch(bi: dict, bio_f: torch.Tensor, yb: torch.Tensor,
                alpha: float = 0.3):
    """C4/C5 Mixup augmentation — Zhang et al. (2018).

    λ ~ Beta(α, α) 로 두 샘플 선형 보간.
    decision boundary 주변 샘플 생성 → ArcFace margin 강화.
    """
    lam = float(np.random.beta(alpha, alpha))
    B   = yb.shape[0]
    idx = torch.randperm(B, device=yb.device)
    mixed_bi  = {k: lam * v + (1 - lam) * v[idx] for k, v in bi.items()}
    mixed_bio = lam * bio_f + (1 - lam) * bio_f[idx]
    return mixed_bi, mixed_bio, yb, yb[idx], lam


def train_stage3(backbone, tr_dl, val_dl, te_dl,
                 tag: str = "",
                 curve_dir: Path | None = None):
    """Stage3: C4(흙길) vs C5(잔디) binary ArcFace 학습.

    전략 A: FFT Branch (Zheng et al. 2021) — 주파수 도메인 패턴
    전략 B: Mixup (Zhang et al. 2018) — decision boundary 강화
    ArcFace (Deng et al. CVPR 2019) — angular margin

    학습 2단계:
      Step1: ArcFace + Mixup (S3_EPOCHS ep)
      Step2: CE 마무리 (20ep, backbone 고정)
    """
    feat_dim = _get_feat_dim(backbone)
    model    = Stage3Model(backbone, feat_dim,
                           bio_dim=128, embed_dim=128,
                           fft_dim=S3_FFT_DIM).to(DEVICE)
    arcface  = ArcFaceLoss(feat_dim=128, num_classes=2,
                           s=S3_ARCFACE_S, m=S3_ARCFACE_M).to(DEVICE)

    params = list(model.parameters()) + list(arcface.parameters())
    opt    = torch.optim.AdamW(params, lr=S3_LR,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, S3_EPOCHS, warmup=10, base_lr=S3_LR)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))
    curve  = CurveTracker(f"S3_{tag.replace('[','').replace(']','')}")

    best_va    = 0.0
    best_state = None
    patience   = 0
    t0         = time.time()

    log(f"  {tag} Stage3 C4vsC5 ArcFace+FFT+Mixup"
        f" ({S3_EPOCHS}ep, s={S3_ARCFACE_S}, m={S3_ARCFACE_M},"
        f" α={S3_MIXUP_ALPHA})")

    # Step1: ArcFace (Mixup 제거 — 리뷰2: ArcFace와 수학적 충돌)
    for ep in range(1, S3_EPOCHS + 1):
        model.train(); arcface.train()
        for bi, bio_f, yb in tr_dl:
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                embed = model.forward_embed(bi, bio_f)
                loss  = arcface(embed, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
        sch.step()

        # val — balanced accuracy (C4/C5 편향 방지)
        model.eval()
        va_preds, va_labels = [], []
        with torch.inference_mode():
            for bi, bio_f, yb in val_dl:
                bi, bio_f, yb = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model(bi, bio_f)
                va_preds.append(logits.argmax(1).cpu())
                va_labels.append(yb.cpu())
        vp = torch.cat(va_preds).numpy()
        vl = torch.cat(va_labels).numpy()
        # balanced: C4 recall + C5 recall 평균 → C4 편향 early stop 방지
        c4_rec = ((vp == 0) & (vl == 0)).sum() / max((vl == 0).sum(), 1)
        c5_rec = ((vp == 1) & (vl == 1)).sum() / max((vl == 1).sum(), 1)
        va = (c4_rec + c5_rec) / 2.0
        curve.record(acc=va)

        if va > best_va:
            best_va    = va
            best_state = _clone_state(model)
            patience   = 0
        else:
            patience += 1
            if patience >= S3_PATIENCE and ep > 20:
                log(f"  {tag} S3 EarlyStop ep{ep}"); break

        if ep % 30 == 0:
            log(f"  {tag} S3 ep{ep:03d}/{S3_EPOCHS}"
                f"  bal_acc={va:.4f}  best={best_va:.4f}"
                f"  C4={c4_rec:.3f}  C5={c5_rec:.3f}"
                f"  ({time.time()-t0:.0f}s)")

    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    log(f"  {tag} S3 ArcFace 완료  best_val={best_va:.4f}")

    # Step2: CE 마무리 2-phase
    # Phase A: backbone 완전 동결 — head/embed/fft만 먼저 정착 (10ep)
    # Phase B: backbone fc 레이어만 선택 해제 — 세밀 조정 (10ep)
    # 총 S3_CE_FT_EP=20ep (40→20 축소: ArcFace margin 구조 보존)
    for p in model.backbone.parameters(): p.requires_grad = False
    head_params = (list(model.embed.parameters()) +
                   list(model.bio_head.parameters()) +
                   list(model.fft_branch.parameters()) +
                   list(model.classifier.parameters()))
    opt2   = torch.optim.AdamW(head_params, lr=S3_LR * 0.3,
                                weight_decay=config.WEIGHT_DECAY)
    # C5 upweight: C4_recall 97% vs C5_recall 30% 불균형 → C5 집중
    s3_crit_weight = torch.tensor([0.8, S3_C5_WEIGHT], dtype=torch.float32, device=DEVICE)
    crit2  = FocalLoss(gamma=1.0, weight=s3_crit_weight)
    best_va2    = 0.0
    best_state2 = None
    patience2   = 0
    phase_a_ep  = S3_CE_FT_EP // 2   # 10ep

    log(f"  {tag} S3 CE 마무리 Phase A ({phase_a_ep}ep, head only)")
    for ep in range(1, phase_a_ep + 1):
        model.train()
        for bi, bio_f, yb in tr_dl:
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            opt2.zero_grad(set_to_none=True)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                loss = crit2(model(bi, bio_f), yb)
            scaler.scale(loss).backward()
            scaler.step(opt2); scaler.update()

        model.eval()
        va_c = va_n = 0
        with torch.inference_mode():
            for bi, bio_f, yb in val_dl:
                bi, bio_f, yb = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model(bi, bio_f)
                va_c += (logits.argmax(1) == yb).sum().item()
                va_n += len(yb)
        va2 = va_c / max(va_n, 1)
        if va2 > best_va2:
            best_va2    = va2
            best_state2 = _clone_state(model)
            patience2   = 0
        else:
            patience2 += 1
            if patience2 >= 5: break

    # Phase B: backbone fc 레이어 선택 해제
    for name, p in model.backbone.named_parameters():
        if any(k in name for k in ("fc", "clf", "out", "head", "linear")):
            p.requires_grad = True
    all_ft_params = head_params + [p for p in model.backbone.parameters()
                                   if p.requires_grad]
    opt3 = torch.optim.AdamW(all_ft_params, lr=S3_LR * 0.1,
                              weight_decay=config.WEIGHT_DECAY)
    phase_b_ep = S3_CE_FT_EP - phase_a_ep   # 10ep
    patience3  = 0

    log(f"  {tag} S3 CE 마무리 Phase B ({phase_b_ep}ep, +backbone fc unfreeze)")
    for ep in range(1, phase_b_ep + 1):
        model.train()
        for bi, bio_f, yb in tr_dl:
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            opt3.zero_grad(set_to_none=True)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                loss = crit2(model(bi, bio_f), yb)
            scaler.scale(loss).backward()
            scaler.step(opt3); scaler.update()

        model.eval()
        va_c = va_n = 0
        with torch.inference_mode():
            for bi, bio_f, yb in val_dl:
                bi, bio_f, yb = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model(bi, bio_f)
                va_c += (logits.argmax(1) == yb).sum().item()
                va_n += len(yb)
        va3 = va_c / max(va_n, 1)
        if va3 > best_va2:
            best_va2    = va3
            best_state2 = _clone_state(model)
            patience3   = 0
        else:
            patience3 += 1
            if patience3 >= 5: break

    if best_state2:
        model.load_state_dict(best_state2); model.to(DEVICE)
    for p in model.backbone.parameters(): p.requires_grad = True
    log(f"  {tag} S3 완료  best_val={best_va2:.4f}")
    if curve_dir: curve.save(curve_dir)

    # 최종 test 평가 + score fusion용 probs 반환
    model.eval()
    preds_list, labels_list, probs_list = [], [], []
    with torch.inference_mode():
        for bi, bio_f, yb in te_dl:
            bi, bio_f, yb = _to_device(bi, bio_f, yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model(bi, bio_f)
            probs_list.append(torch.softmax(logits.float(), dim=1).cpu())
            preds_list.append(logits.argmax(1).cpu())
            labels_list.append(yb.cpu())

    s3_probs = torch.cat(probs_list).numpy() if probs_list else np.zeros((0, 2))
    return (torch.cat(preds_list).numpy(),
            torch.cat(labels_list).numpy(),
            s3_probs,
            model)


def main() -> None:
    args = parse_args()
    apply_args(args)

    config.print_config()
    log(
        f"  ★ Hierarchical v9.0  K-Fold (3-Stage + ArcFace + C5Fix)\n"
        f"  Stage1: 3cls CE {S1_EPOCHS}ep (LR={S1_LR:.0e})"
        f"  Soft-Routing threshold={S1_SOFT_THRESHOLD}\n"
        f"  Stage2: CE-Warmup({S2_WARMUP_EP}ep)"
        f" → Focal({S2_FOCAL_EP}ep, γ={FOCAL_GAMMA}, LR={S2_FOCAL_LR:.0e})"
        f" → CE({S2_FINETUNE_EP}ep, backbone fc partial unfreeze)\n"
        f"  Stage3: binary [C4 vs C5] ArcFace+FFT"
        f" (s={S3_ARCFACE_S}, m={S3_ARCFACE_M}, {S3_EPOCHS}ep)"
        f"  CE마무리 {S3_CE_FT_EP}ep Focal(C5w={S3_C5_WEIGHT})\n"
        f"  결합: Hard Routing  Subject-aware majority vote(window={args.vote_window})\n"
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

        # ── Stage 1 ───────────────────────────────
        tag         = f"[F{fi}][Hier]"
        backbone_s1 = M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        s1_preds, s1_labels, s1_probs, s1_model = train_stage1(
            backbone_s1, tr_dl, val_dl, te_dl, tag, curve_dir=curve_dir)
        s1_acc = accuracy_score(s1_labels, s1_preds)
        log(f"  {tag} Stage1 Acc={s1_acc:.4f}")

        # ── Stage 2/3 데이터 준비 ─────────────────
        te_y6 = y[te_idx]
        tr_y6 = y[inner_tr_idx]
        va_y6 = y[inner_va_idx]

        def _make_flat_ds_3cls(base_ds, y6, mask):
            y_3cls = np.array([S2_3CLS_MAP[c] for c in y6[mask]])
            return FlatBranchDataset(base_ds, bio_extractor, mask, y_3cls), y_3cls

        def _make_binary_ds(base_ds, y6, mask):
            y_bin = np.array([S3_BINARY_MAP[c] for c in y6[mask]])
            return FlatBranchDataset(base_ds, bio_extractor, mask, y_bin), y_bin

        tr_flat_mask   = np.isin(tr_y6, FLAT_CLASSES)
        va_flat_mask   = np.isin(va_y6, FLAT_CLASSES)
        te_flat_mask   = (s1_preds == 0)
        te_oracle_mask = np.isin(te_y6, FLAT_CLASSES)

        tr_flat_ds, tr_y_3cls     = _make_flat_ds_3cls(tr_ds, tr_y6, tr_flat_mask)
        va_flat_ds, _             = _make_flat_ds_3cls(val_ds, va_y6, va_flat_mask)
        te_oracle_ds, te_oracle_y3 = _make_flat_ds_3cls(te_ds, te_y6, te_oracle_mask)

        te_y6_flat = te_y6[te_flat_mask]
        te_y_3cls  = np.array([S2_3CLS_MAP.get(c, 2) for c in te_y6_flat])
        te_flat_ds = FlatBranchDataset(te_ds, bio_extractor, te_flat_mask, te_y_3cls)

        tr_flat_dl_bal = make_flat_loader(tr_flat_ds, True,  balanced=True)
        tr_flat_dl_ce  = make_flat_loader(tr_flat_ds, True,  balanced=False)
        va_flat_dl     = make_flat_loader(va_flat_ds, False)
        te_flat_dl     = make_flat_loader(te_flat_ds, False)
        te_oracle_dl   = make_flat_loader(te_oracle_ds, False)

        log(f"  {tag} Stage2 (3cls: C1/C6/C4C5)"
            f"  tr_flat={tr_flat_mask.sum()}"
            f"  va_flat={va_flat_mask.sum()}"
            f"  te_pipeline={te_flat_mask.sum()}"
            f"  te_oracle={te_oracle_mask.sum()}")

        # ── Stage2 모델 (S1 backbone 전이) ────────
        # S1 backbone 전이: flat 특성 이미 학습 → warmup 수렴 가속
        backbone_s2 = M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        backbone_s2.load_state_dict(s1_model.backbone.state_dict())
        feat_dim = _get_feat_dim(backbone_s2)
        s2_model = Stage2Model(backbone_s2, feat_dim,
                               bio_dim=128, num_classes=3).to(DEVICE)

        s2_weights = auto_class_weights(tr_y_3cls)

        # Stage2: CE Warmup(80ep) → Focal(100ep, full params) → CE(50ep)
        warmup_stage2_ce(     s2_model, tr_flat_dl_ce, va_flat_dl, tag, s2_weights,
                              curve_dir=curve_dir)
        focal_finetune_stage2(s2_model, tr_flat_dl_ce, va_flat_dl, tag, s2_weights)
        s2_preds, s2_labels, s2_te_probs = finetune_stage2(
            s2_model, tr_flat_dl_ce, va_flat_dl, te_flat_dl, tag, s2_weights,
            curve_dir=curve_dir)

        # Stage2 Oracle 평가
        s2_pipeline_acc = accuracy_score(s2_labels, s2_preds)
        s2_model.eval()
        oracle_preds_list = []
        with torch.inference_mode():
            for bi, bio_f, yb in te_oracle_dl:
                bi, bio_f, _ = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    oracle_preds_list.append(
                        s2_model(bi, bio_f).argmax(1).cpu())
        oracle_preds  = torch.cat(oracle_preds_list).numpy()
        s2_oracle_acc = accuracy_score(te_oracle_y3, oracle_preds)
        log(f"  {tag} Stage2  oracle={s2_oracle_acc:.4f}"
            f"  pipeline={s2_pipeline_acc:.4f}")

        # ── Stage3 데이터 준비 ────────────────────
        tr_c4c5_mask          = np.isin(tr_y6, [3, 4])
        va_c4c5_mask          = np.isin(va_y6, [3, 4])
        te_c4c5_pipeline_mask = te_flat_mask.copy()
        te_c4c5_pipeline_mask[te_flat_mask] = (s2_preds == 2)
        te_c4c5_oracle_mask   = np.isin(te_y6, [3, 4])

        tr_c4c5_ds, tr_y_bin          = _make_binary_ds(tr_ds, tr_y6, tr_c4c5_mask)
        va_c4c5_ds, _                 = _make_binary_ds(val_ds, va_y6, va_c4c5_mask)
        te_c4c5_oracle_ds, te_c4c5_oracle_y = _make_binary_ds(
            te_ds, te_y6, te_c4c5_oracle_mask)

        te_y6_c4c5 = te_y6[te_c4c5_pipeline_mask]
        te_y_bin   = np.array([S3_BINARY_MAP.get(c, 0) for c in te_y6_c4c5])
        te_c4c5_ds = FlatBranchDataset(
            te_ds, bio_extractor, te_c4c5_pipeline_mask, te_y_bin)

        tr_c4c5_dl        = make_flat_loader(tr_c4c5_ds, True, balanced=True)
        va_c4c5_dl        = make_flat_loader(va_c4c5_ds, False)
        te_c4c5_dl        = make_flat_loader(te_c4c5_ds, False)
        te_c4c5_oracle_dl = make_flat_loader(te_c4c5_oracle_ds, False)

        log(f"  {tag} Stage3 (C4vsC5 binary)"
            f"  tr={tr_c4c5_mask.sum()}"
            f"  va={va_c4c5_mask.sum()}"
            f"  te_pipeline={te_c4c5_pipeline_mask.sum()}"
            f"  te_oracle={te_c4c5_oracle_mask.sum()}")

        # ── Stage3 (S2 backbone 전이 + ArcFace) ───
        backbone_s3 = M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        backbone_s3.load_state_dict(s2_model.backbone.state_dict())
        s3_preds, _, s3_te_probs, s3_model = train_stage3(
            backbone_s3, tr_c4c5_dl, va_c4c5_dl, te_c4c5_dl, tag,
            curve_dir=curve_dir)

        # Stage3 Oracle 평가 + C4/C5 recall 분리
        s3_model.eval()
        s3_oracle_preds_list = []
        with torch.inference_mode():
            for bi, bio_f, yb in te_c4c5_oracle_dl:
                bi, bio_f, _ = _to_device(bi, bio_f, yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    s3_oracle_preds_list.append(
                        s3_model(bi, bio_f).argmax(1).cpu())
        s3_oracle_preds = torch.cat(s3_oracle_preds_list).numpy() \
                          if s3_oracle_preds_list else np.array([])
        s3_oracle_acc   = accuracy_score(te_c4c5_oracle_y, s3_oracle_preds) \
                          if len(s3_oracle_preds) > 0 else 0.0
        s3_pipeline_acc = accuracy_score(te_y_bin, s3_preds) \
                          if len(s3_preds) > 0 else 0.0

        # C4/C5 recall 분리 — binary에서 한 클래스만 잘 맞는 착시 방지
        if len(s3_oracle_preds) > 0:
            cm_s3  = confusion_matrix(te_c4c5_oracle_y, s3_oracle_preds, labels=[0, 1])
            rec_c4 = cm_s3[0, 0] / max(cm_s3[0].sum(), 1)
            rec_c5 = cm_s3[1, 1] / max(cm_s3[1].sum(), 1)
        else:
            rec_c4 = rec_c5 = 0.0
        log(f"  {tag} Stage3  oracle={s3_oracle_acc:.4f}"
            f"  pipeline={s3_pipeline_acc:.4f}"
            f"  C4_recall={rec_c4:.3f}  C5_recall={rec_c5:.3f}"
            f"  (C4={(te_y_bin==0).sum()}  C5={(te_y_bin==1).sum()})")

        # ── 최종 결합: Hard Routing (검증된 방식) ──
        # Score Fusion 실험 결과: p_flat×p_s2 < p_up 조건에서 역전 발생
        # → 확률 미보정 상태에서 hard routing이 더 안정적
        final_preds = combine_predictions_3stage(s1_preds, s2_preds, s3_preds)
        final_preds_raw = final_preds.copy()

        # Subject-aware majority vote (fold 전체 아닌 subject 단위)
        if args.vote_window > 0:
            final_preds = majority_vote_by_subject(
                final_preds, te_idx=te_idx, groups=groups,
                window=args.vote_window)

        acc_raw = accuracy_score(te_y6, final_preds_raw)
        acc     = accuracy_score(te_y6, final_preds)
        f1      = f1_score(te_y6, final_preds, average="macro", zero_division=0)
        vote_info = f"  (before vote={acc_raw:.4f})" if args.vote_window > 0 else ""
        log(f"  {tag} ★ 최종 6cls  Acc={acc:.4f}  F1={f1:.4f}{vote_info}")

        all_preds.append(final_preds)
        all_labels.append(te_y6)
        fold_meta.append({
            "fold":             fi,
            "test_subjects":    te_s,
            "s1_acc":           round(s1_acc, 4),
            "s2_oracle_acc":    round(s2_oracle_acc, 4),
            "s2_pipeline_acc":  round(s2_pipeline_acc, 4),
            "s3_oracle_acc":    round(s3_oracle_acc, 4),
            "s3_pipeline_acc":  round(s3_pipeline_acc, 4),
            "final_acc_raw":    round(acc_raw, 4),
            "final_acc":        round(acc, 4),
            "final_f1":         round(f1, 4),
            "fold_time_min":    round((time.time() - t_fold) / 60, 1),
        })

        del backbone_s1, backbone_s2, backbone_s3, s2_model, s3_model, s1_model
        del tr_ds, val_ds, te_ds
        del tr_flat_ds, va_flat_ds, te_flat_ds, te_oracle_ds
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
    print(f"  ★ Hierarchical v8.9  {config.KFOLD}-Fold")
    print(f"  총 소요: {total_min:.1f}분")
    print(f"  Acc={acc_all:.4f}  MacroF1={f1_all:.4f}")
    print(f"{'='*60}")
    print(f"  ── 클래스별 Recall ──")
    for i, r in enumerate(recalls):
        print(f"    {CLASS_NAMES_ALL.get(i, f'C{i+1}'):<14} {r*100:.1f}%")

    rep = classification_report(
        labels_all, preds_all,
        target_names=[f"C{c}" for c in le.classes_], digits=4, zero_division=0)
    (out / "report_v89.txt").write_text(
        f"Hierarchical v8.9\nAcc={acc_all:.4f}  F1={f1_all:.4f}\n\n{rep}")

    le_out = LabelEncoder()
    le_out.fit(sorted(set(labels_all.tolist())))
    save_cm(preds_all, labels_all, le_out, "Hierarchical_v89_KFold", out)

    summary = {
        "experiment":  "hierarchical_kfold_v89",
        "version":     "v8.9",
        "method":      "Hierarchical Soft-Routing + CE Warmup + FocalLoss + ArcFace + FFT + ScoreFusion + SubjectVote",
        "routing":     f"soft (Stage1 flat prob >= {S1_SOFT_THRESHOLD} → Stage2 → Stage3) + Score Fusion 6cls",
        "postprocess": f"majority_vote_by_subject(window={args.vote_window})" if args.vote_window > 0 else "none",
        "hyperparams": {
            "s1_epochs":        S1_EPOCHS,
            "s1_lr":            S1_LR,
            "s1_soft_threshold": S1_SOFT_THRESHOLD,
            "s2_warmup_ep":     S2_WARMUP_EP,
            "s2_focal_ep":      S2_FOCAL_EP,
            "s2_focal_lr":      S2_FOCAL_LR,
            "s2_finetune_ep":   S2_FINETUNE_EP,
            "focal_gamma":      FOCAL_GAMMA,
            "s3_epochs":        S3_EPOCHS,
            "s3_ce_ft_ep":      S3_CE_FT_EP,
            "s3_arcface_s":     S3_ARCFACE_S,
            "s3_arcface_m":     S3_ARCFACE_M,
            "n_bio":            BioMechFeatures.N_BIO,
            "hf_bin":           "30Hz",
        },
        "total_minutes": round(total_min, 1),
        "overall": {"acc": round(acc_all, 4), "f1": round(f1_all, 4)},
        "per_class_recall": {
            CLASS_NAMES_ALL.get(i, f"C{i+1}"): round(float(r), 4)
            for i, r in enumerate(recalls)
        },
        "fold_meta": fold_meta,
    }
    path_json = out / "summary_v89.json"
    path_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  ✅ {path_json}")
    h5data.close()


if __name__ == "__main__":
    main()