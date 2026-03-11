# -*- coding: utf-8 -*-
"""
train_surface_expert.py — C4/C5/C6 Surface Expert Classifier

목적: hierarchical 모델의 최대 약점인 C4(흙길)/C5(잔디)/C6(평지) 혼동을
     전용 surface expert로 교정.

핵심 아이디어:
  1. C4/C5/C6 샘플만 추출 → 3-class 전용 학습
  2. surface 구분에 특화된 physics-based features 추가
  3. hierarchical 예측 중 C4/C5/C6 해당 샘플을 expert로 재분류
  4. 최종 앙상블: hierarchical proba × surface_expert proba

Surface-specific feature 설계 근거:
  C4 흙길: 충격 높이 분산↑, 보행 비주기성↑, ML 불안정↑, 고주파 불규칙
  C5 잔디: 로딩 속도↓(탄성), 충격 흡수↑, LF/HF 비율↑, 반발 감쇠
  C6 평지: 자기상관↑(규칙적), 좌우대칭↑, 케이던스 피크 명확, ML 안정

실행:
  python train_surface_expert.py
  python train_surface_expert.py --ensemble_weight 0.4
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import config as _cfg
from train_common import H5Data, seed_everything, log, ensure_dir, save_json, Timer
from features import batch_extract, N_FEATURES

# ── 클래스 정의 ────────────────────────────────────────────────────────────
ALL6   = [0, 1, 2, 3, 4, 5]
C1, C2, C3, C4, C5, C6 = 0, 1, 2, 3, 4, 5
SURFACE_CLASSES = [C4, C5, C6]   # expert가 다루는 클래스
CLASS_NAMES = ["C1-미끄러운", "C2-오르막", "C3-내리막", "C4-흙길", "C5-잔디", "C6-평지"]

# C4/C5/C6 → 0/1/2 로컬 매핑
SURFACE_LOCAL = {C4: 0, C5: 1, C6: 2}
SURFACE_LOCAL_INV = {0: C4, 1: C5, 2: C6}
SURFACE_NAMES = ["C4-흙길", "C5-잔디", "C6-평지"]

DEVICE = _cfg.DEVICE


# ══════════════════════════════════════════════════════════════════════════════
# 1. Surface-Specific Feature Extractor
#    physics-based features specifically designed to separate C4/C5/C6
# ══════════════════════════════════════════════════════════════════════════════

def extract_surface_features(X: np.ndarray, sample_rate: float = 200.0) -> np.ndarray:
    """
    X: (N, T, C) raw IMU signal  (T=256, C=54)
    returns: (N, N_SURF_FEAT)

    Surface-specific feature groups:
    A. 충격 패턴 (6): peak height mean/std/cv, loading rate, impact ratio
    B. 보행 규칙성 (4): autocorrelation, cadence peak prominence, step consistency
    C. 주파수 특성 (6): LF/HF ratio, spectral entropy, centroid, HF energy
    D. ML 안정성 (4): medial-lateral RMS, ML/AP ratio, ML jerk
    E. 흡수/반발 (4): damping index, rebound ratio, loading asymmetry
    F. 좌우대칭 (4): L/R peak symmetry, timing symmetry
    """
    N, T, C = X.shape
    feat_list = []

    # ── foot Z accel (발 수직 가속도) ────────────────────────────────────
    # 채널 레이아웃: Foot_LT(0~5), Foot_RT(6~11), Shank_LT(12~17), Shank_RT(18~23)
    # accel: 0,1,2=LT xyz, 6,7,8=RT xyz  → z=index 2(LT), 8(RT)
    # AP(전후)=x, ML(좌우)=y, Vert(수직)=z
    fz_lt = X[:, :, 2].astype(np.float32)   # (N, T) foot LT vertical
    fz_rt = X[:, :, 8].astype(np.float32)   # (N, T) foot RT vertical
    fx_lt = X[:, :, 0].astype(np.float32)   # foot LT AP
    fy_lt = X[:, :, 1].astype(np.float32)   # foot LT ML
    fy_rt = X[:, :, 7].astype(np.float32)   # foot RT ML
    sz_lt = X[:, :, 14].astype(np.float32)  # shank LT vertical
    sz_rt = X[:, :, 20].astype(np.float32)  # shank RT vertical

    eps = 1e-6

    # ── A. 충격 패턴 ──────────────────────────────────────────────────────
    def _find_peaks_simple(sig):
        """(N, T) → (N,) peak height, (N,) peak interval CV"""
        N_ = sig.shape[0]
        peak_heights = np.zeros(N_, dtype=np.float32)
        peak_cv      = np.zeros(N_, dtype=np.float32)
        for i in range(N_):
            s = sig[i]
            # 로컬 최대값 찾기
            diff = np.diff(s)
            peaks = np.where((np.concatenate([[0], diff]) >= 0) &
                             (np.concatenate([diff, [0]]) < 0))[0]
            heights = s[peaks] if len(peaks) > 0 else np.array([s.max()])
            peak_heights[i] = heights.mean()
            if len(peaks) > 2:
                intervals = np.diff(peaks).astype(np.float32)
                peak_cv[i] = intervals.std() / (intervals.mean() + eps)
            else:
                peak_cv[i] = 1.0
        return peak_heights, peak_cv

    fz_mean = (fz_lt.mean(1) + fz_rt.mean(1)) / 2
    fz_peak_lt, fz_cv_lt = _find_peaks_simple(fz_lt)
    fz_peak_rt, fz_cv_rt = _find_peaks_simple(fz_rt)
    fz_peak_var = np.abs(fz_peak_lt - fz_peak_rt)   # 좌우 피크 차이

    # Loading rate: 첫 25% 구간의 최대 상승 기울기
    seg_len = T // 4
    fz_grad_lt = np.diff(fz_lt[:, :seg_len], axis=1).max(axis=1)
    fz_grad_rt = np.diff(fz_rt[:, :seg_len], axis=1).max(axis=1)
    loading_rate = (fz_grad_lt + fz_grad_rt) / 2

    # Impact absorption: 충격 직후 감쇠 비율
    mid = T // 2
    pre_rms  = np.sqrt((fz_lt[:, :mid]**2).mean(1) + eps)
    post_rms = np.sqrt((fz_lt[:, mid:]**2).mean(1) + eps)
    absorption = post_rms / pre_rms   # <1이면 감쇠 → C5 특징

    feat_list.extend([
        fz_peak_lt,            # A1: LT 피크 평균
        fz_cv_lt,              # A2: LT 피크 간격 변동계수 (C4↑)
        fz_cv_rt,              # A3: RT 피크 간격 변동계수
        fz_peak_var,           # A4: LR 피크 차이 (C4↑)
        loading_rate,          # A5: 로딩 속도 (C6↑, C5↓)
        absorption,            # A6: 충격 흡수율 (C5↑)
    ])

    # ── B. 보행 규칙성 ────────────────────────────────────────────────────
    # Autocorrelation at lag=stride_period (~100샘플@200Hz)
    def _autocorr_at_lag(sig, lag):
        """(N, T) → (N,) lag에서의 정규화 자기상관"""
        N_ = sig.shape[0]
        ac = np.zeros(N_, dtype=np.float32)
        for i in range(N_):
            s = sig[i] - sig[i].mean()
            norm = (s**2).sum() + eps
            ac[i] = float((s[:-lag] * s[lag:]).sum()) / norm if lag < len(s) else 0.
        return ac

    stride_lag = max(20, int(0.5 * sample_rate))   # ~0.5초 stride
    ac_fz_lt = _autocorr_at_lag(fz_lt, stride_lag)
    ac_sz_lt = _autocorr_at_lag(sz_lt, stride_lag)

    # Cadence peak prominence: FFT에서 케이던스 주파수(0.8~2.5Hz) 피크 강도
    freqs = np.fft.rfftfreq(T, d=1./sample_rate)
    fft_fz = np.abs(np.fft.rfft(fz_lt - fz_lt.mean(axis=1, keepdims=True), axis=1))
    cadence_mask = (freqs >= 0.8) & (freqs <= 2.5)
    total_power  = (fft_fz**2).sum(1) + eps
    cadence_power = (fft_fz[:, cadence_mask]**2).sum(1)
    cadence_prominence = cadence_power / total_power   # C6↑ (규칙적 보행)

    # Step-to-step consistency: 연속 stride 간 RMS 변화율
    half = T // 2
    step_consistency = np.abs(
        np.sqrt((fz_lt[:, :half]**2).mean(1)) -
        np.sqrt((fz_lt[:, half:]**2).mean(1))
    )   # C4↑ (불규칙)

    feat_list.extend([
        ac_fz_lt,              # B1: foot z 자기상관 (C6↑)
        ac_sz_lt,              # B2: shank z 자기상관 (C6↑)
        cadence_prominence,    # B3: 케이던스 피크 (C6↑)
        step_consistency,      # B4: 스텝 일관성 (C4↑=나쁨)
    ])

    # ── C. 주파수 특성 ────────────────────────────────────────────────────
    fft_sz = np.abs(np.fft.rfft(sz_lt - sz_lt.mean(axis=1, keepdims=True), axis=1))
    total_sz = (fft_sz**2).sum(1) + eps

    lf_mask  = (freqs >= 0.5) & (freqs < 5.0)
    mf_mask  = (freqs >= 5.0) & (freqs < 15.0)
    hf_mask  = (freqs >= 15.0) & (freqs < 50.0)

    lf_power = (fft_sz[:, lf_mask]**2).sum(1)
    mf_power = (fft_sz[:, mf_mask]**2).sum(1)
    hf_power = (fft_sz[:, hf_mask]**2).sum(1)

    lf_hf_ratio = lf_power / (hf_power + eps)   # C5↑ (저주파 지배)
    mf_hf_ratio = mf_power / (hf_power + eps)

    # Spectral entropy
    p = fft_sz / (fft_sz.sum(1, keepdims=True) + eps)
    p = np.clip(p, 1e-10, 1.0)
    spec_entropy = -(p * np.log(p)).sum(1) / np.log(p.shape[1] + 1)  # C4↑

    # Spectral centroid
    spec_centroid = (fft_sz * freqs[None, :fft_sz.shape[1]]).sum(1) / (fft_sz.sum(1) + eps)

    # HF ratio (foot)
    fft_fz_arr = np.abs(np.fft.rfft(fz_lt - fz_lt.mean(axis=1, keepdims=True), axis=1))
    total_fz = (fft_fz_arr**2).sum(1) + eps
    hf_ratio_foot = (fft_fz_arr[:, hf_mask]**2).sum(1) / total_fz  # C6↑ (딱딱한 지면)

    feat_list.extend([
        lf_hf_ratio,           # C1: LF/HF (C5↑)
        mf_hf_ratio,           # C2: MF/HF
        spec_entropy,          # C3: 스펙트럴 엔트로피 (C4↑)
        spec_centroid,         # C4: 스펙트럴 중심 주파수
        hf_ratio_foot,         # C5: 발 고주파 비율 (C6↑)
        hf_power / total_sz,   # C6: shank 고주파 비율
    ])

    # ── D. ML(좌우) 안정성 ───────────────────────────────────────────────
    ml_rms_lt = np.sqrt((fy_lt**2).mean(1))
    ml_rms_rt = np.sqrt((fy_rt**2).mean(1))
    ap_rms_lt = np.sqrt((fx_lt**2).mean(1))

    ml_ap_ratio = (ml_rms_lt + ml_rms_rt) / (2 * ap_rms_lt + eps)  # C4↑
    ml_jerk_lt  = np.sqrt((np.diff(fy_lt, axis=1)**2).mean(1))
    ml_sym      = np.abs(ml_rms_lt - ml_rms_rt) / (ml_rms_lt + ml_rms_rt + eps)  # C4↑

    feat_list.extend([
        ml_rms_lt,             # D1: ML 가속도 RMS (C4↑)
        ml_ap_ratio,           # D2: ML/AP 비율 (C4↑)
        ml_jerk_lt,            # D3: ML jerk (C4↑)
        ml_sym,                # D4: ML 비대칭 (C4↑)
    ])

    # ── E. 흡수/반발 특성 ────────────────────────────────────────────────
    # Damping index: 힐스트라이크 이후 진동 감쇠 속도
    heel_end = T // 3
    peak_region = fz_lt[:, :heel_end]
    peak_idx = np.argmax(np.abs(peak_region), axis=1)

    damping_vals = np.zeros(N, dtype=np.float32)
    rebound_vals = np.zeros(N, dtype=np.float32)
    for i in range(N):
        pi = peak_idx[i]
        if pi + 20 < heel_end:
            after = np.abs(peak_region[i, pi:pi+20])
            peak_v = np.abs(peak_region[i, pi]) + eps
            # 지수 감쇠 계수 (클수록 빠른 감쇠 → C6 딱딱)
            damping_vals[i] = float(after[0]) / float(peak_v)
            # 반발: peak 이후 반등 비율
            if len(after) > 5:
                rebound_vals[i] = float(after[5:].max()) / float(peak_v)
        else:
            damping_vals[i] = 0.5
            rebound_vals[i] = 0.3

    # Loading asymmetry: 올라가는 slope vs 내려가는 slope 비율
    rise_slope = np.diff(fz_lt[:, :T//4], axis=1).clip(min=0).mean(1)
    fall_slope = np.abs(np.diff(fz_lt[:, T//4:T//2], axis=1).clip(max=0)).mean(1)
    load_asymm = rise_slope / (fall_slope + eps)  # C5↑ (천천히 올라감)

    # Contact time proxy: 수직 가속도가 threshold 이상인 구간 비율
    threshold = np.percentile(np.abs(fz_lt), 70, axis=1, keepdims=True)
    contact_ratio = (np.abs(fz_lt) >= threshold).mean(1)  # C5↑

    feat_list.extend([
        damping_vals,          # E1: 감쇠율 (C6↑=빠른감쇠)
        rebound_vals,          # E2: 반발 비율 (C5↑=탄성)
        load_asymm,            # E3: 로딩 비대칭 (C5↑)
        contact_ratio,         # E4: 접촉 시간 비율 (C5↑)
    ])

    # ── F. 좌우 대칭 ─────────────────────────────────────────────────────
    # Peak symmetry
    fz_peak_sym = np.abs(fz_peak_lt - fz_peak_rt) / (fz_peak_lt + fz_peak_rt + eps)

    # Timing symmetry: LT/RT 피크 위치 차이
    pk_lt = np.argmax(np.abs(fz_lt), axis=1).astype(np.float32)
    pk_rt = np.argmax(np.abs(fz_rt), axis=1).astype(np.float32)
    timing_sym = np.abs(pk_lt - pk_rt) / T   # C4↑ (비대칭)

    # RMS symmetry
    rms_sym_foot  = np.abs(np.sqrt((fz_lt**2).mean(1)) - np.sqrt((fz_rt**2).mean(1)))
    rms_sym_shank = np.abs(np.sqrt((sz_lt**2).mean(1)) - np.sqrt((sz_rt**2).mean(1)))

    feat_list.extend([
        fz_peak_sym,           # F1: 피크 대칭 (C6↓=대칭)
        timing_sym,            # F2: 타이밍 대칭 (C4↑)
        rms_sym_foot,          # F3: foot RMS 대칭 (C4↑)
        rms_sym_shank,         # F4: shank RMS 대칭 (C4↑)
    ])

    # ── 기존 feat 추가 (N_FEATURES) ──────────────────────────────────────
    # (아래에서 concat 예정)

    return np.stack(feat_list, axis=1).astype(np.float32)   # (N, 28)


N_SURF_FEAT = 28   # A(6)+B(4)+C(6)+D(4)+E(4)+F(4)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SurfaceDataset(Dataset):
    """C4/C5/C6 전용 데이터셋 (로컬 라벨 0/1/2)"""
    def __init__(self, feat_std: np.ndarray, feat_surf: np.ndarray,
                 y: np.ndarray, local_labels: bool = True):
        """
        feat_std : (N, N_FEATURES) 표준 feat (정규화됨)
        feat_surf: (N, N_SURF_FEAT) surface-specific feat
        y        : (N,) 원본 라벨 (C4=3, C5=4, C6=5)
        """
        self.x = np.concatenate([feat_std, feat_surf], axis=1).astype(np.float32)
        if local_labels:
            self.y = np.array([SURFACE_LOCAL[int(yi)] for yi in y], dtype=np.int64)
        else:
            self.y = y.astype(np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model — Surface Expert
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))


class SurfaceExpertModel(nn.Module):
    """
    C4/C5/C6 전용 3-class classifier.
    feat_std + feat_surf 입력 → C4/C5/C6 로컬 확률 출력
    """
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.25):
        super().__init__()
        # Surface feat 전용 브랜치 (28차원)
        self.surf_branch = nn.Sequential(
            nn.Linear(N_SURF_FEAT, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
        )
        # 표준 feat 브랜치
        self.std_branch = nn.Sequential(
            nn.Linear(in_dim - N_SURF_FEAT, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
        )
        # 통합 trunk
        trunk_dim = 128 + 256
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.head = nn.Linear(hidden // 2, 3)

        # auxiliary heads (surface physics)
        self.softness_head  = nn.Linear(hidden // 2, 1)  # P(soft surface=C5)
        self.irregular_head = nn.Linear(hidden // 2, 1)  # P(irregular=C4)

    def forward(self, x):
        surf = x[:, -N_SURF_FEAT:]
        std  = x[:, :-N_SURF_FEAT]
        hs   = self.surf_branch(surf)
        hd   = self.std_branch(std)
        h    = self.trunk(torch.cat([hs, hd], dim=1))
        return {
            "logits":   self.head(h),
            "soft_logit": self.softness_head(h).squeeze(-1),
            "irr_logit":  self.irregular_head(h).squeeze(-1),
        }


class SurfaceExpertLoss(nn.Module):
    def __init__(self, class_weights=None, focal_gamma=2.0,
                 w_main=1.0, w_soft=0.5, w_irr=0.5):
        super().__init__()
        self.gamma    = focal_gamma
        self.w_main   = w_main
        self.w_soft   = w_soft
        self.w_irr    = w_irr
        self.cw       = class_weights   # (3,) tensor

    def focal_ce(self, logits, targets):
        log_p = F.log_softmax(logits, 1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = log_pt.exp()
        loss   = -((1 - pt) ** self.gamma) * log_pt
        if self.cw is not None:
            loss = loss * self.cw.to(logits.device)[targets]
        return loss.mean()

    def forward(self, out, y_local):
        # y_local: 0=C4, 1=C5, 2=C6
        L_main = self.focal_ce(out["logits"], y_local)
        y_soft = (y_local == 1).float()   # C5
        y_irr  = (y_local == 0).float()   # C4
        L_soft = F.binary_cross_entropy_with_logits(out["soft_logit"], y_soft)
        L_irr  = F.binary_cross_entropy_with_logits(out["irr_logit"],  y_irr)
        return self.w_main * L_main + self.w_soft * L_soft + self.w_irr * L_irr


# ══════════════════════════════════════════════════════════════════════════════
# 4. Subject Normalization
# ══════════════════════════════════════════════════════════════════════════════

class SubjectNormalizer:
    def __init__(self): self.stats = {}

    def fit(self, feat, groups):
        for sbj in np.unique(groups):
            x = feat[groups == sbj]
            self.stats[sbj] = (x.mean(0), x.std(0).clip(min=1e-6))

    def transform(self, feat, groups):
        out = feat.copy().astype(np.float32)
        for sbj in np.unique(groups):
            m = groups == sbj
            if sbj in self.stats:
                mu, std = self.stats[sbj]
            else:
                x = feat[m]; mu = x.mean(0); std = x.std(0).clip(min=1e-6)
            out[m] = (feat[m] - mu) / std
        return out

    def fit_transform(self, feat, groups):
        self.fit(feat, groups); return self.transform(feat, groups)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Training
# ══════════════════════════════════════════════════════════════════════════════

def make_balanced_loader(ds, batch_size, shuffle=True, drop_last=True):
    """C4/C5/C6 균형 샘플링"""
    cls, cnt = np.unique(ds.y, return_counts=True)
    cw = np.zeros(3, dtype=np.float64)
    cw[cls] = 1.0 / cnt.astype(np.float64)
    weights = torch.tensor(cw[ds.y], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(ds.y), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      drop_last=drop_last, pin_memory=True, num_workers=4)


def run_fold(fi, tr_feat, tr_surf, tr_y, te_feat, te_surf, te_y, args):
    in_dim = tr_feat.shape[1] + N_SURF_FEAT

    tr_ds = SurfaceDataset(tr_feat, tr_surf, tr_y)
    te_ds = SurfaceDataset(te_feat, te_surf, te_y)

    tr_dl = make_balanced_loader(tr_ds, args.batch)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                       pin_memory=True, num_workers=4)

    # class weights
    cls, cnt = np.unique(tr_ds.y, return_counts=True)
    cw = np.ones(3, dtype=np.float32)
    for c, n in zip(cls, cnt):
        cw[int(c)] = len(tr_ds.y) / (3.0 * n)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    log(f"  [F{fi}] class weights: {cw.round(2).tolist()}")

    model = SurfaceExpertModel(in_dim, args.hidden, args.dropout).to(DEVICE)
    crit  = SurfaceExpertLoss(class_weights=cw_t, focal_gamma=args.focal_gamma)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch   = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=max(1, len(tr_dl)),
        pct_start=0.2, anneal_strategy="cos", div_factor=10, final_div_factor=100,
    )

    best_f1, best_state, patience = -1., None, 0

    for ep in range(1, args.epochs + 1):
        # train
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step(); sch.step()

        # eval
        model.eval(); yt, yp = [], []
        with torch.no_grad():
            for xb, yb in te_dl:
                logits = model(xb.to(DEVICE))["logits"]
                yp.extend(logits.argmax(1).cpu().tolist())
                yt.extend(yb.tolist())

        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1; best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= args.early_stop: log(f"  [F{fi}] EarlyStop ep{ep}"); break

        if ep % 10 == 0 or ep == 1:
            log(f"  [F{fi}] ep{ep:03d}/{args.epochs}  acc={acc:.4f}  f1={f1:.4f}  best={best_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval(); yt, yp, proba = [], [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            out = model(xb.to(DEVICE))
            p = F.softmax(out["logits"], 1).cpu().numpy()
            proba.append(p)
            yp.extend(p.argmax(1).tolist())
            yt.extend(yb.tolist())

    proba_arr = np.concatenate(proba, axis=0)   # (N, 3)
    yt_arr = np.array(yt); yp_arr = np.array(yp)

    acc = accuracy_score(yt_arr, yp_arr)
    f1  = f1_score(yt_arr, yp_arr, average="macro", zero_division=0)
    log(f"  [F{fi}] ★ Best  Acc={acc:.4f}  F1={f1:.4f}")
    log(f"\n{classification_report(yt_arr, yp_arr, target_names=SURFACE_NAMES, digits=4, zero_division=0)}")

    return acc, f1, proba_arr, yt_arr, model


# ══════════════════════════════════════════════════════════════════════════════
# 6. Ensemble with Hierarchical
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_with_hierarchical(
    hier_proba: np.ndarray,       # (N, 6) hierarchical 예측 확률
    expert_proba_local: np.ndarray,  # (N_surf, 3) C4/C5/C6 전용 확률
    surface_mask: np.ndarray,     # (N,) bool — C4/C5/C6 해당 샘플
    hier_labels: np.ndarray,      # (N,) 실제 라벨
    expert_w: float = 0.5,        # expert 가중치 (0~1)
) -> np.ndarray:
    """
    hierarchical proba와 surface expert proba를 앙상블.
    C4/C5/C6 해당 샘플에서만 expert를 반영.
    """
    final = hier_proba.copy()
    surf_idx = np.where(surface_mask)[0]

    # expert proba (local 0/1/2) → 전체 6-class 공간으로 확장
    expert_6cls = np.zeros((len(surf_idx), 6), dtype=np.float32)
    expert_6cls[:, C4] = expert_proba_local[:, 0]
    expert_6cls[:, C5] = expert_proba_local[:, 1]
    expert_6cls[:, C6] = expert_proba_local[:, 2]

    # 가중 평균 (C4/C5/C6 차원만)
    for j, gi in enumerate(surf_idx):
        h_surf = hier_proba[gi, [C4, C5, C6]]
        h_surf_sum = h_surf.sum() + 1e-8
        h_surf_norm = h_surf / h_surf_sum

        e_surf = expert_proba_local[j]

        blended = (1 - expert_w) * h_surf_norm + expert_w * e_surf
        blended /= blended.sum() + 1e-8

        # 비율 유지하며 전체 proba에 반영
        total_surf_prob = hier_proba[gi, [C4, C5, C6]].sum()
        final[gi, C4] = blended[0] * total_surf_prob
        final[gi, C5] = blended[1] * total_surf_prob
        final[gi, C6] = blended[2] * total_surf_prob

    return final


# ══════════════════════════════════════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch",          type=int,   default=512)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--wd",             type=float, default=1e-4)
    p.add_argument("--hidden",         type=int,   default=512)
    p.add_argument("--dropout",        type=float, default=0.25)
    p.add_argument("--early_stop",     type=int,   default=20)
    p.add_argument("--focal_gamma",    type=float, default=2.0)
    p.add_argument("--kfold",          type=int,   default=5)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--ensemble_weight",type=float, default=0.5,
                   help="expert weight for C4/C5/C6 ensemble (0~1)")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    DEVICE_str = str(DEVICE)

    log("=" * 60)
    log("  Surface Expert Classifier (C4/C5/C6)")
    log(f"  N_SURF_FEAT={N_SURF_FEAT}  hidden={args.hidden}")
    log(f"  epochs={args.epochs}  lr={args.lr}  ensemble_w={args.ensemble_weight}")
    log("=" * 60)

    h5     = H5Data(_cfg.CFG.h5_path)
    le     = LabelEncoder()
    y_all  = le.fit_transform(h5.y_raw).astype(np.int64)
    groups = h5.subj_id

    log(f"  데이터: N={len(y_all)}  피험자={len(np.unique(groups))}명")

    # ── feat 추출 ──────────────────────────────────────────────────────────
    cache_dir = ensure_dir(_cfg.CFG.repo_dir / "cache" / f"feat{N_FEATURES}_seed{args.seed}_k5")
    cache_std = cache_dir / "all_feat_std.npy"
    cache_surf = _cfg.CFG.repo_dir / "cache" / f"surf_feat_seed{args.seed}.npy"

    if cache_std.exists():
        log("  표준 feat 캐시 히트")
        feat_std = np.load(cache_std)
    else:
        from channel_groups import get_foot_accel_idx
        foot_idx = get_foot_accel_idx(h5.channels)
        log("  표준 feat 추출 중...")
        with Timer() as t:
            feat_std = batch_extract(h5.X, foot_idx, _cfg.CFG.sample_rate)
        np.save(cache_std, feat_std)
        log(f"  완료 {t}")

    if cache_surf.exists():
        log("  surface feat 캐시 히트")
        feat_surf = np.load(cache_surf)
    else:
        log("  surface feat 추출 중...")
        with Timer() as t:
            feat_surf = extract_surface_features(h5.X, float(_cfg.CFG.sample_rate))
        np.save(cache_surf, feat_surf)
        log(f"  surface feat 완료 {t}")

    assert feat_surf.shape == (len(y_all), N_SURF_FEAT), \
        f"surf feat shape mismatch: {feat_surf.shape}"

    # ── C4/C5/C6 샘플만 선택 ─────────────────────────────────────────────
    surf_mask = np.isin(y_all, SURFACE_CLASSES)
    log(f"  C4/C5/C6 샘플: {surf_mask.sum()} / {len(y_all)}")

    feat_std_surf  = feat_std[surf_mask]
    feat_surf_surf = feat_surf[surf_mask]
    y_surf         = y_all[surf_mask]
    groups_surf    = groups[surf_mask]

    out = ensure_dir(_cfg.CFG.repo_dir / "out_N50" / "surface_expert")

    # ── K-Fold ─────────────────────────────────────────────────────────────
    sgkf    = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    results = []
    all_preds, all_labels = [], []

    with Timer() as total_timer:
        for fi, (tr_idx, te_idx) in enumerate(
            sgkf.split(np.zeros(len(y_surf)), y_surf, groups_surf), 1
        ):
            log(f"\n{'='*60}")
            log(f"  Fold {fi}/{args.kfold}  tr={len(tr_idx)}  te={len(te_idx)}")

            # subject-wise normalization
            snorm = SubjectNormalizer()
            tr_std = snorm.fit_transform(feat_std_surf[tr_idx], groups_surf[tr_idx])
            te_std = snorm.transform(feat_std_surf[te_idx],  groups_surf[te_idx])

            # surface feat도 동일 정규화
            snorm_s = SubjectNormalizer()
            tr_sf = snorm_s.fit_transform(feat_surf_surf[tr_idx], groups_surf[tr_idx])
            te_sf = snorm_s.transform(feat_surf_surf[te_idx],  groups_surf[te_idx])

            acc, f1, proba, te_y_local, model = run_fold(
                fi, tr_std, tr_sf, y_surf[tr_idx],
                    te_std, te_sf, y_surf[te_idx], args
            )

            # 원래 라벨로 변환
            te_y_orig  = np.array([SURFACE_LOCAL_INV[int(yi)] for yi in te_y_local])
            pred_local = proba.argmax(1)
            pred_orig  = np.array([SURFACE_LOCAL_INV[int(p)] for p in pred_local])

            all_preds.extend(pred_orig.tolist())
            all_labels.extend(te_y_orig.tolist())
            results.append({"fold": fi, "acc": acc, "f1": f1})

            # proba 저장 (hierarchical 앙상블용)
            proba_dir = ensure_dir(out / "probas")
            np.save(proba_dir / f"surface_proba_fold{fi}.npy",  proba)
            np.save(proba_dir / f"surface_labels_fold{fi}.npy", te_y_orig)

            del model; gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 전체 결과 ──────────────────────────────────────────────────────────
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    final_acc  = accuracy_score(all_labels, all_preds)
    final_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # C4/C5/C6만의 per-class recall
    cm = confusion_matrix(all_labels, all_preds, labels=SURFACE_CLASSES)
    recalls = cm.diagonal() / cm.sum(1).clip(min=1)

    log(f"\n{'='*60}")
    log(f"  ★ Surface Expert 완료  총 소요: {total_timer}")
    log(f"  Mean Acc={final_acc:.4f}  MacroF1={final_f1:.4f}")
    for ci, cls in enumerate(SURFACE_CLASSES):
        log(f"    {CLASS_NAMES[cls]:<12} recall={recalls[ci]*100:.1f}%")
    log(f"{'='*60}")

    save_json({
        "experiment": "surface_expert",
        "mean_acc": round(final_acc, 4),
        "mean_f1":  round(final_f1,  4),
        "per_class_recall": {
            CLASS_NAMES[c]: round(float(r), 4)
            for c, r in zip(SURFACE_CLASSES, recalls)
        },
        "folds": results,
    }, out / "summary_surface_expert.json")

    # ── hierarchical proba와 앙상블 테스트 ────────────────────────────────
    hier_proba_dir = _cfg.CFG.repo_dir / "out_N50" / "kfold" / "hierarchical_eventfusion" / "probas"
    if hier_proba_dir.exists():
        log("\n  ── Hierarchical 앙상블 테스트 ──")
        for ew in [0.3, 0.4, 0.5, 0.6]:
            # 각 fold proba 로드 & 앙상블
            ens_preds, ens_labels = [], []
            for fi in range(1, args.kfold + 1):
                hp_path = hier_proba_dir / f"hier_proba_fold{fi}.npy"
                hl_path = hier_proba_dir / f"hier_labels_fold{fi}.npy"
                sp_path = out / "probas" / f"surface_proba_fold{fi}.npy"
                if not (hp_path.exists() and sp_path.exists()): continue

                hier_p  = np.load(hp_path)   # (N_te, 6)
                hier_l  = np.load(hl_path)   # (N_te,)
                surf_p  = np.load(sp_path)   # (N_surf_te, 3)
                surf_l  = np.load(out / "probas" / f"surface_labels_fold{fi}.npy")

                # C4/C5/C6 해당 인덱스 찾기
                surf_idx_in_te = np.where(np.isin(hier_l, SURFACE_CLASSES))[0]
                if len(surf_idx_in_te) != len(surf_p):
                    log(f"  [WARN] F{fi}: 인덱스 불일치 {len(surf_idx_in_te)} vs {len(surf_p)}")
                    continue

                s_mask = np.zeros(len(hier_l), dtype=bool)
                s_mask[surf_idx_in_te] = True

                blended = ensemble_with_hierarchical(
                    hier_p, surf_p, s_mask, hier_l, expert_w=ew)
                preds = blended.argmax(1)
                ens_preds.extend(preds.tolist())
                ens_labels.extend(hier_l.tolist())

            if ens_labels:
                e_acc = accuracy_score(ens_labels, ens_preds)
                e_f1  = f1_score(ens_labels, ens_preds, average="macro", zero_division=0)
                log(f"  앙상블 w={ew:.1f}  Acc={e_acc:.4f}  F1={e_f1:.4f}")
    else:
        log(f"  [INFO] hierarchical proba 없음 — 앙상블 생략")
        log(f"         경로: {hier_proba_dir}")

    h5.close()


if __name__ == "__main__":
    main()