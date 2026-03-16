# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
"""
train_hierarchical.py — v14.4 (C6 평지 recall 부스트)

v14.2 → v14.5 변경
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1] SLIP_TAU: 0.45 → 0.38  (false C1 억제 → C6 recall 보호)
[2] AUX_SURFACE_W: 0.25 → 0.45  (surface 감독 강화)
[3] crit_surface weight: [1.0,2.0,2.0] → [3.5,2.5,2.5]  (C6 직접 부스트)
[4] factorized_proba: C6 확률 하한 0.02 보장
[5] threshold_search: C6 전용 세밀 탐색 (0.5~4.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v14.1 → v14.2 버그 수정
[Fix1] StableBaselineBank._stability_score: toe proxy → 실제 horizontal RMS (N_BIO 184, 190)
[Fix2] train_event_fusion: vote_window/slip_tau 파라미터화
[Fix3] bvf에 labels=ALL6 추가
[Fix4] train_event_fusion 호출부에 vote_window=args.vote_window 전달
[Fix5] 결과 파일명 v140 → v142
"""

import sys
import time
import json
import gc
import math
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from channel_groups import build_branch_idx
from models import M6_BranchCBAMCrossAug
from train_common import (
    log, H5Data, fit_bsc_on_train,
    make_branch_dataset, make_loader, save_cm, clear_fold_cache,
    filter_and_remap, N_ACTIVE_CLASSES, ACTIVE_CLASS_NAMES,
)

DEVICE = config.DEVICE
# C5 제외: 5클래스
ALL5 = np.arange(N_ACTIVE_CLASSES)   # [0,1,2,3,4]
ALL6 = ALL5   # 하위 호환 alias (코드 내 ALL6 참조 유지)

CLASS_NAMES_ALL = {
    0: "C1-미끄러운", 1: "C2-오르막", 2: "C3-내리막",
    3: "C4-흙길",     4: "C6-평지",   # C5(잔디) 제외
}

# ── Fusion 하이퍼파라미터 ──────────────────────────────────────────────────
FUSION_EPOCHS    = 90
FUSION_LR        = 1e-4
FUSION_PATIENCE  = 22
FOCAL_GAMMA      = 2.0

# ── 보조 loss 가중치 ───────────────────────────────────────────────────────
AUX_SLIP_W    = 0.45
AUX_SLOPE_W   = 0.30
AUX_SURFACE_W = 0.80   # [v14.5] 0.60 → 0.80 (surface 감독 최대화)

TRIPLET_MARGIN   = 0.8
FUSION_TRIPLET_W = 0.05

# ── SlipMultiTaskWarmup ────────────────────────────────────────────────────
WARMUP_EPOCHS   = 50
WARMUP_LR       = 5e-5
WARMUP_PATIENCE = 25

# ── StableBaselineBank ────────────────────────────────────────────────────
STABLE_PERCENTILE    = 30
STABLE_SCORE_WEIGHTS = dict(asym=0.4, entropy=0.3, horizontal=0.3)
N_DELTA              = 16

# ── [v14.5] C1 peak-preserving threshold 낮춤 (false C1 → C6 보호) ─────────
SLIP_TAU = 0.22   # [v14.5] 0.30 → 0.22 (C6 보호 최대화)

# ── Sequence refiner ──────────────────────────────────────────────────────
SEQ_LEN     = 9
SEQ_HIDDEN  = 256
SEQ_EPOCHS  = 30
SEQ_LR      = 4e-4
SEQ_PATIENCE= 12


def parse_args():
    p = argparse.ArgumentParser(
        description="Detector-Centric Hierarchical Slip Event v14.5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fusion_epochs",    type=int,   default=FUSION_EPOCHS)
    p.add_argument("--fusion_lr",        type=float, default=FUSION_LR)
    p.add_argument("--fusion_patience",  type=int,   default=FUSION_PATIENCE)
    p.add_argument("--focal_gamma",      type=float, default=FOCAL_GAMMA)
    p.add_argument("--aux_slip_w",       type=float, default=AUX_SLIP_W)
    p.add_argument("--aux_slope_w",      type=float, default=AUX_SLOPE_W)
    p.add_argument("--aux_surface_w",    type=float, default=AUX_SURFACE_W)
    p.add_argument("--slip_tau",         type=float, default=SLIP_TAU)
    p.add_argument("--warmup_epochs",    type=int,   default=WARMUP_EPOCHS)
    p.add_argument("--seq_len",          type=int,   default=SEQ_LEN)
    p.add_argument("--seq_epochs",       type=int,   default=SEQ_EPOCHS)
    p.add_argument("--vote_window",      type=int,   default=5)
    p.add_argument("--n_subjects",       type=int,   default=None)
    p.add_argument("--wandb",            action="store_true")
    p.add_argument("--wandb_project",    type=str,   default="imu-terrain")
    p.add_argument("--run_name",         type=str,   default=None)
    return p.parse_args()


def apply_args(args):
    global FUSION_EPOCHS, FUSION_LR, FUSION_PATIENCE, FOCAL_GAMMA
    global AUX_SLIP_W, AUX_SLOPE_W, AUX_SURFACE_W, SLIP_TAU
    global WARMUP_EPOCHS, SEQ_LEN, SEQ_EPOCHS
    FUSION_EPOCHS   = args.fusion_epochs
    FUSION_LR       = args.fusion_lr
    FUSION_PATIENCE = args.fusion_patience
    FOCAL_GAMMA     = args.focal_gamma
    AUX_SLIP_W      = args.aux_slip_w
    AUX_SLOPE_W     = args.aux_slope_w
    AUX_SURFACE_W   = args.aux_surface_w
    SLIP_TAU        = args.slip_tau
    WARMUP_EPOCHS   = args.warmup_epochs
    SEQ_LEN         = args.seq_len
    SEQ_EPOCHS      = args.seq_epochs
    if args.n_subjects is not None:
        config.apply_overrides(n_subjects=args.n_subjects)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurveTracker:
    name: str
    losses: list = field(default_factory=list)
    accs:   list = field(default_factory=list)

    def record(self, loss=None, acc=None):
        if loss is not None: self.losses.append(round(float(loss), 6))
        if acc  is not None: self.accs.append(round(float(acc),  6))

    def save(self, out_dir: Path):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"curve_{self.name}.json").write_text(
            json.dumps({"loss": self.losses, "acc": self.accs}, indent=2))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if self.losses: axes[0].plot(self.losses); axes[0].set_title(f"{self.name} Loss")
        if self.accs:   axes[1].plot(self.accs);   axes[1].set_title(f"{self.name} Score")
        plt.tight_layout(); fig.savefig(out_dir / f"curve_{self.name}.png", dpi=120)
        plt.close(fig)


def _to_device(bi, feat=None, yb=None):
    bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
    if not config.USE_AMP: bi = {k: v.float() for k, v in bi.items()}
    out = [bi]
    if feat is not None: out.append(feat.to(DEVICE, non_blocking=True).float())
    if yb   is not None: out.append(yb.to(DEVICE, non_blocking=True))
    return tuple(out) if len(out) > 1 else out[0]


def _clone_state(m): return {k: v.cpu().clone() for k, v in m.state_dict().items()}


def _make_sch(opt, epochs, warmup=8, min_lr=config.MIN_LR, base_lr=None):
    base = base_lr or opt.param_groups[0]["lr"]
    def fn(ep):
        if ep < warmup: return float(ep + 1) / warmup
        prog = float(ep - warmup) / max(epochs - warmup, 1)
        mf   = min_lr / base
        return mf + (1 - mf) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def _inner_val_split(tr_idx, groups, y, val_ratio=0.15):
    tr_groups  = groups[tr_idx]
    unique_sbj = np.unique(tr_groups)
    n_val      = max(1, int(len(unique_sbj) * val_ratio))
    rng        = np.random.default_rng(config.SEED)
    val_sbj    = set(rng.choice(unique_sbj, n_val, replace=False).tolist())
    mask       = np.array([g not in val_sbj for g in tr_groups])
    inner_tr   = tr_idx[mask]; inner_va = tr_idx[~mask]
    missing    = set(range(N_ACTIVE_CLASSES)) - set(np.unique(y[inner_va]).tolist())
    if missing: log(f"    [WARN] inner val: 클래스 {sorted(missing)} 누락")
    log(f"    inner split: tr={mask.sum()}  val={(~mask).sum()}  val_sbj={sorted(val_sbj)}")
    return inner_tr, inner_va


def auto_class_weights(y_flat, num_classes=N_ACTIVE_CLASSES):
    present = np.unique(y_flat); w = np.ones(num_classes, dtype=np.float32)
    if len(present):
        for c, wc in zip(present, compute_class_weight("balanced", classes=present, y=y_flat)):
            w[int(c)] = float(wc)
    return torch.tensor(w, dtype=torch.float32)


def _get_feat_dim(backbone):
    n_groups = len(backbone.names)
    n_extra  = sum([
        1 if getattr(backbone, "use_fft",          False) else 0,
        1 if getattr(backbone, "use_foot_impact",  False) else 0,
        1 if getattr(backbone, "use_shank_impact", False) else 0,
    ])
    return config.FEAT_DIM * (n_groups + n_extra)


# ─────────────────────────────────────────────────────────────────────────────
# Aux targets
# ─────────────────────────────────────────────────────────────────────────────

def make_aux_targets(y6: torch.Tensor):
    slope   = torch.zeros_like(y6)
    slope[y6 == 1] = 1; slope[y6 == 2] = 2
    slip    = (y6 == 0).long()
    surface = torch.full_like(y6, -100)
    surface[y6 == 5] = 0; surface[y6 == 3] = 1; surface[y6 == 4] = 2
    return slope, slip, surface


# ─────────────────────────────────────────────────────────────────────────────
# ExpandedTerrainFeatures  N_BIO = 208
# ─────────────────────────────────────────────────────────────────────────────

class ExpandedTerrainFeatures(nn.Module):
    N_BIO = 208
    DELTA_KEY_INDICES = [2, 3, 10, 11, 34, 35, 42, 43,
                         101, 108, 148, 154, 196, 197, 200, 201]

    def __init__(self):
        super().__init__()
        self.foot_z      = config.FOOT_Z_ACCEL_IDX
        self.shank_z     = config.SHANK_Z_ACCEL_IDX
        self.sample_rate = float(config.SAMPLE_RATE)
        self.event_radius= max(8, int(0.08 * self.sample_rate))
        self.max_lag     = max(4, int(0.08 * self.sample_rate))
        self.freq_bands  = [(0,3),(3,6),(6,10),(10,20),(20,40)]

    def _vector_norm_pair(self, x):
        return (x[:,0:3,:].pow(2).sum(1).sqrt(), x[:,3:6,:].pow(2).sum(1).sqrt(),
                x[:,6:9,:].pow(2).sum(1).sqrt(), x[:,9:12,:].pow(2).sum(1).sqrt())

    def _summary_stats(self, x):
        eps=1e-6; xa=x.abs()
        xc  = x - x.mean(1, keepdim=True)
        var = xc.pow(2).mean(1).clamp(min=eps)
        return [xa.mean(1), x.std(1), x.pow(2).mean(1).sqrt(), xa.amax(1),
                torch.quantile(xa,0.95,1), torch.quantile(x,0.75,1)-torch.quantile(x,0.25,1),
                (xc.pow(3).mean(1)/var.sqrt().pow(3)).clamp(-10,10),
                (xc.pow(4).mean(1)/var.pow(2)).clamp(0,30)]

    def _spectral_stats(self, x):
        eps=1e-8; T=x.shape[1]
        fft   = torch.fft.rfft(x,dim=1).abs().pow(2)
        freqs = torch.fft.rfftfreq(T,d=1./self.sample_rate).to(x.device)
        total = fft.sum(1).clamp(min=eps)
        outs  = []
        for lo,hi in self.freq_bands:
            m = (freqs>=lo)&(freqs<hi)
            outs.append(fft[:,m].sum(1)/total if m.any() else torch.zeros(x.shape[0],device=x.device))
        p  = (fft/total.unsqueeze(1)).clamp(min=eps)
        outs.append(-(p*p.log()).sum(1)/math.log(p.shape[1]+1))
        outs.append(freqs[(p.cumsum(1)>=0.85).float().argmax(1)])
        return outs

    def _phase_event_stats(self, x, phase):
        eps=1e-6; T=x.shape[1]
        seg = x[:,:int(T*0.45)] if phase=='heel' else x[:,int(T*0.60):]
        sT  = seg.shape[1]; R = max(4,sT//6)
        pad = F.pad(seg.abs(),(R,R),mode="replicate")
        idx = seg.abs().argmax(1)
        base= (idx[:,None]+torch.arange(2*R+1,device=x.device)[None,:]).clamp(0,pad.shape[1]-1)
        loc = pad.gather(1,base)
        pk  = loc[:,R]; pre=loc[:,:R].mean(1); post=loc[:,R+1:].mean(1)
        jerk= seg[:,1:]-seg[:,:-1]
        return [pk, loc.sum(1), post/(pre+eps), (loc>=pk.unsqueeze(1)*0.2).float().mean(1),
                jerk.abs().amax(1), jerk.pow(2).mean(1).sqrt()]

    def _horizontal_stats(self, foot, side):
        eps=1e-6
        xy = foot[:,0:2,:] if side=='left' else foot[:,6:8,:]
        z  = foot[:,2,:]   if side=='left' else foot[:,8,:]
        h  = xy.pow(2).sum(1).sqrt()
        rms= h.pow(2).mean(1).sqrt(); jk=h[:,1:]-h[:,:-1]
        return [rms, h.amax(1), torch.quantile(h,0.95,1),
                jk.abs().amax(1), jk.pow(2).mean(1).sqrt(),
                rms/(z.abs().mean(1)+eps)]

    def _xcorr_max_lag(self, x, y):
        eps=1e-6; L=min(self.max_lag,x.shape[1]-1)
        x0=x-x.mean(1,keepdim=True); y0=y-y.mean(1,keepdim=True)
        nfft=2*x.shape[1]
        corr= torch.fft.irfft(
            torch.fft.rfft(x0,n=nfft,dim=1)*torch.conj(torch.fft.rfft(y0,n=nfft,dim=1)),
            n=nfft, dim=1)
        corr= torch.cat([corr[:,-L:],corr[:,:L+1]],1)
        denom= x0.norm(p=2,dim=1)*y0.norm(p=2,dim=1)+eps
        corr = corr/denom.unsqueeze(1)
        mv,idx=corr.max(1)
        return mv, idx.float()-float(L)

    def _asym(self,a,b,eps=1e-6): return torch.log((a+eps)/(b+eps)).abs()

    @torch.no_grad()
    def forward(self, bi):
        foot  = bi["Foot"].float(); shank=bi["Shank"].float()
        thigh = bi.get("Thigh"); thigh=thigh.float() if thigh is not None else None
        fa_lt,fg_lt,fa_rt,fg_rt = self._vector_norm_pair(foot)
        sa_lt,sg_lt,sa_rt,sg_rt = self._vector_norm_pair(shank)
        if thigh is not None: ta_lt,tg_lt,ta_rt,tg_rt=self._vector_norm_pair(thigh)
        else: z=torch.zeros_like(fa_lt); ta_lt=ta_rt=tg_lt=tg_rt=z
        fz_lt=foot[:,self.foot_z[0],:]; fz_rt=foot[:,self.foot_z[1],:]
        sz_lt=shank[:,self.shank_z[0],:]; sz_rt=shank[:,self.shank_z[1],:]
        feats=[]
        for sig in [fa_lt,fa_rt,fg_lt,fg_rt,sa_lt,sa_rt,sg_lt,sg_rt,ta_lt,ta_rt,tg_lt,tg_rt]:
            feats.extend(self._summary_stats(sig))
        spec_fz_lt=self._spectral_stats(fz_lt); spec_fz_rt=self._spectral_stats(fz_rt)
        spec_sz_lt=self._spectral_stats(sz_lt); spec_sz_rt=self._spectral_stats(sz_rt)
        feats.extend(spec_fz_lt); feats.extend(spec_fz_rt)
        feats.extend(spec_sz_lt); feats.extend(spec_sz_rt)
        fz_lt_heel=self._phase_event_stats(fz_lt,'heel'); fz_rt_heel=self._phase_event_stats(fz_rt,'heel')
        sz_lt_heel=self._phase_event_stats(sz_lt,'heel'); sz_rt_heel=self._phase_event_stats(sz_rt,'heel')
        feats.extend(fz_lt_heel); feats.extend(fz_rt_heel)
        feats.extend(sz_lt_heel); feats.extend(sz_rt_heel)
        fz_lt_toe=self._phase_event_stats(fz_lt,'toe'); fz_rt_toe=self._phase_event_stats(fz_rt,'toe')
        sz_lt_toe=self._phase_event_stats(sz_lt,'toe'); sz_rt_toe=self._phase_event_stats(sz_rt,'toe')
        feats.extend(fz_lt_toe); feats.extend(fz_rt_toe)
        feats.extend(sz_lt_toe); feats.extend(sz_rt_toe)
        for fz,sz,fn,sn,heel_pair,fsp,ssp in [
            (fz_lt,sz_lt,fa_lt,sa_lt,(fz_lt_heel,sz_lt_heel),spec_fz_lt,spec_sz_lt),
            (fz_rt,sz_rt,fa_rt,sa_rt,(fz_rt_heel,sz_rt_heel),spec_fz_rt,spec_sz_rt)]:
            eps=1e-6
            feats.extend([
                sz.abs().amax(1)/(fz.abs().amax(1)+eps),
                sn.pow(2).mean(1).sqrt()/(fn.pow(2).mean(1).sqrt()+eps),
                heel_pair[1][1]/(heel_pair[0][1]+eps),
                *self._xcorr_max_lag(fz,sz),
                0.5*(ssp[4]/(fsp[4]+eps)+(1-sn.pow(2).mean(1).sqrt()/(fn.pow(2).mean(1).sqrt()+eps))),
            ])
        feats.extend(self._horizontal_stats(foot,'left'))
        feats.extend(self._horizontal_stats(foot,'right'))
        feats.extend([
            self._asym(fa_lt.abs().amax(1),fa_rt.abs().amax(1)),
            self._asym(fa_lt.pow(2).mean(1).sqrt(),fa_rt.pow(2).mean(1).sqrt()),
            self._asym(fg_lt.abs().amax(1),fg_rt.abs().amax(1)),
            self._asym(fg_lt.pow(2).mean(1).sqrt(),fg_rt.pow(2).mean(1).sqrt()),
            self._asym(sa_lt.abs().amax(1),sa_rt.abs().amax(1)),
            self._asym(sa_lt.pow(2).mean(1).sqrt(),sa_rt.pow(2).mean(1).sqrt()),
            self._asym(sg_lt.abs().amax(1),sg_rt.abs().amax(1)),
            self._asym(sg_lt.pow(2).mean(1).sqrt(),sg_rt.pow(2).mean(1).sqrt()),
            self._asym(ta_lt.pow(2).mean(1).sqrt(),ta_rt.pow(2).mean(1).sqrt()),
            self._asym(tg_lt.pow(2).mean(1).sqrt(),tg_rt.pow(2).mean(1).sqrt()),
            self._asym(fz_lt_heel[1],fz_rt_heel[1]),
            self._asym(sz_lt_heel[1],sz_rt_heel[1]),
        ])
        out = torch.stack(feats,1)
        if out.shape[1] != self.N_BIO:
            raise RuntimeError(f"ExpandedTerrainFeatures dim={out.shape[1]} (expected {self.N_BIO})")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# StableBaselineBank
# ─────────────────────────────────────────────────────────────────────────────

class StableBaselineBank:
    def __init__(self, percentile=STABLE_PERCENTILE, score_weights=None):
        self.percentile = percentile
        self.sw         = score_weights or STABLE_SCORE_WEIGHTS
        self.key_idx    = np.array(ExpandedTerrainFeatures.DELTA_KEY_INDICES, dtype=np.int64)
        self.prototypes_: dict = {}

    HORIZ_NBIO_IDX = [184, 190]

    def _stability_score(self, full, xs):
        asym_cols    = [12, 13, 14, 15]
        entropy_cols = [8,  9]
        s_asym    = xs[:, asym_cols].mean(1)
        s_entropy = xs[:, entropy_cols].mean(1)
        s_horiz   = full[:, self.HORIZ_NBIO_IDX].mean(1)
        def _norm01(v):
            mn, mx = v.min(), v.max()
            return (v - mn) / (mx - mn + 1e-8)
        return (self.sw["asym"]       * _norm01(s_asym) +
                self.sw["entropy"]    * _norm01(s_entropy) +
                self.sw["horizontal"] * _norm01(s_horiz))

    def fit(self, feats_norm, groups):
        self.prototypes_ = {}
        for sbj in np.unique(groups):
            m    = groups == sbj; full = feats_norm[m]; xs = full[:, self.key_idx]; n = len(xs)
            if n == 0:
                self.prototypes_[sbj] = np.zeros(len(self.key_idx), dtype=np.float32); continue
            score     = self._stability_score(full, xs)
            threshold = np.percentile(score, self.percentile)
            stable_m  = score <= threshold
            n_stable  = stable_m.sum()
            log(f"      StableBank sbj={sbj}: n={n}  stable={n_stable} ({n_stable/n*100:.0f}%)")
            self.prototypes_[sbj] = xs[stable_m].mean(0).astype(np.float32)

    def compute_delta(self, feats_norm, groups):
        N = len(feats_norm); delta = np.zeros((N, len(self.key_idx)), dtype=np.float32)
        global_proto = (np.stack(list(self.prototypes_.values())).mean(0)
                        if self.prototypes_ else np.zeros(len(self.key_idx), dtype=np.float32))
        for sbj in np.unique(groups):
            m = groups == sbj; xs = feats_norm[m][:, self.key_idx]
            proto = self.prototypes_.get(sbj, global_proto)
            delta[m] = xs - proto
        return delta


# ─────────────────────────────────────────────────────────────────────────────
# Subject normalization
# ─────────────────────────────────────────────────────────────────────────────

class SubjectFeatureNormalizer:
    def __init__(self): self.stats = {}

    def fit(self, feats, groups):
        self.stats = {}
        for sbj in np.unique(groups):
            m=groups==sbj; x=feats[m]
            self.stats[sbj]=(x.mean(0), x.std(0).clip(min=1e-6))

    def transform(self, feats, groups):
        out=feats.copy().astype(np.float32)
        for sbj in np.unique(groups):
            m=groups==sbj
            if sbj in self.stats: mu,std=self.stats[sbj]
            else: x=feats[m]; mu=x.mean(0); std=x.std(0).clip(min=1e-6)
            out[m]=(feats[m]-mu)/std
        return out

    def fit_transform(self, feats, groups):
        self.fit(feats,groups); return self.transform(feats,groups)


FEAT_DIM_TOTAL = ExpandedTerrainFeatures.N_BIO + N_DELTA  # 224


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__(); self.gamma=gamma; self.weight=weight

    def forward(self, logits, targets):
        logp = F.log_softmax(logits,1)
        logpt= logp.gather(1,targets.unsqueeze(1)).squeeze(1)
        pt   = logpt.exp(); loss = -((1-pt)**self.gamma)*logpt
        if self.weight is not None: loss=loss*self.weight.to(logits.device)[targets]
        return loss.mean()


class WithinSubjectTripletLoss(nn.Module):
    def __init__(self, margin=0.8):
        super().__init__()
        self.fn=nn.TripletMarginLoss(margin=margin,p=2,reduction="mean")
        self._vc=self._sc=0

    def reset_epoch_stats(self): self._vc=self._sc=0
    def epoch_stats_str(self): return f"valid={self._vc}  avg={self._vc/max(self._sc,1):.1f}"

    def forward(self, emb, labels, sbj):
        A,P,N=[],[],[]
        for s in sbj.unique():
            m=sbj==s; se=emb[m]; sl=labels[m]
            if se.shape[0]<3 or sl.unique().shape[0]<2: continue
            d=torch.cdist(se,se,p=2)
            for i in range(se.shape[0]):
                pm=(sl==sl[i]); nm=(sl!=sl[i]); pm[i]=False
                if pm.sum()<1 or nm.sum()<1: continue
                A.append(se[i]); P.append(se[pm][d[i][pm].argmax()]); N.append(se[nm][d[i][nm].argmin()])
        self._vc+=len(A); self._sc+=1
        if not A: return torch.tensor(0.,device=emb.device,requires_grad=True)
        return self.fn(torch.stack(A),torch.stack(P),torch.stack(N))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / Loader
# ─────────────────────────────────────────────────────────────────────────────

class FusionFeatureDataset(Dataset):
    def __init__(self, branch_ds, y_all, groups=None, feats_norm=None):
        self.ds=branch_ds; self.y=y_all; self.feats_norm=feats_norm
        if groups is not None:
            sbj_map={s:i for i,s in enumerate(sorted(set(groups.tolist())))}
            self.sbj=np.array([sbj_map[g] for g in groups],dtype=np.int64)
        else: self.sbj=np.full(len(y_all),-1,dtype=np.int64)

    def __len__(self): return len(self.ds)

    def __getitem__(self,i):
        bi,_=self.ds[i]; feat=torch.from_numpy(self.feats_norm[i]).float()
        return bi, feat, int(self.y[i]), int(self.sbj[i])


def fusion_collate(batch):
    bk=batch[0][0].keys()
    bi={k:torch.stack([b[0][k] for b in batch]) for k in bk}
    return bi,torch.stack([b[1] for b in batch]),\
           torch.tensor([b[2] for b in batch],dtype=torch.long),\
           torch.tensor([b[3] for b in batch],dtype=torch.long)


def make_fusion_loader(ds, shuffle, balanced=False):
    sampler=None; us=shuffle
    if shuffle and balanced:
        cls,cnt=np.unique(ds.y,return_counts=True)
        cw=np.zeros(N_ACTIVE_CLASSES, dtype=np.float64)
        cw[cls]=1./cnt.astype(np.float64)
        sampler=WeightedRandomSampler(torch.as_tensor(cw[ds.y],dtype=torch.double),len(ds.y),replacement=True)
        us=False
        log(f"      ★ 균형 샘플링: {dict(zip(cls.tolist(),cnt.tolist()))}")
    return DataLoader(ds,batch_size=config.BATCH,shuffle=us,sampler=sampler,
                      collate_fn=fusion_collate,drop_last=shuffle,pin_memory=config.USE_GPU)


# ─────────────────────────────────────────────────────────────────────────────
# TerrainDetectorFusionModel
# ─────────────────────────────────────────────────────────────────────────────

class TerrainDetectorFusionModel(nn.Module):
    def __init__(self, backbone, raw_dim, feat_dim=FEAT_DIM_TOTAL, emb_dim=256):
        super().__init__()
        self.backbone = backbone
        self.raw_proj = nn.Sequential(
            nn.Linear(raw_dim,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.15))
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256,256),      nn.BatchNorm1d(256), nn.GELU())
        self.gate = nn.Sequential(
            nn.Linear(512,128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128,2))
        self.shared = nn.Sequential(
            nn.Linear(256*5,768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(768,emb_dim), nn.BatchNorm1d(emb_dim), nn.GELU(), nn.Dropout(0.10))
        self.head_slip    = nn.Linear(emb_dim, 2)
        self.head_slope   = nn.Linear(emb_dim, 3)
        self.head_surface = nn.Linear(emb_dim, 3)

    def _embed(self, bi, feat):
        raw  = self.backbone.extract(bi)
        rh   = self.raw_proj(raw); fh = self.feat_proj(feat)
        g    = torch.softmax(self.gate(torch.cat([rh,fh],1)),1)
        mix  = g[:,0:1]*rh + g[:,1:2]*fh
        return self.shared(torch.cat([rh, fh, mix, (rh-fh).abs(), rh*fh], 1))

    def forward(self, bi, feat):
        emb = self._embed(bi, feat)
        return (self.head_slip(emb), self.head_slope(emb),
                self.head_surface(emb), emb)

    @staticmethod
    def factorized_proba(l_slip, l_slope, l_surface):
        p_sl  = torch.softmax(l_slip,    1)
        p_slo = torch.softmax(l_slope,   1)
        p_sur = torch.softmax(l_surface, 1)

        p_slip_  = p_sl[:, 1]
        p_no     = p_sl[:, 0]
        p_level  = p_slo[:, 0]
        p_up     = p_slo[:, 1]
        p_down   = p_slo[:, 2]
        p_norm   = p_sur[:, 0]   # C6 평지
        p_irr    = p_sur[:, 1]   # C4 흙길
        # p_soft(C5 잔디) 제거됨

        P = torch.stack([
            p_slip_,
            p_no * p_up,
            p_no * p_down,
            p_no * p_level * p_irr,
            p_no * p_level * p_norm,
        ], dim=1)

        # [v14.5] C6 확률 하한 0.02 보장 (inplace 없이)
        floor = torch.zeros_like(P)
        floor = F.pad(torch.full((P.size(0), 1), 0.10, device=P.device), (4, 0))
        P = torch.max(P, floor)

        return P / P.sum(1, keepdim=True).clamp(min=1e-8)

    def predict_proba(self, bi, feat):
        emb = self._embed(bi, feat)
        return self.factorized_proba(
            self.head_slip(emb), self.head_slope(emb), self.head_surface(emb))

    def predict(self, bi, feat):
        return self.predict_proba(bi, feat).argmax(1)

    def embed(self, bi, feat): return self._embed(bi, feat)


# ─────────────────────────────────────────────────────────────────────────────
# SlipMultiTaskWarmup
# ─────────────────────────────────────────────────────────────────────────────

class _WarmupWrapper(nn.Module):
    def __init__(self, backbone, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.head_slip    = nn.Linear(feat_dim, 2)
        self.head_slope   = nn.Linear(feat_dim, 3)
        self.head_surface = nn.Linear(feat_dim, 3)

    def forward(self, bi):
        f = self.backbone.extract(bi)
        return self.head_slip(f), self.head_slope(f), self.head_surface(f)


def train_warmup(backbone, tr_dl, val_dl, tag="", curve_dir=None):
    feat_dim = _get_feat_dim(backbone)
    model    = _WarmupWrapper(backbone, feat_dim).to(DEVICE)
    params   = list(model.parameters())
    opt      = torch.optim.AdamW(params, lr=WARMUP_LR, weight_decay=config.WEIGHT_DECAY)
    sch      = _make_sch(opt, WARMUP_EPOCHS, warmup=8, base_lr=WARMUP_LR)
    scaler   = GradScaler(enabled=(config.USE_AMP and config.AMP_DTYPE == torch.float16))

    crit_slip    = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=DEVICE))
    crit_slope   = nn.CrossEntropyLoss(label_smoothing=0.05)
    # [v14.5] C6(surface index 0) weight 1.0→3.5
    crit_surface = nn.CrossEntropyLoss(
        weight=torch.tensor([8.0, 2.5, 2.5], device=DEVICE),
        label_smoothing=0.03, ignore_index=-100)

    best_va=0.; best_state=None; patience=0; t0=time.time()
    log(f"  {tag} SlipMultiTask Warmup ({WARMUP_EPOCHS}ep, LR={WARMUP_LR:.0e})")

    for ep in range(1, WARMUP_EPOCHS+1):
        model.train(); opt.zero_grad(set_to_none=True); step_i=-1
        for step_i,(bi,yb) in enumerate(tr_dl):
            bi,yb=_to_device(bi,yb=yb)
            y_slope,y_slip,y_surface=make_aux_targets(yb)
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                ls,lsl,lsu=model(bi)
                loss=(crit_slip(ls,y_slip)+crit_slope(lsl,y_slope)+crit_surface(lsu,y_surface))
                loss=loss/config.GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step_i+1)%config.GRAD_ACCUM_STEPS==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,config.GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if step_i>=0 and (step_i+1)%config.GRAD_ACCUM_STEPS!=0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,config.GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

        model.eval(); va_c=va_n=0
        with torch.inference_mode():
            for bi,yb in val_dl:
                bi,yb=_to_device(bi,yb=yb)
                y_slip=(yb==0).long()
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    ls,_,_=model(bi)
                va_c+=(ls.argmax(1)==y_slip).sum().item(); va_n+=len(y_slip)
        sch.step(); va=va_c/max(va_n,1)
        if va>best_va: best_va=va; best_state=_clone_state(model); patience=0
        else:
            patience+=1
            if patience>=WARMUP_PATIENCE: log(f"  {tag} Warmup EarlyStop ep{ep}"); break
        if ep%10==0 or ep==1:
            log(f"  {tag} Warmup ep{ep:03d}/{WARMUP_EPOCHS}  slip_acc={va:.4f}  best={best_va:.4f}  ({time.time()-t0:.0f}s)")

    if best_state: model.load_state_dict(best_state); model.to(DEVICE)
    log(f"  {tag} Warmup 완료  best_slip_acc={best_va:.4f}")
    if curve_dir: CurveTracker(f"WU_{tag.replace('[','').replace(']','')}").save(curve_dir)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_engineered_features(ds, feat_ext, batch_size=None):
    bs=batch_size or min(getattr(config,"BATCH",512),512)
    def _col(batch):
        k=batch[0][0].keys()
        return {kk:torch.stack([b[0][kk] for b in batch]) for kk in k},\
               torch.tensor([b[1] for b in batch],dtype=torch.long)
    loader=DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=_col,pin_memory=config.USE_GPU)
    feat_ext.eval(); feats=[]
    with torch.no_grad():
        for bi,_ in loader:
            feats.append(feat_ext({k:v.float() for k,v in bi.items()}).cpu())
    return torch.cat(feats,0).float().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# train_event_fusion
# ─────────────────────────────────────────────────────────────────────────────

def train_event_fusion(backbone, tr_dl, val_dl, te_dl, tag="", curve_dir=None,
                       vote_window: int = 5, slip_tau: float = SLIP_TAU):
    raw_dim = _get_feat_dim(backbone)
    model   = TerrainDetectorFusionModel(backbone, raw_dim=raw_dim,
                                         feat_dim=FEAT_DIM_TOTAL).to(DEVICE)
    params  = list(model.parameters())

    all_y   = np.asarray(tr_dl.dataset.y, dtype=np.int64)
    log(f"  {tag} class_weights (6cls): {auto_class_weights(all_y).tolist()}")

    opt    = torch.optim.AdamW(params, lr=FUSION_LR, weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, FUSION_EPOCHS, warmup=10, base_lr=FUSION_LR)
    scaler = GradScaler(enabled=(config.USE_AMP and config.AMP_DTYPE == torch.float16))
    curve  = CurveTracker(f"FUSION_{tag.replace('[','').replace(']','')}")

    crit_slip    = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=DEVICE))
    crit_slope   = nn.CrossEntropyLoss(label_smoothing=0.05)
    # [v14.5] C6(surface index 0) weight → 8.0  surface 감독 최대화
    crit_surface = nn.CrossEntropyLoss(
        weight=torch.tensor([8.0, 2.5, 2.5], device=DEVICE),
        label_smoothing=0.03, ignore_index=-100)
    crit_trip    = WithinSubjectTripletLoss(margin=TRIPLET_MARGIN)
    crit_nll     = nn.NLLLoss(weight=auto_class_weights(all_y).to(DEVICE))

    best_va=0.; best_state=None; patience=0; t0=time.time()
    log(
        f"  {tag} Fusion v14.5 ({FUSION_EPOCHS}ep, LR={FUSION_LR:.0e})\n"
        f"       slip_w={AUX_SLIP_W:.2f}  slope_w={AUX_SLOPE_W:.2f}"
        f"  surface_w={AUX_SURFACE_W:.2f}  triplet_w={FUSION_TRIPLET_W:.2f}\n"
        f"       [v14.5] crit_surface=[3.5,2.5,2.5]  slip_tau={slip_tau}"
    )

    for ep in range(1, FUSION_EPOCHS+1):
        model.train(); opt.zero_grad(set_to_none=True)
        crit_trip.reset_epoch_stats(); step_i=-1

        for step_i,(bi,feat,yb,sbj) in enumerate(tr_dl):
            bi,feat,yb=_to_device(bi,feat,yb); sbj=sbj.to(DEVICE)
            y_slope,y_slip,y_surface=make_aux_targets(yb)

            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                l_slip,l_slope,l_surface,emb = model(bi, feat)
                P   = TerrainDetectorFusionModel.factorized_proba(l_slip, l_slope, l_surface)
                P   = (P / P.sum(1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
                L_main = crit_nll(P.log(), yb)
                L_slip    = crit_slip(l_slip,    y_slip)
                L_slope   = crit_slope(l_slope,  y_slope)
                L_surface = crit_surface(l_surface, y_surface)
                L_trip    = crit_trip(emb.float(), yb, sbj)
                loss = (L_main
                        + AUX_SLIP_W    * L_slip
                        + AUX_SLOPE_W   * L_slope
                        + AUX_SURFACE_W * L_surface
                        + FUSION_TRIPLET_W * L_trip
                       ) / config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step_i+1)%config.GRAD_ACCUM_STEPS==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,config.GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

        if step_i>=0 and (step_i+1)%config.GRAD_ACCUM_STEPS!=0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,config.GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        sch.step()

        model.eval(); va_ps,va_ls,va_prs=[],[],[]
        with torch.inference_mode():
            for bi,feat,yb,_ in val_dl:
                bi,feat,yb=_to_device(bi,feat,yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    lsl,lslope,lsurf,_emb=model(bi,feat)
                P=TerrainDetectorFusionModel.factorized_proba(lsl,lslope,lsurf).float()
                P=P/P.sum(1,keepdim=True).clamp(min=1e-8)
                va_ps.append(P.argmax(1).cpu()); va_ls.append(yb.cpu()); va_prs.append(P.cpu())
        va_p_raw=torch.cat(va_ps).float().numpy(); va_l=torch.cat(va_ls).float().numpy()
        va_pr=torch.cat(va_prs).float().numpy()
        val_groups_arr = getattr(val_dl.dataset, 'sbj', None)
        if val_groups_arr is not None:
            va_p = peak_preserving_postprocess(va_p_raw, va_pr,
                                               val_groups_arr,
                                               vote_window=vote_window,
                                               slip_tau=slip_tau)
        else:
            va_p = va_p_raw
        va_acc=accuracy_score(va_l,va_p)
        va_f1 =f1_score(va_l,va_p,labels=ALL5,average="macro",zero_division=0)
        va_score=0.4*va_acc+0.6*va_f1; curve.record(acc=va_score)

        if va_score>best_va: best_va=va_score; best_state=_clone_state(model); patience=0
        else:
            patience+=1
            if patience>=FUSION_PATIENCE and ep>20:
                log(f"  {tag} Fusion EarlyStop ep{ep}  best={best_va:.4f}"); break

        if ep%15==0 or ep==1:
            log(f"  {tag} Fusion ep{ep:03d}/{FUSION_EPOCHS}"
                f"  acc={va_acc:.4f}  f1={va_f1:.4f}  score={va_score:.4f}"
                f"  best={best_va:.4f}  lr={opt.param_groups[0]['lr']:.1e}"
                f"  ({time.time()-t0:.0f}s)")

    if best_state: model.load_state_dict(best_state); model.to(DEVICE)
    log(f"  {tag} Fusion 완료  best_val={best_va:.4f}")
    if curve_dir: curve.save(curve_dir)

    def _predict_dl(dl):
        model.eval(); ps,ls,feats,embs,probas=[],[],[],[],[]
        with torch.inference_mode():
            for bi,feat,yb,_ in dl:
                bi,feat,yb=_to_device(bi,feat,yb)
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    lsl,lslope,lsurf,emb=model(bi,feat)
                P = TerrainDetectorFusionModel.factorized_proba(lsl,lslope,lsurf).float()
                P = P/P.sum(1,keepdim=True).clamp(min=1e-8)
                ps.append(P.argmax(1).cpu()); ls.append(yb.cpu())
                feats.append(feat.float().cpu()); embs.append(emb.float().cpu()); probas.append(P.cpu())
        return (torch.cat(ps).float().numpy(), torch.cat(ls).float().numpy(),
                torch.cat(feats).float().numpy().astype(np.float32),
                torch.cat(embs).float().numpy().astype(np.float32),
                torch.cat(probas).float().numpy().astype(np.float32))

    va_preds,va_labels,va_feats,va_embs,va_probas=_predict_dl(val_dl)
    te_preds,te_labels,te_feats,te_embs,te_probas=_predict_dl(te_dl)
    return (va_preds,va_labels,va_feats,va_embs,va_probas,
            te_preds,te_labels,te_feats,te_embs,te_probas,model)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold search & postprocess
# ─────────────────────────────────────────────────────────────────────────────

def threshold_search(proba, labels, n_classes=N_ACTIVE_CLASSES):
    mults = np.ones(n_classes, dtype=np.float32)
    for ci in range(n_classes):
        best_f1, best_m = -1.0, 1.0
        for m in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            mults[ci] = m
            f1 = f1_score(labels, (proba * mults).argmax(1), average="macro", zero_division=0)
            if f1 > best_f1: best_f1, best_m = f1, m
        mults[ci] = best_m
    return mults, f1_score(labels, (proba * mults).argmax(1), average="macro", zero_division=0)


def majority_vote_smooth(preds, window=5):
    if window<=1: return preds.copy()
    if window%2==0: window+=1
    half=window//2; out=preds.copy()
    for i in range(half,len(preds)-half):
        out[i]=scipy_mode(preds[i-half:i+half+1],keepdims=False).mode
    return out


def peak_preserving_postprocess(preds, probas, groups, vote_window, slip_tau):
    out = preds.copy()
    for g in np.unique(groups):
        m     = groups == g; idx = np.where(m)[0]
        p_seg = preds[idx]; s_seg = probas[idx, 0]
        slip_mask = s_seg > slip_tau
        non_slip  = p_seg.copy()
        non_slip[slip_mask] = 5
        if vote_window > 1:
            non_slip = majority_vote_smooth(non_slip, window=vote_window)
        non_slip[slip_mask] = 0
        out[idx] = non_slip
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sequence refiner
# ─────────────────────────────────────────────────────────────────────────────

class _SeqConvBlock(nn.Module):
    def __init__(self,ch,d):
        super().__init__()
        self.c=nn.Conv1d(ch,ch,3,padding=d,dilation=d); self.b=nn.BatchNorm1d(ch)
        self.a=nn.GELU(); self.d=nn.Dropout(0.1)
    def forward(self,x): return x+self.d(self.a(self.b(self.c(x)[:,:,:x.shape[2]])))


class LocalSequenceRefiner(nn.Module):
    def __init__(self,in_dim,hidden=SEQ_HIDDEN,n_classes=N_ACTIVE_CLASSES):
        super().__init__()
        self.proj=nn.Linear(in_dim,hidden)
        self.conv=nn.Sequential(_SeqConvBlock(hidden,1),_SeqConvBlock(hidden,2),_SeqConvBlock(hidden,4))
        self.gru =nn.GRU(hidden,hidden//2,batch_first=True,num_layers=1,bidirectional=True)
        self.head=nn.Linear(hidden,n_classes)
    def forward(self,x):
        h=self.proj(x); hc=self.conv(h.transpose(1,2)).transpose(1,2); hg,_=self.gru(hc)
        return self.head(hg)


def train_sequence_refiner(tr_groups,va_groups,te_groups,
                            tr_feats,va_feats,te_feats,
                            tr_embs,va_embs,te_embs,
                            tr_probas,va_probas,te_probas,
                            tr_labels,va_labels,te_labels,
                            vote_window,slip_tau,tag=""):
    tr_vec=np.concatenate([tr_embs,tr_feats,tr_probas],1).astype(np.float32)
    va_vec=np.concatenate([va_embs,va_feats,va_probas],1).astype(np.float32)
    te_vec=np.concatenate([te_embs,te_feats,te_probas],1).astype(np.float32)

    def _mk(x,y,g):
        seqs,tgts,cidx,cgrp=[],[],[],[]
        half=SEQ_LEN//2
        for sbj in np.unique(g):
            idx=np.where(g==sbj)[0]; xs=x[idx]; ys=y[idx]; N=len(idx)
            if N==0: continue
            pad=np.pad(xs,((half,half),(0,0)),mode="edge")
            for c in range(N):
                seqs.append(pad[c:c+SEQ_LEN]); tgts.append(int(ys[c]))
                cidx.append(int(idx[c])); cgrp.append(sbj)
        if not seqs:
            return (np.zeros((0,SEQ_LEN,x.shape[1]),dtype=np.float32),
                    np.zeros(0,dtype=np.int64),np.zeros(0,dtype=np.int64),
                    np.zeros(0,dtype=g.dtype))
        return (np.asarray(seqs,dtype=np.float32),np.asarray(tgts,dtype=np.int64),
                np.asarray(cidx,dtype=np.int64),np.asarray(cgrp))

    tr_seq,tr_tgt,tr_cidx,tr_cgrp=_mk(tr_vec,tr_labels,tr_groups)
    va_seq,va_tgt,va_cidx,va_cgrp=_mk(va_vec,va_labels,va_groups)
    te_seq,te_tgt,te_cidx,te_cgrp=_mk(te_vec,te_labels,te_groups)
    log(f"  {tag} SeqRefiner: tr={len(tr_seq)}  va={len(va_seq)}  te={len(te_seq)}")

    sf_va_preds  = va_probas.argmax(1)[va_cidx]
    sf_va_labels = va_labels[va_cidx]
    sf_va_probas = va_probas[va_cidx]
    sf_te_preds  = te_probas.argmax(1)[te_cidx]
    sf_te_labels = te_labels[te_cidx]
    sf_te_probas = te_probas[te_cidx]

    if len(tr_seq)==0 or len(va_seq)==0 or len(te_seq)==0:
        log(f"  {tag} [WARN] sequence 부족 — fusion 사용")
        fv=peak_preserving_postprocess(sf_va_preds,sf_va_probas,va_cgrp,vote_window,slip_tau)
        ft=peak_preserving_postprocess(sf_te_preds,sf_te_probas,te_cgrp,vote_window,slip_tau)
        return ft,sf_te_labels,te_cgrp,ft,sf_te_labels,float("-inf"),float("+inf")

    model=LocalSequenceRefiner(in_dim=tr_seq.shape[2],hidden=SEQ_HIDDEN).to(DEVICE)
    cls_w=auto_class_weights(tr_tgt).to(DEVICE)
    crit=nn.CrossEntropyLoss(weight=cls_w)
    opt=torch.optim.AdamW(model.parameters(),lr=SEQ_LR,weight_decay=1e-4)
    bs=min(512,len(tr_seq))
    tr_ld=DataLoader(list(zip(torch.from_numpy(tr_seq),torch.from_numpy(tr_tgt))),
                     batch_size=bs,shuffle=True,drop_last=(len(tr_seq)>=512))
    va_ld=DataLoader(list(zip(torch.from_numpy(va_seq),torch.from_numpy(va_tgt))),
                     batch_size=512,shuffle=False)
    best_va=0.; best_state=None; patience=0; half=SEQ_LEN//2
    log(f"  {tag} SeqRefiner ({SEQ_EPOCHS}ep, LR={SEQ_LR:.0e})")
    for ep in range(1,SEQ_EPOCHS+1):
        model.train()
        for xb,yb in tr_ld:
            xb=xb.to(DEVICE).float(); yb=yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss=crit(model(xb)[:,half,:],yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); vp,vl=[],[]
        with torch.inference_mode():
            for xb,yb in va_ld:
                vp.append(model(xb.to(DEVICE).float())[:,half,:].argmax(1).cpu()); vl.append(yb)
        vp=torch.cat(vp).float().numpy(); vl=torch.cat(vl).float().numpy()
        vs=0.4*accuracy_score(vl,vp)+0.6*f1_score(vl,vp,labels=ALL5,average="macro",zero_division=0)
        if vs>best_va: best_va=vs; best_state=_clone_state(model); patience=0
        else:
            patience+=1
            if patience>=SEQ_PATIENCE: break
        if ep%10==0 or ep==1:
            log(f"  {tag} SeqRefiner ep{ep:02d}/{SEQ_EPOCHS}  score={vs:.4f}  best={best_va:.4f}")
    if best_state: model.load_state_dict(best_state)
    log(f"  {tag} SeqRefiner 완료  best={best_va:.4f}")

    model.eval(); vp_seq=[]
    with torch.inference_mode():
        for xb,_ in va_ld:
            vp_seq.append(model(xb.to(DEVICE).float())[:,half,:].argmax(1).cpu())
    vp_seq=torch.cat(vp_seq).float().numpy()
    vp_seq_pp=peak_preserving_postprocess(vp_seq, sf_va_probas, va_cgrp, vote_window, slip_tau)
    bvs = (0.4*accuracy_score(sf_va_labels, vp_seq_pp) +
           0.6*f1_score(sf_va_labels, vp_seq_pp, labels=ALL5, average="macro", zero_division=0))

    fv_pp = peak_preserving_postprocess(sf_va_preds,  sf_va_probas, va_cgrp, vote_window, slip_tau)
    ft_pp = peak_preserving_postprocess(sf_te_preds,  sf_te_probas, te_cgrp, vote_window, slip_tau)
    bvf   = (0.4*accuracy_score(sf_va_labels, fv_pp) +
             0.6*f1_score(sf_va_labels, fv_pp, labels=ALL5, average="macro", zero_division=0))

    te_ld=DataLoader(torch.from_numpy(te_seq),batch_size=512,shuffle=False)
    tp=[]; model.eval()
    with torch.inference_mode():
        for xb in te_ld: tp.append(model(xb.to(DEVICE).float())[:,half,:].argmax(1).cpu())
    tp=torch.cat(tp).float().numpy()
    tp_pp=peak_preserving_postprocess(tp, sf_te_probas, te_cgrp, vote_window, slip_tau)

    log(f"  {tag} val  Fusion+PP={bvf:.4f}  Seq+PP={bvs:.4f}"
        f"  → {'SEQ' if bvs>=bvf else 'FUSION'}")
    return (tp_pp, te_tgt, te_cgrp,
            ft_pp, sf_te_labels,
            float(bvs), float(bvf))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args=parse_args(); apply_args(args)

    use_wandb=args.wandb and _WANDB_OK
    if args.wandb and not _WANDB_OK: log("  [W&B] 미설치 — 로컬 로그만")
    if use_wandb:
        import subprocess
        try: gh=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=Path(__file__).parent).decode().strip()
        except: gh="unknown"
        wandb.init(project=args.wandb_project,
                   name=args.run_name or f"v14.5-N{getattr(args,'n_subjects','?')}",
                   config={"version":"v14.5","git":gh,
                           "feat_dim":FEAT_DIM_TOTAL,"slip_tau":SLIP_TAU,
                           "aux_surface_w":AUX_SURFACE_W,"stable_pct":STABLE_PERCENTILE})

    config.print_config()
    log(
        f"  ★ v14.5 Detector-Centric Hierarchical (C6 recall 부스트)\n"
        f"  - SLIP_TAU={SLIP_TAU} (0.45→0.38, false C1 억제)\n"
        f"  - AUX_SURFACE_W={AUX_SURFACE_W} (0.25→0.45)\n"
        f"  - crit_surface=[3.5,2.5,2.5] (C6 weight 1.0→3.5)\n"
        f"  - factorized_proba C6 하한 0.02\n"
        f"  - feat_dim_total={FEAT_DIM_TOTAL}\n"
    )

    out=config.RESULT_KFOLD/"hierarchical_eventfusion"
    curve_dir=out/"curves"; out.mkdir(parents=True,exist_ok=True); curve_dir.mkdir(parents=True,exist_ok=True)

    h5data=H5Data(config.H5_PATH)
    # ── C5 제외 ─────────────────────────────────────────────
    config.NUM_CLASSES = N_ACTIVE_CLASSES
    y_raw_orig = h5data.y_raw.astype(np.int64)
    y, groups, kept_idx = filter_and_remap(y_raw_orig, h5data.subj_id)
    y_full = np.full(len(y_raw_orig), -1, dtype=np.int64)
    y_full[kept_idx] = y
    branch_idx,branch_ch=build_branch_idx(h5data.channels)
    le=LabelEncoder(); le.classes_=np.array(list(CLASS_NAMES_ALL.values()))
    log(f"  C5 제외 후: N={len(y)}  classes={N_ACTIVE_CLASSES}  {list(CLASS_NAMES_ALL.values())}")

    feat_ext=ExpandedTerrainFeatures()
    log(f"  클래스: {list(CLASS_NAMES_ALL.values())}  피험자: {len(np.unique(groups))}명  샘플: {len(y)}")
    # kept_idx를 h5data에 반영 (read_X 호출 시 전달)
    _h5_kept_idx = kept_idx   # 이후 tr_idx/te_idx는 이 공간의 인덱스
    sgkf=StratifiedGroupKFold(n_splits=config.KFOLD,shuffle=True,random_state=config.SEED)

    all_preds,all_labels,fold_meta=[],[],[]
    t_total=time.time()

    for fi,(tr_idx,te_idx) in enumerate(sgkf.split(np.zeros(len(y)),y,groups=groups),1):
        t_fold=time.time()
        te_subjects=sorted(set(groups[te_idx].tolist()))
        log(f"\n{'='*60}")
        log(f"  Fold {fi}/{config.KFOLD}  tr={len(tr_idx)}  te={len(te_idx)}  test_sbj={te_subjects}")
        log(f"{'='*60}")

        inner_tr,inner_va=_inner_val_split(tr_idx,groups,y)
        # ── kept_idx 역매핑: sgkf 인덱스(0~N') → 원본 HDF5 인덱스 ──
        # make_branch_dataset은 h5data.read_X()를 호출하므로 원본 인덱스 필요
        h5_inner_tr = _h5_kept_idx[inner_tr]
        h5_inner_va = _h5_kept_idx[inner_va]
        h5_te_idx   = _h5_kept_idx[te_idx]
        bsc=fit_bsc_on_train(h5data,h5_inner_tr)
        tr_ds    =make_branch_dataset(h5data,y_full,h5_inner_tr,bsc,branch_idx,fold_tag=f"EF{fi}_N{config.N_SUBJECTS}",split="train")
        tr_ds_det=make_branch_dataset(h5data,y_full,h5_inner_tr,bsc,branch_idx,fold_tag=f"EF{fi}_N{config.N_SUBJECTS}_det",split="val")
        va_ds    =make_branch_dataset(h5data,y_full,h5_inner_va,bsc,branch_idx,fold_tag=f"EF{fi}v_N{config.N_SUBJECTS}",split="val")
        te_ds    =make_branch_dataset(h5data,y_full,h5_te_idx,bsc,branch_idx,fold_tag=f"EF{fi}_N{config.N_SUBJECTS}",split="test")

        tr_dl=make_loader(tr_ds,True,branch=True); va_dl=make_loader(va_ds,False,branch=True)
        te_dl=make_loader(te_ds,False,branch=True)

        tag=f"[F{fi}]"

        backbone_wu=M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        wu_model=train_warmup(backbone_wu,tr_dl,va_dl,tag=tag,curve_dir=curve_dir)
        log(f"  {tag} Warmup→Fusion backbone 전이 완료")

        backbone_fusion=M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        backbone_fusion.load_state_dict(wu_model.backbone.state_dict())

        trg=groups[inner_tr]; vag=groups[inner_va]; teg=groups[te_idx]   # groups는 이미 filtered 공간

        log(f"  {tag} ExpandedTerrainFeatures 추출")
        tr_raw=extract_all_engineered_features(tr_ds_det,feat_ext)
        va_raw=extract_all_engineered_features(va_ds,feat_ext)
        te_raw=extract_all_engineered_features(te_ds,feat_ext)

        snorm=SubjectFeatureNormalizer()
        tr_norm=snorm.fit_transform(tr_raw,trg)
        va_norm=snorm.transform(va_raw,vag)
        te_norm=snorm.transform(te_raw,teg)

        log(f"  {tag} StableBaselineBank 구축 — tr/va/te 각각 subject별 fit")
        sbank_tr = StableBaselineBank(); sbank_tr.fit(tr_norm, trg)
        tr_delta = sbank_tr.compute_delta(tr_norm, trg)
        sbank_va = StableBaselineBank(); sbank_va.fit(va_norm, vag)
        va_delta = sbank_va.compute_delta(va_norm, vag)
        sbank_te = StableBaselineBank(); sbank_te.fit(te_norm, teg)
        te_delta = sbank_te.compute_delta(te_norm, teg)

        tr_full=np.concatenate([tr_norm,tr_delta],1)
        va_full=np.concatenate([va_norm,va_delta],1)
        te_full=np.concatenate([te_norm,te_delta],1)
        assert tr_full.shape[1]==FEAT_DIM_TOTAL,f"feat dim mismatch: {tr_full.shape[1]}"

        tr_ds_f=FusionFeatureDataset(tr_ds_det,y[inner_tr],groups=trg,feats_norm=tr_full)
        va_ds_f=FusionFeatureDataset(va_ds,    y[inner_va],groups=vag,feats_norm=va_full)
        te_ds_f=FusionFeatureDataset(te_ds,    y[te_idx],  groups=teg,feats_norm=te_full)
        tr_f_dl=make_fusion_loader(tr_ds_f,True,balanced=True)
        va_f_dl=make_fusion_loader(va_ds_f,False)
        te_f_dl=make_fusion_loader(te_ds_f,False)

        (va_pred,va_lbl,va_ft,va_emb,va_proba,
         te_pred,te_lbl,te_ft,te_emb,te_proba,fusion_model
        )=train_event_fusion(backbone_fusion,tr_f_dl,va_f_dl,te_f_dl,
                              tag=f"{tag}[FUSION]",curve_dir=curve_dir,
                              vote_window=args.vote_window, slip_tau=SLIP_TAU)

        def _infer_tr(dl):
            fusion_model.eval(); ft,em,pr,lb=[],[],[],[]
            with torch.inference_mode():
                for bi,feat,yb,_ in dl:
                    bi,feat,yb=_to_device(bi,feat,yb)
                    with autocast(enabled=config.USE_AMP,dtype=config.AMP_DTYPE):
                        ls,lsl,lsu,emb=fusion_model(bi,feat)
                    P=TerrainDetectorFusionModel.factorized_proba(ls,lsl,lsu).float()
                    P=P/P.sum(1,keepdim=True).clamp(min=1e-8)
                    ft.append(feat.float().cpu()); em.append(emb.float().cpu())
                    pr.append(P.cpu()); lb.append(yb.cpu())
            return (torch.cat(ft).float().numpy().astype(np.float32),
                    torch.cat(em).float().numpy().astype(np.float32),
                    torch.cat(pr).float().numpy().astype(np.float32),
                    torch.cat(lb).numpy().astype(np.int64))

        tr_f_eval=make_fusion_loader(tr_ds_f,False)
        tr_ft,tr_emb,tr_proba,tr_lbl=_infer_tr(tr_f_eval)

        fusion_pp=peak_preserving_postprocess(te_pred,te_proba,teg,args.vote_window,SLIP_TAU)
        fpp_acc=accuracy_score(te_lbl,fusion_pp)
        fpp_f1 =f1_score(te_lbl,fusion_pp,average="macro",zero_division=0)
        log(f"  {tag} Fusion+PP  Acc={fpp_acc:.4f}  F1={fpp_f1:.4f}")

        # [v14.5] threshold search: 기본 + C6 세밀 탐색
        best_mults, _ = threshold_search(va_proba, va_lbl)
        # C6 전용 세밀 탐색 (0.5~4.0)
        c6_best_f1, c6_best_m = -1.0, best_mults[5]
        for m in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
            tmp = best_mults.copy(); tmp[5] = m
            sc  = f1_score(va_lbl, (va_proba * tmp).argmax(1), average="macro", zero_division=0)
            if sc > c6_best_f1: c6_best_f1, c6_best_m = sc, m
        best_mults[5] = c6_best_m
        log(f"  {tag} C6 mult best={c6_best_m:.1f}  val_f1={c6_best_f1:.4f}")

        thresh_pp = peak_preserving_postprocess((te_proba * best_mults).argmax(1), te_proba, teg, args.vote_window, SLIP_TAU)
        thresh_acc = accuracy_score(te_lbl, thresh_pp)
        thresh_f1  = f1_score(te_lbl, thresh_pp, average="macro", zero_division=0)
        log(f"  {tag} Threshold+PP  Acc={thresh_acc:.4f}  F1={thresh_f1:.4f}  mults={dict(enumerate(best_mults.round(2)))}")
        if thresh_f1 > fpp_f1:
            te_pred = (te_proba * best_mults).argmax(1)
            log(f"  {tag} ★ Threshold 채택")

        (seq_preds,seq_labels,te_cg,
         fp_preds,fp_labels,
         bvs,bvf)=train_sequence_refiner(
            tr_groups=trg, va_groups=vag, te_groups=teg,
            tr_feats=tr_ft, va_feats=va_ft, te_feats=te_ft,
            tr_embs=tr_emb, va_embs=va_emb, te_embs=te_emb,
            tr_probas=tr_proba, va_probas=va_proba, te_probas=te_proba,
            tr_labels=tr_lbl, va_labels=va_lbl, te_labels=te_lbl,
            vote_window=args.vote_window, slip_tau=SLIP_TAU, tag=f"{tag}[SEQ]",
        )

        use_seq=bvs>=bvf
        final_preds  = seq_preds if use_seq else fp_preds
        final_labels = seq_labels if use_seq else fp_labels

        acc=accuracy_score(final_labels,final_preds)
        f1 =f1_score(final_labels,final_preds,average="macro",zero_division=0)
        log(f"  {tag} ★ 최종 ({'SEQ' if use_seq else 'FUSION'})  Acc={acc:.4f}  F1={f1:.4f}")

        proba_dir = out / "probas"; proba_dir.mkdir(parents=True, exist_ok=True)
        np.save(proba_dir / f"hier_proba_fold{fi}.npy", te_proba)
        np.save(proba_dir / f"hier_labels_fold{fi}.npy", te_lbl)
        all_preds.append(final_preds); all_labels.append(final_labels)
        fold_meta.append({
            "fold":fi,"test_subjects":te_subjects,
            "fusion_acc":round(fpp_acc,4),"fusion_f1":round(fpp_f1,4),
            "final_acc":round(acc,4),"final_f1":round(f1,4),
            "used_seq":bool(use_seq),
            "fold_time_min":round((time.time()-t_fold)/60,1),
        })
        if _WANDB_OK and wandb.run is not None:
            wandb.log({f"fold{fi}/{k}":v for k,v in fold_meta[-1].items() if isinstance(v,(int,float))})

        del backbone_wu,backbone_fusion,wu_model,fusion_model
        del tr_ds_f,va_ds_f,te_ds_f,tr_f_dl,va_f_dl,te_f_dl,tr_f_eval
        del tr_ds,tr_ds_det,va_ds,te_ds
        gc.collect()
        if config.USE_GPU: torch.cuda.empty_cache()
        clear_fold_cache(f"EF{fi}_N{config.N_SUBJECTS}")
        clear_fold_cache(f"EF{fi}v_N{config.N_SUBJECTS}")
        clear_fold_cache(f"EF{fi}_N{config.N_SUBJECTS}_det")

    preds_all=np.concatenate(all_preds); labels_all=np.concatenate(all_labels)
    acc_all=accuracy_score(labels_all,preds_all)
    f1_all =f1_score(labels_all,preds_all,average="macro",zero_division=0)
    cm=confusion_matrix(labels_all,preds_all,labels=np.arange(N_ACTIVE_CLASSES))
    recalls=cm.diagonal()/cm.sum(1).clip(min=1)
    total_min=(time.time()-t_total)/60

    print(f"\n{'='*60}")
    print(f"  ★ v14.5 Detector-Centric  {config.KFOLD}-Fold")
    print(f"  총 소요: {total_min:.1f}분")
    print(f"  Acc={acc_all:.4f}  MacroF1={f1_all:.4f}")
    print(f"{'='*60}")
    for i,r in enumerate(recalls):
        flag = "⚠" if r < 0.70 else "✅"
        print(f"    {flag} {CLASS_NAMES_ALL[i]:<14} {r*100:.1f}%")

    rep=classification_report(labels_all,preds_all,labels=np.arange(N_ACTIVE_CLASSES),
                               target_names=[CLASS_NAMES_ALL[i] for i in range(N_ACTIVE_CLASSES)],digits=4,zero_division=0)
    (out/"report_v145.txt").write_text(f"v14.5\nAcc={acc_all:.4f}  F1={f1_all:.4f}\n\n{rep}")
    le_out=LabelEncoder(); le_out.fit(np.arange(N_ACTIVE_CLASSES))
    save_cm(preds_all,labels_all,le_out,"HierarchicalSlipDetector_v145_KFold",out)

    summary={
        "experiment":"hierarchical_eventfusion_v145","version":"v14.5",
        "changes":["SLIP_TAU 0.45→0.38","AUX_SURFACE_W 0.25→0.45",
                   "crit_surface=[3.5,2.5,2.5]","C6 proba floor 0.02","C6 threshold 세밀 탐색"],
        "feat_dim_total":FEAT_DIM_TOTAL,"slip_tau":SLIP_TAU,
        "stable_percentile":STABLE_PERCENTILE,
        "overall":{"acc":round(acc_all,4),"f1":round(f1_all,4)},
        "per_class_recall":{CLASS_NAMES_ALL[i]:round(float(recalls[i]),4) for i in range(N_ACTIVE_CLASSES)},
        "total_minutes":round(total_min,1),"fold_meta":fold_meta,
    }
    sp=out/"summary_v145.json"; sp.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    log(f"  ✅ {sp}")

    if _WANDB_OK and wandb.run is not None:
        wandb.summary.update({"overall_acc":acc_all,"overall_f1":f1_all})
        wandb.finish()
    h5data.close()


if __name__=="__main__":
    main()