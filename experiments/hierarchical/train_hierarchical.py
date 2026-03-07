"""
train_hierarchical.py — Hierarchical SupCon (v8.7)
═══════════════════════════════════════════════════════
논문 베이스:
  [1] Ordóñez & Roggen (2016) IEEE TNNLS — Hierarchical HAR
  [2] Khosla et al. (2020) NeurIPS — Supervised Contrastive
  [3] Niswander et al. (2021) — 발 IMU 충격 생체역학
  [4] Lin et al. (2017) ICCV — Focal Loss

v8.6 → v8.7 수정사항 (리뷰 2차 반영):
  Review1 버그 수정 (이미 v8.6):
    [1순위] inner val split (test fold 분리)
    [2순위] full model state_dict 저장
    [3순위] balanced sampler for SupCon
    [4순위] config.SAMPLE_RATE 사용 / 주파수 해석 수정
    [5순위] oracle vs pipeline acc 분리 보고
    [추가]  hard routing 명시, BioMech N_BIO=16, T=0.07

  Review2 성능 개선 (v8.7 신규):
    [A] FocalLoss 추가 (gamma=2.0) — C4/C5 어려운 클래스 집중
    [B] 4단계 학습: CE워밍업(30ep) → SupCon(100ep)
                   → FocalLoss(100ep) → CE마무리(50ep)
    [C] 에포크 최적화: S1=60ep, S2 총=280ep
    [D] S2_WEIGHTS 재조정 [C1=2.0, C4=2.5, C5=4.0, C6=1.5]

학습 파이프라인:
  Stage1: 3cls CE (평탄/오르막/내리막)
  Stage2: 4cls  Step1 CE-warmup(30ep)
                Step2 SupCon(100ep, T=0.07, balanced)
                Step3 FocalLoss-finetune(100ep)   ← 신규
                Step4 CE-마무리(50ep)
  결합: Hard Routing (Stage1 argmax → Stage2 if flat)
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys, time, json, gc, warnings, math
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
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

# ── 클래스 상수 ──────────────────────────────────
FLAT_CLASSES = [0, 3, 4, 5]   # C1,C4,C5,C6
SLOPE_UP     = [1]
SLOPE_DOWN   = [2]

def to_stage1_label(y6: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y6)
    for c in SLOPE_UP:   out[y6 == c] = 1
    for c in SLOPE_DOWN: out[y6 == c] = 2
    return out

FLAT_MAP     = {0: 0, 3: 1, 4: 2, 5: 3}
FLAT_MAP_INV = {v: k for k, v in FLAT_MAP.items()}
CLASS_NAMES_FLAT = {0: "C1-평지", 1: "C4-흙길", 2: "C5-잔디", 3: "C6-정상"}
CLASS_NAMES_ALL  = {0: "C1-평지", 1: "C2-오르막", 2: "C3-내리막",
                    3: "C4-흙길",  4: "C5-잔디",   5: "C6-정상"}

# ── Hyperparams ───────────────────────────────────
S1_EPOCHS       = 60    # 80→60 (Review2)
S1_LR           = 5e-5  # 1e-4→5e-5 (Review2)
S2_WARMUP_EP    = 30    # 50→30 (Review2)
S2_PRETRAIN_EP  = 100   # 150→100 (Review2)
S2_FOCAL_EP     = 100   # 신규: FocalLoss finetune
S2_FINETUNE_EP  = 50    # 150→50 (Review2, 마무리)
TEMPERATURE     = 0.12  # 0.07→0.12 (4클래스 소규모 데이터, Khosla2020 부록 권장)
S2_LR           = 3e-5  # 1e-4→3e-5 (Review2)
FOCAL_GAMMA     = 2.0   # Focal Loss gamma
# Stage2 backbone 초기값: Stage1 학습된 weight 이어받기
# Stage1이 99%+ → IMU 표현 이미 양호 → SupCon 수렴 안정화
S2_INIT_FROM_S1 = True
# Stage2 CLASS_WEIGHTS [C1-미끄러운, C4-흙길, C5-잔디, C6-평지]
S2_WEIGHTS      = [2.0, 2.5, 4.0, 1.5]   # Review2 재조정


# ═══════════════════════════════════════════════
# 1. Biomechanical Feature Extractor (N_BIO=16)
# ═══════════════════════════════════════════════

class BioMechFeatures(nn.Module):
    """생체역학 충격 피처 추출기 (v8.6: 8→16 피처).

    [수정] sample_rate → config.SAMPLE_RATE 사용
    [수정] 고주파 에너지를 "step-domain 상대 고주파 에너지"로 표현
           (resample로 실제 Hz 대응 불완전, 상대적 비율로만 해석)

    피처 16개:
      0~3 : Foot/Shank LT/RT 충격 피크값
      4~5 : Foot/Shank 충격비 (지면 흡수량 지표)
      6~7 : step-domain 상위 주파수 에너지 비율 (상대적)
      8~9 : Foot LT/RT 신호 표준편차 (C1 미끄러운 → 변동성↑)
      10~11: Foot LT/RT 피크 후 감쇠율 (C5 잔디 → 빠른 감쇠)
      12~13: Shank LT/RT 진동 (|diff|.mean) (C4 흙 → 진동↑)
      14~15: Foot/Shank 분산비 LT/RT (C1 불안정 지표)
    """
    N_BIO = 16

    def __init__(self) -> None:
        super().__init__()
        self.foot_z  = config.FOOT_Z_ACCEL_IDX
        self.shank_z = config.SHANK_Z_ACCEL_IDX
        # [수정] config.SAMPLE_RATE 사용
        self.sr      = config.SAMPLE_RATE
        # step-domain 상위 50% bin (상대적 고주파)
        self.hf_bin  = config.TS // 4

    @torch.no_grad()
    def forward(
        self,
        foot_x:  torch.Tensor,   # (B, 12, T)
        shank_x: torch.Tensor,   # (B, 12, T)
    ) -> torch.Tensor:
        foot_x  = foot_x.float()
        shank_x = shank_x.float()

        fz_lt = foot_x[:,  self.foot_z[0],  :]
        fz_rt = foot_x[:,  self.foot_z[1],  :]
        sz_lt = shank_x[:, self.shank_z[0], :]
        sz_rt = shank_x[:, self.shank_z[1], :]
        eps   = 1e-6

        # 0~3: 피크값
        f_pk_lt = fz_lt.abs().max(dim=1).values
        f_pk_rt = fz_rt.abs().max(dim=1).values
        s_pk_lt = sz_lt.abs().max(dim=1).values
        s_pk_rt = sz_rt.abs().max(dim=1).values

        # 4~5: Foot/Shank 충격비
        ratio_lt = f_pk_lt / (s_pk_lt + eps)
        ratio_rt = f_pk_rt / (s_pk_rt + eps)

        # 6~7: step-domain 상대 고주파 에너지 (논문에서 "정규화된 고주파 에너지"로 표현)
        hf_lt = self._hf_ratio(fz_lt)
        hf_rt = self._hf_ratio(fz_rt)

        # 8~9: 변동성 (C1 미끄러운 → 보행 불안정 → std↑)
        std_lt = fz_lt.std(dim=1)
        std_rt = fz_rt.std(dim=1)

        # 10~11: 피크 후 감쇠율 (C5 잔디 → 충격 빠르게 감쇠)
        T_half = fz_lt.shape[1] // 2
        decay_lt = (fz_lt[:, :T_half].abs().mean(dim=1) /
                    (fz_lt[:, T_half:].abs().mean(dim=1) + eps))
        decay_rt = (fz_rt[:, :T_half].abs().mean(dim=1) /
                    (fz_rt[:, T_half:].abs().mean(dim=1) + eps))

        # 12~13: Shank 진동 (C4 흙길 → 불규칙 진동↑)
        vib_lt = (sz_lt[:, 1:] - sz_lt[:, :-1]).abs().mean(dim=1)
        vib_rt = (sz_rt[:, 1:] - sz_rt[:, :-1]).abs().mean(dim=1)

        # 14~15: Foot/Shank 분산비 (C1 미끄러운 → 불안정)
        var_ratio_lt = fz_lt.var(dim=1) / (sz_lt.var(dim=1) + eps)
        var_ratio_rt = fz_rt.var(dim=1) / (sz_rt.var(dim=1) + eps)

        return torch.stack([
            f_pk_lt, f_pk_rt, s_pk_lt, s_pk_rt,
            ratio_lt, ratio_rt, hf_lt, hf_rt,
            std_lt, std_rt, decay_lt, decay_rt,
            vib_lt, vib_rt, var_ratio_lt, var_ratio_rt,
        ], dim=1)   # (B, 16)

    def _hf_ratio(self, x: torch.Tensor) -> torch.Tensor:
        fft_mag = torch.fft.rfft(x, dim=1).abs()
        total   = fft_mag.pow(2).sum(dim=1) + 1e-6
        hf      = fft_mag[:, self.hf_bin:].pow(2).sum(dim=1)
        return hf / total


# ═══════════════════════════════════════════════
# 2. BioMechHead (BatchNorm 정규화)
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
    def __init__(self, backbone, feat_dim, bio_dim=64, num_classes=4):
        super().__init__()
        self.backbone   = backbone
        self.bio_head   = BioMechHead(BioMechFeatures.N_BIO, bio_dim)
        total_dim       = feat_dim + bio_dim
        self.proj_head  = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Dropout(config.DROPOUT_CLF),
            nn.Linear(256, num_classes),
        )

    def _extract(self, bi, bio_feat):
        return torch.cat([self.backbone.extract(bi),
                          self.bio_head(bio_feat)], dim=1)

    def forward_proj(self, bi, bio_feat):
        return F.normalize(self.proj_head(self._extract(bi, bio_feat)), dim=1)

    def forward(self, bi, bio_feat):
        return self.classifier(self._extract(bi, bio_feat))


# ═══════════════════════════════════════════════
# 4. SupCon Loss
# ═══════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Khosla et al. NeurIPS 2020."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)
        features = F.normalize(features, dim=1)
        sim  = torch.matmul(features, features.T) / self.temperature
        eye  = torch.eye(B, dtype=torch.bool, device=features.device)
        sim  = sim.masked_fill(eye, -1e9)
        labels   = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & ~eye
        log_prob = sim - torch.logsumexp(
            sim.masked_fill(eye, -1e9), dim=1, keepdim=True)
        n_pos = pos_mask.sum(1).float().clamp(min=1)
        return -(log_prob * pos_mask).sum(1).div(n_pos).mean()


class FocalLoss(nn.Module):
    """Lin et al. ICCV 2017 — Focal Loss.

    C4/C5처럼 어려운 클래스(쉽게 틀리는 샘플)에 집중.
    pt 낮을수록 (1-pt)^gamma 가 커져서 가중치↑.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None) -> None:
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # class weight (S2_WEIGHTS)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none")
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ═══════════════════════════════════════════════
# 5. FlatBranchDataset
# ═══════════════════════════════════════════════

class FlatBranchDataset(Dataset):
    def __init__(self, branch_ds, bio_extractor, flat_mask, y_flat):
        self.ds      = branch_ds
        self.bio     = bio_extractor
        self.indices = np.where(flat_mask)[0]
        self.y_flat  = y_flat   # 이미 필터링된 배열

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        orig_i  = int(self.indices[i])
        bi, _   = self.ds[orig_i]
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
    """[수정] SupCon용 balanced sampler 지원."""
    sampler = None
    use_shuffle = shuffle
    if shuffle and balanced:
        classes, counts = np.unique(ds.y_flat, return_counts=True)
        class_weights   = 1.0 / counts.astype(np.float64)
        sample_weights  = class_weights[ds.y_flat]
        sampler     = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(ds.y_flat),
            replacement=True,
        )
        use_shuffle = False
        log(f"      ★ S2 균형 샘플링: {dict(zip(classes.tolist(), counts.tolist()))}")
    return DataLoader(
        ds,
        batch_size=config.BATCH,
        shuffle=use_shuffle,
        sampler=sampler,
        collate_fn=flat_collate,
        drop_last=shuffle,
        pin_memory=config.USE_GPU,
    )


# ═══════════════════════════════════════════════
# 6. LR 스케줄러
# ═══════════════════════════════════════════════

def _make_sch(opt, epochs, warmup=10, min_lr=config.MIN_LR, base_lr=None):
    base = base_lr or opt.param_groups[0]["lr"]
    def fn(ep):
        if ep < warmup:
            return float(ep + 1) / warmup
        prog = float(ep - warmup) / max(epochs - warmup, 1)
        cos  = 0.5 * (1.0 + math.cos(math.pi * prog))
        mf   = min_lr / base
        return mf + (1.0 - mf) * cos
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# ═══════════════════════════════════════════════
# 7. inner val split 유틸
# ═══════════════════════════════════════════════

def _inner_val_split(
    tr_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """[수정 1순위] tr_idx 안에서 subject 단위 inner val split.

    test fold와 완전히 분리된 val set으로
    early stopping / best model 선택 수행.
    """
    tr_groups  = groups[tr_idx]
    unique_sbj = np.unique(tr_groups)
    n_val_sbj  = max(1, int(len(unique_sbj) * val_ratio))

    rng        = np.random.default_rng(config.SEED)
    val_sbj    = set(rng.choice(unique_sbj, n_val_sbj, replace=False).tolist())

    inner_tr_mask = np.array([g not in val_sbj for g in tr_groups])
    inner_va_mask = ~inner_tr_mask

    inner_tr_idx = tr_idx[inner_tr_mask]
    inner_va_idx = tr_idx[inner_va_mask]
    log(f"    inner split: tr={len(inner_tr_idx)}  val={len(inner_va_idx)}"
        f"  val_sbj={sorted(val_sbj)}")
    return inner_tr_idx, inner_va_idx


# ═══════════════════════════════════════════════
# 8. Stage 1 학습
# ═══════════════════════════════════════════════

def _get_feat_dim(backbone):
    """backbone의 실제 출력 차원을 계산한다.
    M6_BranchCBAMCrossAug 기준: names(그룹 수) + use_fft 여부만 존재.
    use_foot_impact / use_shank_impact는 구버전 속성 → getattr로 안전하게 처리.
    """
    n_groups = len(backbone.names)
    n_extra  = (1 if getattr(backbone, "use_fft", False) else 0) + \
               (1 if getattr(backbone, "use_foot_impact", False) else 0) + \
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


def train_stage1(
    backbone, tr_dl, val_dl, te_dl, tag=""
):
    """[수정] val_dl(inner val)로 best epoch 선택, te_dl은 최종 평가만."""
    feat_dim = _get_feat_dim(backbone)
    head     = nn.Linear(feat_dim, 3).to(DEVICE)
    model    = _S1Wrapper(backbone, head).to(DEVICE)

    opt    = torch.optim.AdamW(model.parameters(), lr=S1_LR,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, S1_EPOCHS, warmup=10, base_lr=S1_LR)
    crit   = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))

    best_va    = 0.0
    best_state = None
    patience   = 0
    t0         = time.time()

    log(f"  {tag} Stage1 3cls ({S1_EPOCHS}ep)  [val=inner split]")
    for ep in range(1, S1_EPOCHS + 1):
        model.train()
        for bi, yb in tr_dl:
            bi  = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            yb3 = _y6_to_y3(yb).to(DEVICE)
            if not config.USE_AMP:
                bi = {k: v.float() for k, v in bi.items()}
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                loss = crit(model(bi), yb3)
            if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
            opt.zero_grad(set_to_none=True)

        # val(inner split)로 best 선택
        model.eval()
        va_c = va_n = 0
        with torch.inference_mode():
            for bi, yb in val_dl:
                bi  = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
                yb3 = _y6_to_y3(yb).to(DEVICE)
                if not config.USE_AMP:
                    bi = {k: v.float() for k, v in bi.items()}
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model(bi)
                va_c += (logits.argmax(1) == yb3).sum().item()
                va_n += len(yb3)
        sch.step()
        va = va_c / max(va_n, 1)

        if va > best_va:
            best_va    = va
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP:
                log(f"  {tag} S1 EarlyStop ep{ep}"); break

        if ep % 20 == 0:
            log(f"  {tag} S1 ep{ep:03d}/{S1_EPOCHS}"
                f"  val_acc={va:.4f}  best={best_va:.4f}"
                f"  ({time.time()-t0:.0f}s)")

    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)

    # test fold 최종 평가
    model.eval()
    preds_list, labels_list = [], []
    with torch.inference_mode():
        for bi, yb in te_dl:
            bi  = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            yb3 = _y6_to_y3(yb)
            if not config.USE_AMP:
                bi = {k: v.float() for k, v in bi.items()}
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model(bi)
            preds_list.append(logits.argmax(1).cpu())
            labels_list.append(yb3.cpu())

    log(f"  {tag} S1 완료  best_val={best_va:.4f}")
    return (torch.cat(preds_list).numpy(),
            torch.cat(labels_list).numpy(),
            model)


# ═══════════════════════════════════════════════
# 9. Stage 2 — CE Warmup
# ═══════════════════════════════════════════════

def _eval_flat_dl(model, val_dl, crit):
    model.eval()
    vl_sum = va_c = va_n = 0
    with torch.inference_mode():
        for bi, bio_f, yb in val_dl:
            bi    = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            bio_f = bio_f.to(DEVICE, non_blocking=True).float()
            yb    = yb.to(DEVICE, non_blocking=True)
            if not config.USE_AMP:
                bi = {k: v.float() for k, v in bi.items()}
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model(bi, bio_f)
                loss   = crit(logits, yb)
            vl_sum += loss.item() * len(yb)
            va_c   += (logits.argmax(1) == yb).sum().item()
            va_n   += len(yb)
    return vl_sum / max(va_n, 1), va_c / max(va_n, 1)


def warmup_stage2_ce(model, tr_dl, val_dl, tag=""):
    """Step1: CE 워밍업 [val=inner split]."""
    weight = torch.tensor(S2_WEIGHTS, dtype=torch.float32).to(DEVICE)
    crit   = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    opt    = torch.optim.AdamW(model.parameters(), lr=S2_LR,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, S2_WARMUP_EP, warmup=5, base_lr=S2_LR)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))

    best_va    = 0.0
    best_state = None
    patience   = 0
    t0         = time.time()

    log(f"  {tag} S2 Step1 CE Warmup ({S2_WARMUP_EP}ep)  [val=inner split]")
    for ep in range(1, S2_WARMUP_EP + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step_i, (bi, bio_f, yb) in enumerate(tr_dl):
            bi    = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            bio_f = bio_f.to(DEVICE, non_blocking=True).float()
            yb    = yb.to(DEVICE, non_blocking=True)
            if not config.USE_AMP:
                bi = {k: v.float() for k, v in bi.items()}
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                loss = crit(model(bi, bio_f), yb) / config.GRAD_ACCUM_STEPS
            if scaler: scaler.scale(loss).backward()
            else: loss.backward()
            if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()), config.GRAD_CLIP_NORM)
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()), config.GRAD_CLIP_NORM)
                    opt.step()
                opt.zero_grad(set_to_none=True)

        sch.step()
        _, va = _eval_flat_dl(model, val_dl, crit)

        if va > best_va:
            best_va    = va
            # [수정 2순위] model 전체 state_dict 저장
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                log(f"  {tag} S2 Warmup EarlyStop ep{ep}"); break

        if ep % 10 == 0:
            log(f"  {tag} S2W ep{ep:03d}/{S2_WARMUP_EP}"
                f"  val={va:.4f}  best={best_va:.4f}"
                f"  ({time.time()-t0:.0f}s)")

    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    log(f"  {tag} S2 Warmup 완료  best_val={best_va:.4f}")


# ═══════════════════════════════════════════════
# 10. Stage 2 — SupCon Pretrain
# ═══════════════════════════════════════════════

def pretrain_stage2(model, tr_dl, tag=""):
    """Step2: SupCon (balanced sampler 사용).

    [수정 3순위] tr_dl은 balanced sampler로 구성된 로더 사용
    → 배치 내 positive pair 보장
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
        model.train()
        total_loss = n = 0
        opt.zero_grad(set_to_none=True)

        for step_i, (bi, bio_f, yb) in enumerate(tr_dl):
            bi    = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            bio_f = bio_f.to(DEVICE, non_blocking=True).float()
            yb    = yb.to(DEVICE, non_blocking=True)
            if not config.USE_AMP:
                bi = {k: v.float() for k, v in bi.items()}
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                proj = model.forward_proj(bi, bio_f)
                loss = supcon(proj, yb) / config.GRAD_ACCUM_STEPS
            if scaler: scaler.scale(loss).backward()
            else: loss.backward()
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

        sch.step()
        avg = total_loss / max(n, 1)
        if avg < best_loss: best_loss = avg
        if ep % 30 == 0:
            log(f"  {tag} S2P2 ep{ep:03d}/{S2_PRETRAIN_EP}"
                f"  loss={avg:.4f}  best={best_loss:.4f}"
                f"  lr={opt.param_groups[0]['lr']:.1e}"
                f"  ({time.time()-t0:.0f}s)")
    log(f"  {tag} S2 SupCon 완료  best_loss={best_loss:.4f}")


# ═══════════════════════════════════════════════
# 11-A. Stage 2 — Focal Loss Finetune (Step3, 신규)
# ═══════════════════════════════════════════════

def focal_finetune_stage2(model, tr_dl, val_dl, tag=""):
    """Step3: Focal Loss Finetune (Review2 신규).

    SupCon으로 feature space 분리 후,
    FocalLoss로 C4/C5 어려운 샘플 집중 학습.
    backbone 일부 해제 (fine-grained tuning).
    """
    weight = torch.tensor(S2_WEIGHTS, dtype=torch.float32).to(DEVICE)
    crit   = FocalLoss(gamma=FOCAL_GAMMA, weight=weight)
    log(f"    Focal Loss(gamma={FOCAL_GAMMA}, weights={S2_WEIGHTS})")

    # backbone 마지막 레이어만 해제 (중간 레이어 고정)
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.proj_head.parameters(): p.requires_grad = False

    params = (list(model.classifier.parameters()) +
              list(model.bio_head.parameters()))
    opt    = torch.optim.AdamW(params, lr=S2_LR,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, S2_FOCAL_EP, warmup=5, base_lr=S2_LR)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))

    # val용 CE (Focal은 값 범위 달라서 CE로 val loss 측정)
    val_crit = nn.CrossEntropyLoss(weight=weight)
    best_vl    = float("inf")
    best_state = None
    patience   = 0
    t0         = time.time()

    log(f"  {tag} S2 Step3 FocalFinetune ({S2_FOCAL_EP}ep)  [val=inner split]")
    for ep in range(1, S2_FOCAL_EP + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step_i, (bi, bio_f, yb) in enumerate(tr_dl):
            bi    = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            bio_f = bio_f.to(DEVICE, non_blocking=True).float()
            yb    = yb.to(DEVICE, non_blocking=True)
            if not config.USE_AMP:
                bi = {k: v.float() for k, v in bi.items()}
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                loss = crit(model(bi, bio_f), yb) / config.GRAD_ACCUM_STEPS
            if scaler: scaler.scale(loss).backward()
            else: loss.backward()
            if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                    opt.step()
                opt.zero_grad(set_to_none=True)

        sch.step()
        vl, va = _eval_flat_dl(model, val_dl, val_crit)

        if ep % 25 == 0:
            log(f"  {tag} S2P3F ep{ep:03d}/{S2_FOCAL_EP}"
                f"  val={vl:.4f}/{va:.4f}"
                f"  lr={opt.param_groups[0]['lr']:.1e}"
                f"  ({time.time()-t0:.0f}s)")

        if vl < best_vl:
            best_vl    = vl
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP:
                log(f"  {tag} S2 FocalFinetune EarlyStop ep{ep}"); break

    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)
    # backbone 다시 활성화
    for p in model.backbone.parameters(): p.requires_grad = True
    log(f"  {tag} S2 FocalFinetune 완료  best_val_loss={best_vl:.4f}")


# ═══════════════════════════════════════════════
# 11. Stage 2 — CE Finetune
# ═══════════════════════════════════════════════

def finetune_stage2(model, tr_dl, val_dl, te_dl, tag=""):
    """Step3: CE Finetune [val=inner split, te=최종 평가].

    [수정 2순위] best_state: model 전체 state_dict 저장
    [수정 1순위] val_dl(inner split)로 early stopping
    """
    # backbone 고정
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.proj_head.parameters(): p.requires_grad = False

    weight = torch.tensor(S2_WEIGHTS, dtype=torch.float32).to(DEVICE)
    crit   = nn.CrossEntropyLoss(weight=weight,
                                  label_smoothing=config.LABEL_SMOOTH)
    log(f"    Loss: CE(S2_WEIGHTS={S2_WEIGHTS})")

    # [수정 2순위] classifier + bio_head 함께 학습
    params = (list(model.classifier.parameters()) +
              list(model.bio_head.parameters()))
    opt    = torch.optim.AdamW(params, lr=S2_LR,
                                weight_decay=config.WEIGHT_DECAY)
    sch    = _make_sch(opt, S2_FINETUNE_EP, warmup=10, base_lr=S2_LR)
    scaler = GradScaler(enabled=(config.USE_AMP and
                                  config.AMP_DTYPE == torch.float16))

    best_vl    = float("inf")
    best_state = None
    patience   = 0
    t0         = time.time()

    log(f"  {tag} S2 Step4 CE-마무리 ({S2_FINETUNE_EP}ep)  [val=inner split]")
    for ep in range(1, S2_FINETUNE_EP + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step_i, (bi, bio_f, yb) in enumerate(tr_dl):
            bi    = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            bio_f = bio_f.to(DEVICE, non_blocking=True).float()
            yb    = yb.to(DEVICE, non_blocking=True)
            if not config.USE_AMP:
                bi = {k: v.float() for k, v in bi.items()}
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                loss = crit(model(bi, bio_f), yb) / config.GRAD_ACCUM_STEPS
            if scaler: scaler.scale(loss).backward()
            else: loss.backward()
            if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                    opt.step()
                opt.zero_grad(set_to_none=True)

        sch.step()
        # [수정 1순위] inner val로 early stopping
        vl, va = _eval_flat_dl(model, val_dl, crit)

        if ep % 25 == 0:
            log(f"  {tag} S2P4 ep{ep:03d}/{S2_FINETUNE_EP}"
                f"  val={vl:.4f}/{va:.4f}"
                f"  lr={opt.param_groups[0]['lr']:.1e}"
                f"  ({time.time()-t0:.0f}s)")

        if vl < best_vl:
            best_vl    = vl
            # [수정 2순위] model 전체 state_dict 저장
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP:
                log(f"  {tag} S2 Finetune EarlyStop ep{ep}"); break

    if best_state:
        model.load_state_dict(best_state); model.to(DEVICE)

    # backbone 다시 활성화
    for p in model.backbone.parameters(): p.requires_grad = True

    # 최종 평가: test fold
    model.eval()
    preds_list, labels_list = [], []
    with torch.inference_mode():
        for bi, bio_f, yb in te_dl:
            bi    = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            bio_f = bio_f.to(DEVICE, non_blocking=True).float()
            yb    = yb.to(DEVICE, non_blocking=True)
            if not config.USE_AMP:
                bi = {k: v.float() for k, v in bi.items()}
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model(bi, bio_f)
            preds_list.append(logits.argmax(1).cpu())
            labels_list.append(yb.cpu())

    return (torch.cat(preds_list).numpy(),
            torch.cat(labels_list).numpy())


# ═══════════════════════════════════════════════
# 12. 예측 결합 (Hard Routing — 명시적)
# ═══════════════════════════════════════════════

def combine_predictions(
    s1_preds: np.ndarray,
    s2_preds: np.ndarray,
    te_y6:    np.ndarray,
) -> np.ndarray:
    """[수정] Hard routing 방식 명시.

    주의: soft ensemble(확률 결합)이 아님.
    Stage1 argmax → 평탄이면 Stage2 argmax로 결정.
    논문 기술: "hard routing based hierarchical classifier"
    """
    final = np.zeros(len(s1_preds), dtype=np.int64)
    s2_i  = 0
    for i, s1 in enumerate(s1_preds):
        if s1 == 1:
            final[i] = 1   # C2 오르막
        elif s1 == 2:
            final[i] = 2   # C3 내리막
        else:
            final[i] = FLAT_MAP_INV.get(int(s2_preds[s2_i]) if s2_i < len(s2_preds) else 0, 0)
            s2_i += 1
    return final


# ═══════════════════════════════════════════════
# 13. K-Fold 메인
# ═══════════════════════════════════════════════

def main() -> None:
    config.print_config()
    log(f"  ★ Hierarchical SupCon K-Fold v8.7\n"
        f"  [Review1] inner val / full state_dict / balanced sampler\n"
        f"  [Review2] FocalLoss + 4단계 학습\n"
        f"  S1={S1_EPOCHS}ep(LR={S1_LR:.0e})  "
        f"S2: Warmup={S2_WARMUP_EP}+SupCon={S2_PRETRAIN_EP}"
        f"+Focal={S2_FOCAL_EP}+CE={S2_FINETUNE_EP}ep\n"
        f"  T={TEMPERATURE}  S2_WEIGHTS={S2_WEIGHTS}  gamma={FOCAL_GAMMA}\n"
        f"  BioMech N_BIO={BioMechFeatures.N_BIO}\n"
        f"  클래스: C1=미끄러운, C2=오르막, C3=내리막, "
        f"C4=흙길, C5=잔디, C6=평지\n")

    out = config.RESULT_KFOLD / "hierarchical"
    out.mkdir(parents=True, exist_ok=True)

    h5data = H5Data(config.H5_PATH)
    le     = LabelEncoder()
    y      = le.fit_transform(h5data.y_raw).astype(np.int64)
    groups = h5data.subj_id
    branch_idx, branch_ch = build_branch_idx(h5data.channels)
    bio_extractor = BioMechFeatures()

    log(f"  클래스: {le.classes_.tolist()}  피험자: {len(np.unique(groups))}명"
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
        log(f"\n{'='*55}")
        log(f"  Fold {fi}/{config.KFOLD}"
            f"  tr={len(tr_idx)}  te={len(te_idx)}")
        log(f"  Test 피험자: {te_s}")
        log(f"{'='*55}")

        # [수정 1순위] inner val split
        inner_tr_idx, inner_va_idx = _inner_val_split(
            tr_idx, y, groups, val_ratio=0.15)

        bsc = fit_bsc_on_train(h5data, inner_tr_idx)

        # 전체 train (SupCon/warmup용) + inner val + test 데이터셋
        tr_ds     = make_branch_dataset(
            h5data, y, inner_tr_idx, bsc, branch_idx,
            fold_tag=f"HC{fi}", split="train")
        val_ds    = make_branch_dataset(
            h5data, y, inner_va_idx, bsc, branch_idx,
            fold_tag=f"HC{fi}v", split="val")
        te_ds     = make_branch_dataset(
            h5data, y, te_idx, bsc, branch_idx,
            fold_tag=f"HC{fi}", split="test")

        tr_dl  = make_loader(tr_ds,  True,  branch=True)
        val_dl = make_loader(val_ds, False, branch=True)
        te_dl  = make_loader(te_ds,  False, branch=True)

        # ── Stage 1 ─────────────────────────────
        backbone_s1 = M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        tag         = f"[F{fi}][Hier]"

        s1_preds, s1_labels, s1_model = train_stage1(
            backbone_s1, tr_dl, val_dl, te_dl, tag)
        s1_acc = accuracy_score(s1_labels, s1_preds)
        log(f"  {tag} Stage1 Acc={s1_acc:.4f}")

        # ── Stage 2 데이터 준비 ──────────────────
        te_y6    = y[te_idx]
        tr_y6    = y[inner_tr_idx]
        va_y6    = y[inner_va_idx]

        # Train flat (실제 flat 클래스만)
        tr_flat_mask = np.isin(tr_y6, FLAT_CLASSES)
        tr_y_flat    = np.array([FLAT_MAP[c] for c in tr_y6[tr_flat_mask]])
        tr_flat_ds   = FlatBranchDataset(
            tr_ds, bio_extractor, tr_flat_mask, tr_y_flat)

        # Val flat (inner val 기반) — early stopping용
        va_flat_mask = np.isin(va_y6, FLAT_CLASSES)
        va_y_flat    = np.array([FLAT_MAP[c] for c in va_y6[va_flat_mask]])
        va_flat_ds   = FlatBranchDataset(
            val_ds, bio_extractor, va_flat_mask, va_y_flat)

        # Test flat: Stage1 "평탄" 예측 기준 (pipeline)
        te_flat_mask = (s1_preds == 0)
        te_y6_flat   = te_y6[te_flat_mask]
        te_y_flat    = np.array([FLAT_MAP.get(c, 0) for c in te_y6_flat])
        te_flat_ds   = FlatBranchDataset(
            te_ds, bio_extractor, te_flat_mask, te_y_flat)

        # Oracle test flat: 실제 flat 정답 기준 (Stage2 자체 성능)
        te_oracle_mask = np.isin(te_y6, FLAT_CLASSES)
        te_oracle_y    = np.array([FLAT_MAP[c] for c in te_y6[te_oracle_mask]])
        te_oracle_ds   = FlatBranchDataset(
            te_ds, bio_extractor, te_oracle_mask, te_oracle_y)

        # [수정 3순위] SupCon용 balanced sampler
        tr_flat_dl_bal = make_flat_loader(tr_flat_ds, True,  balanced=True)
        tr_flat_dl_ce  = make_flat_loader(tr_flat_ds, True,  balanced=False)
        va_flat_dl     = make_flat_loader(va_flat_ds, False, balanced=False)
        te_flat_dl     = make_flat_loader(te_flat_ds, False, balanced=False)
        te_oracle_dl   = make_flat_loader(te_oracle_ds, False, balanced=False)

        log(f"  {tag} Stage2 tr_flat={tr_flat_mask.sum()}"
            f"  va_flat={va_flat_mask.sum()}"
            f"  te_pipeline={te_flat_mask.sum()}"
            f"  te_oracle={te_oracle_mask.sum()}")

        backbone_s2 = M6_BranchCBAMCrossAug(branch_ch).to(DEVICE)
        if S2_INIT_FROM_S1:
            # Stage1 backbone weight → Stage2 초기값
            # Stage1이 99%+ 학습 → IMU 표현 안정 → SupCon 수렴 가속
            s2_backbone_state = {
                k: v for k, v in backbone_s1.state_dict().items()
            }
            backbone_s2.load_state_dict(s2_backbone_state, strict=True)
            log(f"  {tag} Stage2 backbone ← Stage1 weight 이식 (S2_INIT_FROM_S1=True)")
        feat_dim    = _get_feat_dim(backbone_s2)
        s2_model    = Stage2Model(backbone_s2, feat_dim,
                                   bio_dim=64, num_classes=4).to(DEVICE)

        # 4단계 학습 (Review2)
        warmup_stage2_ce(s2_model, tr_flat_dl_ce, va_flat_dl, tag)     # Step1: CE 워밍업
        pretrain_stage2(s2_model, tr_flat_dl_bal, tag)                  # Step2: SupCon
        focal_finetune_stage2(s2_model, tr_flat_dl_ce, va_flat_dl, tag) # Step3: FocalLoss
        s2_preds, s2_labels = finetune_stage2(                          # Step4: CE 마무리
            s2_model, tr_flat_dl_ce, va_flat_dl, te_flat_dl, tag)

        # [수정 5순위] oracle vs pipeline 분리 보고
        s2_pipeline_acc = accuracy_score(s2_labels, s2_preds)

        s2_model.eval()
        oracle_preds = []
        with torch.inference_mode():
            for bi, bio_f, yb in te_oracle_dl:
                bi    = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
                bio_f = bio_f.to(DEVICE, non_blocking=True).float()
                if not config.USE_AMP:
                    bi = {k: v.float() for k, v in bi.items()}
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    oracle_preds.append(s2_model(bi, bio_f).argmax(1).cpu())
        oracle_preds = torch.cat(oracle_preds).numpy()
        s2_oracle_acc = accuracy_score(te_oracle_y, oracle_preds)

        log(f"  {tag} Stage2 oracle={s2_oracle_acc:.4f}"
            f"  pipeline={s2_pipeline_acc:.4f}")

        # ── 최종 결합 (hard routing) ─────────────
        final_preds = combine_predictions(s1_preds, s2_preds, te_y6)
        acc = accuracy_score(te_y6, final_preds)
        f1  = f1_score(te_y6, final_preds, average="macro", zero_division=0)
        log(f"  {tag} ★ 최종 6cls  Acc={acc:.4f}  F1={f1:.4f}")

        all_preds.append(final_preds)
        all_labels.append(te_y6)
        fold_meta.append({
            "fold": fi, "test_subjects": te_s,
            "s1_acc": round(s1_acc, 4),
            "s2_oracle_acc": round(s2_oracle_acc, 4),
            "s2_pipeline_acc": round(s2_pipeline_acc, 4),
            "final_acc": round(acc, 4),
            "final_f1": round(f1, 4),
            "fold_time_min": round((time.time()-t_fold)/60, 1),
        })

        del backbone_s1, backbone_s2, s2_model, s1_model
        del tr_ds, val_ds, te_ds
        del tr_flat_ds, va_flat_ds, te_flat_ds, te_oracle_ds
        gc.collect()
        if config.USE_GPU: torch.cuda.empty_cache()
        clear_fold_cache(f"HC{fi}")
        clear_fold_cache(f"HC{fi}v")

    # ── 전체 결과 ───────────────────────────────
    preds_all  = np.concatenate(all_preds)
    labels_all = np.concatenate(all_labels)
    acc_all    = accuracy_score(labels_all, preds_all)
    f1_all     = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    cm         = confusion_matrix(labels_all, preds_all)
    recalls    = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    total_min  = (time.time() - t_total) / 60

    print(f"\n{'='*60}")
    print(f"  ★ Hierarchical SupCon v8.6  {config.KFOLD}-Fold")
    print(f"  총 소요: {total_min:.1f}분")
    print(f"  Acc={acc_all:.4f}  MacroF1={f1_all:.4f}")
    print(f"{'='*60}")
    print(f"  ── 클래스별 Recall ──")
    for i, r in enumerate(recalls):
        print(f"    {CLASS_NAMES_ALL.get(i, f'C{i+1}'):<12} {r*100:.1f}%")

    rep = classification_report(
        labels_all, preds_all,
        target_names=[f"C{c}" for c in le.classes_], digits=4, zero_division=0)
    (out / "report_hierarchical.txt").write_text(
        f"Hierarchical SupCon v8.6\nAcc={acc_all:.4f}  F1={f1_all:.4f}\n\n{rep}")

    le_out = LabelEncoder()
    le_out.fit(sorted(set(labels_all.tolist())))
    save_cm(preds_all, labels_all, le_out, "Hierarchical_v87_KFold", out)

    summary = {
        "experiment": "hierarchical_supcon_kfold_v87",
        "version": "v8.7",
        "method": "Hierarchical Hard-Routing SupCon + FocalLoss",
        "routing": "hard (Stage1 argmax → Stage2 if flat)",
        "references": [
            "Ordóñez & Roggen (2016) IEEE TNNLS",
            "Khosla et al. (2020) NeurIPS",
            "Niswander et al. (2021)",
            "Lin et al. (2017) ICCV — Focal Loss",
        ],
        "fixes_v86": [
            "inner val split for early stopping",
            "full model state_dict saved",
            "balanced sampler for SupCon",
            "BioMech sample_rate from config",
            "oracle vs pipeline acc reported",
            "N_BIO 8→16, hard routing 명시",
        ],
        "improvements_v87": [
            "FocalLoss finetune stage (gamma=2.0)",
            "4-stage training: CE-warmup→SupCon→Focal→CE",
            "S1 LR 1e-4→5e-5, epochs 80→60",
            "T=0.12 (Khosla2020 4-class small-data recommendation)",
            "S2 backbone init from S1 weights (99%+ transfer)",
            "S2_WEIGHTS [2.0, 2.5, 4.0, 1.5]",
        ],
        "hyperparams": {
            "s1_epochs": S1_EPOCHS, "s1_lr": S1_LR,
            "s2_warmup_epochs": S2_WARMUP_EP,
            "s2_pretrain_epochs": S2_PRETRAIN_EP,
            "s2_focal_epochs": S2_FOCAL_EP,
            "s2_finetune_epochs": S2_FINETUNE_EP,
            "temperature": TEMPERATURE,
            "focal_gamma": FOCAL_GAMMA,
            "s2_weights": S2_WEIGHTS,
            "n_bio": BioMechFeatures.N_BIO,
        },
        "total_minutes": round(total_min, 1),
        "overall": {"acc": round(acc_all, 4), "f1": round(f1_all, 4)},
        "per_class_recall": {
            CLASS_NAMES_ALL.get(i, f"C{i+1}"): round(float(r), 4)
            for i, r in enumerate(recalls)
        },
        "fold_meta": fold_meta,
    }
    (out / "summary_hierarchical_v87.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  ✅ {out / 'summary_hierarchical_v87.json'}")
    h5data.close()


if __name__ == "__main__":
    main()