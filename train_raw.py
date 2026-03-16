"""train_raw.py — Raw 시계열 CNN + Transformer (v1)

입력: (N, 256, 54) 원시 IMU 시계열 → 6클래스 지형 분류

구조:
  [1] MultiScaleCNN: 3개 커널 크기(3,7,15)로 다중 스케일 특징 추출
      → C4/C5/C6의 충격 패턴을 여러 시간 스케일에서 포착
  [2] Transformer Encoder: 시간축 self-attention
      → 보행 주기 내 장거리 의존성 학습
  [3] SurfaceBranch: C4/C5/C6 전용 추가 분류 헤드
  [4] Mixup + TTA + 2단계 학습
  [5] 학습 완료 후 proba 저장 → train_kfold.py 앙상블에 통합

실행:
  cd ~/project/repo
  python train_raw.py
  python train_raw.py --epochs 100 --batch 256
"""
from __future__ import annotations

import sys, gc, argparse, os, copy, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

import config as _cfg
from config import CFG
from train_common import (H5Data, seed_everything, log, ensure_dir,
                          save_json, Timer,
                          filter_and_remap, N_ACTIVE_CLASSES, ACTIVE_CLASS_NAMES)

# C5 제외: 5클래스 학습
N_CLASSES  = N_ACTIVE_CLASSES     # 5
C4, C6     = 3, 4                 # 새 인덱스 기준 (C5 제거)
CLASS_NAMES = ACTIVE_CLASS_NAMES  # ["C1-미끄러운","C2-오르막","C3-내리막","C4-흙길","C6-평지"]

# ══════════════════════════════════════════════════════════════
# 1. Dataset
# ══════════════════════════════════════════════════════════════

class RawIMUDataset(Dataset):
    """(N, T, C) 원시 시계열 데이터셋."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 indices: np.ndarray, augment: bool = False,
                 mean: np.ndarray = None, std: np.ndarray = None):
        self.X = X[indices].astype(np.float32)   # (N, T, C)
        self.y = y[indices].astype(np.int64)
        self.augment = augment
        self.indices = indices

        if mean is None:
            # train: 자체 통계로 정규화
            self.mean = self.X.mean(axis=(0, 1), keepdims=True)  # (1,1,C)
            self.std  = self.X.std(axis=(0, 1), keepdims=True) + 1e-6
        else:
            # test: train 통계 사용
            self.mean = mean
            self.std  = std
        self.X = (self.X - self.mean) / self.std

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]   # (T, C)
        if self.augment:
            # 시간축 jitter
            if np.random.random() < 0.5:
                x = x + np.random.randn(*x.shape).astype(np.float32) * 0.05
            # 채널 dropout
            if np.random.random() < 0.3:
                drop_ch = np.random.randint(0, x.shape[1], size=3)
                x = x.copy(); x[:, drop_ch] = 0.0
        return {"x": torch.from_numpy(x), "y": torch.tensor(self.y[i])}


def make_sampler(y: np.ndarray) -> WeightedRandomSampler:
    classes, counts = np.unique(y, return_counts=True)
    w = np.zeros(N_CLASSES, dtype=np.float64)
    w[classes] = 1.0 / counts
    for c in [C4, C6]:
        w[c] *= 5.0
    sample_w = w[y]
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


# ══════════════════════════════════════════════════════════════
# 2. Model — MultiScaleCNN + Transformer
# ══════════════════════════════════════════════════════════════

class MultiScaleCNN(nn.Module):
    """3개 커널 크기로 다중 스케일 특징 추출."""

    def __init__(self, in_ch: int, out_ch: int, kernels=(3, 7, 15)):
        super().__init__()
        per = out_ch // len(kernels)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, per, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(per),
                nn.GELU(),
            ) for k in kernels
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(per * len(kernels), out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):  # (B, C, T)
        outs = [conv(x) for conv in self.convs]
        return self.proj(torch.cat(outs, dim=1))


class CNNBlock(nn.Module):
    def __init__(self, ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, stride=stride,
                      padding=kernel // 2, bias=False, groups=ch),
            nn.Conv1d(ch, ch, 1, bias=False),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.ds = nn.Conv1d(ch, ch, 1, stride=stride, bias=False) \
                  if stride > 1 else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.ds(x)


class RawTerrainModel(nn.Module):
    """
    Raw IMU (T, C) → 6클래스 분류

    [구조]
    (B, T, C)
      → transpose → (B, C, T)
      → MultiScaleCNN (입력 임베딩)
      → 3× CNNBlock stride=2 (T 축소: 256→32)
      → transpose → (B, 32, d_model)
      → Transformer Encoder (4 layer)
      → CLS token → FC head
      → surface branch (C4/C6 보강, C5 제외)
    """

    def __init__(self, in_ch: int = 54, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.embed = MultiScaleCNN(in_ch, d_model)

        # CNN downsampling: T 256→32 (stride=2 × 3)
        self.cnn_blocks = nn.Sequential(
            CNNBlock(d_model, 7, stride=2),   # 256→128
            CNNBlock(d_model, 5, stride=2),   # 128→64
            CNNBlock(d_model, 3, stride=2),   # 64→32
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, 33, d_model))  # 32+1(CLS)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,   # Pre-LN (더 안정적)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 분류 헤드
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_CLASSES),
        )

        # Surface 전용 브랜치 (C4/C5/C6)
        self.surface_branch = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),  # C5 제외 → C4/C6 2클래스
        )
        self.surf_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, T, C)
        h = x.transpose(1, 2)              # (B, C, T)
        h = self.embed(h)                  # (B, d, T)
        h = self.cnn_blocks(h)             # (B, d, T//8)
        h = h.transpose(1, 2)             # (B, T//8, d)

        # CLS token 추가
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)    # (B, T//8+1, d)
        h = h + self.pos_emb[:, :h.size(1)]

        h = self.transformer(h)           # (B, T//8+1, d)
        cls_out = h[:, 0]                 # CLS token (B, d)

        logits = self.head(cls_out)       # (B, 6)
        surf   = self.surface_branch(cls_out)  # (B, 2): C4/C6
        scale  = torch.clamp(self.surf_scale, 0.0, 3.0)

        logits_final = torch.cat([
            logits[:, :3],
            logits[:, 3:6] + scale * surf,
        ], dim=1)

        return {"logits": logits_final, "surface_logits": surf, "hidden": cls_out}


# ══════════════════════════════════════════════════════════════
# 3. Loss
# ══════════════════════════════════════════════════════════════

class TerrainFocalLoss(nn.Module):
    def __init__(self, gamma_default=2.0, gamma_surface=3.0, label_smooth=0.05):
        super().__init__()
        gammas = [gamma_default] * N_CLASSES
        gammas[C4] = gammas[C6] = gamma_surface   # C5 제외
        self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float32))
        self.ls = label_smooth

    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            smooth = torch.full_like(logits, self.ls / (n - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=1)
        ce    = -(smooth * log_p).sum(1)
        pt    = torch.exp(-ce)
        gamma = self.gammas[targets]
        return ((1 - pt) ** gamma * ce).mean()


class SurfaceAuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # C5 제외 → C4/C6 2클래스 surface 보조 손실
        w = torch.tensor([1.2, 1.0], dtype=torch.float32)  # C4, C6
        self.crit = nn.CrossEntropyLoss(weight=w, ignore_index=-100, label_smoothing=0.03)

    def forward(self, surf_logits, targets):
        self.crit.weight = self.crit.weight.to(surf_logits.device)
        surf_tgt = torch.full_like(targets, -100)
        surf_tgt[targets == C4] = 0   # C4 흙길 → 0
        surf_tgt[targets == C6] = 1   # C6 평지 → 1  (C5 제외)
        return self.crit(surf_logits, surf_tgt)


def mixup(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(len(x), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ══════════════════════════════════════════════════════════════
# 4. Threshold search
# ══════════════════════════════════════════════════════════════

def threshold_search(proba, labels, vals=None):
    if vals is None:
        vals = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]
    mults = np.ones(N_CLASSES, dtype=np.float32)
    for ci in range(N_CLASSES):
        best_f1, best_m = -1.0, 1.0
        for m in vals:
            tmp = mults.copy(); tmp[ci] = m
            f1  = f1_score(labels, (proba * tmp).argmax(1),
                           average="macro", zero_division=0)
            if f1 > best_f1: best_f1, best_m = f1, m
        mults[ci] = best_m
    return mults, f1_score(labels, (proba * mults).argmax(1),
                           average="macro", zero_division=0)


# ══════════════════════════════════════════════════════════════
# 5. Train / Eval
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, crit, crit_surf, device,
                grad_clip=5.0, aux_w=0.3):
    model.train(); total_loss = 0.0; n = 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        opt.zero_grad()

        if np.random.random() < 0.5:
            x_mix, ya, yb, lam = mixup(x, y, 0.4)
            out  = model(x_mix)
            loss = lam * crit(out["logits"], ya) + (1-lam) * crit(out["logits"], yb)
            loss += aux_w * crit_surf(out["surface_logits"],
                                      ya if lam >= 0.5 else yb)
        else:
            out  = model(x)
            loss = crit(out["logits"], y)
            loss += aux_w * crit_surf(out["surface_logits"], y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total_loss += float(loss) * len(y); n += len(y)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, crit, device, mults=None, tta_n=1):
    model.eval()
    total_loss = 0.0; n = 0
    all_proba, all_y = [], []
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        probas = []
        for ti in range(tta_n):
            xi = x if ti == 0 else x + torch.randn_like(x) * 0.02
            out = model(xi)
            probas.append(F.softmax(out["logits"], dim=1))
        proba = torch.stack(probas).mean(0)

        out_c = model(x)
        loss  = crit(out_c["logits"], y)
        total_loss += float(loss) * len(y); n += len(y)
        all_proba.append(proba.cpu().numpy())
        all_y.extend(y.cpu().tolist())

    all_proba = np.concatenate(all_proba)
    all_y     = np.array(all_y, dtype=np.int64)
    preds = (all_proba * mults).argmax(1) if mults is not None else all_proba.argmax(1)
    acc = accuracy_score(all_y, preds)
    f1  = f1_score(all_y, preds, average="macro", zero_division=0)
    return total_loss / max(n, 1), acc, f1, all_y, preds, all_proba


# ══════════════════════════════════════════════════════════════
# 6. Run fold
# ══════════════════════════════════════════════════════════════

def run_fold(fi, tr_ds, te_ds, args, device):
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, sampler=make_sampler(tr_ds.y),
                       num_workers=4, pin_memory=True, drop_last=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                       num_workers=4, pin_memory=True)

    model     = RawTerrainModel(in_ch=54, d_model=args.d_model,
                                n_heads=args.n_heads, n_layers=args.n_layers,
                                dropout=args.dropout).to(device)
    crit      = TerrainFocalLoss().to(device)
    crit_surf = SurfaceAuxLoss().to(device)

    # warmup + cosine LR
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup = 8
    min_lr_ratio = 0.01  # 최소 lr = args.lr * 0.01

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(args.epochs - warmup, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_f1, best_state = -1.0, None
    best_mults = np.ones(N_CLASSES, dtype=np.float32)
    patience = 0

    for ep in range(1, args.epochs + 1):
        tl = train_epoch(model, tr_dl, opt, crit, crit_surf, device,
                         args.grad_clip, args.aux_w)
        sched.step()

        if ep % 5 == 0 or ep == args.epochs:
            _, _, _, vy, _, vp = eval_epoch(model, te_dl, crit, device, tta_n=1)
            mults, _ = threshold_search(vp, vy)
        else:
            mults = best_mults

        _, va, vf, _, _, _ = eval_epoch(model, te_dl, crit, device, mults, tta_n=1)

        log(f"  [F{fi}] ep{ep:03d}/{args.epochs}"
            f"  tl={tl:.4f}  acc={va:.4f}  f1={vf:.4f}"
            f"  lr={sched.get_last_lr()[0]*args.lr:.2e}")

        if vf > best_f1:
            best_f1 = vf; best_mults = mults.copy()
            best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                log(f"  [F{fi}] EarlyStop ep{ep}")
                break

    # Phase 2: surface fine-tune
    model.load_state_dict(best_state)
    surf_params = (list(model.surface_branch.parameters()) +
                   [model.surf_scale])
    opt2 = torch.optim.AdamW(surf_params, lr=args.lr * 0.05, weight_decay=args.wd)

    for ep2 in range(1, 6):
        model.train()
        for batch in tr_dl:
            x, y = batch["x"].to(device), batch["y"].to(device)
            opt2.zero_grad()
            loss = crit_surf(model(x)["surface_logits"], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surf_params, args.grad_clip)
            opt2.step()
        _, va2, vf2, _, _, vp2 = eval_epoch(model, te_dl, crit, device,
                                             best_mults, tta_n=1)
        log(f"  [F{fi}] Phase2 ep{ep2}/5  acc={va2:.4f}  f1={vf2:.4f}")
        if vf2 > best_f1:
            best_f1 = vf2
            m2, _ = threshold_search(vp2, te_ds.y)
            best_mults = m2
            best_state = copy.deepcopy(model.state_dict())

    # 최종 TTA 평가
    model.load_state_dict(best_state)
    _, acc, f1, yt, yp, proba = eval_epoch(model, te_dl, crit, device,
                                            best_mults, tta_n=5)
    log(f"  [F{fi}] ★ Best(TTA×5)  Acc={acc:.4f}  F1={f1:.4f}")

    report = classification_report(yt, yp, target_names=CLASS_NAMES,
                                   digits=4, zero_division=0)
    cm  = confusion_matrix(yt, yp, labels=list(range(N_CLASSES)))
    log(f"\n{report}")
    recalls = cm.diagonal() / cm.sum(1).clip(min=1)
    for i, r in enumerate(recalls):
        flag = "✅" if r >= 0.85 else ("⚠" if r >= 0.70 else "❌")
        log(f"  {flag} {CLASS_NAMES[i]:<16} recall={r*100:.1f}%")

    return {"acc": acc, "f1": f1, "mults": best_mults.tolist(),
            "report": report, "proba": proba, "labels": yt,
            "te_idx": te_ds.indices}


# ══════════════════════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=80)
    p.add_argument("--batch",      type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--wd",         type=float, default=1e-4)
    p.add_argument("--d_model",    type=int,   default=256)
    p.add_argument("--n_heads",    type=int,   default=8)
    p.add_argument("--n_layers",   type=int,   default=4)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--aux_w",      type=float, default=0.3)
    p.add_argument("--grad_clip",  type=float, default=5.0)
    p.add_argument("--early_stop", type=int,   default=20)
    p.add_argument("--kfold",      type=int,   default=5)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = _cfg.DEVICE

    log("=" * 60)
    log("  RawTerrainModel v1 (MultiScaleCNN + Transformer)")
    log(f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}")
    log(f"  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    log(f"  device={device}")
    log("=" * 60)

    # 데이터 로드
    h5     = H5Data(CFG.h5_path)
    # ── C5 제외 ──────────────────────────────────────────────
    CFG.num_classes = N_ACTIVE_CLASSES
    y_raw_orig = h5.y_raw.astype(np.int64)
    y_all, groups, kept_idx = filter_and_remap(y_raw_orig, h5.subj_id)
    X_all  = h5.X[kept_idx]   # (N', T, C)
    le = LabelEncoder(); le.classes_ = np.array(CLASS_NAMES)
    log(f"  C5 제외 후: X={X_all.shape}  y={y_all.shape}  subjects={len(np.unique(groups))}")

    # 출력 디렉토리
    out   = ensure_dir(CFG.repo_dir / "out_N50" / "raw_cnn_transformer")
    proba_dir = ensure_dir(out / "probas")

    # K-Fold
    sgkf    = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    results = []

    with Timer() as total_t:
        for fi, (tr_idx, te_idx) in enumerate(
            sgkf.split(np.zeros(len(y_all)), y_all, groups), 1
        ):
            log(f"\n{'='*60}")
            log(f"  Fold {fi}/{args.kfold}  tr={len(tr_idx)}  te={len(te_idx)}")

            tr_ds = RawIMUDataset(X_all, y_all, tr_idx, augment=True)
            te_ds = RawIMUDataset(X_all, y_all, te_idx, augment=False,
                                  mean=tr_ds.mean, std=tr_ds.std)

            res = run_fold(fi, tr_ds, te_ds, args, device)

            # proba 저장 (앙상블용)
            np.save(proba_dir / f"raw_proba_fold{fi}.npy",  res["proba"])
            np.save(proba_dir / f"raw_labels_fold{fi}.npy", res["labels"])
            np.save(proba_dir / f"raw_te_idx_fold{fi}.npy", res["te_idx"])
            log(f"  [F{fi}] proba 저장 → {proba_dir}/raw_proba_fold{fi}.npy")

            results.append({"fold": fi, "acc": res["acc"], "f1": res["f1"]})

            del tr_ds, te_ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 요약
    mean_acc = float(np.mean([r["acc"] for r in results]))
    mean_f1  = float(np.mean([r["f1"]  for r in results]))

    log(f"\n{'='*60}")
    log(f"  ★ RawTerrainModel v1 완료  총 소요: {total_t}")
    log(f"  Mean Acc={mean_acc:.4f}  Mean F1={mean_f1:.4f}")
    for r in results:
        log(f"    Fold{r['fold']}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}")
    log(f"{'='*60}")

    save_json({
        "experiment": "raw_cnn_transformer_v1",
        "mean_acc": round(mean_acc, 4),
        "mean_f1":  round(mean_f1, 4),
        "folds": results,
    }, out / "summary_raw.json")

    h5.close()


if __name__ == "__main__":
    main()