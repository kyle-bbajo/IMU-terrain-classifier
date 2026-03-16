"""train_surface.py — C4/C5/C6 전용 3분류 모델 (v1)

전략:
  [1] C4/C5/C6 샘플만 추출 → 3분류 집중 학습
      → 6클래스 동시 분류 대비 모델 capacity 집중
  [2] DualBranchMLP + surface 피처 강조
  [3] TTA × 5 + per-class threshold search
  [4] 출력 proba를 6클래스로 복원 후 저장
      → train_kfold.py 앙상블에 surface 전용 proba 추가

실행:
  cd ~/project/repo
  python train_surface.py
"""
from __future__ import annotations

import sys, gc, argparse, copy, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

import config as _cfg
from config import CFG
from train_common import (H5Data, seed_everything, log, ensure_dir,
                          save_json, Timer,
                          filter_and_remap, N_ACTIVE_CLASSES, ACTIVE_CLASS_NAMES)
from features import batch_extract, N_FEATURES

# ── 상수 ──────────────────────────────────────────────────────
# C5 제외 후 surface 모델: C4/C6 2클래스 집중 학습
N_SURFACE   = 2                       # C4, C6 (C5 잔디 제외)
N_CLASSES   = N_ACTIVE_CLASSES        # 전체 학습 클래스 = 5
# 새 인덱스 기준: C4=3, C6=4 (filter_and_remap 이후)
SURFACE_MAP = {3: 0, 4: 1}           # 새 레이블 → surface 0/1
INV_MAP     = {0: 3, 1: 4}           # surface → 새 레이블 (C4→3, C6→4)
SURFACE_NAMES = ["C4-흙길", "C6-평지"]   # C5 제외


# ══════════════════════════════════════════════════════════════
# 1. Dataset
# ══════════════════════════════════════════════════════════════

class SurfaceDataset(Dataset):
    def __init__(self, feat: np.ndarray, y_orig: np.ndarray,
                 indices: np.ndarray, scaler: StandardScaler | None = None,
                 fit_scaler: bool = False):
        # C4/C5/C6만 필터
        # filter_and_remap 이후 새 인덱스: C4=3, C6=4
        mask = np.isin(y_orig[indices], list(SURFACE_MAP.keys()))
        self.indices_all = indices          # 전체 fold 인덱스 보존
        self.surf_indices = indices[mask]   # surface만

        raw = feat[self.surf_indices]
        if fit_scaler:
            self.scaler = StandardScaler()
            self.feat   = self.scaler.fit_transform(raw).astype(np.float32)
        elif scaler is not None:
            self.scaler = scaler
            self.feat   = scaler.transform(raw).astype(np.float32)
        else:
            self.scaler = None
            self.feat   = raw.astype(np.float32)

        # surface 레이블 (0,1,2)
        self.y      = np.array([SURFACE_MAP[lb] for lb in y_orig[self.surf_indices]],
                               dtype=np.int64)
        self.y_orig = y_orig[self.surf_indices]   # 원래 레이블 보존

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {"x": torch.from_numpy(self.feat[i]),
                "y": torch.tensor(self.y[i])}


def make_sampler(y: np.ndarray) -> WeightedRandomSampler:
    classes, counts = np.unique(y, return_counts=True)
    w = np.zeros(N_SURFACE, dtype=np.float64)
    w[classes] = 1.0 / counts.astype(np.float64)
    return WeightedRandomSampler(w[y], len(y), replacement=True)


# ══════════════════════════════════════════════════════════════
# 2. Model — SurfaceMLP (C4/C5/C6 전용)
# ══════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim),
        )
    def forward(self, x): return x + self.net(x)


class SurfaceExpertMLP(nn.Module):
    """C4/C5/C6 전용 3분류 MLP.
    더 깊은 네트워크 + 강한 정규화.
    """
    def __init__(self, in_dim: int, hidden: int = 512,
                 n_blocks: int = 6, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout)
                                      for _ in range(n_blocks)])
        # 3클래스 헤드
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, N_SURFACE),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.blocks(self.proj(x)))


# ══════════════════════════════════════════════════════════════
# 3. Loss — Surface 전용 Focal
# ══════════════════════════════════════════════════════════════

class SurfaceFocalLoss(nn.Module):
    """C4(흙길)에 가장 높은 gamma — 가장 혼동이 심한 클래스."""
    def __init__(self, gamma=(4.0, 3.0, 3.0), label_smooth=0.05):
        super().__init__()
        self.register_buffer("gammas",
                             torch.tensor(gamma, dtype=torch.float32))
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


def mixup(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(len(x), device=x.device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam


# ══════════════════════════════════════════════════════════════
# 4. Threshold search (3클래스)
# ══════════════════════════════════════════════════════════════

def threshold_search(proba, labels):
    vals   = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
    mults  = np.ones(N_SURFACE, dtype=np.float32)
    for ci in range(N_SURFACE):
        best_f1, best_m = -1.0, 1.0
        for m in vals:
            tmp = mults.copy(); tmp[ci] = m
            f1  = f1_score(labels, (proba * tmp).argmax(1),
                           average="macro", zero_division=0)
            if f1 > best_f1: best_f1, best_m = f1, m
        mults[ci] = best_m
    return mults


# ══════════════════════════════════════════════════════════════
# 5. Train / Eval
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, crit, device, grad_clip=5.0):
    model.train(); total = 0.0; n = 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        opt.zero_grad()
        if np.random.random() < 0.5:
            xm, ya, yb, lam = mixup(x, y, 0.4)
            lg   = model(xm)
            loss = lam * crit(lg, ya) + (1-lam) * crit(lg, yb)
        else:
            loss = crit(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += float(loss) * len(y); n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, crit, device, mults=None, tta_n=1):
    model.eval()
    total = 0.0; n = 0; all_p, all_y = [], []
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        probas = []
        for ti in range(tta_n):
            xi = x if ti == 0 else x + torch.randn_like(x) * 0.02
            probas.append(F.softmax(model(xi), dim=1))
        proba = torch.stack(probas).mean(0)
        loss  = crit(model(x), y)
        total += float(loss) * len(y); n += len(y)
        all_p.append(proba.cpu().numpy())
        all_y.extend(y.cpu().tolist())
    all_p  = np.concatenate(all_p)
    all_y  = np.array(all_y, dtype=np.int64)
    preds  = (all_p * mults).argmax(1) if mults is not None else all_p.argmax(1)
    acc = accuracy_score(all_y, preds)
    f1  = f1_score(all_y, preds, average="macro", zero_division=0)
    return total / max(n,1), acc, f1, all_y, preds, all_p


# ══════════════════════════════════════════════════════════════
# 6. Run fold
# ══════════════════════════════════════════════════════════════

def run_fold(fi, tr_ds, te_ds, args, device):
    tr_dl = DataLoader(tr_ds, batch_size=args.batch,
                       sampler=make_sampler(tr_ds.y),
                       num_workers=4, pin_memory=True, drop_last=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                       num_workers=4, pin_memory=True)

    model = SurfaceExpertMLP(tr_ds.feat.shape[1],
                             args.hidden, args.n_blocks, args.dropout).to(device)
    crit  = SurfaceFocalLoss().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    warmup = 8
    def lr_lambda(ep):
        if ep < warmup: return (ep+1) / warmup
        prog = (ep - warmup) / max(args.epochs - warmup, 1)
        return 1e-6/args.lr + (1-1e-6/args.lr)*0.5*(1+math.cos(math.pi*prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_f1, best_state = -1.0, None
    best_mults = np.ones(N_SURFACE, dtype=np.float32)
    patience = 0

    for ep in range(1, args.epochs + 1):
        tl = train_epoch(model, tr_dl, opt, crit, device)
        sched.step()

        if ep % 5 == 0 or ep == args.epochs:
            _, _, _, vy, _, vp = eval_epoch(model, te_dl, crit, device, tta_n=1)
            mults = threshold_search(vp, vy)
        else:
            mults = best_mults

        _, va, vf, _, _, _ = eval_epoch(model, te_dl, crit, device, mults, tta_n=1)
        log(f"  [F{fi}] ep{ep:03d}/{args.epochs}"
            f"  tl={tl:.4f}  acc={va:.4f}  f1={vf:.4f}")

        if vf > best_f1:
            best_f1 = vf; best_mults = mults.copy()
            best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                log(f"  [F{fi}] EarlyStop ep{ep}"); break

    # 최종 TTA 평가
    model.load_state_dict(best_state)
    _, acc, f1, yt, yp, surf_proba = eval_epoch(
        model, te_dl, crit, device, best_mults, tta_n=5)

    log(f"  [F{fi}] ★ Best(TTA×5)  Acc={acc:.4f}  F1={f1:.4f}")
    rep = classification_report(yt, yp, target_names=SURFACE_NAMES,
                                digits=4, zero_division=0)
    log(f"\n{rep}")

    cm = confusion_matrix(yt, yp, labels=[0, 1])  # C4=0, C6=1
    recalls = cm.diagonal() / cm.sum(1).clip(min=1)
    for i, r in enumerate(recalls):
        flag = "✅" if r >= 0.80 else ("⚠" if r >= 0.65 else "❌")
        log(f"  {flag} {SURFACE_NAMES[i]:<12} recall={r*100:.1f}%")

    # ── 5클래스(전체) proba로 복원 ───────────────────────────
    # C4/C6가 아닌 자리는 0으로 채움 (앙상블 통합용)
    n_surf = len(surf_proba)
    probaFull = np.zeros((n_surf, N_CLASSES), dtype=np.float32)
    for si in range(N_SURFACE):
        orig_cls = INV_MAP[si]      # 0→3(C4), 1→4(C6) 새 인덱스
        probaFull[:, orig_cls] = surf_proba[:, si]

    return {
        "acc": acc, "f1": f1,
        "mults": best_mults.tolist(),
        "proba6": probaFull,        # (N_surf, 5) — C4/C6 열만 채워짐 (앙상블용)
        "surf_proba": surf_proba,   # (N_surf, 2) C4/C6 2클래스
        "labels_surf": yt,          # surface 레이블 (0=C4, 1=C6)
        "labels_orig": te_ds.y_orig,  # 새 인덱스 레이블 (3=C4, 4=C6)
        "surf_indices": te_ds.surf_indices,  # 전체 데이터셋 내 인덱스
    }


# ══════════════════════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=150)
    p.add_argument("--batch",      type=int,   default=512)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--wd",         type=float, default=1e-4)
    p.add_argument("--hidden",     type=int,   default=512)
    p.add_argument("--n_blocks",   type=int,   default=6)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--early_stop", type=int,   default=30)
    p.add_argument("--kfold",      type=int,   default=5)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = _cfg.DEVICE

    log("=" * 60)
    log("  SurfaceExpertMLP v1 — C4/C5/C6 전용 3분류")
    log(f"  hidden={args.hidden}  n_blocks={args.n_blocks}  N_FEATURES={N_FEATURES}")
    log(f"  epochs={args.epochs}  device={device}")
    log("=" * 60)

    h5     = H5Data(CFG.h5_path)
    # ── C5 제외 적용 ──────────────────────────────────────────
    CFG.num_classes = N_ACTIVE_CLASSES
    y_raw_orig = h5.y_raw.astype(np.int64)
    y_all, groups, kept_idx = filter_and_remap(y_raw_orig, h5.subj_id)
    X_all  = h5.X[kept_idx]
    log(f"  C5 제외 후: N={len(y_all)}  C4(3)/C6(4) 2클래스 surface 학습")

    # surface 샘플 비율 로그
    for ci, name in SURFACE_NAMES_DICT.items():
        n = (y_all == ci).sum()
        log(f"  {name}: {n}개 ({n/len(y_all)*100:.1f}%)")

    # 피처 추출 (캐시)
    cache_dir  = ensure_dir(CFG.repo_dir / "cache" /
                            f"feat{N_FEATURES}_seed{args.seed}_final")
    cache_path = cache_dir / "all_feat.npy"
    if cache_path.exists():
        log(f"[feat] 캐시 히트 → {cache_path}")
        feat_all = np.load(cache_path)
    else:
        log("[feat] 추출 중...")
        from channel_groups import get_foot_accel_idx
        foot_idx = get_foot_accel_idx(h5.channels)
        with Timer() as t:
            feat_all = batch_extract(X_all, foot_idx, CFG.sample_rate,
                                     h5_path=str(CFG.h5_path))
        np.save(cache_path, feat_all)
        log(f"[feat] 완료 shape={feat_all.shape}  ({t})")

    out       = ensure_dir(CFG.repo_dir / "out_N50" / "surface_expert")
    proba_dir = ensure_dir(out / "probas")

    sgkf    = StratifiedGroupKFold(n_splits=args.kfold,
                                   shuffle=True, random_state=args.seed)
    results = []

    with Timer() as total_t:
        for fi, (tr_idx, te_idx) in enumerate(
            sgkf.split(np.zeros(len(y_all)), y_all, groups), 1
        ):
            log(f"\n{'='*60}")
            # surface 샘플 수 로그
            tr_surf = np.isin(y_all[tr_idx], list(SURFACE_MAP.keys())).sum()
            te_surf = np.isin(y_all[te_idx], list(SURFACE_MAP.keys())).sum()
            log(f"  Fold {fi}/{args.kfold}  "
                f"tr_surf={tr_surf}  te_surf={te_surf}")

            tr_ds = SurfaceDataset(feat_all, y_all, tr_idx, fit_scaler=True)
            te_ds = SurfaceDataset(feat_all, y_all, te_idx,
                                   scaler=tr_ds.scaler)

            if len(tr_ds) == 0 or len(te_ds) == 0:
                log(f"  [WARN] surface 샘플 없음, 스킵"); continue

            res = run_fold(fi, tr_ds, te_ds, args, device)

            # proba 저장
            np.save(proba_dir / f"surface_proba_fold{fi}.npy",   res["proba6"])    # kfold 앙상블용
            np.save(proba_dir / f"surface_labels_fold{fi}.npy",  res["labels_orig"])
            np.save(proba_dir / f"surface_proba2_fold{fi}.npy",  res["surf_proba"]) # C4/C6 2분류
            np.save(proba_dir / f"surface_indices_fold{fi}.npy", res["surf_indices"])
            log(f"  [F{fi}] proba 저장 → {proba_dir}")

            results.append({"fold": fi, "acc": res["acc"], "f1": res["f1"]})
            del tr_ds, te_ds; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    mean_acc = float(np.mean([r["acc"] for r in results]))
    mean_f1  = float(np.mean([r["f1"]  for r in results]))

    log(f"\n{'='*60}")
    log(f"  ★ SurfaceExpert 완료  {total_t}")
    log(f"  Mean Acc={mean_acc:.4f}  Mean F1={mean_f1:.4f}")
    for r in results:
        log(f"    Fold{r['fold']}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}")

    save_json({"mean_acc": round(mean_acc,4), "mean_f1": round(mean_f1,4),
               "folds": results}, out / "summary_surface.json")
    h5.close()


# C5 제외 후 새 인덱스 기준: C4=3, C6=4
SURFACE_NAMES_DICT = {3: "C4-흙길", 4: "C6-평지"}

if __name__ == "__main__":
    main()