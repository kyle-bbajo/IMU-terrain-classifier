"""
train_supcon.py — Supervised Contrastive Learning (v8.4)
═══════════════════════════════════════════════════════
Khosla et al. (2020) "Supervised Contrastive Learning" NeurIPS

★ C4(흙길)/C5(잔디)/C6(정상지면) 혼동 문제 해결
★ Phase 1: SupConLoss → feature space에서 클래스 분리
★ Phase 2: CE Loss → 분류기 fine-tuning
★ 기존 M5/M6 Branch 구조 재사용 (extract() 메서드)
★ K-Fold Subject-wise 교차검증

학술 근거:
  Khosla et al., NeurIPS 2020 (인용 5000+)
  "Supervised Contrastive Learning"
  → 유사 클래스(hard negative)에서 Cross-Entropy 대비
    명확한 성능 향상 입증
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys, time, json, gc, warnings, argparse, math
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from channel_groups import build_branch_idx
from models import M5_BranchCBAMCross, M6_BranchCBAMCrossAug, count_parameters
from train_common import (
    log, H5Data,
    fit_bsc_on_train,
    make_branch_dataset, make_loader, collate_branch,
    save_report, save_cm, save_history,
    clear_fold_cache, _prepare_model, _unwrap,
    _mem_str, _gpu_mem_str,
)

DEVICE = config.DEVICE


# ═══════════════════════════════════════════════
# 1. Supervised Contrastive Loss
# ═══════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    같은 클래스 샘플은 embedding 공간에서 가깝게,
    다른 클래스는 멀게 학습.

    C4/C5/C6처럼 feature가 유사한 hard negative를
    명시적으로 분리하는 데 탁월.

    Parameters
    ----------
    temperature : float
        Softmax temperature. 낮을수록 분리 강도 증가.
        일반적으로 0.07~0.2.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor
            L2-normalized feature vectors. (B, D)
        labels : torch.Tensor
            Class labels. (B,)
        """
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        # L2 정규화 (cosine similarity 기반)
        features = F.normalize(features, dim=1)

        # Similarity matrix: (B, B)
        sim = torch.matmul(features, features.T) / self.temperature

        # 자기 자신 제외 마스크
        eye = torch.eye(B, dtype=torch.bool, device=features.device)
        sim = sim.masked_fill(eye, -1e9)

        # 같은 클래스 마스크 (positive pairs)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & ~eye   # (B, B)

        # log-sum-exp over all non-self pairs (denominator)
        log_prob = sim - torch.logsumexp(sim.masked_fill(eye, -1e9), dim=1, keepdim=True)

        # positive pair에 대한 평균 log-likelihood
        n_pos = pos_mask.sum(1).float().clamp(min=1)
        loss = -(log_prob * pos_mask).sum(1) / n_pos

        return loss.mean()


# ═══════════════════════════════════════════════
# 2. Projection Head (SupCon 표준 구조)
# ═══════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """SupCon 표준 Projection MLP.

    feature → projection space (128D)에서 contrastive loss 계산.
    fine-tuning 시 제거하고 classifier만 학습.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


# ═══════════════════════════════════════════════
# 3. SupCon Wrapper Model
# ═══════════════════════════════════════════════

class SupConModel(nn.Module):
    """Branch 모델 + Projection Head 래퍼.

    Phase 1 (pretrain): extract() → proj_head → SupConLoss
    Phase 2 (finetune): extract() → classifier → CELoss
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone  = backbone
        self.proj_head = ProjectionHead(feat_dim, 256, 128)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_CLF),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        """Feature 추출 (backbone extract)."""
        return self.backbone.extract(bi)

    def forward_proj(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        """Phase 1: SupConLoss용 projection."""
        return self.proj_head(self.forward_features(bi))

    def forward(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        """Phase 2: 분류 logits."""
        return self.classifier(self.forward_features(bi))


# ═══════════════════════════════════════════════
# 4. LR 스케줄러 (Warmup + Cosine)
# ═══════════════════════════════════════════════

def _make_scheduler(
    opt: torch.optim.Optimizer,
    epochs: int,
    warmup: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def _lr_lambda(ep: int) -> float:
        if ep < warmup:
            return float(ep + 1) / float(warmup)
        progress = float(ep - warmup) / float(max(epochs - warmup, 1))
        cos_val   = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_f     = config.MIN_LR / config.LR
        return min_f + (1.0 - min_f) * cos_val
    return torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)


# ═══════════════════════════════════════════════
# 5. Phase 1 — Contrastive Pretraining
# ═══════════════════════════════════════════════

PRETRAIN_EPOCHS = 150    # 100 → 150
PRETRAIN_LR     = 1e-4
TEMPERATURE     = 0.05   # 0.07 → 0.05 (더 강한 분리)


def pretrain_supcon(
    model: SupConModel,
    tr_dl: DataLoader,
    tag: str = "",
) -> None:
    """Phase 1: Supervised Contrastive Pretraining.

    backbone + proj_head만 학습. classifier는 고정.
    """
    supcon = SupConLoss(temperature=TEMPERATURE)
    params = list(model.backbone.parameters()) + list(model.proj_head.parameters())
    opt = torch.optim.AdamW(params, lr=PRETRAIN_LR, weight_decay=config.WEIGHT_DECAY)
    sch = _make_scheduler(opt, PRETRAIN_EPOCHS, warmup=10)

    use_scaler = config.USE_AMP and config.AMP_DTYPE == torch.float16
    scaler = GradScaler(enabled=use_scaler)

    log(f"  {tag} Phase1 SupCon Pretrain ({PRETRAIN_EPOCHS}ep, T={TEMPERATURE})")
    t0 = time.time()
    best_loss = float("inf")

    for ep in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)

        for step_i, (bi, yb) in enumerate(tr_dl):
            bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            yb = yb.to(DEVICE, non_blocking=True)
            if not config.USE_AMP:
                bi = {k: v.float() for k, v in bi.items()}

            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                proj = model.forward_proj(bi)   # (B, 128)
                loss = supcon(proj, yb) / config.GRAD_ACCUM_STEPS

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                    opt.step()
                opt.zero_grad(set_to_none=True)

            total_loss += loss.item() * config.GRAD_ACCUM_STEPS * len(yb)
            n += len(yb)

        sch.step()
        avg = total_loss / max(n, 1)
        if avg < best_loss:
            best_loss = avg

        if ep % 10 == 0:
            log(f"  {tag} P1 ep{ep:03d}/{PRETRAIN_EPOCHS}"
                f"  loss={avg:.4f}  best={best_loss:.4f}"
                f"  lr={opt.param_groups[0]['lr']:.1e}"
                f"  ({time.time()-t0:.0f}s)"
                f"  RAM={_mem_str()}{_gpu_mem_str()}")


# ═══════════════════════════════════════════════
# 6. Phase 2 — Classifier Fine-tuning
# ═══════════════════════════════════════════════

FINETUNE_EPOCHS = 150


def _make_weight_tensor() -> torch.Tensor | None:
    if hasattr(config, "CLASS_WEIGHTS") and config.CLASS_WEIGHTS:
        return torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    return None


def finetune_classifier(
    model: SupConModel,
    tr_dl: DataLoader,
    te_dl: DataLoader,
    tag: str = "",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Phase 2: backbone 고정 + classifier만 fine-tuning.

    backbone은 SupCon으로 학습된 좋은 representation 유지.
    """
    # backbone 고정
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.proj_head.parameters():
        p.requires_grad = False

    weight = _make_weight_tensor()
    crit = nn.CrossEntropyLoss(
        weight=weight,
        label_smoothing=config.LABEL_SMOOTH,
    )
    log(f"    Loss: CE(smooth={config.LABEL_SMOOTH}"
        f", weight={'ON' if weight is not None else 'OFF'})")

    opt = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=config.LR, weight_decay=config.WEIGHT_DECAY,
    )
    sch = _make_scheduler(opt, FINETUNE_EPOCHS, warmup=10)

    use_scaler = config.USE_AMP and config.AMP_DTYPE == torch.float16
    scaler = GradScaler(enabled=use_scaler)

    hist = {"tl": [], "ta": [], "vl": [], "va": []}
    best_vl = float("inf")
    best_state = None
    patience = 0
    t0 = time.time()

    log(f"  {tag} Phase2 Finetune ({FINETUNE_EPOCHS}ep)")

    for ep in range(1, FINETUNE_EPOCHS + 1):
        # Train
        model.train()
        tl_sum = ta_n = ta_c = 0
        opt.zero_grad(set_to_none=True)

        for step_i, (bi, yb) in enumerate(tr_dl):
            bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
            yb = yb.to(DEVICE, non_blocking=True)
            if not config.USE_AMP:
                bi = {k: v.float() for k, v in bi.items()}

            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model(bi)
                loss = crit(logits, yb) / config.GRAD_ACCUM_STEPS

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step_i + 1) % config.GRAD_ACCUM_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        model.classifier.parameters(), config.GRAD_CLIP_NORM)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.classifier.parameters(), config.GRAD_CLIP_NORM)
                    opt.step()
                opt.zero_grad(set_to_none=True)

            bs = len(yb)
            tl_sum += loss.item() * config.GRAD_ACCUM_STEPS * bs
            ta_c   += (logits.argmax(1) == yb).sum().item()
            ta_n   += bs

        # Val
        model.eval()
        vl_sum = va_n = va_c = 0
        preds_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        with torch.inference_mode():
            for bi, yb in te_dl:
                bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
                yb = yb.to(DEVICE, non_blocking=True)
                if not config.USE_AMP:
                    bi = {k: v.float() for k, v in bi.items()}
                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model(bi)
                    loss   = crit(logits, yb)
                bs = len(yb)
                vl_sum += loss.item() * bs
                va_c   += (logits.argmax(1) == yb).sum().item()
                va_n   += bs
                preds_list.append(logits.argmax(1).cpu())
                labels_list.append(yb.cpu())

        sch.step()

        tl = tl_sum / max(ta_n, 1)
        ta = ta_c   / max(ta_n, 1)
        vl = vl_sum / max(va_n, 1)
        va = va_c   / max(va_n, 1)
        hist["tl"].append(tl); hist["ta"].append(ta)
        hist["vl"].append(vl); hist["va"].append(va)

        if ep % 10 == 0:
            log(f"  {tag} P2 ep{ep:03d}/{FINETUNE_EPOCHS}"
                f"  loss={tl:.4f}/{ta:.4f}"
                f"  val={vl:.4f}/{va:.4f}"
                f"  lr={opt.param_groups[0]['lr']:.1e}"
                f"  ({time.time()-t0:.0f}s)")

        if vl < best_vl:
            best_vl = vl
            best_state = {
                k: v.cpu().clone()
                for k, v in model.classifier.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP:
                log(f"  {tag} ★ EarlyStop ep{ep}")
                break

    if best_state:
        model.classifier.load_state_dict(best_state)
        model.to(DEVICE)

    # backbone 다시 활성화 (다음 fold 대비)
    for p in model.backbone.parameters():
        p.requires_grad = True

    preds  = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()
    return preds, labels, hist


# ═══════════════════════════════════════════════
# 7. K-Fold 메인
# ═══════════════════════════════════════════════

CLASS_NAMES = {0: "C1-평지", 1: "C2-오르막", 2: "C3-내리막",
               3: "C4-흙길", 4: "C5-잔디",   5: "C6-정상"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SupCon K-Fold")
    p.add_argument("--n_subjects", type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--batch",      type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_subjects: config.N_SUBJECTS = args.n_subjects
    if args.seed:       config.SEED       = args.seed
    if args.batch:      config.BATCH      = args.batch

    config.print_config()
    log(f"  ★ SupCon K-Fold {config.KFOLD}-Fold\n"
        f"  Phase1={PRETRAIN_EPOCHS}ep (T={TEMPERATURE})"
        f"  Phase2={FINETUNE_EPOCHS}ep\n")

    out = config.RESULT_KFOLD / "supcon"
    out.mkdir(parents=True, exist_ok=True)
    config.snapshot(out)

    h5data  = H5Data(config.H5_PATH)
    le      = LabelEncoder()
    y       = le.fit_transform(h5data.y_raw).astype(np.int64)
    groups  = h5data.subj_id
    branch_idx, branch_ch = build_branch_idx(h5data.channels)

    log(f"  클래스: {le.classes_.tolist()}  피험자: {len(np.unique(groups))}명"
        f"  샘플: {len(y)}")

    sgkf = StratifiedGroupKFold(
        n_splits=config.KFOLD, shuffle=True, random_state=config.SEED)

    all_preds:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_hist:   list[dict]       = []
    fold_meta:  list[dict]       = []
    t_total = time.time()

    for fi, (tr_idx, te_idx) in enumerate(
        sgkf.split(np.zeros(len(y)), y, groups=groups), 1
    ):
        t_fold = time.time()
        te_s   = sorted(set(groups[te_idx].tolist()))
        tr_s   = sorted(set(groups[tr_idx].tolist()))

        log(f"\n{'='*55}")
        log(f"  Fold {fi}/{config.KFOLD}"
            f"  tr={len(tr_idx)}({len(tr_s)}명)"
            f"  te={len(te_idx)}({len(te_s)}명)")
        log(f"  Test 피험자: {te_s}")
        log(f"{'='*55}")

        bsc = fit_bsc_on_train(h5data, tr_idx)

        tr_ds = make_branch_dataset(
            h5data, y, tr_idx, bsc, branch_idx,
            fold_tag=f"SC{fi}", split="train")
        te_ds = make_branch_dataset(
            h5data, y, te_idx, bsc, branch_idx,
            fold_tag=f"SC{fi}", split="test")

        tr_dl = make_loader(tr_ds, True,  branch=True)
        te_dl = make_loader(te_ds, False, branch=True)

        # 모델 생성
        backbone = M6_BranchCBAMCrossAug(branch_ch)
        feat_dim = config.FEAT_DIM * (
            len(branch_ch)
            + (1 if config.USE_FFT_BRANCH else 0)
            + (1 if "Foot"  in branch_ch else 0)
            + (1 if "Shank" in branch_ch else 0)
        )
        model = SupConModel(backbone, feat_dim, config.NUM_CLASSES)
        model = model.to(DEVICE)
        log(f"  backbone params={count_parameters(backbone):,}"
            f"  feat_dim={feat_dim}")

        tag = f"[F{fi}][SupCon]"

        # Phase 1: Contrastive Pretraining
        pretrain_supcon(model, tr_dl, tag)

        # Phase 2: Classifier Fine-tuning
        preds, labels, hist = finetune_classifier(model, tr_dl, te_dl, tag)

        acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
        f1  = f1_score(labels, preds, average="macro", zero_division=0) if len(labels) > 0 else 0.0
        log(f"  {tag} Acc={acc:.4f}  F1={f1:.4f}")

        all_preds.append(preds)
        all_labels.append(labels)
        all_hist.append(hist)

        fold_time = round((time.time() - t_fold) / 60, 1)
        fold_meta.append({
            "fold": fi, "test_subjects": te_s,
            "acc": round(acc, 4), "f1": round(f1, 4),
            "fold_time_min": fold_time,
        })

        del model, backbone, bsc, tr_ds, te_ds; gc.collect()
        if config.USE_GPU:
            torch.cuda.empty_cache()
        clear_fold_cache(f"SC{fi}")

    # 전체 결과
    preds_all  = np.concatenate(all_preds)
    labels_all = np.concatenate(all_labels)

    acc_all = accuracy_score(labels_all, preds_all)
    f1_all  = f1_score(labels_all, preds_all, average="macro", zero_division=0)

    # 클래스별 recall
    cm   = confusion_matrix(labels_all, preds_all)
    recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    total_min = (time.time() - t_total) / 60

    print(f"\n{'='*60}")
    print(f"  ★ SupCon {config.KFOLD}-Fold  {config.DEVICE_NAME}")
    print(f"  총 소요: {total_min:.1f}분")
    print(f"  Acc={acc_all:.4f}  MacroF1={f1_all:.4f}")
    print(f"{'='*60}")
    print(f"  ── 클래스별 Recall ──")
    for i, r in enumerate(recalls):
        print(f"    {CLASS_NAMES.get(i, f'C{i+1}'):<12} {r*100:.1f}%")

    rep = classification_report(
        labels_all, preds_all,
        target_names=[f"C{c}" for c in le.classes_], digits=4, zero_division=0)
    (out / "report_supcon.txt").write_text(
        f"SupCon KFold\nAcc={acc_all:.4f}  F1={f1_all:.4f}\n\n{rep}")

    # Confusion Matrix 저장
    save_cm(preds_all, labels_all, le, "SupCon_KFold", out)

    summary = {
        "experiment": "supcon_kfold",
        "version": "v8.4",
        "method": "Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)",
        "temperature": TEMPERATURE,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "config": config.snapshot(),
        "total_minutes": round(total_min, 1),
        "overall": {"acc": round(acc_all, 4), "f1": round(f1_all, 4)},
        "per_class_recall": {
            CLASS_NAMES.get(i, f"C{i+1}"): round(float(r), 4)
            for i, r in enumerate(recalls)
        },
        "fold_meta": fold_meta,
    }
    (out / "summary_supcon.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  ✅ {out / 'summary_supcon.json'}")
    h5data.close()


if __name__ == "__main__":
    main()