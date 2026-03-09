"""src/eval_utils.py — 평가 결과 저장 / 시각화 유틸."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
)

from utils import save_json, ensure_dir


# ─────────────────────────────────────────────
# 저장 함수
# ─────────────────────────────────────────────

def save_report(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    tag: str,
    out_dir: Path,
) -> tuple[float, float]:
    """분류 결과 저장 → (acc, macro_f1) 반환."""
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0, output_dict=True)
    save_json({
        "tag": tag, "acc": round(acc, 4), "macro_f1": round(f1, 4),
        "report": report,
    }, out_dir / f"report_{tag}.json")
    return acc, f1


def save_cm(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    tag: str,
    out_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {tag}")
    plt.tight_layout()
    plt.savefig(out_dir / f"cm_{tag}.png", dpi=150)
    plt.close()


def save_history(
    hist: Dict[str, List[List[dict]]],   # {model_name: [fold_hist, ...]}
    out_dir: Path,
) -> None:
    """각 모델의 학습 곡선 PNG 저장."""
    for model_name, folds in hist.items():
        if not folds:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for fi, fold_h in enumerate(folds, 1):
            eps     = [r["ep"] for r in fold_h]
            tr_acc  = [r.get("tr_acc",  r.get("train_acc", 0))  for r in fold_h]
            val_acc = [r.get("val_acc", 0) for r in fold_h]
            val_loss= [r.get("val_loss", 0) for r in fold_h]
            axes[0].plot(eps, tr_acc,  "--", alpha=0.5, label=f"F{fi}-tr")
            axes[0].plot(eps, val_acc,        alpha=0.9, label=f"F{fi}-val")
            axes[1].plot(eps, val_loss,        alpha=0.7, label=f"F{fi}")
        axes[0].set_title(f"{model_name} — Accuracy")
        axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
        axes[1].set_title(f"{model_name} — Val Loss")
        axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"history_{model_name}.png", dpi=150)
        plt.close()


def save_summary_table(
    results: Dict[str, tuple[float, float]],
    out_dir: Path,
    fname: str = "summary.csv",
) -> None:
    """모델별 Acc/F1 CSV 저장."""
    lines = ["model,acc,macro_f1"]
    for model, (acc, f1) in sorted(results.items()):
        lines.append(f"{model},{acc:.4f},{f1:.4f}")
    (out_dir / fname).write_text("\n".join(lines))


def save_per_subject_heatmap(
    per_subj: Dict[int, Dict[str, float]],
    results: Dict[str, tuple[float, float]],
    out_dir: Path,
) -> None:
    """LOSO 피험자별 Accuracy heatmap PNG."""
    subjs = sorted(per_subj)
    tags  = list(results)
    data  = np.array([[per_subj[s].get(t, 0) for t in tags] for s in subjs])
    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 1.5), max(8, len(subjs) * 0.35)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(len(tags)));     ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_yticks(range(len(subjs)));    ax.set_yticklabels([f"S{s:03d}" for s in subjs], fontsize=7)
    ax.set_title("LOSO Per-Subject Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "per_subject_heatmap.png", dpi=150)
    plt.close()
    # CSV도 저장
    lines = ["subject," + ",".join(tags)]
    for s in subjs:
        vals = [str(round(per_subj[s].get(t, 0), 4)) for t in tags]
        lines.append(f"S{s:03d},{','.join(vals)}")
    (out_dir / "per_subject_accuracy.csv").write_text("\n".join(lines))