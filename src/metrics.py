"""src/metrics.py — 평가 지표."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    recall_score, classification_report,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc":              float(accuracy_score(y_true, y_pred)),
        "macro_f1":         float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion":        confusion_matrix(y_true, y_pred).tolist(),
    }


def print_report(y_true: np.ndarray, y_pred: np.ndarray, target_names=None) -> None:
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))