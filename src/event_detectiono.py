"""src/event_detection.py — Heel-Strike / Toe-Off 이벤트 검출."""
from __future__ import annotations
import numpy as np


def moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    return np.convolve(np.pad(x, (pad, pad), "edge"), np.ones(k) / k, mode="valid")


def detect_hs_to_rule(
    gyro_pitch: np.ndarray,    # 발등 센서 Gyro_y (pitch 축)
    acc_mag: np.ndarray,       # 가속도 크기
    hs_thresh: float = -1.0,
    to_thresh: float = -0.5,
    min_step_gap: int = 30,    # 최소 스텝 간격 (샘플)
) -> tuple[list[int], list[int]]:
    """Rule-based Heel-Strike / Toe-Off 검출.

    Returns
    -------
    hs_idx : Heel-Strike 인덱스 리스트
    to_idx : 대응하는 Toe-Off 인덱스 리스트
    """
    gx = moving_average(gyro_pitch.astype(np.float32), 5)
    n  = len(gx)

    hs_raw: list[int] = []
    for i in range(1, n - 1):
        if gx[i] < hs_thresh and gx[i] <= gx[i-1] and gx[i] <= gx[i+1]:
            if not hs_raw or (i - hs_raw[-1]) >= min_step_gap:
                hs_raw.append(i)

    hs_idx: list[int] = []
    to_idx: list[int] = []
    for hs in hs_raw:
        for j in range(hs + 1, min(hs + 120, n - 1)):
            if gx[j] < to_thresh and gx[j] <= gx[j-1] and gx[j] <= gx[j+1]:
                hs_idx.append(hs)
                to_idx.append(j)
                break

    return hs_idx, to_idx