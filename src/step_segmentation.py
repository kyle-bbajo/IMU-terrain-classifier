# -*- coding: utf-8 -*-
"""
step_segmentation.py — heel-strike step segmentation  (v10.0-trialwise)
════════════════════════════════════════════════════════════════════════════
핵심 구조:
  [1] 파일 전체 → walking bout(trial) 분할
      - stop / turn 구간 자동 제거
      - C1(4 bout), C6(2 bout) 구조 인식
      - bout 수 미달 시 파라미터 자동 재시도

  [2] 각 bout 안에서만 HS 검출
      - LT/RT dual-signal consensus
      - polarity flip 자동 재시도
      - trial 경계 오염 방지

  [3] HDF5 메타 저장
      source_file / trial_id / bout_start / bout_end /
      step_start / step_end / side / support

사용:
    python3 step_segmentation.py
    python3 step_segmentation.py --force
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys
import time
import re
import json
import gc
import argparse
from pathlib import Path
from typing import Optional, NamedTuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import numpy as np
import pandas as pd
import h5py
from scipy.signal import find_peaks, resample, butter, filtfilt


# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

SEGMENTATION_VERSION   = "v10.0-trialwise-bout-step"
CSV_SKIPROWS = getattr(config, "CSV_SKIPROWS", 2)  # Noraxon export: 헤더 2줄 skip
_SUPPORTED_H5_FORMATS  = {"subject_group_v10", "subject_group_v9", "subject_group_v8"}

BOUT_SMOOTH_SEC    = 0.25
BOUT_MIN_WALK_SEC  = 2.0
BOUT_MAX_GAP_SEC   = 0.60
BOUT_PAD_SEC       = 0.20
BOUT_ENERGY_Z      = -0.2
TURN_Z_THR         = 2.0
TURN_MIN_SEC       = 0.25
MIN_STEPS_PER_BOUT = 2

# C1: 4회 왕복, C6: 중간 방향 전환 1회 → 2 bout
COND_EXPECTED_BOUTS: dict[int, int] = {1: 4, 6: 2}

_FLIP_RATIO = 0.60


# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_csv_with_retry(path: Path, skiprows: int = CSV_SKIPROWS, max_retries: int = 3, **kw) -> pd.DataFrame:
    last: Exception = RuntimeError("unknown")
    for attempt in range(max_retries):
        try:
            return pd.read_csv(path, skiprows=skiprows, low_memory=False, **kw)  # skiprows default=CSV_SKIPROWS
        except Exception as e:
            last = e
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
    raise last


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    return np.convolve(x, np.ones(win) / win, mode="same")


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return (x - med) / (1.4826 * mad + 1e-8)


def find_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs


def merge_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def subtract_runs(
    base: list[tuple[int, int]], cuts: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    if not base:
        return []
    if not cuts:
        return base[:]
    out: list[tuple[int, int]] = []
    for bs, be in base:
        segs = [(bs, be)]
        for cs, ce in cuts:
            new_segs = []
            for ss, se in segs:
                if ce <= ss or cs >= se:
                    new_segs.append((ss, se))
                else:
                    if ss < cs:
                        new_segs.append((ss, cs))
                    if ce < se:
                        new_segs.append((ce, se))
            segs = new_segs
            if not segs:
                break
        out.extend((s, e) for s, e in segs if e > s)
    return out


# ──────────────────────────────────────────────────────────────
# 컬럼 정규화
# ──────────────────────────────────────────────────────────────

RENAME_MAP: dict[str, str] = {
    "Noraxon MyoMotion-Joints-Knee LT-Rotation Ext (deg)": "Knee Rotation Ext LT (deg)",
    "Noraxon MyoMotion-Joints-Knee RT-Rotation Ext (deg)": "Knee Rotation Ext RT (deg)",
}
_COL_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+Segments[-\s]+"),     ""),
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+Joints[-\s]+"),       ""),
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+Trajectories[-\s]+"), "Trajectories "),
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+"),                   ""),
    (re.compile(r"Acceleration\b"),                                "Accel Sensor"),
]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    renames = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    if renames:
        df = df.rename(columns=renames)
    new: dict[str, str] = {}
    for c in df.columns:
        nc = c
        for pat, repl in _COL_RULES:
            nc = pat.sub(repl, nc)
        if nc != c:
            new[c] = nc
    if new:
        existing = set(df.columns)
        safe = {k: v for k, v in new.items() if v not in existing or v == k}
        if safe:
            df = df.rename(columns=safe)
    return df


# ──────────────────────────────────────────────────────────────
# 파일 탐색
# ──────────────────────────────────────────────────────────────


def interpolate_sensor_gaps(df, max_gap=100):
    """센서 NaN 구간을 linear interpolation으로 복구 (max_gap 샘플 이하만)."""
    import pandas as pd
    for col in df.select_dtypes(include=["float64","float32","int64","int32"]).columns:
        if df[col].isna().any():
            s = df[col]
            nan_mask = s.isna()
            groups = nan_mask != nan_mask.shift()
            group_id = groups.cumsum()
            group_sizes = nan_mask.groupby(group_id).transform("sum")
            interp_mask = nan_mask & (group_sizes <= max_gap)
            if interp_mask.any():
                df[col] = df[col].where(~interp_mask, s.interpolate(method="linear", limit_direction="both"))
    return df

def parse_filename(fname: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    m = re.search(r"S(\d+)C(\d+)T(\d+)", fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"S(\d+)C(\d+)", fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), 1
    return None, None, None


def discover_csvs(data_dir: Path, n_subjects: int) -> list[dict]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    records: list[dict] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        sid, cond, trial = parse_filename(csv_path.name)
        if sid is None or sid > n_subjects:
            continue
        if cond is None or cond < 1 or cond > config.NUM_CLASSES:
            continue
        records.append({"path": csv_path, "sid": sid, "cond": cond,
                         "trial": trial, "label": cond - 1})
    return records


# ──────────────────────────────────────────────────────────────
# 기존 HDF5 / processed_files
# ──────────────────────────────────────────────────────────────

def load_existing_h5_info(h5_path: Path) -> tuple[Optional[list[str]], int]:
    if not h5_path.exists():
        return None, 0
    try:
        with h5py.File(h5_path, "r") as f:
            # format 호환성 검증
            fmt = f.attrs.get("format", "")
            if fmt and fmt not in _SUPPORTED_H5_FORMATS:
                raise ValueError(f"unsupported HDF5 format: '{fmt}'. Use --force.")
            step_def = f.attrs.get("step_definition", "")
            expected = "trialwise_dual_signal_consensus_bilateral_merge"
            if step_def and step_def != expected:
                raise ValueError(f"step_definition mismatch (saved='{step_def}'). Use --force.")
            if "subjects" in f and "channels" in f:
                n_existing = sum(f[f"subjects/{sk}/X"].shape[0] for sk in f["subjects"])
                channels = [c.decode() if isinstance(c, bytes) else c for c in f["channels"][:]]
                return channels, n_existing
            return None, 0
    except ValueError:
        raise
    except Exception as e:
        log(f"  ⚠ failed reading HDF5: {e}")
        return None, 0


def make_file_key(rec: dict) -> str:
    return f"S{rec['sid']:04d}|C{rec['cond']:02d}|T{rec['trial']:02d}|{rec['path'].name}"


_ZERO_STEP_MAX_RETRIES = 3   # 이 횟수 초과 시 해당 파일 영구 skip


def load_processed_files(path: Path) -> tuple[set[str], str, dict[str, int]]:
    """(done_files, status, zero_retries) 반환.
    zero_retries: {file_key: 누적 실패 횟수}
    """
    if not path.exists():
        return set(), "missing", {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = raw.get("version", "unknown")
        if version != SEGMENTATION_VERSION:
            log(f"  ⚠ processed_files version mismatch ({version} vs {SEGMENTATION_VERSION})")
            return set(), "invalid", {}
        zr_raw = raw.get("zero_retries", {})
        zr = {str(k): int(v) for k, v in zr_raw.items()
              if isinstance(v, (int, float, str))}
        return set(raw.get("files", [])), "valid", zr
    except Exception as e:
        log(f"  ⚠ failed loading processed_files: {e}")
        return set(), "invalid", {}


def save_processed_files(
    path: Path, files: set[str], zero_retries: dict[str, int]
) -> None:
    path.write_text(json.dumps({
        "version":      SEGMENTATION_VERSION,
        "files":        sorted(files),
        "zero_retries": zero_retries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# 공통 채널
# ──────────────────────────────────────────────────────────────

def find_common_channels(records: list[dict]) -> list[str]:
    log("  [Pass 0] scanning common channels...")
    common: Optional[set[str]] = None
    first_ordered: Optional[list[str]] = None
    for i, rec in enumerate(records):
        try:
            cols_df = read_csv_with_retry(rec["path"], nrows=0)
            cols_df = rename_columns(cols_df)
            cols = cols_df.columns.tolist()
            drop = set(config.resolve_drop_cols(cols))
            data_cols = set(c for c in cols if c not in drop)
            if common is None:
                common = data_cols
                first_ordered = [c for c in cols if c not in drop]
            else:
                common &= data_cols
        except Exception as e:
            log(f"    ⚠ scan failed: {rec['path'].name} ({e})")
        if (i + 1) % 100 == 0:
            log(f"    scanned {i+1}/{len(records)}")
    if not common or not first_ordered:
        raise ValueError("no common channels")
    channels = [c for c in first_ordered if c in common]
    log(f"  common channels: {len(channels)}")
    return channels


def _verify_channels(records: list[dict], required_channels: list[str]) -> None:
    """채널 호환성 사전 점검. 누락 채널이 있는 파일을 조기에 경고한다."""
    req_set = set(required_channels)
    warn_count = 0
    for rec in records:
        try:
            cols_df = read_csv_with_retry(rec["path"], nrows=0)
            cols_df = rename_columns(cols_df)
            cols = set(c for c in cols_df.columns if c not in set(config.resolve_drop_cols(cols_df.columns.tolist())))
            missing = req_set - cols
            if missing:
                warn_count += 1
                if warn_count <= 5:
                    log(f"    ⚠ channel missing: {rec['path'].name} ({len(missing)} channels)")
        except Exception:
            continue
    if warn_count == 0:
        log(f"    ✅ channel check OK ({len(records)} files)")
    else:
        log(f"    ⚠ channel shortage: {warn_count} files")


# ──────────────────────────────────────────────────────────────
# 신호 추출
# ──────────────────────────────────────────────────────────────

def compute_foot_acc_norm(df: pd.DataFrame, side: str = "LT") -> np.ndarray:
    cols = config.resolve_foot_acc_cols(df.columns.tolist(), side)
    ax = df[cols["x"]].values.astype(np.float64)
    ay = df[cols["y"]].values.astype(np.float64)
    az = df[cols["z"]].values.astype(np.float64)
    return np.sqrt(ax**2 + ay**2 + az**2)


# signal_type 동의어 맵 — 장비별 표기 변형 허용
_SIGNAL_TYPE_SYNONYMS: dict[str, list[str]] = {
    "gyroscope": ["gyro", "gyroscope"],
    "accel":     ["accel", "acceleration"],
}
# side 토큰 구분자 — substring 오매칭 방지 (예: "lt"가 "delta"에 걸리는 것 방지)


def _resolve_sensor_axis(columns: list[str], sensor: str, axis: str,
                          side: str, signal_type: str = "gyroscope") -> str | None:
    """센서/축/side를 기준으로 컬럼명 매칭.
    - signal_type 동의어 허용 (gyro/gyroscope, accel/acceleration)
    - side는 토큰 경계 기반으로 확인 (substring 오매칭 방지)
    - 축 표기: -x, _x, (x), 공백+x 등 변형 허용
    """
    sl  = sensor.lower()
    xl  = axis.lower()
    sdl = side.lower()
    type_tokens = _SIGNAL_TYPE_SYNONYMS.get(signal_type.lower(),
                                             [signal_type.lower()])

    for c in columns:
        cl = c.lower()

        # 1) 센서명 포함
        if sl not in cl:
            continue

        # 2) side 토큰 확인 — pelvis는 side 무관
        if sl != "pelvis":
            # 구분자로 둘러싸인 side 토큰이어야 함
            parts = re.split(r"[\s\-_()\[\]/]", cl)
            if sdl not in parts:
                continue

        # 3) signal_type 동의어 매칭
        if not any(tok in cl for tok in type_tokens):
            continue

        # 4) 축 매칭 — 구분자+축 또는 (축) 형태
        axis_ok = (cl.endswith(f"-{xl}") or cl.endswith(f"_{xl}")
                   or f" {xl} " in cl or f"-{xl} " in cl
                   or f"_{xl} " in cl or f"({xl})" in cl
                   or cl.endswith(f" {xl}"))
        if axis_ok:
            return c
    return None


def extract_ml_gyro(df: pd.DataFrame, side: str) -> np.ndarray | None:
    col = _resolve_sensor_axis(df.columns.tolist(), config.HS_GYRO_SENSOR,
                                config.HS_GYRO_AXIS, side, "gyroscope")
    return None if col is None else df[col].values.astype(np.float64)


def extract_ap_accel(df: pd.DataFrame, side: str) -> np.ndarray | None:
    col = _resolve_sensor_axis(df.columns.tolist(), config.HS_ACCEL_SENSOR,
                                config.HS_ACCEL_AXIS, side, "accel")
    return None if col is None else df[col].values.astype(np.float64)


def _find_pelvis_gyro_col(columns: list[str]) -> str | None:
    """pelvis yaw gyro 컬럼 탐색. side 조건 없이 직접 매칭.
    z(yaw) 우선, 없으면 y(pitch) — 루프를 axis별로 분리해 순서 보장.
    signal_type은 _SIGNAL_TYPE_SYNONYMS["gyroscope"]와 동일 기준 적용."""
    gyro_tokens = _SIGNAL_TYPE_SYNONYMS["gyroscope"]
    for axis in ("x", "z", "y"):  # Noraxon pelvis: yaw=x
        for c in columns:
            cl = c.lower()
            if "pelvis" not in cl:
                continue
            if not any(tok in cl for tok in gyro_tokens):
                continue
            if (cl.endswith(f"-{axis}") or cl.endswith(f"_{axis}")
                    or f" {axis} " in cl or f"-{axis} " in cl
                    or f"({axis})" in cl):
                return c
    return None


def extract_turn_signal(df: pd.DataFrame) -> np.ndarray:
    """Turn proxy: pelvis yaw gyro 우선, 없으면 LT/RT ML gyro 절댓값 평균.
    pelvis는 side 개념이 없으므로 _find_pelvis_gyro_col()로 직접 탐색."""
    col = _find_pelvis_gyro_col(df.columns.tolist())
    if col is not None:
        return np.abs(df[col].values.astype(np.float64))
    lt = extract_ml_gyro(df, "LT")
    rt = extract_ml_gyro(df, "RT")
    if lt is not None and rt is not None:
        return 0.5 * (np.abs(lt) + np.abs(rt))
    for sig in (lt, rt):
        if sig is not None:
            return np.abs(sig)
    return np.zeros(len(df), dtype=np.float64)


def bandpass_filter(
    signal: np.ndarray, fs: int = config.SAMPLE_RATE,
    low: float = config.BANDPASS_LOW, high: float = config.BANDPASS_HIGH,
    order: int = config.BANDPASS_ORDER,
) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    mask = np.isnan(signal)
    if mask.all():
        return signal.copy()
    sig_clean = signal.copy()
    sig_clean[mask] = np.nanmean(signal)
    try:
        filtered = filtfilt(b, a, sig_clean)
    except ValueError:
        return sig_clean
    filtered[mask] = np.nan
    return filtered


# ──────────────────────────────────────────────────────────────
# Walking bout 검출
# ──────────────────────────────────────────────────────────────

def compute_motion_energy(df: pd.DataFrame, fs: int) -> np.ndarray:
    feats: list[np.ndarray] = []
    for side in ("LT", "RT"):
        try:
            feats.append(compute_foot_acc_norm(df, side))
        except Exception:
            pass
        try:
            g = extract_ml_gyro(df, side)
            if g is not None:
                feats.append(np.abs(g))
        except Exception:
            pass
    if not feats:
        return np.zeros(len(df), dtype=np.float64)
    arr = np.vstack([
        np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.any(~np.isnan(x)) else 0.0)
        for x in feats
    ])
    return moving_average(np.mean(arr, axis=0), max(3, int(fs * BOUT_SMOOTH_SEC)))


def _run_bout_detection(
    df: pd.DataFrame, fs: int,
    energy_z_thr: float, turn_z_thr: float,
    min_walk_sec: float, max_gap_sec: float,
) -> list[tuple[int, int]]:
    n = len(df)
    energy   = compute_motion_energy(df, fs)
    turn_sig = extract_turn_signal(df)

    ez = robust_zscore(energy)
    tz = robust_zscore(moving_average(turn_sig, max(3, int(fs * 0.15))))

    walk_mask = ez > energy_z_thr
    turn_mask = tz > turn_z_thr

    min_turn = int(fs * TURN_MIN_SEC)
    turn_runs = [(s, e) for s, e in find_true_runs(turn_mask) if (e - s) >= min_turn]

    walk_runs = find_true_runs(walk_mask)
    walk_runs = merge_close_runs(walk_runs, max_gap=int(fs * max_gap_sec))
    walk_runs = subtract_runs(walk_runs, turn_runs)

    min_walk = int(fs * min_walk_sec)
    walk_runs = [(s, e) for s, e in walk_runs if (e - s) >= min_walk]
    walk_runs = merge_close_runs(walk_runs, max_gap=int(fs * max_gap_sec))
    walk_runs = [(s, e) for s, e in walk_runs if (e - s) >= min_walk]

    pad = int(fs * BOUT_PAD_SEC)
    padded = [(max(0, s - pad), min(n, e + pad)) for s, e in walk_runs]
    padded = merge_close_runs(sorted(padded), max_gap=int(fs * max_gap_sec))

    return padded if padded else [(0, n)]


def detect_walking_bouts(
    df: pd.DataFrame,
    fs: int = config.SAMPLE_RATE,
    cond: Optional[int] = None,
) -> list[tuple[int, int]]:
    """
    파일 → walking bout 목록.

    C1(4 bout), C6(2 bout) 구조를 인식:
    - bout 수 < expected → turn_z_thr 완화로 재시도 (더 잘게 자름)
    - bout 수 > expected → max_gap 확대로 재시도 (더 합침)
    """
    n = len(df)
    if n < int(fs * BOUT_MIN_WALK_SEC):
        return [(0, n)]

    expected = COND_EXPECTED_BOUTS.get(cond)
    bouts = _run_bout_detection(df, fs, BOUT_ENERGY_Z, TURN_Z_THR,
                                 BOUT_MIN_WALK_SEC, BOUT_MAX_GAP_SEC)

    if expected is None:
        return bouts

    # bout 수 부족 → turn threshold 낮춰서 더 잘게 자름
    if len(bouts) < expected:
        for thr in (1.5, 1.0, 0.75):
            retry = _run_bout_detection(df, fs, BOUT_ENERGY_Z, thr,
                                         BOUT_MIN_WALK_SEC, BOUT_MAX_GAP_SEC)
            if len(retry) >= expected:
                log(f"    ↺ bout↑ (turn_thr={thr:.2f}): {len(bouts)}→{len(retry)} (exp={expected})")
                bouts = retry
                break

    # bout 수 초과 → gap 확대해서 합침
    elif len(bouts) > expected:
        for gap in (1.5, 3.0, 5.0):
            retry = _run_bout_detection(df, fs, BOUT_ENERGY_Z, TURN_Z_THR,
                                         BOUT_MIN_WALK_SEC, gap)
            if len(retry) <= expected:
                log(f"    ↺ bout↓ (max_gap={gap:.1f}s): {len(bouts)}→{len(retry)} (exp={expected})")
                bouts = retry
                break

    if len(bouts) != expected:
        log(f"    ⚠ C{cond} bout={len(bouts)} (expected={expected})")

    return bouts


def _log_bout_mismatch(step_logger, rec: dict, n_detected: int, n_expected: int) -> None:
    """bout 수 불일치를 step_log에 기록 (QC 추적용).
    delta = detected - expected, severity: minor(±1) / major(±2+)."""
    delta = n_detected - n_expected
    if delta == 0:
        severity = "none"
    elif abs(delta) == 1:
        severity = "minor"
    else:
        severity = "major"
    step_logger.write({
        "type": "bout_mismatch",
        "file": rec["path"].name,
        "sid": rec["sid"],
        "cond": rec["cond"],
        "bout_detected": n_detected,
        "bout_expected": n_expected,
        "delta": delta,
        "severity": severity,
    })


# ──────────────────────────────────────────────────────────────
# HS 검출 — dual-signal consensus
# ──────────────────────────────────────────────────────────────

def _adaptive_peaks(
    sig: np.ndarray, fs: int, negate: bool = False, loose_dist_ms: float = 250.0,
) -> tuple[np.ndarray, float]:
    src = -sig if negate else sig
    sigma = float(np.std(src))
    if sigma == 0:
        return np.array([], dtype=int), 0.0
    loose_dist = int(fs * loose_dist_ms / 1000)
    cands, _ = find_peaks(src, prominence=0.2 * sigma, distance=loose_dist)
    if len(cands) >= 4:
        med = float(np.median(np.diff(cands).astype(float)))
        min_dist = int(np.clip(med * 0.40, fs * 0.25, fs * 0.80))
    else:
        min_dist = int(fs * 0.35)
    peaks, _ = find_peaks(src, prominence=0.3 * sigma, distance=min_dist)
    return peaks, sigma


def _detect_hs_acc(ap_accel: np.ndarray, fs: int) -> np.ndarray:
    n = len(ap_accel)
    sig = ap_accel.copy()
    mask = np.isnan(sig)
    if mask.any() and not mask.all():
        x = np.arange(n)
        sig[mask] = np.interp(x[mask], x[~mask], sig[~mask])
    sig = bandpass_filter(sig, fs, low=0.5, high=20.0)
    peaks, sigma = _adaptive_peaks(sig, fs, negate=True)
    if len(peaks) == 0 or sigma == 0:
        return np.array([], dtype=int)
    win = int(0.10 * fs)
    trusted = []
    for p in peaks:
        w = sig[max(0, p - win): min(n, p + win)]
        if len(w) > 0 and float(np.nanmax(w)) - float(sig[p]) > 0.3 * sigma:
            trusted.append(p)
    return np.array(trusted, dtype=int)


def _detect_hs_gyro(
    ml_gyro: np.ndarray, fs: int, force_flip: bool = False,
) -> np.ndarray:
    n = len(ml_gyro)
    sig_raw = ml_gyro.copy()
    mask = np.isnan(sig_raw)
    if mask.any() and not mask.all():
        x = np.arange(n)
        sig_raw[mask] = np.interp(x[mask], x[~mask], sig_raw[~mask])
    sig_raw = bandpass_filter(sig_raw, fs, low=0.5, high=15.0)

    def _score(s: np.ndarray) -> tuple[np.ndarray, float]:
        sigma = float(np.std(s))
        if sigma == 0:
            return np.array([], dtype=int), 0.0
        swings, props = find_peaks(s, prominence=0.2 * sigma, distance=int(fs * 0.25))
        if len(swings) == 0:
            return np.array([], dtype=int), 0.0
        cv = (float(np.std(np.diff(swings).astype(float))) /
              max(float(np.mean(np.diff(swings))), 1.0)) if len(swings) >= 2 else 1.0
        return swings, float(np.sum(props["prominences"])) / (1.0 + cv)

    sw_orig, sc_orig = _score(sig_raw)
    sw_flip, sc_flip = _score(-sig_raw)

    use_flip = (sc_flip > sc_orig) ^ force_flip
    sig = -sig_raw if use_flip else sig_raw
    swing_peaks = sw_flip if use_flip else sw_orig

    sigma = float(np.std(sig))
    if sigma == 0:
        return np.array([], dtype=int)

    hs_cands, _ = _adaptive_peaks(sig, fs, negate=True)
    if len(hs_cands) == 0 or len(swing_peaks) == 0:
        return hs_cands

    half_win = fs // 2
    thresh = config.HS_TRUSTED_SWING * float(np.percentile(sig[swing_peaks], 90))
    trusted = []
    for cand in hs_cands:
        nearby = swing_peaks[np.abs(swing_peaks - cand) < half_win]
        if len(nearby) > 0 and sig[nearby].max() > thresh:
            trusted.append(cand)
        elif len(nearby) == 0:
            w = sig[max(0, cand - half_win): min(n, cand + half_win)]
            if len(w) > 0 and float(np.percentile(w, 99)) > thresh:
                trusted.append(cand)
    return np.array(trusted, dtype=int)


def _reconcile_candidates(
    hs_acc: np.ndarray, hs_gyro: np.ndarray, tol_ms: float, fs: int,
) -> list[tuple[int, str]]:
    tol = int(fs * tol_ms / 1000)
    result: list[tuple[int, str]] = []
    used_gyro: set[int] = set()
    for a in hs_acc:
        if len(hs_gyro) == 0:
            result.append((int(a), "acc"))
            continue
        dists = np.abs(hs_gyro - a)
        ni = int(np.argmin(dists))
        if dists[ni] <= tol and ni not in used_gyro:
            used_gyro.add(ni)
            result.append((int(a), "both"))
        else:
            result.append((int(a), "acc"))
    for gi, g in enumerate(hs_gyro):
        if gi not in used_gyro:
            result.append((int(g), "gyro"))
    result.sort(key=lambda x: x[0])
    return result


def _boutwise_stride_filter(
    candidates: list[tuple[int, str]], fs: int,
) -> list[tuple[int, str]]:
    if len(candidates) < 2:
        return candidates
    times = np.array([t for t, _ in candidates], dtype=float)
    trusted_times = np.array([t for t, s in candidates if s == "both"], dtype=float)
    ref = trusted_times if len(trusted_times) >= 4 else times
    intervals = np.diff(ref)
    if len(intervals) >= 4:
        med = float(np.median(intervals))
        mad = float(np.median(np.abs(intervals - med)))
        k = 2.5
        i_min = max(med - k * mad, config.HS_MIN_STRIDE_SAM)
        i_max = min(med + k * mad, config.HS_MAX_STRIDE_SAM)
        p5, p95 = float(np.percentile(intervals, 5)), float(np.percentile(intervals, 95))
        i_min = max(int(min(p5,  i_min)), config.HS_MIN_STRIDE_SAM)
        i_max = min(int(max(p95, i_max)), config.HS_MAX_STRIDE_SAM)
    else:
        i_min, i_max = config.HS_MIN_STRIDE_SAM, config.HS_MAX_STRIDE_SAM

    _ord = {"both": 2, "acc": 1, "gyro": 1, "norm": 0}
    kept: list[tuple[int, str]] = [candidates[0]]
    for cand in candidates[1:]:
        gap = cand[0] - kept[-1][0]
        if i_min <= gap <= i_max:
            kept.append(cand)
        elif gap < i_min:
            if _ord.get(cand[1], 0) > _ord.get(kept[-1][1], 0):
                kept[-1] = cand
        else:
            kept.append(cand)
    return kept


def detect_steps(
    ml_gyro: np.ndarray, ap_accel: np.ndarray,
    fs: int = config.SAMPLE_RATE, force_flip: bool = False,
) -> tuple[list[tuple[int, int]], list[str]]:
    n = len(ml_gyro)
    if n < fs:
        return [], []
    nan_ratio = (np.isnan(ml_gyro).sum() + np.isnan(ap_accel).sum()) / (2 * n)
    if nan_ratio > config.HS_NAN_THRESHOLD:
        return [], []

    hs_acc  = _detect_hs_acc(ap_accel, fs)
    hs_gyro = _detect_hs_gyro(ml_gyro, fs, force_flip=force_flip)
    if len(hs_acc) == 0 and len(hs_gyro) == 0:
        return [], []

    candidates = _reconcile_candidates(hs_acc, hs_gyro, 250.0, fs)
    candidates = _boutwise_stride_filter(candidates, fs)  # bout 내부 stride 필터
    if len(candidates) < 2:
        return [], []

    _ord = {"both": 3, "acc": 2, "gyro": 1, "norm": 0}
    steps: list[tuple[int, int]] = []
    supports: list[str] = []
    for i in range(len(candidates) - 1):
        s_t, s_sup = candidates[i]
        e_t, e_sup = candidates[i + 1]
        length = e_t - s_t
        if length <= 0:
            continue
        nan_ml = float(np.isnan(ml_gyro[s_t:e_t]).sum()) / length
        nan_ap = float(np.isnan(ap_accel[s_t:e_t]).sum()) / length
        if max(nan_ml, nan_ap) > config.HS_NAN_THRESHOLD:
            continue
        steps.append((s_t, e_t))
        # 보수적 선택: start/end 중 신뢰도 낮은 쪽을 step support로 채택
        supports.append(s_sup if _ord.get(s_sup, 0) <= _ord.get(e_sup, 0) else e_sup)
    return steps, supports


def _fallback_detect_steps(norm_f: np.ndarray, fs: int) -> tuple[list[tuple[int, int]], list[str]]:
    valid = norm_f[~np.isnan(norm_f)]
    if len(valid) == 0:
        return [], []
    mu, sigma = float(np.mean(valid)), float(np.std(valid))
    loose_peaks, _ = find_peaks(norm_f, height=mu + 0.3 * sigma, distance=int(fs * 0.25))
    if len(loose_peaks) >= 4:
        med   = float(np.median(np.diff(loose_peaks).astype(float)))
        s_min = max(int(med * 0.50), config.HS_MIN_STRIDE_SAM)
        s_max = min(int(med * 1.50), config.HS_MAX_STRIDE_SAM)
        min_d = int(np.clip(med * 0.40, fs * 0.25, fs * 0.80))
    else:
        s_min, s_max = config.HS_MIN_STRIDE_SAM, config.HS_MAX_STRIDE_SAM
        min_d = config.HS_MIN_STRIDE_SAM
    peaks, _ = find_peaks(norm_f, height=mu + 0.5 * sigma, distance=min_d)
    steps = [(int(peaks[j]), int(peaks[j + 1]))
             for j in range(len(peaks) - 1)
             if s_min <= peaks[j + 1] - peaks[j] <= s_max]
    return steps, ["norm"] * len(steps)


# ──────────────────────────────────────────────────────────────
# 품질 점수 + 병합 + bilateral sanity
# ──────────────────────────────────────────────────────────────

def estimate_stride_params_from_steps(steps: list[tuple[int, int]]) -> tuple[float, float, float]:
    if not steps:
        return 0.0, float(config.HS_MIN_STRIDE_SAM), float(config.HS_MAX_STRIDE_SAM)
    lengths = np.array([e - s for s, e in steps], dtype=float)
    mean = float(np.mean(lengths))
    lo = float(max(np.percentile(lengths, 5),  config.HS_MIN_STRIDE_SAM)) if len(lengths) >= 4 else float(config.HS_MIN_STRIDE_SAM)
    hi = float(min(np.percentile(lengths, 95), config.HS_MAX_STRIDE_SAM)) if len(lengths) >= 4 else float(config.HS_MAX_STRIDE_SAM)
    return mean, lo, hi


_SUPPORT_WEIGHT: dict[str, float] = {"both": 1.00, "acc": 0.80, "gyro": 0.75, "norm": 0.65}


class ScoredStep(NamedTuple):
    start:   int
    end:     int
    side:    str    # "LT" / "RT"
    score:   float
    support: str    # "both" / "acc" / "gyro" / "norm"


def _step_quality(
    start: int, end: int, signal: np.ndarray,
    stride_mean: float, stride_min: float, stride_max: float, support: str,
) -> float:
    length = end - start
    if length <= 0:
        return 0.0
    q_nan = 1.0 - float(np.isnan(signal[start:end]).sum()) / length
    if stride_mean > 0:
        center, rng = stride_mean, max(stride_max - stride_min, 1.0)
    elif stride_min > 0 and stride_max > stride_min:
        center, rng = (stride_min + stride_max) / 2.0, max(stride_max - stride_min, 1.0)
    else:
        center, rng = float(length), 1.0
    q_int = 1.0 / (1.0 + abs(length - center) / rng)
    return q_nan * q_int * _SUPPORT_WEIGHT.get(support, 0.75)


def score_steps_by_side(
    steps: list[tuple[int, int]], supports: list[str], side: str,
    signal: np.ndarray, stride_mean: float, stride_min: float, stride_max: float,
) -> list[ScoredStep]:
    scored: list[ScoredStep] = []
    for i, (s, e) in enumerate(steps):
        if i >= len(supports):
            log(f"  ⚠ score_steps [{side}]: supports 불일치 (steps={len(steps)}, supports={len(supports)}), {len(steps)-i}개 스킵")
            break
        scored.append(ScoredStep(start=s, end=e, side=side, support=supports[i],
                                  score=_step_quality(s, e, signal, stride_mean, stride_min, stride_max, supports[i])))
    return scored


def merge_bilateral_steps(
    lt_scored: list[ScoredStep], rt_scored: list[ScoredStep],
    overlap_threshold: float = 0.5,
) -> list[ScoredStep]:
    def _dedup(steps: list[ScoredStep]) -> list[ScoredStep]:
        if len(steps) < 2:
            return steps
        steps = sorted(steps, key=lambda x: x.start)
        kept = [0]
        for ci in range(1, len(steps)):
            pi = kept[-1]
            ov = max(0, min(steps[pi].end, steps[ci].end) - max(steps[pi].start, steps[ci].start))
            if ov == 0:
                kept.append(ci)
            else:
                shorter = min(steps[pi].end - steps[pi].start, steps[ci].end - steps[ci].start)
                if shorter == 0 or ov / shorter > overlap_threshold:
                    if steps[ci].score >= steps[pi].score:
                        kept[-1] = ci
                else:
                    kept.append(ci)
        return [steps[i] for i in kept]
    merged = _dedup(lt_scored) + _dedup(rt_scored)
    merged.sort(key=lambda x: x.start)
    return merged


def bilateral_sanity_check(steps: list[ScoredStep], fs: int = config.SAMPLE_RATE) -> list[ScoredStep]:
    if len(steps) < 2:
        return steps
    tol = int(fs * 0.05)
    cleaned: list[ScoredStep] = [steps[0]]
    for curr in steps[1:]:
        prev = cleaned[-1]
        if prev.side != curr.side and abs(curr.start - prev.start) <= tol:
            if curr.score > prev.score:
                cleaned[-1] = curr
        else:
            cleaned.append(curr)
    lt_n = sum(1 for s in cleaned if s.side == "LT")
    rt_n = len(cleaned) - lt_n
    if lt_n + rt_n > 0:
        r = lt_n / (lt_n + rt_n)
        if r > 0.70 or r < 0.30:
            log(f"    ⚠ bilateral imbalance: LT={lt_n} RT={rt_n} ({r:.0%}/{1-r:.0%})")
    return cleaned


# ──────────────────────────────────────────────────────────────
# 리샘플링
# ──────────────────────────────────────────────────────────────

def resample_step(data_segment: np.ndarray, target_length: int = config.TS) -> np.ndarray:
    L, C = data_segment.shape
    if L == target_length:
        return data_segment.astype(np.float32)
    result = np.zeros((target_length, C), dtype=np.float32)
    for c in range(C):
        col = data_segment[:, c].copy()
        nans = np.isnan(col)
        if nans.all():
            continue
        if nans.any():
            x = np.arange(L)
            col[nans] = np.interp(x[nans], x[~nans], col[~nans])
        result[:, c] = resample(col, target_length)
    return result


# ──────────────────────────────────────────────────────────────
# 로그 라이터
# ──────────────────────────────────────────────────────────────

class StepLogWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, entry: dict) -> None:
        self._fh.write(json.dumps(entry, ensure_ascii=False, default=int) + "\n")
        self.count += 1
        if self.count % 500 == 0:
            self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> "StepLogWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────
# HDF5 쓰기
# ──────────────────────────────────────────────────────────────

def _append_or_create(grp: h5py.Group, name: str, arr: np.ndarray, dtype=None) -> None:
    n = len(arr)
    if name in grp:
        ds = grp[name]
        old = ds.shape[0]
        ds.resize(old + n, axis=0)
        ds[old:old + n] = arr
    else:
        grp.create_dataset(name, data=arr, maxshape=(None,),
                           chunks=(min(1024, max(1, n)),),
                           dtype=dtype if dtype else arr.dtype)


def write_subject_group(
    hf: h5py.File, sid: int,
    X_arr: np.ndarray, y_arr: np.ndarray,
    meta: dict[str, np.ndarray], n_ch: int,
) -> None:
    grp_name = f"subjects/S{sid:04d}"
    str_dt = h5py.string_dtype(encoding="utf-8")
    if grp_name in hf:
        grp = hf[grp_name]
        old_n = grp["X"].shape[0]
        new_n = old_n + len(X_arr)
        grp["X"].resize(new_n, axis=0); grp["y"].resize(new_n, axis=0)
        grp["X"][old_n:new_n] = X_arr; grp["y"][old_n:new_n] = y_arr
        for k in ("trial_id","bout_start","bout_end","step_start","step_end","trial_step_index"):
            _append_or_create(grp, k, meta[k])
        for k in ("trial_key","source_file","side","support"):
            _append_or_create(grp, k, meta[k], dtype=str_dt)
    else:
        grp = hf.create_group(grp_name)
        n_new = len(X_arr)
        grp.create_dataset("X", data=X_arr, maxshape=(None, config.TS, n_ch),
                           chunks=(min(64, max(1, n_new)), config.TS, n_ch))
        grp.create_dataset("y", data=y_arr, maxshape=(None,))
        for k in ("trial_id","bout_start","bout_end","step_start","step_end","trial_step_index"):
            grp.create_dataset(k, data=meta[k], maxshape=(None,))
        for k in ("trial_key","source_file","side","support"):
            grp.create_dataset(k, data=meta[k], maxshape=(None,), dtype=str_dt)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_subjects", type=int, default=None)
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config.apply_overrides(n_subjects=args.n_subjects)
    force = args.force

    log("=" * 60)
    log(f"  step_segmentation  {SEGMENTATION_VERSION}")
    log(f"  {'--force' if force else 'incremental'}  "
        f"N={config.N_SUBJECTS}  TS={config.TS}pt  {config.SAMPLE_RATE}Hz")
    log("=" * 60)

    all_records = discover_csvs(config.DATA_DIR, config.N_SUBJECTS)
    if not all_records:
        raise FileNotFoundError(f"no CSV in {config.DATA_DIR}")

    cond_count: dict[int, int] = defaultdict(int)
    subj_set:   set[int]       = set()
    for r in all_records:
        cond_count[r["cond"]] += 1
        subj_set.add(r["sid"])
    log(f"  CSV: {len(all_records)} files  subjects: {len(subj_set)}")
    for c, n in sorted(cond_count.items()):
        exp_str = f"  (expected {COND_EXPECTED_BOUTS[c]} bouts/file)" if c in COND_EXPECTED_BOUTS else ""
        log(f"    C{c}: {n} files{exp_str}")

    t0 = time.time()
    pf_path = config.BATCH_DIR / "processed_files.json"

    # 항상 먼저 기본값으로 초기화 (done_files 미정의 버그 방지)
    existing_channels, existing_n, done_files = None, 0, set()
    zero_retries: dict[str, int] = {}

    if force:
        log("  ★ --force: full rebuild")
    else:
        existing_channels, existing_n = load_existing_h5_info(config.H5_PATH)
        done_files, pf_status, zero_retries = load_processed_files(pf_path)
        if pf_status != "valid":
            if config.H5_PATH.exists():
                log("  ⚠ invalid processed_files → safe rebuild")
                force = True
            else:
                log("  ⚠ invalid processed_files, no HDF5 → fresh start")
            existing_channels, existing_n, done_files = None, 0, set()
        elif pf_status == "valid":
            log(f"  ★ processed={len(done_files)}  HDF5={existing_n} steps")

    # zero_retries 초과 파일은 재시도 제외 (--force 시엔 무시)
    _skip_keys = {k for k, v in zero_retries.items() if v >= _ZERO_STEP_MAX_RETRIES}
    if _skip_keys and not force:
        log(f"  ⚠ zero_retries 한계 초과 skip: {len(_skip_keys)}개 파일 (--force로 재시도 가능)")
    target_records = all_records if force else [
        r for r in all_records
        if make_file_key(r) not in done_files and make_file_key(r) not in _skip_keys
    ]
    if not target_records:
        log(f"  ✅ all files done, steps={existing_n}")
        return

    if existing_channels:
        channels = existing_channels
        log(f"  [Pass 0] reuse channels: {len(channels)}")
        # 재사용 채널도 실제 파일과 대조 검증
        _verify_channels(target_records, channels)
    else:
        all_common = find_common_channels(all_records)
        from channel_groups import filter_raw_channels
        channels = filter_raw_channels(all_common)
        log(f"  [Pass 0] {len(all_common)} → {len(channels)} raw IMU channels")
        if not channels:
            raise ValueError("no raw IMU channels")

    n_ch = len(channels)
    log(f"  [Pass 2] {len(target_records)} files → bout → step")

    subj_bufs: dict[int, dict[str, list]] = defaultdict(lambda: {
        "X":[], "y":[], "trial_id":[], "trial_key":[], "bout_start":[], "bout_end":[],
        "step_start":[], "step_end":[], "trial_step_index":[],
        "source_file":[], "side":[], "support":[],
    })

    new_steps = 0
    raw_lens: list[int] = []
    processed_now:  set[str] = set()   # step >= 1인 파일만
    processed_zero: set[str] = set()   # zero_steps 파일 (재시도 대상)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = config.BATCH_DIR / f"step_log_{ts}.jsonl"
    log(f"  step log → {log_path}")

    with StepLogWriter(log_path) as step_logger:
        for i, rec in enumerate(target_records):
            file_key = make_file_key(rec)
            try:
                df = read_csv_with_retry(rec["path"])
                df = rename_columns(df)
                df = interpolate_sensor_gaps(df)
            except Exception as e:
                log(f"  ⚠ read fail: {rec['path'].name} ({e})")
                zero_retries[file_key] = zero_retries.get(file_key, 0) + 1
                continue

            sid, cond, label = rec["sid"], rec["cond"], rec["label"]

            missing_ch = set(channels) - set(df.columns)
            if missing_ch:
                log(f"  ⚠ {rec['path'].name}: {len(missing_ch)} missing channels, skip")
                zero_retries[file_key] = zero_retries.get(file_key, 0) + 1
                del df
                continue

            data_np = df[channels].values.astype(np.float32)

            # ── bout 분할 (C1/C6 구조 인식) ──
            bouts = detect_walking_bouts(df, fs=config.SAMPLE_RATE, cond=cond)
            expected_n = COND_EXPECTED_BOUTS.get(cond)
            if expected_n is not None and len(bouts) != expected_n:
                _log_bout_mismatch(step_logger, rec, len(bouts), expected_n)
            file_steps = 0

            for local_trial_id, (bout_start, bout_end) in enumerate(bouts, 1):
                if (bout_end - bout_start) < int(config.SAMPLE_RATE * BOUT_MIN_WALK_SEC):
                    continue

                df_bout   = df.iloc[bout_start:bout_end].reset_index(drop=True)
                data_bout = data_np[bout_start:bout_end]

                lt_steps, lt_sups = [], []
                rt_steps, rt_sups = [], []
                side_signals: dict[str, np.ndarray] = {}
                side_counts:  dict[str, int]        = {"LT": 0, "RT": 0}

                for side, s_steps, s_sups in [
                    ("LT", lt_steps, lt_sups),
                    ("RT", rt_steps, rt_sups),
                ]:
                    try:
                        ml_g = extract_ml_gyro(df_bout, side)
                        ap_a = extract_ap_accel(df_bout, side)
                        if ml_g is None or ap_a is None:
                            norm_f = bandpass_filter(compute_foot_acc_norm(df_bout, side))
                            raw, sups = _fallback_detect_steps(norm_f, config.SAMPLE_RATE)
                            sig_q = norm_f
                        else:
                            raw, sups = detect_steps(ml_g.copy(), ap_a.copy())
                            sig_q = ml_g
                        s_steps.extend(raw); s_sups.extend(sups)
                        side_signals[side] = sig_q
                        side_counts[side]  = len(raw)
                    except Exception:
                        side_signals[side] = np.full(len(data_bout), np.nan, dtype=np.float32)

                # ── polarity flip 재시도 (발별 품질 기준) ──────────────────
                for side, s_steps, s_sups, other in [
                    ("LT", lt_steps, lt_sups, "RT"),
                    ("RT", rt_steps, rt_sups, "LT"),
                ]:
                    other_n = side_counts[other]
                    this_n  = side_counts[side]
                    if other_n < 2 or this_n >= other_n * _FLIP_RATIO:
                        continue
                    try:
                        ml_g = extract_ml_gyro(df_bout, side)
                        ap_a = extract_ap_accel(df_bout, side)
                        if ml_g is None or ap_a is None:
                            continue
                        retry, rsups = detect_steps(ml_g.copy(), ap_a.copy(), force_flip=True)
                        if len(retry) <= this_n:
                            continue

                        # ── 품질 비교: 단순 개수 증가 아닌 3가지 기준 ──
                        # 1) 과폭증 방지: retry가 other_n의 1.5배 초과하면 거부
                        if len(retry) > other_n * 1.5:
                            log(f"    ✗ flip [{side}] 거부 (폭증 {this_n}→{len(retry)}, other={other_n})")
                            continue

                        # 2) stride CV 비교: retry가 더 규칙적이어야 채택
                        def _stride_cv(steps: list) -> float:
                            if len(steps) < 3:
                                return 1.0
                            ivs = np.diff([s for s, _ in steps]).astype(float)
                            return float(np.std(ivs)) / max(float(np.mean(ivs)), 1.0)

                        cv_orig  = _stride_cv([(s, e) for s, e in s_steps])
                        cv_retry = _stride_cv([(s, e) for s, e in retry])
                        # CV가 나빠지면서 개수만 늘어난 경우 거부
                        # 불균형 심할수록 CV 허용 완화 (this_n이 other_n의 절반 미만이면 2.0까지 허용)
                        cv_tolerance = 2.0 if this_n < other_n * 0.5 else 1.3
                        if cv_retry > cv_orig * cv_tolerance and len(retry) < this_n * 2:
                            log(f"    ✗ flip [{side}] 거부 (CV 악화 {cv_orig:.2f}→{cv_retry:.2f})")
                            continue

                        # 3) both support 비율 비교: 나빠지면 거부
                        def _both_ratio(sups: list) -> float:
                            return sups.count("both") / max(len(sups), 1)

                        br_orig  = _both_ratio(s_sups)
                        br_retry = _both_ratio(rsups)
                        if br_retry < br_orig * 0.5 and br_orig > 0.1:
                            log(f"    ✗ flip [{side}] 거부 (both 비율 저하 {br_orig:.2f}→{br_retry:.2f})")
                            continue

                        log(f"    ↺ flip [{side}] 채택: {this_n}→{len(retry)}"
                            f"  CV:{cv_orig:.2f}→{cv_retry:.2f}"
                            f"  both:{br_orig:.2f}→{br_retry:.2f}")
                        s_steps.clear(); s_sups.clear()
                        s_steps.extend(retry); s_sups.extend(rsups)
                        side_signals[side] = ml_g
                        side_counts[side]  = len(retry)
                    except Exception:
                        pass

                if side_counts["LT"] + side_counts["RT"] < MIN_STEPS_PER_BOUT:
                    continue

                nan_sig = np.full(len(data_bout), np.nan, dtype=np.float32)
                lt_m, lt_lo, lt_hi = estimate_stride_params_from_steps(lt_steps)
                rt_m, rt_lo, rt_hi = estimate_stride_params_from_steps(rt_steps)

                # ── 발별 baseline 통계 로그 ────────────────────────────────
                for _side, _steps, _sups, _m in [
                    ("LT", lt_steps, lt_sups, lt_m),
                    ("RT", rt_steps, rt_sups, rt_m),
                ]:
                    if _steps:
                        _lens = [e - s for s, e in _steps]
                        _cv   = float(np.std(_lens)) / max(float(np.mean(_lens)), 1.0)
                        _both = _sups.count("both") / max(len(_sups), 1)
                        step_logger.write({
                            "type": "bout_side_stats",
                            "file": rec["path"].name, "sid": sid, "cond": cond,
                            "trial_id": local_trial_id, "side": _side,
                            "n": len(_steps),
                            "stride_mean_ms": round(_m / config.SAMPLE_RATE * 1000, 1),
                            "stride_cv": round(_cv, 3),
                            "both_ratio": round(_both, 3),
                        })

                # ── LT/RT ratio hard filter ──────────────────────────────
                lt_n_raw = side_counts["LT"]
                rt_n_raw = side_counts["RT"]
                total_raw = lt_n_raw + rt_n_raw
                if total_raw >= MIN_STEPS_PER_BOUT:
                    lt_ratio = lt_n_raw / total_raw
                    if lt_ratio > 0.70 or lt_ratio < 0.30:
                        log(f"    ✗ bout{local_trial_id} skip: "
                            f"LT/RT={lt_n_raw}/{rt_n_raw} ({lt_ratio:.0%}/{1-lt_ratio:.0%}) — 극단 불균형")
                        continue

                lt_scored = score_steps_by_side(lt_steps, lt_sups, "LT",
                                                side_signals.get("LT", nan_sig), lt_m, lt_lo, lt_hi)
                rt_scored = score_steps_by_side(rt_steps, rt_sups, "RT",
                                                side_signals.get("RT", nan_sig), rt_m, rt_lo, rt_hi)

                merged = bilateral_sanity_check(merge_bilateral_steps(lt_scored, rt_scored))

                trial_step_idx = 0
                for step in merged:
                    s, e, side, score, support = step
                    raw_len = e - s
                    if raw_len < config.MIN_STEP_LEN:
                        continue
                    g_s, g_e = bout_start + s, bout_start + e
                    seg = resample_step(data_bout[s:e])

                    buf = subj_bufs[sid]
                    buf["X"].append(seg); buf["y"].append(label)
                    buf["trial_id"].append(local_trial_id)
                    buf["trial_key"].append(f"{rec['path'].name}__trial{local_trial_id}")
                    buf["bout_start"].append(bout_start); buf["bout_end"].append(bout_end)
                    buf["step_start"].append(g_s); buf["step_end"].append(g_e)
                    buf["trial_step_index"].append(trial_step_idx)
                    buf["source_file"].append(rec["path"].name)
                    buf["side"].append(side); buf["support"].append(support)

                    file_steps += 1; new_steps += 1; raw_lens.append(raw_len)
                    step_logger.write({
                        "file": rec["path"].name, "sid": sid, "cond": cond, "label": label,
                        "trial_id": local_trial_id,
                        "bout_start": bout_start, "bout_end": bout_end,
                        "step_start": g_s, "step_end": g_e,
                        "trial_step_index": trial_step_idx,
                        "side": side, "score": round(float(score), 6), "support": support,
                        "lt_raw_n": side_counts["LT"], "rt_raw_n": side_counts["RT"],
                    })
                    trial_step_idx += 1

            if file_steps > 0:
                processed_now.add(file_key)
            else:
                processed_zero.add(file_key)
                zero_retries[file_key] = zero_retries.get(file_key, 0) + 1
                step_logger.write({"file": rec["path"].name, "sid": sid,
                                   "cond": cond, "status": "zero_steps",
                                   "retry_count": zero_retries[file_key]})

            del df, data_np
            # gc는 20파일마다 1번 (매 파일 강제 GC 불필요)
            if (i + 1) % 20 == 0 or (i + 1) == len(target_records):
                gc.collect()
            if (i + 1) % 20 == 0 or (i + 1) == len(target_records):
                log(f"    {i+1}/{len(target_records)}  file_steps={file_steps}  total={new_steps}")

    log(f"\n  [Pass 2] done: {new_steps} steps")

    if force and config.H5_PATH.exists():
        config.H5_PATH.unlink()

    t2 = time.time()
    total_steps = existing_n

    with h5py.File(config.H5_PATH, "a") as hf:
        hf.require_group("subjects")
        for sid, buf in sorted(subj_bufs.items()):
            if not buf["X"]:
                continue
            X_arr = np.stack(buf["X"]).astype(np.float32)
            y_arr = np.array(buf["y"], dtype=np.int64)
            meta = {
                "trial_id":         np.array(buf["trial_id"],         dtype=np.int32),
                "trial_key":        np.array(buf["trial_key"],        dtype=object),
                "bout_start":       np.array(buf["bout_start"],       dtype=np.int32),
                "bout_end":         np.array(buf["bout_end"],         dtype=np.int32),
                "step_start":       np.array(buf["step_start"],       dtype=np.int32),
                "step_end":         np.array(buf["step_end"],         dtype=np.int32),
                "trial_step_index": np.array(buf["trial_step_index"], dtype=np.int32),
                "source_file":      np.array(buf["source_file"],      dtype=object),
                "side":             np.array(buf["side"],             dtype=object),
                "support":          np.array(buf["support"],          dtype=object),
            }
            write_subject_group(hf, sid, X_arr, y_arr, meta, n_ch)
            total_steps += len(X_arr)

        if "channels" in hf:
            del hf["channels"]
        hf.create_dataset("channels", data=np.array(channels, dtype="S"))
        hf.attrs.update({
            "segmentation":    "heel_strike",
            "sample_rate":     config.SAMPLE_RATE,
            "target_ts":       config.TS,
            "n_classes":       config.NUM_CLASSES,
            "format":          "subject_group_v10",
            "label_base":      0,
            "label_semantics": "terrain_condition",
            "step_definition": "trialwise_dual_signal_consensus_bilateral_merge",
            "trial_definition":"walking_bout_in_file_excluding_stop_turn",
            "trial_key_definition": "source_file__trial{trial_id}",
            "expected_bouts":  json.dumps(COND_EXPECTED_BOUTS, ensure_ascii=False),
            "label_mapping":   json.dumps({f"C{i+1}": i for i in range(config.NUM_CLASSES)},
                                           ensure_ascii=False),
        })

    log(f"  HDF5 write: {time.time()-t2:.1f}s")
    # processed_now = file_steps > 0만 (processed_zero와 배타적이므로 단순 저장)
    all_done = processed_now if force else (done_files | processed_now)
    # force 시엔 zero_retries 리셋
    save_zero_retries = {} if force else zero_retries
    save_processed_files(pf_path, all_done, save_zero_retries)
    if processed_zero:
        log(f"  ⚠ zero_steps: {len(processed_zero)}개 파일 (다음 실행에서 재시도)")
        zero_path = config.BATCH_DIR / "zero_steps_files.json"
        zero_path.write_text(
            json.dumps(sorted(processed_zero), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log(f"     zero_steps 목록 → {zero_path}")

    size_mb = config.H5_PATH.stat().st_size / 1024**2
    log(f"  ✅ {config.H5_PATH}  {size_mb:.1f} MB")
    log(f"     existing={existing_n} + new={new_steps} = total={total_steps}")

    if total_steps > 0:
        with h5py.File(config.H5_PATH, "r") as f:
            lc: dict[int, int] = defaultdict(int)
            for sk in f["subjects"]:
                for lbl in f[f"subjects/{sk}/y"][:]:
                    lc[int(lbl)] += 1
        log(f"     label dist: {dict(sorted(lc.items()))}")

    if raw_lens:
        log(f"     step length: min={min(raw_lens)} max={max(raw_lens)} "
            f"mean={np.mean(raw_lens):.0f} ({np.mean(raw_lens)/config.SAMPLE_RATE*1000:.0f}ms)")

    log(f"  elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()# 이미 중간에 있어야 함 - 확인 필요
