"""
step_segmentation.py — 힐스트라이크 스텝 분할 (v9.4-filewise)
═══════════════════════════════════════════════════════════════
아키텍처:
    Pass 0 : 공통 채널 (기존 HDF5 있으면 재사용)
    Pass 2 : 파일별 적응형 스텝 검출 → HDF5 추가

완전 파일 적응형 (v9.4-filewise):
  - detect_steps()     : 이 파일 신호 통계만으로 피크·stride 기준 결정
  - score_steps_by_side: 이 파일 raw_steps 간격으로 stride 파라미터 추정
  - 증분 판정         : processed_files.json (파일 키 기반)
  - terrain_params / Pass 1 완전 제거

사용법:
    python3 step_segmentation.py           # 증분 (미처리 파일만)
    python3 step_segmentation.py --force   # 전체 재생성
═══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys
import time
import re
import json
import gc
import argparse
from pathlib import Path
from types import TracebackType                     # ⑤
from typing import Optional
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

SEGMENTATION_VERSION   = "v9.4-filewise-consensus-2"  # processed_files 버전 — 로직 변경 시 올림
_SUPPORTED_H5_FORMATS = {
    "subject_group_v9",
    "subject_group_v8",
}


# ──────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    """타임스탬프 포함 로그 출력."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_csv_with_retry(
    path: Path,
    skiprows: int = 2,
    max_retries: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """일시적 I/O 오류에 대비한 재시도 CSV 읽기.

    Args:
        path: CSV 파일 경로
        skiprows: 헤더 스킵 행 수
        max_retries: 최대 재시도 횟수
        **kwargs: pd.read_csv 에 그대로 전달 (예: nrows=0)

    Returns:
        읽어들인 DataFrame

    Raises:
        최종 시도까지 실패하면 마지막 예외를 재발생
    """
    last_exc: Exception = RuntimeError("알 수 없는 오류")
    for attempt in range(max_retries):
        try:
            return pd.read_csv(path, skiprows=skiprows, low_memory=False, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
    raise last_exc


# ──────────────────────────────────────────────────────────────
# 컬럼명 통일 (Noraxon 버전/내보내기 설정 차이 보정)
# ──────────────────────────────────────────────────────────────

RENAME_MAP: dict[str, str] = {
    "Noraxon MyoMotion-Joints-Knee LT-Rotation Ext (deg)":
        "Knee Rotation Ext LT (deg)",
    "Noraxon MyoMotion-Joints-Knee RT-Rotation Ext (deg)":
        "Knee Rotation Ext RT (deg)",
}

_COL_NORMALIZE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+Segments[-\s]+"), ""),
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+Joints[-\s]+"), ""),
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+Trajectories[-\s]+"), "Trajectories "),
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+"), ""),
    (re.compile(r"Acceleration\b"), "Accel Sensor"),
]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Noraxon export 버전 간 컬럼명 차이를 통일한다."""
    renames = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    if renames:
        df = df.rename(columns=renames)

    new_names: dict[str, str] = {}
    for c in df.columns:
        new_c = c
        for pat, repl in _COL_NORMALIZE_RULES:
            new_c = pat.sub(repl, new_c)
        if new_c != c:
            new_names[c] = new_c

    if new_names:
        existing = set(df.columns)
        collisions = [
            (k, v) for k, v in new_names.items() if v in existing and v != k
        ]
        if collisions:
            for old, new in collisions[:5]:
                log(f"    ⚠ 컬럼 rename 충돌: '{old}' -> '{new}' (스킵)")
            if len(collisions) > 5:
                log(f"    ⚠ 추가 rename 충돌 {len(collisions) - 5}건")
        safe_renames = {
            k: v for k, v in new_names.items()
            if v not in existing or v == k
        }
        if safe_renames:
            df = df.rename(columns=safe_renames)

    return df


# ──────────────────────────────────────────────────────────────
# 1. 파일 탐색
# ──────────────────────────────────────────────────────────────

def parse_filename(fname: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """파일명에서 피험자·지형·시행 번호를 추출한다."""
    m = re.search(r"S(\d+)C(\d+)T(\d+)", fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"S(\d+)C(\d+)", fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), 1
    return None, None, None


def discover_csvs(data_dir: Path, n_subjects: int) -> list[dict]:
    """데이터 디렉토리에서 유효한 CSV 레코드를 수집한다."""
    if not data_dir.exists():
        raise FileNotFoundError(f"데이터 디렉토리 없음: {data_dir}")

    records: list[dict] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        sid, cond, trial = parse_filename(csv_path.name)
        if sid is None or sid > n_subjects:
            continue
        if cond is None or cond < 1 or cond > config.NUM_CLASSES:
            continue
        records.append({
            "path": csv_path,
            "sid": sid,
            "cond": cond,
            "trial": trial,
            "label": cond - 1,    # 학습용 class label: 0-based
        })
    return records


# ──────────────────────────────────────────────────────────────
# 2. 기존 HDF5 상태 확인
# ──────────────────────────────────────────────────────────────

def load_existing_h5_info(
    h5_path: Path,
) -> tuple[Optional[list[str]], int]:
    """기존 HDF5에서 채널 목록과 총 스텝 수를 읽는다.

    역할: 채널 일관성 검증 + step 수 집계 + format/step_definition 검증.
    완료 파일 판정은 processed_files.json 에서 담당하므로 done_sids 반환 없음.
    """
    if not h5_path.exists():
        return None, 0

    try:
        with h5py.File(h5_path, "r") as f:
            h5_format = f.attrs.get("format", "unknown")
            if h5_format == "unknown":
                log(
                    "  ℹ HDF5 format attr 없음: flat v7 이전 구조로 추정. "
                    "하위 호환으로 읽지만 step_definition 의미가 다를 수 있습니다."
                )
            elif h5_format not in _SUPPORTED_H5_FORMATS:
                log(
                    f"  ⚠ HDF5 포맷 비호환: '{h5_format}' "
                    f"(지원={_SUPPORTED_H5_FORMATS}). 데이터 구조가 다를 수 있습니다."
                )
            elif h5_format == "subject_group_v8":
                log(
                    "  ⚠ HDF5 포맷 v8 감지: 구조 확인용 읽기만 허용됩니다. "
                    "step_definition이 현재 버전과 다르면 append는 차단되므로 "
                    "--force로 전체 재생성을 권장합니다."
                )

            step_def = f.attrs.get("step_definition", "")
            if step_def and step_def != "filewise_dual_signal_consensus_bilateral_merge":
                log(
                    f"  ℹ step_definition='{step_def}' "
                    f"(현재 버전은 'filewise_dual_signal_consensus_bilateral_merge'). "
                    f"병합 로직이 다른 버전으로 생성된 데이터입니다."
                )
                raise ValueError(
                    f"step_definition 불일치 (저장='{step_def}'). "
                    f"--force 로 전체 재생성하거나 기존 HDF5를 백업 후 삭제하세요."
                )

            if "subjects" in f and "channels" in f:
                n_existing = sum(
                    f[f"subjects/{sk}/X"].shape[0] for sk in f["subjects"]
                )
                channels = [
                    c.decode() if isinstance(c, bytes) else c
                    for c in f["channels"][:]
                ]
                return channels, n_existing

            for key in ("X", "y", "subject_id", "channels"):
                if key not in f:
                    log(f"  ⚠ 기존 HDF5에 '{key}' 없음, 처음부터 생성")
                    return None, 0

            raise ValueError(
                "flat v7 HDF5가 감지되었습니다. "
                "v9 group 구조와 혼용하면 파일이 손상됩니다. "
                "--force로 전체 재생성하세요."
            )

    except ValueError:
        raise
    except Exception as e:
        log(f"  ⚠ 기존 HDF5 읽기 실패: {e}")
        return None, 0


# ──────────────────────────────────────────────────────────────
# 2b. 파일별 증분 처리 — processed_files.json
# ──────────────────────────────────────────────────────────────

def make_file_key(rec: dict) -> str:
    """CSV 레코드를 고유 파일 키로 변환한다.

    형식: S{sid:04d}|C{cond:02d}|T{trial:02d}|{filename}
    동일 피험자의 동일 조건 동일 trial 이라도 파일명이 다르면 다른 키.
    """
    return f"S{rec['sid']:04d}|C{rec['cond']:02d}|T{rec['trial']:02d}|{rec['path'].name}"


def load_processed_files(path: Path) -> tuple[set[str], str]:
    """processed_files.json 에서 완료된 파일 키 집합을 읽는다.

    Returns:
        (files, status)  status ∈ {"valid", "invalid", "missing"}
        "valid"   : 정상 로드
        "missing" : 파일 없음 (첫 실행)
        "invalid" : 버전 불일치 또는 파싱 실패
    """
    if not path.exists():
        return set(), "missing"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = raw.get("version", "unknown")
        if version != SEGMENTATION_VERSION:
            log(
                f"  ⚠ processed_files 버전 불일치 "
                f"(저장={version}, 현재={SEGMENTATION_VERSION}) "
                f"→ 기존 처리 이력 무효"
            )
            return set(), "invalid"
        return set(raw.get("files", [])), "valid"
    except Exception as e:
        log(f"  ⚠ processed_files 로드 실패: {e}")
        return set(), "invalid"


def save_processed_files(path: Path, files: set[str]) -> None:
    """완료된 파일 키 집합을 processed_files.json 에 저장한다."""
    payload = {
        "version": SEGMENTATION_VERSION,
        "files": sorted(files),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ──────────────────────────────────────────────────────────────
# 3. 공통 채널 교집합 계산
# ──────────────────────────────────────────────────────────────

def find_common_channels(records: list[dict]) -> list[str]:
    """모든 CSV의 컬럼 교집합을 구한다 (첫 파일 순서 보존)."""
    log("  [Pass 0] 전 파일 공통 채널 계산...")
    common: Optional[set[str]] = None
    first_cols_ordered: Optional[list[str]] = None

    for i, rec in enumerate(records):
        try:
            cols_df = read_csv_with_retry(rec["path"], nrows=0)
            cols_df = rename_columns(cols_df)
            cols = cols_df.columns.tolist()
            drop = set(config.resolve_drop_cols(cols))
            data_cols = set(c for c in cols if c not in drop)

            if common is None:
                common = data_cols
                first_cols_ordered = [c for c in cols if c not in drop]
            else:
                common = common & data_cols
        except Exception as e:
            log(f"    ⚠ 스캔 실패: {rec['path'].name} ({e})")
            continue

        if (i + 1) % 100 == 0:
            log(f"    {i+1}/{len(records)} 파일 스캔")

    if common is None or first_cols_ordered is None or len(common) == 0:
        raise ValueError("공통 채널이 없습니다. CSV 파일을 확인하세요.")

    channels = [c for c in first_cols_ordered if c in common]
    log(f"  공통 채널: {len(channels)}개")
    return channels


def verify_channels(records: list[dict], required_channels: list[str]) -> None:
    """신규 CSV들이 기존 채널을 모두 포함하는지 검증한다."""
    req_set = set(required_channels)
    warn_count = 0

    for rec in records:
        try:
            cols_df = read_csv_with_retry(rec["path"], nrows=0)
            cols_df = rename_columns(cols_df)
            cols = cols_df.columns.tolist()
            flex_drop = set(config.resolve_drop_cols(cols))
            available = set(c for c in cols if c not in flex_drop)
            missing = req_set - available
            if missing:
                warn_count += 1
                if warn_count <= 5:
                    log(f"    ⚠ 채널 부족: {rec['path'].name} (누락 {len(missing)}개)")
        except Exception:
            continue

    if warn_count > 0:
        log(f"    ⚠ 총 {warn_count}개 파일 채널 부족 (Pass 2에서 스킵됨)")
    else:
        log(f"    ✅ 신규 {len(records)}개 CSV 채널 호환 확인")


# ──────────────────────────────────────────────────────────────
# 4. 신호 추출 + 대역통과 필터
# ──────────────────────────────────────────────────────────────

def compute_foot_acc_norm(df: pd.DataFrame, side: str = "LT") -> np.ndarray:
    """발 가속도 센서의 3축 norm을 계산한다 (폴백용)."""
    cols = config.resolve_foot_acc_cols(df.columns.tolist(), side)
    ax = df[cols["x"]].values.astype(np.float64)
    ay = df[cols["y"]].values.astype(np.float64)
    az = df[cols["z"]].values.astype(np.float64)
    return np.sqrt(ax**2 + ay**2 + az**2)


def _resolve_sensor_axis(
    columns: list[str],
    sensor: str,
    axis: str,
    side: str,
    signal_type: str = "gyroscope",
) -> str | None:
    """특정 센서의 특정 축 채널명을 찾는다."""
    sensor_l = sensor.lower()
    axis_l = axis.lower()
    side_l = side.lower()
    type_l = signal_type.lower()

    for c in columns:
        cl = c.lower()
        has_sensor = sensor_l in cl
        has_side = side_l in cl or sensor_l == "pelvis"
        has_type = type_l in cl or ("accel" in type_l and "accel" in cl)
        has_axis = (
            cl.endswith(f"-{axis_l}") or
            cl.endswith(f"_{axis_l}") or
            f" {axis_l} " in cl or
            f"-{axis_l} " in cl or
            f" {axis_l} {side_l}" in cl
        )
        if has_sensor and has_side and has_type and has_axis:
            return c
    return None


def extract_ml_gyro(df: pd.DataFrame, side: str) -> np.ndarray | None:
    """ML(Mediolateral) Gyroscope 신호를 추출한다."""
    col = _resolve_sensor_axis(
        df.columns.tolist(), config.HS_GYRO_SENSOR,
        config.HS_GYRO_AXIS, side, "gyroscope",
    )
    if col is None:
        return None
    return df[col].values.astype(np.float64)


def extract_ap_accel(df: pd.DataFrame, side: str) -> np.ndarray | None:
    """AP(Anteroposterior) Accelerometer 신호를 추출한다."""
    col = _resolve_sensor_axis(
        df.columns.tolist(), config.HS_ACCEL_SENSOR,
        config.HS_ACCEL_AXIS, side, "accel",
    )
    if col is None:
        return None
    return df[col].values.astype(np.float64)


def bandpass_filter(
    signal: np.ndarray,
    fs: int = config.SAMPLE_RATE,
    low: float = config.BANDPASS_LOW,
    high: float = config.BANDPASS_HIGH,
    order: int = config.BANDPASS_ORDER,
) -> np.ndarray:
    """Butterworth 대역통과 필터를 적용한다."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    mask = np.isnan(signal)
    if mask.all():
        return signal
    sig_clean = signal.copy()
    sig_clean[mask] = np.nanmean(signal)
    try:
        filtered = filtfilt(b, a, sig_clean)
    except ValueError:
        return sig_clean
    filtered[mask] = np.nan
    return filtered


# ──────────────────────────────────────────────────────────────
# 6. 힐스트라이크 검출 (완전 파일 적응형)
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# 6. 힐스트라이크 검출 — dual-signal consensus (파일 적응형)
# ──────────────────────────────────────────────────────────────

def _adaptive_peaks(
    sig: np.ndarray,
    fs: int,
    negate: bool = False,
    loose_dist_ms: float = 250.0,
) -> tuple[np.ndarray, float]:
    """파일 신호 통계로 prominence·min_dist를 자동 결정하고 피크를 반환한다.

    Returns:
        (peaks, sigma)
    """
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


def _detect_hs_acc(
    ap_accel: np.ndarray,
    fs: int,
) -> np.ndarray:
    """AP Accel 기반 힐스트라이크 후보 검출.

    HS 시점에서 AP 가속도가 감속(음의 피크)하는 것을 검출.
    local dip sharpness 필터로 단순 노이즈성 dip을 제거한다.
    """
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

    # local dip sharpness: ±100ms 창에서 peak depth가 sigma × 0.5 이상인 것만 유지
    win = int(0.10 * fs)
    trusted = []
    for p in peaks:
        l = max(0, p - win)
        r = min(n, p + win)
        w = sig[l:r]
        if len(w) == 0:
            continue
        local_depth = float(np.nanmax(w)) - float(sig[p])
        if local_depth > 0.3 * sigma:  # 0.5→0.3: acc 후보 과탈락 방지
            trusted.append(p)

    return np.array(trusted, dtype=int)


def _detect_hs_gyro(
    ml_gyro: np.ndarray,
    fs: int,
    force_flip: bool = False,
) -> np.ndarray:
    """ML Gyro 기반 힐스트라이크 후보 검출.

    파일별 polarity를 자동 점검한다:
    원신호와 부호반전 신호 중 swing-HS 구조(주기성)가 더 좋은 쪽을 선택.

    Args:
        force_flip: True면 자동 선택 결과를 강제로 반전.
                    한 발 스텝 수가 반대 발보다 현저히 적을 때 외부 재시도용.
    """
    n = len(ml_gyro)
    sig_raw = ml_gyro.copy()
    mask = np.isnan(sig_raw)
    if mask.any() and not mask.all():
        x = np.arange(n)
        sig_raw[mask] = np.interp(x[mask], x[~mask], sig_raw[~mask])

    sig_raw = bandpass_filter(sig_raw, fs, low=0.5, high=15.0)

    def _score_polarity(sig: np.ndarray) -> tuple[np.ndarray, float]:
        """swing 후보 검출 수 × prominence 합 / (1 + CV) — 주기성 보정."""
        sigma = float(np.std(sig))
        if sigma == 0:
            return np.array([], dtype=int), 0.0
        loose_dist = int(fs * 0.25)
        swings, props = find_peaks(sig, prominence=0.2 * sigma, distance=loose_dist)
        if len(swings) == 0:
            return np.array([], dtype=int), 0.0
        proms = props["prominences"]
        if len(swings) >= 2:
            ivs = np.diff(swings).astype(float)
            cv = float(np.std(ivs)) / max(float(np.mean(ivs)), 1.0)
        else:
            cv = 1.0
        score = float(np.sum(proms)) / (1.0 + cv)
        return swings, score

    swings_orig, score_orig = _score_polarity(sig_raw)
    swings_flip, score_flip = _score_polarity(-sig_raw)

    # 자동 선택 후 force_flip이면 반전
    auto_flip = score_flip > score_orig
    use_flip  = auto_flip ^ force_flip  # XOR: force_flip이면 반대 선택

    if use_flip:
        sig = -sig_raw
        swing_peaks = swings_flip
    else:
        sig = sig_raw
        swing_peaks = swings_orig

    sigma = float(np.std(sig))
    if sigma == 0:
        return np.array([], dtype=int)

    hs_cands, _ = _adaptive_peaks(sig, fs, negate=True)
    if len(hs_cands) == 0:
        return np.array([], dtype=int)

    half_win = fs // 2
    if len(swing_peaks) == 0:
        return hs_cands

    peak_ref = float(np.percentile(sig[swing_peaks], 90))
    thresh = config.HS_TRUSTED_SWING * peak_ref

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
    hs_acc: np.ndarray,
    hs_gyro: np.ndarray,
    tol_ms: float,
    fs: int,
) -> list[tuple[int, str]]:
    """AP accel 후보와 ML gyro 후보를 합의(consensus)한다.

    Args:
        hs_acc:  AP accel 기반 후보 (sample index)
        hs_gyro: ML gyro 기반 후보 (sample index)
        tol_ms:  매칭 허용 오차 (ms)
        fs:      샘플링 레이트

    Returns:
        list of (timestamp, support)
        support ∈ {"both", "acc", "gyro"}
        anchor는 AP accel 시점 우선 (충격 기반 정의)
    """
    tol = int(fs * tol_ms / 1000)
    result: list[tuple[int, str]] = []
    used_gyro: set[int] = set()

    # AP 후보 기준으로 gyro 매칭
    for a in hs_acc:
        if len(hs_gyro) == 0:
            result.append((int(a), "acc"))
            continue
        dists = np.abs(hs_gyro - a)
        nearest_idx = int(np.argmin(dists))
        if dists[nearest_idx] <= tol and nearest_idx not in used_gyro:
            used_gyro.add(nearest_idx)
            result.append((int(a), "both"))   # anchor = AP 시점
        else:
            result.append((int(a), "acc"))

    # gyro 전용 후보 추가 (AP와 매칭 안 된 것)
    for gi, g in enumerate(hs_gyro):
        if gi not in used_gyro:
            result.append((int(g), "gyro"))

    result.sort(key=lambda x: x[0])
    return result


def _filewise_stride_filter(
    candidates: list[tuple[int, str]],
    fs: int,
) -> list[tuple[int, str]]:
    """파일 내 stride 통계 기반으로 이상 간격 후보를 제거한다."""
    if len(candidates) < 2:
        return candidates

    times = np.array([t for t, _ in candidates], dtype=float)

    # "both" 신뢰 후보 중심으로 interval 통계 추정, 부족하면 전체로 fallback
    trusted_times = np.array([t for t, s in candidates if s == "both"], dtype=float)
    ref_times = trusted_times if len(trusted_times) >= 4 else times
    intervals = np.diff(ref_times)

    if len(intervals) >= 4:
        med = float(np.median(intervals))
        mad = float(np.median(np.abs(intervals - med)))
        k = 2.5
        i_min = max(med - k * mad, config.HS_MIN_STRIDE_SAM)
        i_max = min(med + k * mad, config.HS_MAX_STRIDE_SAM)
        # p5/p95도 같이 사용
        p5  = float(np.percentile(intervals, 5))
        p95 = float(np.percentile(intervals, 95))
        i_min = max(int(min(p5,  i_min)), config.HS_MIN_STRIDE_SAM)
        i_max = min(int(max(p95, i_max)), config.HS_MAX_STRIDE_SAM)
    else:
        i_min = config.HS_MIN_STRIDE_SAM
        i_max = config.HS_MAX_STRIDE_SAM

    # 시작점은 항상 유지, interval 기준으로 end 후보 제거
    kept: list[tuple[int, str]] = [candidates[0]]
    for i in range(1, len(candidates)):
        gap = candidates[i][0] - kept[-1][0]
        if i_min <= gap <= i_max:
            kept.append(candidates[i])
        # 너무 짧으면 둘 중 support 높은 것 유지
        elif gap < i_min:
            sup_order = {"both": 2, "acc": 1, "gyro": 1, "norm": 0}
            if sup_order.get(candidates[i][1], 0) > sup_order.get(kept[-1][1], 0):
                kept[-1] = candidates[i]
        # 너무 길면 그냥 추가 (중간 이벤트 누락 가능)
        else:
            kept.append(candidates[i])

    return kept


def detect_steps(
    ml_gyro: np.ndarray,
    ap_accel: np.ndarray,
    fs: int = config.SAMPLE_RATE,
    force_flip: bool = False,
) -> tuple[list[tuple[int, int]], list[str]]:
    """Dual-signal consensus 힐스트라이크 검출 — 완전 파일 적응형.

    AP accel과 ML gyro 각각 독립 후보 생성 후 합의.
    최종 anchor는 AP accel 시점 우선.

    Args:
        force_flip: True면 ML gyro polarity 자동 선택을 강제 반전.
                    한 발 스텝이 반대 발보다 현저히 적을 때 외부 재시도용.

    Returns:
        (steps, supports)
        steps:    [(start, end), ...]
        supports: ["both"|"acc"|"gyro", ...]  — steps와 같은 길이
    """
    n = len(ml_gyro)
    if n < fs:
        return [], []

    nan_ratio = (np.isnan(ml_gyro).sum() + np.isnan(ap_accel).sum()) / (2 * n)
    if nan_ratio > config.HS_NAN_THRESHOLD:
        return [], []

    tol_ms = 100.0

    hs_acc  = _detect_hs_acc(ap_accel, fs)
    hs_gyro = _detect_hs_gyro(ml_gyro, fs, force_flip=force_flip)

    if len(hs_acc) == 0 and len(hs_gyro) == 0:
        return [], []

    candidates = _reconcile_candidates(hs_acc, hs_gyro, tol_ms, fs)
    candidates = _filewise_stride_filter(candidates, fs)

    if len(candidates) < 2:
        return [], []

    steps: list[tuple[int, int]] = []
    supports: list[str] = []
    for i in range(len(candidates) - 1):
        start, sup_start = candidates[i]
        end,   sup_end   = candidates[i + 1]
        length = end - start
        if length <= 0:
            continue
        nan_ml  = float(np.isnan(ml_gyro[start:end]).sum()) / length
        nan_ap  = float(np.isnan(ap_accel[start:end]).sum()) / length
        nan_seg = max(nan_ml, nan_ap)
        if nan_seg > config.HS_NAN_THRESHOLD:
            continue
        steps.append((start, end))
        sup_order = {"both": 3, "acc": 2, "gyro": 1, "norm": 0}
        step_sup = sup_start if sup_order.get(sup_start, 0) <= sup_order.get(sup_end, 0) \
                   else sup_end
        supports.append(step_sup)

    return steps, supports


def _fallback_detect_steps(
    norm_f: np.ndarray,
    fs: int,
) -> tuple[list[tuple[int, int]], list[str]]:
    """폴백: 발 가속도 norm 기반 파일 적응형 검출.

    ML gyro 또는 AP accel 중 하나가 없을 때 사용.
    support는 모두 "norm"으로 표시.
    """
    valid = norm_f[~np.isnan(norm_f)]
    if len(valid) == 0:
        return [], []

    mu    = float(np.mean(valid))
    sigma = float(np.std(valid))

    loose_peaks, _ = find_peaks(
        norm_f, height=mu + 0.3 * sigma,
        distance=int(fs * 0.25),
    )
    if len(loose_peaks) >= 4:
        ivs = np.diff(loose_peaks).astype(float)
        med = float(np.median(ivs))
        s_min = max(int(med * 0.50), config.HS_MIN_STRIDE_SAM)
        s_max = min(int(med * 1.50), config.HS_MAX_STRIDE_SAM)
        min_d = int(np.clip(med * 0.40, fs * 0.25, fs * 0.80))  # 상한 clamp
    else:
        s_min = config.HS_MIN_STRIDE_SAM
        s_max = config.HS_MAX_STRIDE_SAM
        min_d = config.HS_MIN_STRIDE_SAM

    peaks, _ = find_peaks(
        norm_f, height=mu + 0.5 * sigma, distance=min_d,
    )
    steps = [
        (int(peaks[j]), int(peaks[j + 1]))
        for j in range(len(peaks) - 1)
        if s_min <= peaks[j + 1] - peaks[j] <= s_max
    ]
    supports = ["norm"] * len(steps)   # foot accel norm 기반 폴백임을 명시
    return steps, supports


# ──────────────────────────────────────────────────────────────
# 6b. 파일별 stride 파라미터 추정
# ──────────────────────────────────────────────────────────────

def estimate_stride_params_from_steps(
    steps: list[tuple[int, int]],
) -> tuple[float, float, float]:
    """raw_steps 간격에서 이 파일의 stride 통계를 직접 추정한다.

    Returns:
        (stride_mean, stride_min, stride_max) — 단위: samples
    """
    if not steps:
        return 0.0, float(config.HS_MIN_STRIDE_SAM), float(config.HS_MAX_STRIDE_SAM)

    lengths = np.array([e - s for s, e in steps], dtype=float)
    stride_mean = float(np.mean(lengths))

    if len(lengths) >= 4:
        stride_min = float(max(np.percentile(lengths, 5),  config.HS_MIN_STRIDE_SAM))
        stride_max = float(min(np.percentile(lengths, 95), config.HS_MAX_STRIDE_SAM))
    else:
        stride_min = float(config.HS_MIN_STRIDE_SAM)
        stride_max = float(config.HS_MAX_STRIDE_SAM)

    return stride_mean, stride_min, stride_max


# ──────────────────────────────────────────────────────────────
# 7. 품질 점수화 + 병합 + bilateral sanity check
# ──────────────────────────────────────────────────────────────

# support → q_support 가중치
_SUPPORT_WEIGHT: dict[str, float] = {
    "both": 1.00,   # AP + gyro 둘 다 확인
    "acc":  0.80,   # AP accel만
    "gyro": 0.75,   # ML gyro만
    "norm": 0.65,   # foot accel norm 폴백 (ML gyro / AP accel 없을 때)
}


def _step_quality(
    start: int,
    end: int,
    signal: np.ndarray,
    stride_mean_sam: float,
    stride_min_sam: float,
    stride_max_sam: float,
    support: str = "both",
) -> float:
    """스텝 품질 점수 = q_nan × q_interval × q_support"""
    length = end - start
    if length <= 0:
        return 0.0

    # q_nan
    nan_ratio = float(np.isnan(signal[start:end]).sum()) / length
    q_nan = 1.0 - nan_ratio

    # q_interval (stride 중심 근접도)
    if stride_mean_sam > 0:
        stride_center = stride_mean_sam
        stride_range  = max(stride_max_sam - stride_min_sam, 1.0)
    elif stride_min_sam > 0 and stride_max_sam > stride_min_sam:
        stride_center = (stride_min_sam + stride_max_sam) / 2.0
        stride_range  = max(stride_max_sam - stride_min_sam, 1.0)
    else:
        stride_center = length
        stride_range  = 1.0
    q_interval = 1.0 / (1.0 + abs(length - stride_center) / stride_range)

    # q_support
    q_support = _SUPPORT_WEIGHT.get(support, 0.75)

    return q_nan * q_interval * q_support


# ScoredStep: NamedTuple
from typing import NamedTuple

class ScoredStep(NamedTuple):
    start:   int
    end:     int
    side:    str    # "LT" 또는 "RT"
    score:   float  # 품질 점수
    support: str    # "both" / "acc" / "gyro" / "norm"


def score_steps_by_side(
    steps: list[tuple[int, int]],
    supports: list[str],
    side: str,
    signal: np.ndarray,
    stride_mean_sam: float,
    stride_min_sam: float,
    stride_max_sam: float,
) -> list[ScoredStep]:
    """각 스텝에 dual-signal support 포함 품질 점수를 부여한다."""
    scored: list[ScoredStep] = []
    for i, (start, end) in enumerate(steps):
        if i >= len(supports):
            log(
                f"  ⚠ score_steps_by_side [{side}]: supports 길이 불일치 "
                f"(steps={len(steps)}, supports={len(supports)}) "
                f"— 이후 {len(steps)-i}개 스텝 스킵"
            )
            break
        sup = supports[i]
        score = _step_quality(
            start, end, signal,
            stride_mean_sam, stride_min_sam, stride_max_sam,
            support=sup,
        )
        scored.append(ScoredStep(start=start, end=end, side=side,
                                 score=score, support=sup))
    return scored


def merge_bilateral_steps(
    lt_scored: list[ScoredStep],
    rt_scored: list[ScoredStep],
    overlap_threshold: float = 0.5,
) -> list[ScoredStep]:
    """LT/RT 스텝을 병합한다.

    겹침 제거는 같은 발(same-side) 내에서만 적용.
    LT↔RT 교차는 정상 보행 패턴이므로 허용.
    """
    def _dedup_same_side(steps: list[ScoredStep]) -> list[ScoredStep]:
        if len(steps) < 2:
            return steps
        steps = sorted(steps, key=lambda x: x.start)
        kept: list[int] = [0]
        for ci in range(1, len(steps)):
            pi = kept[-1]
            overlap = max(0, min(steps[pi].end, steps[ci].end)
                             - max(steps[pi].start, steps[ci].start))
            if overlap == 0:
                kept.append(ci)
                continue
            shorter = min(steps[pi].end - steps[pi].start,
                          steps[ci].end  - steps[ci].start)
            if shorter == 0 or overlap / shorter > overlap_threshold:
                if steps[ci].score >= steps[pi].score:
                    kept[-1] = ci
            else:
                kept.append(ci)
        return [steps[i] for i in kept]

    lt_clean = _dedup_same_side(lt_scored)
    rt_clean = _dedup_same_side(rt_scored)
    merged = lt_clean + rt_clean
    merged.sort(key=lambda x: x.start)
    return merged


def bilateral_sanity_check(
    steps: list[ScoredStep],
    fs: int = config.SAMPLE_RATE,
) -> list[ScoredStep]:
    """LT/RT 교차 패턴 검증.

    체크 1. 동시 이벤트: LT-RT가 50ms 이내 → 낮은 점수 쪽 제거
    체크 2. side balance: 한 발이 70% 이상이면 경고 (제거 안 함)
    """
    if len(steps) < 2:
        return steps

    simultaneous_tol = int(fs * 0.05)  # 50ms

    # 체크 1: 동시 LT-RT 제거
    cleaned: list[ScoredStep] = [steps[0]]
    for curr in steps[1:]:
        prev = cleaned[-1]
        if (prev.side != curr.side
                and abs(curr.start - prev.start) <= simultaneous_tol):
            # 동시 이벤트: 점수 낮은 것 버림
            if curr.score > prev.score:
                cleaned[-1] = curr
            # else: prev 유지
        else:
            cleaned.append(curr)

    # 체크 2: side balance 경고
    lt_n = sum(1 for s in cleaned if s.side == "LT")
    rt_n = sum(1 for s in cleaned if s.side == "RT")
    total = lt_n + rt_n
    if total > 0:
        lt_ratio = lt_n / total
        if lt_ratio > 0.70 or lt_ratio < 0.30:
            log(f"    ⚠ bilateral balance 불균형: LT={lt_n} RT={rt_n} "
                f"({lt_ratio:.0%}/{1-lt_ratio:.0%})")

    return cleaned


# ──────────────────────────────────────────────────────────────
# 8. 리샘플링
# ──────────────────────────────────────────────────────────────

def resample_step(
    data_segment: np.ndarray,
    target_length: int = config.TS,
) -> np.ndarray:
    """가변 길이 스텝 세그먼트를 고정 길이로 리샘플링한다."""
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
# 9. step_log 스트리밍
# ──────────────────────────────────────────────────────────────

class StepLogWriter:
    """JSONL 형식으로 스텝 로그를 스트리밍 기록한다."""

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

    def __exit__(                           # ⑤ TracebackType 명시
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────
# 10. CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    p = argparse.ArgumentParser(description="힐스트라이크 스텝 분할")
    p.add_argument(
        "--n_subjects", type=int, default=None,
        help="피험자 수 (기본: config.N_SUBJECTS)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="기존 HDF5 무시, 전체 재생성",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 11. 메인
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """증분 힐스트라이크 분할 파이프라인을 실행한다."""
    args = parse_args()
    config.apply_overrides(n_subjects=args.n_subjects)
    force = args.force

    log(f"{'='*60}")
    log(f"  step_segmentation.py v9.4-filewise ({'전체 재생성' if force else '증분 모드'})")
    log(f"  N={config.N_SUBJECTS}  TS={config.TS}pt  {config.SAMPLE_RATE}Hz")
    log(f"{'='*60}\n")

    all_records = discover_csvs(config.DATA_DIR, config.N_SUBJECTS)
    log(f"  CSV 파일: {len(all_records)}개")
    if not all_records:
        raise FileNotFoundError(f"CSV 없음: {config.DATA_DIR}")

    cond_count: dict[int, int] = defaultdict(int)
    subj_set: set[int] = set()
    for r in all_records:
        cond_count[r["cond"]] += 1
        subj_set.add(r["sid"])

    log(f"  피험자: {sorted(subj_set)} ({len(subj_set)}명)")
    for c in sorted(cond_count):
        log(f"    C{c}: {cond_count[c]}파일")

    t0 = time.time()
    processed_files_path = config.BATCH_DIR / "processed_files.json"

    if force:
        existing_channels: Optional[list[str]] = None
        existing_n: int = 0
        done_files: set[str] = set()
        log("  ★ --force: 기존 HDF5·processed_files 무시, 전체 재생성")
    else:
        existing_channels, existing_n = load_existing_h5_info(config.H5_PATH)
        done_files, pf_status = load_processed_files(processed_files_path)

        if pf_status != "valid" and config.H5_PATH.exists():
            log("  ⚠ processed_files가 무효하므로 기존 HDF5와 일관성을 보장할 수 없습니다.")
            log("  ⚠ --force와 동일하게 전체 재생성 모드로 전환합니다.")
            force = True
            existing_channels = None
            existing_n = 0
            done_files = set()
            log("  ★ 안전 모드 전환 완료: 이후 전체 재생성 경로로 진행합니다.")
        elif pf_status == "valid":
            if done_files:
                log(f"  ★ 기존 processed_files: {len(done_files)}개 파일 완료")
            if existing_n:
                log(f"  ★ 기존 HDF5: {existing_n}스텝")

    new_records = [r for r in all_records if make_file_key(r) not in done_files]

    target_records = all_records if force else new_records

    if not target_records:
        log(f"\n  ✅ 모든 {len(done_files)}개 파일 처리 완료, 스텝 검출 스킵")
        log(f"     기존 HDF5: {config.H5_PATH} ({existing_n}스텝)")
        return

    target_sids = sorted(set(r["sid"] for r in target_records))
    mode_text = "전체" if force else "신규"
    log(f"\n  {mode_text} 피험자: {target_sids} ({len(target_sids)}명)")
    log(f"  {mode_text} CSV: {len(target_records)}개")

    if existing_channels:
        channels = existing_channels
        log(f"  [Pass 0] 기존 채널 재사용: {len(channels)}개 (일관성 유지)")
        verify_channels(target_records, channels)
    else:
        all_common = find_common_channels(all_records)
        from channel_groups import filter_raw_channels
        channels = filter_raw_channels(all_common)
        log(f"  [Pass 0] Raw IMU 필터: {len(all_common)}ch → {len(channels)}ch")
        if len(channels) == 0:
            raise ValueError("Raw IMU 채널(Accel/Gyro)을 찾을 수 없습니다")

    n_ch = len(channels)

    log(f"  [Pass 2] {mode_text} {len(target_records)}개 CSV → 스텝 검출...")
    log(
        "  ⚠ 주의: v9.0부터 LT/RT 병합 적용 → 데이터셋 의미 변경 "
        "('발별 step' → '파일 단위 시간순 통합 step'). "
        "이전 버전과 직접 비교 불가."
    )
    t1 = time.time()

    subj_bufs: dict[int, dict[str, list]] = defaultdict(lambda: {"X": [], "y": []})
    new_steps = 0
    raw_lens: list[int] = []
    log_path = config.BATCH_DIR / "step_log.jsonl"
    processed_now: set[str] = set()  # 이번 실행에서 성공적으로 처리된 파일 키

    with StepLogWriter(log_path) as step_logger:
        for i, rec in enumerate(target_records):
            file_key = make_file_key(rec)
            try:
                df = read_csv_with_retry(rec["path"])
                df = rename_columns(df)
            except Exception as e:
                log(f"  ⚠ CSV 읽기 실패: {rec['path'].name} ({e})")
                continue  # 읽기 실패 → 완료 처리 안 함

            sid = rec["sid"]
            cond = rec["cond"]
            label = rec["label"]

            data_cols = [c for c in channels if c in df.columns]
            if len(data_cols) < len(channels):
                missing = set(channels) - set(data_cols)
                log(f"  ⚠ {rec['path'].name}: {len(missing)}개 채널 누락, 스킵")
                del df
                gc.collect()
                continue  # 채널 누락 → 완료 처리 안 함

            data_np = df[channels].values.astype(np.float32)

            lt_raw_steps: list[tuple[int, int]] = []
            rt_raw_steps: list[tuple[int, int]] = []
            lt_supports:  list[str] = []
            rt_supports:  list[str] = []
            side_signals: dict[str, np.ndarray] = {}
            side_raw_counts: dict[str, int] = {"LT": 0, "RT": 0}

            for side, raw_bucket, sup_bucket in [
                ("LT", lt_raw_steps, lt_supports),
                ("RT", rt_raw_steps, rt_supports),
            ]:
                try:
                    ml_gyro  = extract_ml_gyro(df, side)
                    ap_accel = extract_ap_accel(df, side)

                    if ml_gyro is None or ap_accel is None:
                        # 폴백: 발 가속도 norm
                        norm   = compute_foot_acc_norm(df, side)
                        norm_f = bandpass_filter(norm)
                        raw_steps, supports = _fallback_detect_steps(
                            norm_f, config.SAMPLE_RATE
                        )
                        sig_q = norm_f
                    else:
                        raw_steps, supports = detect_steps(
                            ml_gyro.copy(), ap_accel.copy()
                        )
                        sig_q = ml_gyro

                    raw_bucket.extend(raw_steps)
                    sup_bucket.extend(supports)
                    side_signals[side] = sig_q
                    side_raw_counts[side] = len(raw_steps)

                except (KeyError, ValueError):
                    side_signals[side] = np.full(len(data_np), np.nan, dtype=np.float32)

            # ── polarity 재시도: 한 발이 반대 발보다 현저히 적으면 flip ──
            # 기준: 한 발이 다른 발의 30% 미만이고, 둘 다 최소 2스텝 이상
            _FLIP_RATIO = 0.30
            _flip_tried: set[str] = set()
            for side, raw_bucket, sup_bucket, other_side in [
                ("LT", lt_raw_steps, lt_supports, "RT"),
                ("RT", rt_raw_steps, rt_supports, "LT"),
            ]:
                other_n = side_raw_counts[other_side]
                this_n  = side_raw_counts[side]
                if other_n >= 2 and this_n < other_n * _FLIP_RATIO:
                    try:
                        ml_g = extract_ml_gyro(df, side)
                        ap_a = extract_ap_accel(df, side)
                        if ml_g is not None and ap_a is not None:
                            retry_steps, retry_sups = detect_steps(
                                ml_g.copy(), ap_a.copy(), force_flip=True
                            )
                            if len(retry_steps) > this_n:
                                log(f"    ↺ polarity flip [{side}]: "
                                    f"{this_n} → {len(retry_steps)}스텝 (flip 적용)")
                                raw_bucket.clear()
                                sup_bucket.clear()
                                raw_bucket.extend(retry_steps)
                                sup_bucket.extend(retry_sups)
                                side_signals[side] = ml_g
                                side_raw_counts[side] = len(retry_steps)
                                _flip_tried.add(side)
                    except (KeyError, ValueError):
                        pass

            # side별 stride 파라미터 (생리적으로 same-side 기준이 맞음)
            lt_stride_mean, lt_stride_min, lt_stride_max = \
                estimate_stride_params_from_steps(lt_raw_steps)
            rt_stride_mean, rt_stride_min, rt_stride_max = \
                estimate_stride_params_from_steps(rt_raw_steps)
            # 파일 전체 통계는 bilateral sanity / 로그용
            file_stride_mean_sam, file_stride_min_sam, file_stride_max_sam = \
                estimate_stride_params_from_steps(lt_raw_steps + rt_raw_steps)

            nan_sig = np.full(len(data_np), np.nan, dtype=np.float32)
            lt_scored: list[ScoredStep] = score_steps_by_side(
                lt_raw_steps, lt_supports, "LT",
                side_signals.get("LT", nan_sig),
                lt_stride_mean, lt_stride_min, lt_stride_max,
            )
            rt_scored: list[ScoredStep] = score_steps_by_side(
                rt_raw_steps, rt_supports, "RT",
                side_signals.get("RT", nan_sig),
                rt_stride_mean, rt_stride_min, rt_stride_max,
            )

            # same-side 중복 제거 → bilateral 병합 → sanity check
            merged_steps = bilateral_sanity_check(
                merge_bilateral_steps(lt_scored, rt_scored)
            )
            file_steps = 0

            for step in merged_steps:
                start, end, side, _score, _support = step
                raw_len = end - start
                if raw_len < config.MIN_STEP_LEN:
                    continue

                seg_256 = resample_step(data_np[start:end])
                subj_bufs[sid]["X"].append(seg_256)
                subj_bufs[sid]["y"].append(label)
                file_steps += 1
                new_steps += 1
                raw_lens.append(raw_len)

                step_logger.write({
                    "file": rec["path"].name,
                    "sid": sid,
                    "cond": cond,
                    "label": label,
                    "side": side,
                    "start": start,
                    "end": end,
                    "raw_len": raw_len,
                    "score": round(float(_score), 6),
                    "support": _support,
                    "stride_mean_sam": round(file_stride_mean_sam, 1),
                    "stride_min_sam": int(file_stride_min_sam),
                    "stride_max_sam": int(file_stride_max_sam),
                    "lt_raw_n": side_raw_counts["LT"],
                    "rt_raw_n": side_raw_counts["RT"],
                })

            del df, data_np
            gc.collect()

            # 처리 상태 분류: "steps" / "zero_steps" (검출 실패와 원래 없는 것 구분 가능)
            proc_status = "steps" if file_steps > 0 else "zero_steps"
            processed_now.add(file_key)  # 파일 읽기·채널 검증 성공 기준

            if proc_status == "zero_steps":
                step_logger.write({
                    "file": rec["path"].name,
                    "sid": sid, "cond": cond, "label": label,
                    "status": "zero_steps",
                    "lt_raw_n": side_raw_counts["LT"],
                    "rt_raw_n": side_raw_counts["RT"],
                })

            if (i + 1) % 20 == 0 or (i + 1) == len(target_records):
                log(
                    f"    {i+1}/{len(target_records)} 파일"
                    f"  최종 병합: {file_steps} 스텝"
                    f"  누적: {new_steps}"
                )

    log(
        f"\n  [Pass 2] 스텝 검출 완료: {new_steps}스텝"
        f"  로그: {step_logger.count}건 → {log_path}"
        f"  ({time.time()-t1:.1f}s)"
    )

    if force and config.H5_PATH.exists():
        config.H5_PATH.unlink()

    t2 = time.time()
    total_steps = existing_n

    with h5py.File(config.H5_PATH, "a") as hf:
        if "subjects" not in hf:
            hf.create_group("subjects")

        for sid, buf in sorted(subj_bufs.items()):
            if not buf["X"]:
                continue

            grp_name = f"subjects/S{sid:04d}"
            X_arr = np.stack(buf["X"], axis=0).astype(np.float32)
            y_arr = np.array(buf["y"], dtype=np.int64)
            n_new = len(X_arr)

            if grp_name in hf:
                grp = hf[grp_name]
                old_n = grp["X"].shape[0]
                new_n = old_n + n_new
                grp["X"].resize(new_n, axis=0)
                grp["y"].resize(new_n, axis=0)
                grp["X"][old_n:new_n] = X_arr
                grp["y"][old_n:new_n] = y_arr
                log(f"    S{sid:04d}: 확장 {old_n}→{new_n}스텝")
            else:
                grp = hf.create_group(grp_name)
                grp.create_dataset(
                    "X", data=X_arr,
                    maxshape=(None, config.TS, n_ch),
                    chunks=(min(64, n_new), config.TS, n_ch),
                )
                grp.create_dataset("y", data=y_arr, maxshape=(None,))
                log(f"    S{sid:04d}: 신규 {n_new}스텝")

            total_steps += n_new
            del X_arr, y_arr

        if "channels" in hf:
            del hf["channels"]
        hf.create_dataset("channels", data=np.array(channels, dtype="S"))

        hf.attrs["segmentation"] = "heel_strike"
        hf.attrs["sample_rate"] = config.SAMPLE_RATE
        hf.attrs["target_ts"] = config.TS
        hf.attrs["n_classes"] = config.NUM_CLASSES
        hf.attrs["format"] = "subject_group_v9"
        hf.attrs["label_base"] = 0
        hf.attrs["label_semantics"] = "terrain_condition"
        hf.attrs["step_definition"] = "filewise_dual_signal_consensus_bilateral_merge"
        hf.attrs["label_mapping"] = json.dumps(
            {f"C{i+1}": i for i in range(config.NUM_CLASSES)},
            ensure_ascii=False,
        )

    log(f"  HDF5 쓰기: {time.time()-t2:.1f}s")

    # processed_files.json 갱신
    all_done_files = processed_now if force else (done_files | processed_now)
    save_processed_files(processed_files_path, all_done_files)
    log(f"  processed_files: {len(all_done_files)}개 → {processed_files_path}")

    size_mb = config.H5_PATH.stat().st_size / 1024**2
    log(f"\n  ✅ HDF5: {config.H5_PATH}")
    if force:
        log(f"     전체 재생성: {new_steps}스텝 (총: {total_steps}스텝)")
    else:
        log(f"     기존: {existing_n}스텝 + 신규: {new_steps}스텝 = 총: {total_steps}스텝")
    log(f"     파일 크기: {size_mb:.1f} MB")

    if total_steps > 0:
        with h5py.File(config.H5_PATH, "r") as f:
            all_sids = sorted(int(k[1:]) for k in f["subjects"])
            label_counts: dict[int, int] = defaultdict(int)
            for skey in f["subjects"]:
                for lbl in f[f"subjects/{skey}/y"][:]:
                    label_counts[int(lbl)] += 1

            log(f"     라벨 분포(0-based): {dict(sorted(label_counts.items()))}")
            log(f"     피험자: {all_sids} ({len(all_sids)}명)")

        if raw_lens:
            log(
                f"     스텝 길이: min={min(raw_lens)} max={max(raw_lens)}"
                f"  mean={np.mean(raw_lens):.0f}"
                f" ({np.mean(raw_lens)/config.SAMPLE_RATE*1000:.0f}ms)"
            )
    else:
        log("  ⚠ 검출된 스텝이 없습니다!")

    log(f"\n  총 소요: {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()