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

SEGMENTATION_VERSION   = "v9.4-filewise-1"  # processed_files 버전 — 로직 변경 시 올림
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
            if step_def and step_def != "bilateral_merged_quality_based_per_side":
                log(
                    f"  ℹ step_definition='{step_def}' "
                    f"(현재 버전은 'bilateral_merged_quality_based_per_side'). "
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

def detect_steps(
    ml_gyro: np.ndarray,
    ap_accel: np.ndarray,
    fs: int = config.SAMPLE_RATE,
) -> list[tuple[int, int]]:
    """ML Gyro + AP Accel 융합 힐스트라이크 검출 — 완전 파일 적응형.

    모든 검출 기준(피크 높이·prominence·stride 범위)을
    이 파일 신호 자체의 통계에서만 계산한다.
    terrain_params / cond 에 의존하지 않는다.
    """
    n = len(ml_gyro)
    if n < fs:
        return []

    # ── NaN 비율 체크 ──
    nan_ratio = (np.isnan(ml_gyro).sum() + np.isnan(ap_accel).sum()) / (2 * n)
    if nan_ratio > config.HS_NAN_THRESHOLD:
        return []

    # ── NaN 선형 보간 ──
    for sig in [ml_gyro, ap_accel]:
        mask = np.isnan(sig)
        if mask.any() and not mask.all():
            x = np.arange(n)
            sig[mask] = np.interp(x[mask], x[~mask], sig[~mask])

    ml_f = bandpass_filter(ml_gyro, fs, low=1.0, high=20.0)
    ap_f = bandpass_filter(ap_accel, fs, low=1.0, high=20.0)

    # ── 파일 신호 통계 ──
    sigma_ml = float(np.std(ml_f))
    if sigma_ml == 0:
        return []
    mean_ap  = float(np.mean(ap_f))
    sigma_ap = float(np.std(ap_f))

    # ── 최소 피크 간격: 이 파일 신호 IQR로 적응 추정 ──
    # 초기 느슨한 거리로 후보 피크를 잡은 뒤, 그 간격 중앙값을 min_dist로 재설정
    loose_dist = int(fs * 0.25)          # 250ms — 매우 빠른 보행도 커버
    cand_raw, _ = find_peaks(-ml_f, prominence=0.2 * sigma_ml, distance=loose_dist)
    if len(cand_raw) >= 4:
        intervals_raw = np.diff(cand_raw).astype(float)
        med_interval  = float(np.median(intervals_raw))
        # 중앙값의 40% 이상은 항상 최소 거리로 허용 (너무 타이트하게 막지 않음)
        min_dist_sam  = max(int(med_interval * 0.40), int(fs * 0.25))
    else:
        min_dist_sam  = int(fs * 0.35)   # 후보 부족 → 보수적 350ms

    # ── Swing 피크 검출 (mid-swing: Gyro 양의 피크) ──
    mid_swings, _ = find_peaks(
        ml_f, prominence=0.3 * sigma_ml, distance=min_dist_sam,
    )
    if len(mid_swings) > 0:
        peak_ref = float(np.percentile(ml_f[mid_swings], 99))
    else:
        peak_ref = sigma_ml
    trusted_thresh = config.HS_TRUSTED_SWING * peak_ref

    # ── 힐스트라이크 후보: Gyro 음의 피크 ──
    hs_cand, _ = find_peaks(
        -ml_f, prominence=config.HS_GYRO_PROMINENCE * sigma_ml, distance=min_dist_sam,
    )
    if len(hs_cand) < 2:
        return []

    # ── Swing 인접성으로 신뢰도 필터 ──
    hs_trusted: list[int] = []
    for cand in hs_cand:
        nearby = [p for p in mid_swings if abs(p - cand) < fs // 2]
        if nearby:
            nearest = min(nearby, key=lambda p: abs(p - cand))
            if ml_f[nearest] > trusted_thresh:
                hs_trusted.append(cand)
        else:
            w_s = max(0, cand - fs // 2)
            w_e = min(n, cand + fs // 2)
            if float(np.percentile(ml_f[w_s:w_e], 99)) > trusted_thresh:
                hs_trusted.append(cand)

    if len(hs_trusted) < 2:
        return []

    # ── AP Accel 검증: 힐스트라이크 시 AP 가속도 음의 피크 확인 ──
    window = config.HS_FUSION_WINDOW_SAM
    hs_final: list[int] = []
    for cand in hs_trusted:
        w_s = max(0, cand - window)
        w_e = min(n, cand + window)
        if np.min(ap_f[w_s:w_e]) < mean_ap - config.HS_ACCEL_THRESHOLD * sigma_ap:
            hs_final.append(cand)

    if len(hs_final) < 2:
        return []

    # ── 파일 적응형 stride 범위 ──
    # 이 파일의 hs_final 간격 분포에서 직접 계산.
    # 조건별 평균값을 쓰지 않으므로 보폭이 특이한 피험자도 처리됨.
    file_intervals = np.diff(hs_final).astype(float)
    if len(file_intervals) >= 4:
        p5  = float(np.percentile(file_intervals, 5))
        p95 = float(np.percentile(file_intervals, 95))
        med = float(np.median(file_intervals))
        stride_min = max(int(min(p5,  med * 0.50)), config.HS_MIN_STRIDE_SAM)
        stride_max = min(int(max(p95, med * 1.50)), config.HS_MAX_STRIDE_SAM)
    else:
        stride_min = config.HS_MIN_STRIDE_SAM
        stride_max = config.HS_MAX_STRIDE_SAM

    # ── 유효 스텝 필터 ──
    valid_steps: list[tuple[int, int]] = []
    for i in range(len(hs_final) - 1):
        start = hs_final[i]
        end   = hs_final[i + 1]
        length = end - start
        if length < stride_min or length > stride_max:
            continue
        if np.isnan(ml_gyro[start:end]).sum() / length > config.HS_NAN_THRESHOLD:
            continue
        valid_steps.append((start, end))

    return valid_steps


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
# 7. LT/RT 중복 스텝 병합 (side별 신호 기반 품질 평가)  ← ①③④
# ──────────────────────────────────────────────────────────────

def _step_quality(
    start: int,
    end: int,
    signal: np.ndarray,
    stride_mean_sam: float,
    stride_min_sam: float,
    stride_max_sam: float,
) -> float:
    """스텝의 품질 점수를 계산한다 (높을수록 좋음).

    품질 = (1 - NaN 비율) × stride 중심 근접도  ← ③
    stride 중심 근접도 = 1 / (1 + |length - stride_center| / stride_range)
    """
    length = end - start
    if length <= 0:
        return 0.0

    nan_ratio = float(np.isnan(signal[start:end]).sum()) / length
    quality_nan = 1.0 - nan_ratio

    # ③ stride_mean 대신 stride range 중심과 범위를 모두 활용
    if stride_mean_sam > 0:
        stride_center = stride_mean_sam
        stride_range = max(stride_max_sam - stride_min_sam, 1.0)
        proximity = 1.0 / (1.0 + abs(length - stride_center) / stride_range)
    elif stride_min_sam > 0 and stride_max_sam > stride_min_sam:
        # ④ stride_mean=0(불안정) 시 min/max 중앙값으로 폴백
        stride_center = (stride_min_sam + stride_max_sam) / 2.0
        stride_range = max(stride_max_sam - stride_min_sam, 1.0)
        proximity = 1.0 / (1.0 + abs(length - stride_center) / stride_range)
    else:
        proximity = 1.0

    return quality_nan * proximity


# ④ ScoredStep: NamedTuple로 필드 명시 (가독성·유지보수성 향상)
from typing import NamedTuple

class ScoredStep(NamedTuple):
    start: int
    end: int
    side: str       # "LT" 또는 "RT"
    score: float    # 품질 점수 (높을수록 좋음)


def score_steps_by_side(
    steps: list[tuple[int, int]],
    side: str,
    signal: np.ndarray,
    stride_mean_sam: float,
    stride_min_sam: float,
    stride_max_sam: float,
) -> list[ScoredStep]:
    """각 발의 스텝에 해당 발 신호 기준 품질 점수를 부여한다.  ← ①

    Args:
        steps: (start, end) 스텝 목록
        side: "LT" 또는 "RT"
        signal: 해당 발의 기준 신호 (LT면 LT 신호, RT면 RT 신호)
        stride_mean_sam: 기대 stride 길이 (samples)
        stride_min_sam: stride 최솟값
        stride_max_sam: stride 최댓값

    Returns:
        (start, end, side, score) 목록
    """
    scored: list[ScoredStep] = []
    for start, end in steps:
        score = _step_quality(
            start, end, signal,
            stride_mean_sam, stride_min_sam, stride_max_sam,
        )
        scored.append(ScoredStep(start=start, end=end, side=side, score=score))
    return scored


def merge_bilateral_steps(
    lt_scored: list[ScoredStep],
    rt_scored: list[ScoredStep],
    overlap_threshold: float = 0.5,
) -> list[ScoredStep]:
    """LT/RT 사전 점수화된 스텝을 시간순으로 병합하고 겹침을 정리한다.  ← ①

    Args:
        lt_scored: (start, end, "LT", score) 목록
        rt_scored: (start, end, "RT", score) 목록
        overlap_threshold: 허용 최대 겹침 비율 (0~1)

    Returns:
        최종 (start, end, side, score) 목록, 시간순 정렬
    """
    tagged = lt_scored + rt_scored
    tagged.sort(key=lambda x: x[0])

    if len(tagged) < 2:
        return tagged

    merged_indices: list[int] = [0]
    for curr_idx in range(1, len(tagged)):
        prev_idx = merged_indices[-1]
        prev_start, prev_end = tagged[prev_idx].start, tagged[prev_idx].end
        curr_start, curr_end = tagged[curr_idx].start, tagged[curr_idx].end

        overlap_len = max(0, min(prev_end, curr_end) - max(prev_start, curr_start))
        if overlap_len == 0:
            merged_indices.append(curr_idx)
            continue

        shorter_len = min(prev_end - prev_start, curr_end - curr_start)
        if shorter_len == 0 or overlap_len / shorter_len > overlap_threshold:
            # 겹침 심하면 품질(score) 높은 쪽 보존
            if tagged[curr_idx].score >= tagged[prev_idx].score:
                merged_indices[-1] = curr_idx
        else:
            merged_indices.append(curr_idx)

    return [tagged[i] for i in merged_indices]


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
            side_signals: dict[str, np.ndarray] = {}   # 검출에 실제 쓴 신호 보존
            side_raw_counts: dict[str, int] = {"LT": 0, "RT": 0}

            for side, raw_bucket in [("LT", lt_raw_steps), ("RT", rt_raw_steps)]:
                try:
                    ml_gyro = extract_ml_gyro(df, side)
                    ap_accel = extract_ap_accel(df, side)

                    if ml_gyro is None or ap_accel is None:
                        # 폴백: 발 가속도 norm으로 파일 적응형 검출
                        norm = compute_foot_acc_norm(df, side)
                        norm_f = bandpass_filter(norm)
                        valid = norm_f[~np.isnan(norm_f)]
                        if len(valid) == 0:
                            raw_steps = []
                        else:
                            mu    = float(np.mean(valid))
                            sigma = float(np.std(valid))
                            loose_peaks, _ = find_peaks(
                                norm_f,
                                height=mu + 0.3 * sigma,
                                distance=int(config.SAMPLE_RATE * 0.25),
                            )
                            if len(loose_peaks) >= 4:
                                ivs = np.diff(loose_peaks).astype(float)
                                med = float(np.median(ivs))
                                s_min = max(int(med * 0.50), config.HS_MIN_STRIDE_SAM)
                                s_max = min(int(med * 1.50), config.HS_MAX_STRIDE_SAM)
                                min_d = max(int(med * 0.40),
                                            int(config.SAMPLE_RATE * 0.25))
                            else:
                                s_min = config.HS_MIN_STRIDE_SAM
                                s_max = config.HS_MAX_STRIDE_SAM
                                min_d = config.HS_MIN_STRIDE_SAM
                            peaks, _ = find_peaks(
                                norm_f,
                                height=mu + 0.5 * sigma,
                                distance=min_d,
                            )
                            raw_steps = [
                                (int(peaks[j]), int(peaks[j + 1]))
                                for j in range(len(peaks) - 1)
                                if s_min <= peaks[j + 1] - peaks[j] <= s_max
                            ]
                        sig_q = norm_f   # 폴백: norm_f 기준 품질 평가
                    else:
                        raw_steps = detect_steps(ml_gyro.copy(), ap_accel.copy())
                        sig_q = ml_gyro  # 정상: ML gyro 기준 품질 평가

                    raw_bucket.extend(raw_steps)
                    side_signals[side] = sig_q   # 검출에 쓴 신호 그대로 보존
                    side_raw_counts[side] = len(raw_steps)

                except (KeyError, ValueError):
                    # 예외 시 NaN 배열로 폴백 → _step_quality NaN 비율 1.0 → 점수 0
                    side_signals[side] = np.full(len(data_np), np.nan, dtype=np.float32)

            # LT+RT 합산으로 파일 기준 stride 파라미터 추정
            file_stride_mean_sam, file_stride_min_sam, file_stride_max_sam = \
                estimate_stride_params_from_steps(lt_raw_steps + rt_raw_steps)

            # 검출에 실제 쓴 신호(side_signals)로 점수화.
            # 정상/폴백 경로에서는 실제 신호가 저장되며,
            # 예외로 side_signals에 키가 없을 때는 NaN 배열을 사용해
            # _step_quality가 NaN 비율 1.0 → 점수 0으로 처리되도록 한다.
            nan_sig = np.full(len(data_np), np.nan, dtype=np.float32)
            lt_scored: list[ScoredStep] = score_steps_by_side(
                lt_raw_steps, "LT", side_signals.get("LT", nan_sig),
                file_stride_mean_sam, file_stride_min_sam, file_stride_max_sam,
            )
            rt_scored: list[ScoredStep] = score_steps_by_side(
                rt_raw_steps, "RT", side_signals.get("RT", nan_sig),
                file_stride_mean_sam, file_stride_min_sam, file_stride_max_sam,
            )

            # ① side별 점수가 붙은 채로 병합
            merged_steps = merge_bilateral_steps(lt_scored, rt_scored)
            file_steps = 0

            for start, end, side, _score in merged_steps:
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
                    # 파일 기준 stride 파라미터 (디버깅 재현 가능)
                    "stride_mean_sam": round(file_stride_mean_sam, 1),
                    "stride_min_sam": int(file_stride_min_sam),
                    "stride_max_sam": int(file_stride_max_sam),
                    "lt_raw_n": side_raw_counts["LT"],
                    "rt_raw_n": side_raw_counts["RT"],
                })

            del df, data_np
            gc.collect()
            processed_now.add(file_key)  # 파일 읽기·채널 검증 성공 기준 (0-step 포함)

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
        hf.attrs["step_definition"] = "bilateral_merged_quality_based_per_side"
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