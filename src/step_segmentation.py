"""
step_segmentation.py — 힐스트라이크 스텝 분할 (v9.4)
═══════════════════════════════════════════════════════════════
[v9.3 → v9.4 보완]
  ① --force 조기 종료 기준      : new_records 대신 all_records 기준으로 통일
  ② HDF5 step_definition 불일치 : --force 없이 append 시 중단 (정책 명문화)
  ③ step_log 디버깅 정보 확장   : stride 파라미터·LT/RT 후보 수 추가 저장
  ④ ScoredStep → NamedTuple     : 필드 명시로 가독성·유지보수성 향상

[v9.2 → v9.3 보완]
  - --force 시 tp_force_all=True
  - step_log에 score 저장
  - flat v7 HDF5 경고 분리
  - step_definition loader 검증

[v9.0–9.1 유지 사항]
  - Pass 1 증분화 + _stable 기반 재추정
  - Threshold 이상치 내성 (percentile 99, fallback 포함)
  - LT/RT 시간순 품질 기반 병합
  - step_log JSONL 스트리밍
  - CSV 읽기 재시도 + **kwargs
  - terrain_params _meta 버전 검증

사용법:
    python3 step_segmentation.py           # 증분 (신규만)
    python3 step_segmentation.py --force   # 전체 재생성

3-패스 아키텍처:
    Pass 0 : 공통 채널 (기존 h5 있으면 재사용)
    Pass 1 : 지면별 적응 파라미터 (신규/불안정 조건 → all_records 기준)
    Pass 2 : 신규 피험자만 스텝 검출 → HDF5 추가
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

TERRAIN_PARAMS_VERSION = "v9.4"
_STRIDE_STABLE_MIN = 20                 # 이 값 미만이면 불안정 → 다음 증분 재계산
_SUPPORTED_H5_FORMATS = {               # ⑥ 호환 가능한 HDF5 포맷 버전
    "subject_group_v9",
    "subject_group_v8",                 # v8 구조도 읽기는 허용 (하위 호환)
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
    allow_step_def_mismatch: bool = False,  # ② True면 step_definition 불일치도 append 허용
) -> tuple[set[int], Optional[list[str]], int]:
    """기존 HDF5에서 처리 완료된 피험자 ID와 채널 목록을 읽는다.

    format attr 및 step_definition을 검증한다.
    step_definition 불일치 시 allow_step_def_mismatch=False(기본)이면 예외 발생.
    """
    if not h5_path.exists():
        return set(), None, 0

    try:
        with h5py.File(h5_path, "r") as f:
            # ⑥ 포맷 버전 검증  (③ flat v7 / 비호환 메시지 분리)
            h5_format = f.attrs.get("format", "unknown")
            if h5_format == "unknown":
                # format attr 없음 = flat v7 이전 구조 (하위 호환 읽기 허용)
                log(
                    "  ℹ HDF5 format attr 없음: flat v7 이전 구조로 추정. "
                    "하위 호환으로 읽지만 step_definition 의미가 다를 수 있습니다."
                )
            elif h5_format not in _SUPPORTED_H5_FORMATS:
                # 알 수 없는 포맷 → 진짜 비호환
                log(
                    f"  ⚠ HDF5 포맷 비호환: '{h5_format}' "
                    f"(지원={_SUPPORTED_H5_FORMATS}). 데이터 구조가 다를 수 있습니다."
                )
            elif h5_format == "subject_group_v8":
                log(
                    "  ⚠ HDF5 포맷 v8 감지: v9.0+ 에서 step_definition이 "
                    "'bilateral_merged_quality_based'로 변경되었습니다. "
                    "이전 실험과 직접 비교 불가."
                )

            # ④ step_definition 검증 (v9.x 내부 의미 차이 경고 + append 차단)
            step_def = f.attrs.get("step_definition", "")
            if step_def and step_def != "bilateral_merged_quality_based_per_side":
                log(
                    f"  ℹ step_definition='{step_def}' "
                    f"(현재 버전은 'bilateral_merged_quality_based_per_side'). "
                    f"병합 로직이 다른 버전으로 생성된 데이터입니다."
                )
                # ② --force 없이 다른 의미의 데이터셋에 append하는 것을 차단
                if not allow_step_def_mismatch:
                    raise ValueError(
                        f"step_definition 불일치 (저장='{step_def}'). "
                        f"--force 로 전체 재생성하거나 기존 HDF5를 백업 후 삭제하세요."
                    )

            if "subjects" in f and "channels" in f:
                done_sids: set[int] = set()
                n_existing = 0
                for skey in f["subjects"]:
                    sid = int(skey[1:])
                    done_sids.add(sid)
                    n_existing += f[f"subjects/{skey}/X"].shape[0]
                channels = [
                    c.decode() if isinstance(c, bytes) else c
                    for c in f["channels"][:]
                ]
                return done_sids, channels, n_existing

            for key in ("X", "y", "subject_id", "channels"):
                if key not in f:
                    log(f"  ⚠ 기존 HDF5에 '{key}' 없음, 처음부터 생성")
                    return set(), None, 0

            # flat v7 구조가 확인됨 → v9 group 방식으로 append하면 구조가 섞임
            raise ValueError(
                "flat v7 HDF5가 감지되었습니다. "
                "v9 group 구조와 혼용하면 파일이 손상됩니다. "
                "--force로 전체 재생성하세요."
            )

    except ValueError:
        raise   # step_definition 불일치 등 정책 위반은 호출자에게 전파
    except Exception as e:
        log(f"  ⚠ 기존 HDF5 읽기 실패: {e}")
        return set(), None, 0


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
# 5. Pass 1: 지면별 통계 수집 (증분 + 안정성 보장)
# ──────────────────────────────────────────────────────────────

def load_existing_terrain_params(
    params_path: Path,
) -> tuple[dict[int, dict], bool]:
    """기존 terrain_params.json을 로드하고 버전·sample_rate 를 검증한다.

    Returns:
        (params_dict, is_valid)
        is_valid=False 이면 호출자가 전체 재계산을 수행해야 한다.
    """
    if not params_path.exists():
        return {}, True

    try:
        raw = json.loads(params_path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"  ⚠ terrain_params.json 파싱 실패: {e}")
        return {}, False

    meta = raw.get("_meta", {})
    stored_version = meta.get("version", "unknown")
    stored_sr = meta.get("sample_rate", None)

    if stored_version != TERRAIN_PARAMS_VERSION:
        log(
            f"  ⚠ terrain_params 버전 불일치 "
            f"(저장={stored_version}, 현재={TERRAIN_PARAMS_VERSION}) → 전체 재계산"
        )
        return {}, False

    if stored_sr is not None and stored_sr != config.SAMPLE_RATE:
        log(
            f"  ⚠ terrain_params sample_rate 불일치 "
            f"(저장={stored_sr}, 현재={config.SAMPLE_RATE}) → 전체 재계산"
        )
        return {}, False

    params = {int(k): v for k, v in raw.items() if k != "_meta"}
    return params, True


def _compute_params_for_conditions(
    records: list[dict],
    conds_to_collect: set[int],
) -> dict[int, dict]:
    """지정된 조건들에 대해 terrain 통계를 수집하고 파라미터를 반환한다."""
    terrain_stats: dict[int, dict[str, list]] = defaultdict(
        lambda: {"means": [], "stds": [], "peak_heights": [], "stride_intervals": []}
    )
    target_records = [r for r in records if r["cond"] in conds_to_collect]

    for i, rec in enumerate(target_records):
        try:
            df = read_csv_with_retry(rec["path"])
            df = rename_columns(df)
        except Exception as e:
            log(f"    ⚠ CSV 읽기 실패: {rec['path'].name} ({e})")
            continue

        cond = rec["cond"]
        for side in ["LT", "RT"]:
            try:
                ml_gyro = extract_ml_gyro(df, side)
                ap_accel = extract_ap_accel(df, side)

                if ml_gyro is None or ap_accel is None:
                    norm = compute_foot_acc_norm(df, side)
                    sig = bandpass_filter(norm)
                    use_gyro_minima = False
                else:
                    sig = bandpass_filter(ml_gyro)
                    use_gyro_minima = True

                nan_mask = np.isnan(sig)
                n_valid = int(np.sum(~nan_mask))
                if n_valid < config.HS_MIN_STRIDE_SAM * 2:
                    continue

                if nan_mask.any():
                    x = np.arange(len(sig))
                    sig = sig.copy()
                    sig[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], sig[~nan_mask])

                mu = float(np.mean(sig))
                std = float(np.std(sig))
                terrain_stats[cond]["means"].append(mu)
                terrain_stats[cond]["stds"].append(std)

                if use_gyro_minima:
                    peaks, props = find_peaks(
                        -sig, prominence=0.4 * std,
                        distance=config.HS_MIN_STRIDE_SAM,
                    )
                    peak_values = props.get("prominences", [])
                else:
                    peaks, props = find_peaks(
                        sig, height=mu + 0.5 * std,
                        distance=config.HS_MIN_STRIDE_SAM,
                    )
                    peak_values = props.get("peak_heights", [])

                if len(peaks) > 0 and len(peak_values) > 0:
                    terrain_stats[cond]["peak_heights"].extend(
                        np.asarray(peak_values).tolist()
                    )
                if len(peaks) > 1:
                    terrain_stats[cond]["stride_intervals"].extend(
                        np.diff(peaks).tolist()
                    )

            except (KeyError, ValueError):
                continue

        del df
        gc.collect()

        if (i + 1) % 50 == 0:
            log(f"    Pass 1: {i+1}/{len(target_records)} 파일")

    result: dict[int, dict] = {}
    for cond in sorted(conds_to_collect):
        s = terrain_stats[cond]
        if not s["means"]:
            result[cond] = {
                "alpha": 1.0,
                "min_dist": config.HS_MIN_STRIDE_SAM,
                "min_peak_ratio": config.HS_PEAK_QUALITY_RATIO,
                "stride_min_sam": config.HS_MIN_STRIDE_SAM,
                "stride_max_sam": config.HS_MAX_STRIDE_SAM,
                "stride_mean_ms": 0.0,
                "stride_std_ms": 0.0,
                "stride_n_samples": 0,
                "_stable": False,
            }
            continue

        avg_mean = float(np.mean(s["means"]))
        avg_std = float(np.mean(s["stds"]))
        avg_peak = (
            float(np.mean(s["peak_heights"])) if s["peak_heights"] else avg_mean + avg_std
        )
        snr = (avg_peak - avg_mean) / avg_std if avg_std > 0 else 0.0
        alpha = max(0.5, min(2.0, snr * 0.5))

        n_stride = len(s["stride_intervals"])
        is_stable = n_stride >= _STRIDE_STABLE_MIN

        if is_stable:
            si = np.array(s["stride_intervals"])
            stride_min = max(40, int(np.percentile(si, 2) * 0.9))
            stride_max = int(np.percentile(si, 98) * 1.1)
            stride_mean = float(np.mean(si))
            stride_std = float(np.std(si))
        else:
            stride_min = config.HS_MIN_STRIDE_SAM
            stride_max = config.HS_MAX_STRIDE_SAM
            stride_mean = 0.0
            stride_std = 0.0
            log(
                f"    ⚠ C{cond}: stride 샘플 부족 ({n_stride}개 < {_STRIDE_STABLE_MIN}), "
                f"기본값 사용. 다음 증분 시 재추정 예정"
            )

        result[cond] = {
            "alpha": round(alpha, 3),
            "min_dist": stride_min,
            "min_peak_ratio": config.HS_PEAK_QUALITY_RATIO,
            "stride_min_sam": stride_min,
            "stride_max_sam": stride_max,
            "stride_mean_ms": round(stride_mean / config.SAMPLE_RATE * 1000, 1),
            "stride_std_ms": round(stride_std / config.SAMPLE_RATE * 1000, 1),
            "stride_n_samples": n_stride,
            "avg_mean": round(avg_mean, 1),
            "avg_std": round(avg_std, 1),
            "avg_peak": round(avg_peak, 1),
            "_stable": is_stable,
        }

        log(
            f"    C{cond}: α={alpha:.2f}  μ={avg_mean:.0f}"
            f"  σ={avg_std:.0f}  peak={avg_peak:.0f}"
            f"  stride={stride_min}~{stride_max}sam"
            f" ({stride_min/config.SAMPLE_RATE*1000:.0f}~"
            f"{stride_max/config.SAMPLE_RATE*1000:.0f}ms)"
            f"  n={n_stride}  stable={'✅' if is_stable else '⚠'}"
        )

    return result


def collect_terrain_stats(
    all_records: list[dict],
    new_records: list[dict],
    existing_params: dict[int, dict],
    force_all: bool = False,            # ② 전체 재계산 강제 플래그
) -> dict[int, dict]:
    """지면 조건별 파라미터를 수집/갱신한다.

    Args:
        all_records: 전체 CSV 레코드 목록
        new_records: 신규 피험자 CSV 레코드 목록
        existing_params: 기존 terrain_params
        force_all: True이면 all_records의 전체 조건 재계산  ← ②

    Returns:
        전체 조건 → 파라미터 dict
    """
    # ② tp_valid=False 시 all_records 전체 조건 재계산
    if force_all:
        all_conds: set[int] = {r["cond"] for r in all_records}
        log(
            f"  [Pass 1] 전체 재계산 (terrain_params 무효화): "
            f"조건 {sorted(all_conds)}, all_records {len(all_records)}개 기준"
        )
        return _compute_params_for_conditions(all_records, all_conds)

    new_conds: set[int] = {r["cond"] for r in new_records}

    # 재계산 대상: 신규 조건 or 기존이지만 _stable=False
    conds_to_collect: set[int] = set()
    for cond in new_conds:
        existing = existing_params.get(cond)
        if existing is None:
            conds_to_collect.add(cond)
        elif not existing.get("_stable", True):
            conds_to_collect.add(cond)
            log(f"  ⚠ C{cond}: 이전에 불안정 판정 → 전체 데이터로 재추정")

    reuse_conds = set(existing_params.keys()) - conds_to_collect
    if reuse_conds:
        log(f"  [Pass 1] 재사용 조건: {sorted(reuse_conds)}")
    if not conds_to_collect:
        log("  [Pass 1] 모든 지면 조건 안정적 → 파라미터 수집 스킵")
        return existing_params

    log(
        f"  [Pass 1] 재계산 조건: {sorted(conds_to_collect)} "
        f"(all_records {len(all_records)}개 기준)"
    )
    new_params = _compute_params_for_conditions(all_records, conds_to_collect)
    return {**existing_params, **new_params}


# ──────────────────────────────────────────────────────────────
# 6. 힐스트라이크 검출
# ──────────────────────────────────────────────────────────────

def detect_steps(
    ml_gyro: np.ndarray,
    ap_accel: np.ndarray,
    terrain_params: dict,
    cond: int,
    fs: int = config.SAMPLE_RATE,
) -> list[tuple[int, int]]:
    """ML Gyro + AP Accel 융합 힐스트라이크 검출."""
    n = len(ml_gyro)
    if n < fs:
        return []

    nan_ratio = (np.isnan(ml_gyro).sum() + np.isnan(ap_accel).sum()) / (2 * n)
    if nan_ratio > config.HS_NAN_THRESHOLD:
        return []

    for sig in [ml_gyro, ap_accel]:
        mask = np.isnan(sig)
        if mask.any() and not mask.all():
            x = np.arange(n)
            sig[mask] = np.interp(x[mask], x[~mask], sig[~mask])

    ml_f = bandpass_filter(ml_gyro, fs, low=1.0, high=20.0)
    ap_f = bandpass_filter(ap_accel, fs, low=1.0, high=20.0)

    sigma_ml = float(np.std(ml_f))
    if sigma_ml == 0:
        return []

    min_dist_sam = int(fs * 0.35)

    mid_swings, _ = find_peaks(
        ml_f, prominence=0.3 * sigma_ml, distance=min_dist_sam,
    )

    if len(mid_swings) > 0:
        peak_ref = float(np.percentile(ml_f[mid_swings], 99))
    else:
        peak_ref = sigma_ml
    trusted_thresh = config.HS_TRUSTED_SWING * peak_ref

    hs_cand, _ = find_peaks(
        -ml_f, prominence=config.HS_GYRO_PROMINENCE * sigma_ml, distance=min_dist_sam,
    )
    if len(hs_cand) < 2:
        return []

    hs_trusted: list[int] = []
    for cand in hs_cand:
        nearby_swings = [p for p in mid_swings if abs(p - cand) < fs // 2]
        if nearby_swings:
            nearest = min(nearby_swings, key=lambda p: abs(p - cand))
            if ml_f[nearest] > trusted_thresh:
                hs_trusted.append(cand)
        else:
            w_start = max(0, cand - fs // 2)
            w_end = min(n, cand + fs // 2)
            local_ref = float(np.percentile(ml_f[w_start:w_end], 99))
            if local_ref > trusted_thresh:
                hs_trusted.append(cand)

    if len(hs_trusted) < 2:
        return []

    mean_ap = float(np.mean(ap_f))
    sigma_ap = float(np.std(ap_f))
    window = config.HS_FUSION_WINDOW_SAM

    hs_final: list[int] = []
    for cand in hs_trusted:
        w_start = max(0, cand - window)
        w_end = min(n, cand + window)
        if np.min(ap_f[w_start:w_end]) < mean_ap - config.HS_ACCEL_THRESHOLD * sigma_ap:
            hs_final.append(cand)

    if len(hs_final) < 2:
        return []

    default_params = {
        "stride_min_sam": config.HS_MIN_STRIDE_SAM,
        "stride_max_sam": config.HS_MAX_STRIDE_SAM,
    }
    params = terrain_params.get(cond, default_params)
    stride_min = params.get("stride_min_sam", config.HS_MIN_STRIDE_SAM)
    stride_max = params.get("stride_max_sam", config.HS_MAX_STRIDE_SAM)

    valid_steps: list[tuple[int, int]] = []
    for i in range(len(hs_final) - 1):
        start = hs_final[i]
        end = hs_final[i + 1]
        length = end - start
        if length < stride_min or length > stride_max:
            continue
        if np.isnan(ml_gyro[start:end]).sum() / length > config.HS_NAN_THRESHOLD:
            continue
        valid_steps.append((start, end))

    return valid_steps


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
    log(f"  step_segmentation.py v9.4 ({'전체 재생성' if force else '증분 모드'})")
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
    params_path = config.BATCH_DIR / "terrain_params.json"

    if force:
        done_sids: set[int] = set()
        existing_channels: Optional[list[str]] = None
        existing_n: int = 0
        existing_terrain_params: dict[int, dict] = {}
        tp_force_all = True             # ① --force면 Pass 1도 명시적 전체 재계산
        log("  ★ --force: 기존 HDF5·terrain_params 무시, 전체 재생성")
    else:
        done_sids, existing_channels, existing_n = load_existing_h5_info(
            config.H5_PATH  # ② else 분기이므로 force는 항상 False → 기본값으로 충분
        )
        existing_terrain_params, tp_valid = load_existing_terrain_params(params_path)
        # ② tp_valid=False면 all_records 전체 조건 재계산 플래그
        tp_force_all = not tp_valid
        if tp_force_all:
            existing_terrain_params = {}
        if done_sids:
            log(f"  ★ 기존 HDF5: {existing_n}스텝, {len(done_sids)}명 완료")
            log(f"    완료: {sorted(done_sids)}")
        if existing_terrain_params:
            log(
                f"  ★ 기존 terrain_params ({TERRAIN_PARAMS_VERSION}): "
                f"{sorted(existing_terrain_params.keys())} 조건 재사용 후보"
            )

    new_records = [r for r in all_records if r["sid"] not in done_sids]

    target_records = all_records if force else new_records  # 이후 모든 처리의 기준

    if not target_records:
        log(f"\n  ✅ 모든 {len(done_sids)}명 처리 완료, 스텝 검출 스킵")
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

    # Pass 1
    terrain_params = collect_terrain_stats(
        all_records, new_records, existing_terrain_params,
        force_all=tp_force_all,
    )

    savable: dict = {
        "_meta": {"version": TERRAIN_PARAMS_VERSION, "sample_rate": config.SAMPLE_RATE},
        **{str(k): v for k, v in terrain_params.items()},
    }
    params_path.write_text(
        json.dumps(savable, indent=2, default=str, ensure_ascii=False)
    )
    log(f"  지면 파라미터 → {params_path}  ({time.time()-t0:.1f}s)\n")

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

    with StepLogWriter(log_path) as step_logger:
        for i, rec in enumerate(target_records):
            try:
                df = read_csv_with_retry(rec["path"])
                df = rename_columns(df)
            except Exception as e:
                log(f"  ⚠ CSV 읽기 실패: {rec['path'].name} ({e})")
                continue

            sid = rec["sid"]
            cond = rec["cond"]
            label = rec["label"]

            data_cols = [c for c in channels if c in df.columns]
            if len(data_cols) < len(channels):
                missing = set(channels) - set(data_cols)
                log(f"  ⚠ {rec['path'].name}: {len(missing)}개 채널 누락, 스킵")
                del df
                gc.collect()
                continue

            data_np = df[channels].values.astype(np.float32)

            # terrain 파라미터 추출 (stride 정보 포함)
            tp = terrain_params.get(cond, {})
            stride_mean_sam = tp.get("stride_mean_ms", 0.0) / 1000.0 * config.SAMPLE_RATE
            stride_min_sam = float(tp.get("stride_min_sam", config.HS_MIN_STRIDE_SAM))
            stride_max_sam = float(tp.get("stride_max_sam", config.HS_MAX_STRIDE_SAM))

            # ① side별 신호로 각각 점수화
            lt_scored: list[ScoredStep] = []
            rt_scored: list[ScoredStep] = []
            side_raw_counts: dict[str, int] = {"LT": 0, "RT": 0}  # ③ 후보 수 추적

            for side, scored_bucket in [("LT", lt_scored), ("RT", rt_scored)]:
                try:
                    ml_gyro = extract_ml_gyro(df, side)
                    ap_accel = extract_ap_accel(df, side)

                    if ml_gyro is None or ap_accel is None:
                        norm = compute_foot_acc_norm(df, side)
                        norm_f = bandpass_filter(norm)
                        valid = norm_f[~np.isnan(norm_f)]
                        sigma = float(np.std(valid)) if len(valid) > 0 else 1.0
                        mu = float(np.mean(valid)) if len(valid) > 0 else 0.0
                        peaks, _ = find_peaks(
                            norm_f, height=mu + 0.5 * sigma,
                            distance=config.HS_MIN_STRIDE_SAM,
                        )
                        raw_steps = [
                            (int(peaks[j]), int(peaks[j + 1]))
                            for j in range(len(peaks) - 1)
                            if config.HS_MIN_STRIDE_SAM
                               <= peaks[j + 1] - peaks[j]
                               <= config.HS_MAX_STRIDE_SAM
                        ]
                        sig_q = norm_f
                    else:
                        raw_steps = detect_steps(
                            ml_gyro.copy(), ap_accel.copy(), terrain_params, cond,
                        )
                        sig_q = ml_gyro   # ← 해당 발 신호 사용

                    # ① 해당 발 신호(sig_q)로만 점수 계산
                    side_raw_counts[side] = len(raw_steps)   # ③
                    scored_bucket.extend(
                        score_steps_by_side(
                            raw_steps, side, sig_q,
                            stride_mean_sam, stride_min_sam, stride_max_sam,
                        )
                    )

                except (KeyError, ValueError):
                    continue

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
                    # ③ 병합 기준 재현에 필요한 stride 파라미터
                    "stride_mean_sam": round(stride_mean_sam, 1),
                    "stride_min_sam": int(stride_min_sam),
                    "stride_max_sam": int(stride_max_sam),
                    "lt_raw_n": side_raw_counts["LT"],
                    "rt_raw_n": side_raw_counts["RT"],
                })

            del df, data_np
            gc.collect()

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