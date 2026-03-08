"""
step_segmentation.py — 힐스트라이크 스텝 분할 (v8.1 증분 모드)
═══════════════════════════════════════════════════════
★ 증분(Incremental) 처리: 기존 HDF5의 피험자는 스킵, 신규만 추가
★ N=40 → N=100 확장 시 기존 40명 재처리 불필요
★ 강제 전체 재생성: --force 플래그
★ NaN 처리: 시간축 유지한 채 선형 보간 → 연속 신호 복원

사용법:
    python3 step_segmentation.py           # 증분 (신규만)
    python3 step_segmentation.py --force   # 전체 재생성

3-패스 아키텍처:
    Pass 0 : 공통 채널 (기존 h5 있으면 재사용)
    Pass 1 : 지면별 적응적 α (NaN 보간 후 연속 신호에서 통계 수집)
    Pass 2 : 신규 피험자만 스텝 검출 → HDF5 추가
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys, time, re, json, gc, argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import numpy as np
import pandas as pd
import h5py
from scipy.signal import find_peaks, resample, butter, filtfilt


def log(msg: str) -> None:
    """타임스탬프 포함 로그 출력."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────
# 컬럼명 통일 (Noraxon 버전/내보내기 설정 차이 보정)
# ─────────────────────────────────────────────

# 정확 매칭 리네임 맵 (기존 호환)
RENAME_MAP: dict[str, str] = {
    "Noraxon MyoMotion-Joints-Knee LT-Rotation Ext (deg)":
        "Knee Rotation Ext LT (deg)",
    "Noraxon MyoMotion-Joints-Knee RT-Rotation Ext (deg)":
        "Knee Rotation Ext RT (deg)",
}

# 패턴 기반 정규화 규칙
# (패턴, 치환) — 컬럼명에서 불필요한 접두사/변형을 제거
_COL_NORMALIZE_RULES: list[tuple[re.Pattern, str]] = [
    # "Noraxon MyoMotion-Segments-" → 제거
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+Segments[-\s]+"), ""),
    # "Noraxon MyoMotion-Joints-" → 제거
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+Joints[-\s]+"), ""),
    # "Noraxon MyoMotion-Trajectories-" → "Trajectories "
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+Trajectories[-\s]+"), "Trajectories "),
    # "Noraxon MyoMotion-" → 제거 (기타)
    (re.compile(r"^Noraxon\s+MyoMotion[-\s]+"), ""),
    # "Accel Sensor" ↔ "Acceleration" ↔ "Accel" 통일
    (re.compile(r"Acceleration\b"), "Accel Sensor"),
]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Noraxon 내보내기 버전 간 컬럼명 차이를 통일한다.

    1차: 정확 매칭 (RENAME_MAP)
    2차: 패턴 기반 정규화 (_COL_NORMALIZE_RULES)
    """
    # 1차: 정확 매칭
    renames = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    if renames:
        df = df.rename(columns=renames)

    # 2차: 패턴 기반 정규화
    new_names: dict[str, str] = {}
    for c in df.columns:
        new_c = c
        for pat, repl in _COL_NORMALIZE_RULES:
            new_c = pat.sub(repl, new_c)
        if new_c != c:
            new_names[c] = new_c

    if new_names:
        # 중복 방지: 이미 존재하는 이름으로 바꾸려면 스킵
        existing = set(df.columns)
        safe_renames = {
            k: v for k, v in new_names.items()
            if v not in existing or v == k
        }
        if safe_renames:
            df = df.rename(columns=safe_renames)

    return df


# ─────────────────────────────────────────────
# 1. 파일 탐색
# ─────────────────────────────────────────────

def parse_filename(fname: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """파일명에서 피험자·지형·시행 번호를 추출한다.

    Parameters
    ----------
    fname : str
        CSV 파일명 (예: '20230101_S01C1T1.csv').

    Returns
    -------
    tuple[int | None, int | None, int | None]
        (subject_id, condition, trial) 또는 파싱 실패 시 (None, None, None).
    """
    m = re.search(r"S(\d+)C(\d+)T(\d+)", fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"S(\d+)C(\d+)", fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), 1
    return None, None, None


def discover_csvs(data_dir: Path, n_subjects: int) -> list[dict]:
    """데이터 디렉토리에서 유효한 CSV 레코드를 수집한다.

    Parameters
    ----------
    data_dir : Path
        CSV 파일이 존재하는 디렉토리.
    n_subjects : int
        최대 피험자 번호.

    Returns
    -------
    list[dict]
        각 요소: {'path', 'sid', 'cond', 'trial', 'label'}.

    Raises
    ------
    FileNotFoundError
        data_dir이 존재하지 않을 때.
    """
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
            "label": cond - 1,
        })
    return records


# ─────────────────────────────────────────────
# 2. 기존 HDF5 상태 확인
# ─────────────────────────────────────────────

def load_existing_h5_info(
    h5_path: Path,
) -> tuple[set[int], Optional[list[str]], int]:
    """기존 HDF5에서 처리 완료된 피험자 ID와 채널 목록을 읽는다.

    v8: Subject-group 형식 (/subjects/S{sid}/X, /subjects/S{sid}/y)
    v7 호환: 구 형식 (flat X, y, subject_id) 도 자동 감지.

    Parameters
    ----------
    h5_path : Path
        HDF5 파일 경로.

    Returns
    -------
    tuple[set[int], list[str] | None, int]
        (완료된 subject_id 집합, 채널 리스트, 기존 스텝 수).
        파일이 없거나 읽기 실패 시 (빈set, None, 0).
    """
    if not h5_path.exists():
        return set(), None, 0

    try:
        with h5py.File(h5_path, "r") as f:
            # v8 subject-group 형식
            if "subjects" in f and "channels" in f:
                done_sids: set[int] = set()
                n_existing = 0
                for skey in f["subjects"]:
                    sid = int(skey[1:])          # "S0001" → 1
                    done_sids.add(sid)
                    n_existing += f[f"subjects/{skey}/X"].shape[0]
                channels = [
                    c.decode() if isinstance(c, bytes) else c
                    for c in f["channels"][:]
                ]
                return done_sids, channels, n_existing

            # v7 flat 형식 (하위 호환)
            for key in ("X", "y", "subject_id", "channels"):
                if key not in f:
                    log(f"  ⚠ 기존 HDF5에 '{key}' 없음, 처음부터 생성")
                    return set(), None, 0

            done_sids = set(np.unique(f["subject_id"][:]).tolist())
            channels = [
                c.decode() if isinstance(c, bytes) else c
                for c in f["channels"][:]
            ]
            n_existing = f["X"].shape[0]
            return done_sids, channels, n_existing

    except Exception as e:
        log(f"  ⚠ 기존 HDF5 읽기 실패: {e}")
        return set(), None, 0


# ─────────────────────────────────────────────
# 3. 공통 채널 교집합 계산
# ─────────────────────────────────────────────

def find_common_channels(records: list[dict]) -> list[str]:
    """모든 CSV의 컬럼 교집합을 구한다 (첫 파일 순서 보존).

    Returns
    -------
    list[str]
        공통 채널명 리스트.

    Raises
    ------
    ValueError
        공통 채널이 0개일 때.
    """
    log("  [Pass 0] 전 파일 공통 채널 계산...")
    common: Optional[set[str]] = None
    first_cols_ordered: Optional[list[str]] = None

    for i, rec in enumerate(records):
        try:
            cols_df = pd.read_csv(rec["path"], skiprows=2, nrows=0)
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


def verify_channels(
    records: list[dict], required_channels: list[str],
) -> None:
    """신규 CSV들이 기존 채널을 모두 포함하는지 검증한다.

    불일치 파일이 있으면 경고 로그를 출력한다 (해당 파일은 Pass 2에서 스킵).
    """
    req_set = set(required_channels)
    warn_count = 0

    for rec in records:
        try:
            cols_df = pd.read_csv(rec["path"], skiprows=2, nrows=0)
            cols_df = rename_columns(cols_df)
            cols = cols_df.columns.tolist()
            flex_drop = set(config.resolve_drop_cols(cols))
            available = set(c for c in cols if c not in flex_drop)
            missing = req_set - available
            if missing:
                warn_count += 1
                if warn_count <= 5:
                    log(f"    ⚠ 채널 부족: {rec['path'].name}"
                        f" (누락 {len(missing)}개)")
        except Exception:
            continue

    if warn_count > 0:
        log(f"    ⚠ 총 {warn_count}개 파일 채널 부족 (Pass 2에서 스킵됨)")
    else:
        log(f"    ✅ 신규 {len(records)}개 CSV 채널 호환 확인")


# ─────────────────────────────────────────────
# 4. 신호 추출 + 대역통과 필터
# ─────────────────────────────────────────────

def compute_foot_acc_norm(df: pd.DataFrame, side: str = "LT") -> np.ndarray:
    """발 가속도 센서의 3축 norm을 계산한다 (폴백용)."""
    cols = config.resolve_foot_acc_cols(df.columns.tolist(), side)
    ax = df[cols["x"]].values.astype(np.float64)
    ay = df[cols["y"]].values.astype(np.float64)
    az = df[cols["z"]].values.astype(np.float64)
    return np.sqrt(ax**2 + ay**2 + az**2)


def _resolve_sensor_axis(
    columns: list[str], sensor: str, axis: str, side: str,
    signal_type: str = "gyroscope",
) -> str | None:
    """특정 센서의 특정 축 채널명을 찾는다.

    Parameters
    ----------
    sensor : str
        센서 부위 (예: "Shank", "Foot")
    axis : str
        축 (예: "x", "y", "z")
    side : str
        좌우 ("LT", "RT")
    signal_type : str
        "gyroscope" 또는 "accel"
    """
    sensor_l = sensor.lower()
    axis_l   = axis.lower()
    side_l   = side.lower()
    type_l   = signal_type.lower()

    for c in columns:
        cl = c.lower()
        # 센서 + 좌우 + 타입 + 축 모두 포함
        has_sensor = sensor_l in cl
        has_side   = side_l in cl or sensor_l == "pelvis"
        has_type   = type_l in cl or ("accel" in type_l and "accel" in cl)
        # 축: 마지막 단어 또는 -x, _x 형태
        has_axis   = (cl.endswith(f"-{axis_l}") or
                     cl.endswith(f"_{axis_l}") or
                     f" {axis_l} " in cl or
                     f"-{axis_l} " in cl or
                     f" {axis_l} {side_l}" in cl)

        if has_sensor and has_side and has_type and has_axis:
            return c
    return None


def extract_ml_gyro(df: pd.DataFrame, side: str) -> np.ndarray | None:
    """ML(Mediolateral) Gyroscope 신호를 추출한다.

    config.HS_GYRO_SENSOR + config.HS_GYRO_AXIS 기반.
    """
    col = _resolve_sensor_axis(
        df.columns.tolist(),
        config.HS_GYRO_SENSOR, config.HS_GYRO_AXIS, side,
        "gyroscope",
    )
    if col is None:
        return None
    return df[col].values.astype(np.float64)


def extract_ap_accel(df: pd.DataFrame, side: str) -> np.ndarray | None:
    """AP(Anteroposterior) Accelerometer 신호를 추출한다.

    config.HS_ACCEL_SENSOR + config.HS_ACCEL_AXIS 기반.
    """
    col = _resolve_sensor_axis(
        df.columns.tolist(),
        config.HS_ACCEL_SENSOR, config.HS_ACCEL_AXIS, side,
        "accel",
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
    """4차 Butterworth 대역통과 필터를 적용한다."""
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


# ─────────────────────────────────────────────
# 5. Pass 1: 지면별 통계 수집
# ─────────────────────────────────────────────

def collect_terrain_stats(records: list[dict]) -> dict[int, dict]:
    """각 지형 조건별 Foot Acc 통계를 수집하고 적응적 α를 계산한다.

    전체 레코드(기존+신규)를 사용하여 최적 임계값을 구한다.

    NaN은 시간축을 유지한 채 선형 보간하여 연속 신호로 복원한 뒤
    통계/피크를 산출한다 (시간축 단절 방지).
    """
    log("  [Pass 1] 지면별 통계 수집...")
    terrain_stats: dict[int, dict[str, list]] = defaultdict(
        lambda: {"means": [], "stds": [], "peak_heights": [], "stride_intervals": []}
    )

    for i, rec in enumerate(records):
        try:
            df = pd.read_csv(rec["path"], skiprows=2, low_memory=False)
            df = rename_columns(df)
        except Exception:
            continue

        cond = rec["cond"]
        for side in ["LT", "RT"]:
            try:
                # v8.1: Type 4 기반 통계 수집 (ML Gyro + Foot Accel)
                ml_gyro  = extract_ml_gyro(df, side)
                ap_accel = extract_ap_accel(df, side)

                if ml_gyro is None or ap_accel is None:
                    # 폴백: Foot Acc Norm
                    norm = compute_foot_acc_norm(df, side)
                    sig = bandpass_filter(norm)
                else:
                    sig = bandpass_filter(ml_gyro)

                # NaN 보간
                nan_mask = np.isnan(sig)
                n_valid = int(np.sum(~nan_mask))
                if n_valid < config.HS_MIN_STRIDE_SAM * 2:
                    continue

                if nan_mask.any():
                    x = np.arange(len(sig))
                    sig = sig.copy()
                    sig[nan_mask] = np.interp(
                        x[nan_mask], x[~nan_mask], sig[~nan_mask]
                    )

                mu  = float(np.mean(sig))
                std = float(np.std(sig))
                terrain_stats[cond]["means"].append(mu)
                terrain_stats[cond]["stds"].append(std)

                # ML Gyro min으로 피크 검출 (Type 4)
                if ml_gyro is not None:
                    peaks, props = find_peaks(
                        -sig, prominence=0.4 * std,
                        distance=config.HS_MIN_STRIDE_SAM,
                    )
                else:
                    peaks, props = find_peaks(
                        sig, height=mu + 0.5 * std,
                        distance=config.HS_MIN_STRIDE_SAM,
                    )

                if len(peaks) > 0 and "peak_heights" in props:
                    terrain_stats[cond]["peak_heights"].extend(
                        props["peak_heights"].tolist()
                    )
                elif len(peaks) > 0 and "prominences" in props:
                    terrain_stats[cond]["peak_heights"].extend(
                        props["prominences"].tolist()
                    )
                # 피크 간 간격 수집 (보폭 분포)
                if len(peaks) > 1:
                    intervals = np.diff(peaks).tolist()
                    terrain_stats[cond]["stride_intervals"].extend(intervals)
            except (KeyError, ValueError):
                continue

        del df; gc.collect()
        if (i + 1) % 50 == 0:
            log(f"    Pass 1: {i+1}/{len(records)} 파일")

    terrain_params: dict[int, dict] = {}
    for cond in sorted(terrain_stats.keys()):
        s = terrain_stats[cond]
        if not s["means"]:
            terrain_params[cond] = {
                "alpha": 1.0,
                "min_dist": config.HS_MIN_STRIDE_SAM,
                "min_peak_ratio": config.HS_PEAK_QUALITY_RATIO,
                "stride_min_sam": config.HS_MIN_STRIDE_SAM,
                "stride_max_sam": config.HS_MAX_STRIDE_SAM,
            }
            continue
        avg_mean = float(np.mean(s["means"]))
        avg_std  = float(np.mean(s["stds"]))
        avg_peak = (float(np.mean(s["peak_heights"]))
                    if s["peak_heights"] else avg_mean + avg_std)
        snr = (avg_peak - avg_mean) / avg_std if avg_std > 0 else 0
        alpha = max(0.5, min(2.0, snr * 0.5))

        # 보폭 범위: 데이터 기반 percentile (하드코딩 폴백)
        if len(s["stride_intervals"]) >= 20:
            si = np.array(s["stride_intervals"])
            stride_min = int(np.percentile(si, 2))   # 2nd percentile
            stride_max = int(np.percentile(si, 98))   # 98th percentile
            # 안전 마진 10%
            stride_min = max(40, int(stride_min * 0.9))   # 최소 40샘플(200ms)
            stride_max = int(stride_max * 1.1)
            stride_mean = float(np.mean(si))
            stride_std  = float(np.std(si))
        else:
            stride_min  = config.HS_MIN_STRIDE_SAM
            stride_max  = config.HS_MAX_STRIDE_SAM
            stride_mean = 0.0
            stride_std  = 0.0

        terrain_params[cond] = {
            "alpha": round(alpha, 3),
            "min_dist": stride_min,
            "min_peak_ratio": config.HS_PEAK_QUALITY_RATIO,
            "stride_min_sam": stride_min,
            "stride_max_sam": stride_max,
            "stride_mean_ms": round(stride_mean / config.SAMPLE_RATE * 1000, 1),
            "stride_std_ms": round(stride_std / config.SAMPLE_RATE * 1000, 1),
            "stride_n_samples": len(s["stride_intervals"]),
            "avg_mean": round(avg_mean, 1),
            "avg_std": round(avg_std, 1),
            "avg_peak": round(avg_peak, 1),
        }
        log(f"    C{cond}: α={alpha:.2f}  μ={avg_mean:.0f}"
            f"  σ={avg_std:.0f}  peak={avg_peak:.0f}"
            f"  stride={stride_min}~{stride_max}sam"
            f" ({stride_min/config.SAMPLE_RATE*1000:.0f}~{stride_max/config.SAMPLE_RATE*1000:.0f}ms)"
            f"  n={len(s['stride_intervals'])}")

    return terrain_params


# ─────────────────────────────────────────────
# 6. 힐스트라이크 검출 (3단계 검증)
# ─────────────────────────────────────────────

def detect_steps(
    ml_gyro: np.ndarray,
    ap_accel: np.ndarray,
    terrain_params: dict,
    cond: int,
    fs: int = config.SAMPLE_RATE,
) -> list[tuple[int, int]]:
    """ML Gyro + AP Accel 융합 힐스트라이크 검출.

    알고리즘:
        1. 입력: Shank ML Gyroscope + Foot AP Accelerometer
        2. 필터: Butterworth bandpass 1-20Hz
        3. HS 후보: ML gyro local min (prominence=0.4σ, dist=0.35s)
        4. 융합 검증: 후보 ±125ms 내 AP accel min < mean - 0.8σ
        5. 보폭 제약: 조건별 stride_min ~ stride_max

    Parameters
    ----------
    ml_gyro : np.ndarray
        Mediolateral gyroscope 원 신호.
    ap_accel : np.ndarray
        Anteroposterior accelerometer 원 신호.
    terrain_params : dict
        조건별 파라미터 (stride_min_sam, stride_max_sam 등).
    cond : int
        지면 조건 번호.
    fs : int
        샘플링 주파수.

    Returns
    -------
    list[tuple[int, int]]
        유효 스텝 구간 리스트 ``[(start, end), ...]``.
    """
    n = len(ml_gyro)
    if n < fs:  # 최소 1초
        return []

    # NaN 체크
    nan_ratio = (np.isnan(ml_gyro).sum() + np.isnan(ap_accel).sum()) / (2 * n)
    if nan_ratio > config.HS_NAN_THRESHOLD:
        return []

    # NaN 보간
    for sig in [ml_gyro, ap_accel]:
        mask = np.isnan(sig)
        if mask.any() and not mask.all():
            x = np.arange(n)
            sig[mask] = np.interp(x[mask], x[~mask], sig[~mask])

    # 1. 대역통과 필터 (1-20Hz: 중력/고주파 노이즈 제거)
    ml_f = bandpass_filter(ml_gyro, fs, low=1.0, high=20.0)
    ap_f = bandpass_filter(ap_accel, fs, low=1.0, high=20.0)

    # 2. ML gyro local minimum → HS 후보
    sigma_ml = float(np.std(ml_f))
    if sigma_ml == 0:
        return []

    min_dist_sam = int(fs * 0.35)  # 최소 350ms 간격

    # Mid-swing peaks (양의 방향 = 스윙 중간)
    mid_swings, _ = find_peaks(ml_f, prominence=0.3 * sigma_ml, distance=min_dist_sam)

    # Trusted swing threshold (Niswander: 20% of max mid-swing)
    trusted_thresh = config.HS_TRUSTED_SWING * (np.max(ml_f) if len(mid_swings) > 0 else sigma_ml)

    # -ml_f의 피크 = ml_f의 minimum
    hs_cand, _ = find_peaks(
        -ml_f,
        prominence=config.HS_GYRO_PROMINENCE * sigma_ml,
        distance=min_dist_sam,
    )
    if len(hs_cand) < 2:
        return []

    # Trusted swing 필터: HS 후보 근처에 충분한 스윙이 있어야 유효
    hs_trusted: list[int] = []
    for cand in hs_cand:
        # 후보 전후 0.5초 내 mid-swing peak 존재 확인
        nearby_swings = [p for p in mid_swings if abs(p - cand) < fs // 2]
        if len(nearby_swings) > 0:
            # 가장 가까운 mid-swing의 크기가 trusted threshold 이상
            nearest = min(nearby_swings, key=lambda p: abs(p - cand))
            if ml_f[nearest] > trusted_thresh:
                hs_trusted.append(cand)
        else:
            # mid-swing 없으면 주변 ml_f 최대값으로 폴백
            w_start = max(0, cand - fs // 2)
            w_end   = min(n, cand + fs // 2)
            if np.max(ml_f[w_start:w_end]) > trusted_thresh:
                hs_trusted.append(cand)

    if len(hs_trusted) < 2:
        return []

    # 3. AP accel 융합 검증
    mean_ap  = float(np.mean(ap_f))
    sigma_ap = float(np.std(ap_f))
    window   = config.HS_FUSION_WINDOW_SAM  # ±25 samples (±125ms)

    hs_final: list[int] = []
    for cand in hs_trusted:
        w_start = max(0, cand - window)
        w_end   = min(n, cand + window)
        # AP accel이 평균보다 충분히 낮은지 (감속 확인)
        if np.min(ap_f[w_start:w_end]) < mean_ap - config.HS_ACCEL_THRESHOLD * sigma_ap:
            hs_final.append(cand)

    if len(hs_final) < 2:
        return []

    # 4. 보폭 제약으로 스텝 구간 생성
    _default_params = {
        "stride_min_sam": config.HS_MIN_STRIDE_SAM,
        "stride_max_sam": config.HS_MAX_STRIDE_SAM,
    }
    params = terrain_params.get(cond) or terrain_params.get(str(cond)) or _default_params
    stride_min = params.get("stride_min_sam", config.HS_MIN_STRIDE_SAM)
    stride_max = params.get("stride_max_sam", config.HS_MAX_STRIDE_SAM)

    valid_steps: list[tuple[int, int]] = []
    for i in range(len(hs_final) - 1):
        start = hs_final[i]
        end   = hs_final[i + 1]
        length = end - start
        if length < stride_min or length > stride_max:
            continue
        # QC: 원본 신호 NaN 비율
        seg_nan = np.isnan(ml_gyro[start:end]).sum() / length
        if seg_nan > config.HS_NAN_THRESHOLD:
            continue
        valid_steps.append((start, end))

    return valid_steps


# ─────────────────────────────────────────────
# 7. 리샘플링
# ─────────────────────────────────────────────

def resample_step(
    data_segment: np.ndarray, target_length: int = config.TS,
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


# ─────────────────────────────────────────────
# 8. 기존 데이터를 tmp에 복사
# ─────────────────────────────────────────────

COPY_CHUNK: int = 2000


# v8: copy_existing_to_tmp 제거 — Subject-group 형식은 복사 불필요


# ─────────────────────────────────────────────
# 9. 메인
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    p = argparse.ArgumentParser(description="힐스트라이크 스텝 분할")
    p.add_argument("--n_subjects", type=int, default=None,
                   help="피험자 수 (기본: config.N_SUBJECTS)")
    p.add_argument("--force", action="store_true",
                   help="기존 HDF5 무시, 전체 재생성")
    return p.parse_args()


def main() -> None:
    """증분 힐스트라이크 분할 파이프라인을 실행한다."""
    args = parse_args()
    config.apply_overrides(n_subjects=args.n_subjects)
    force = args.force

    log(f"{'='*60}")
    log(f"  step_segmentation.py v8.1 ({'전체 재생성' if force else '증분 모드'})")
    log(f"  N={config.N_SUBJECTS}  TS={config.TS}pt  {config.SAMPLE_RATE}Hz")
    log(f"{'='*60}\n")

    # ── 전체 CSV 탐색 ──
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

    # ── 기존 HDF5 확인 ──
    if force:
        done_sids: set[int] = set()
        existing_channels: Optional[list[str]] = None
        existing_n: int = 0
        log("  ★ --force: 기존 HDF5 무시, 전체 재생성")
    else:
        done_sids, existing_channels, existing_n = load_existing_h5_info(
            config.H5_PATH
        )
        if done_sids:
            log(f"  ★ 기존 HDF5: {existing_n}스텝, {len(done_sids)}명 완료")
            log(f"    완료: {sorted(done_sids)}")

    # ── 신규 레코드 필터링 ──
    new_records = [r for r in all_records if r["sid"] not in done_sids]
    new_sids = sorted(set(r["sid"] for r in new_records))

    if not new_records:
        log(f"\n  ✅ 모든 {len(done_sids)}명 처리 완료, 스텝 검출 스킵")
        log(f"     기존 HDF5: {config.H5_PATH} ({existing_n}스텝)")
        return

    log(f"\n  신규 피험자: {new_sids} ({len(new_sids)}명)")
    log(f"  신규 CSV: {len(new_records)}개")

    # ── Pass 0: 채널 결정 ──
    if existing_channels:
        channels = existing_channels
        log(f"  [Pass 0] 기존 채널 재사용: {len(channels)}개 (일관성 유지)")
        verify_channels(new_records, channels)
    else:
        all_common = find_common_channels(all_records)
        # v8.1: Raw IMU 54ch만 필터 (Accel + Gyro)
        from channel_groups import filter_raw_channels
        channels = filter_raw_channels(all_common)
        log(f"  [Pass 0] Raw IMU 필터: {len(all_common)}ch → {len(channels)}ch")
        if len(channels) == 0:
            raise ValueError("Raw IMU 채널(Accel/Gyro)을 찾을 수 없습니다")
    n_ch = len(channels)

    # ── Pass 1: 지면별 통계 (전체 레코드 사용) ──
    terrain_params = collect_terrain_stats(all_records)
    params_path = config.BATCH_DIR / "terrain_params.json"
    params_path.write_text(
        json.dumps(terrain_params, indent=2, default=str, ensure_ascii=False)
    )
    log(f"  지면 파라미터 → {params_path}  ({time.time()-t0:.1f}s)\n")


    # ── Pass 2: 신규 스텝 검출 → HDF5 (Subject-group v8) ──
    log(f"  [Pass 2] 신규 {len(new_records)}개 CSV → 스텝 검출...")
    t1 = time.time()

    # v8: 피험자별 버퍼 (전체 복사 없음)
    subj_bufs: dict[int, dict[str, list]] = defaultdict(
        lambda: {"X": [], "y": []}
    )
    new_steps = 0
    step_log: list[dict] = []
    raw_lens: list[int] = []

    for i, rec in enumerate(new_records):
        try:
            df = pd.read_csv(rec["path"], skiprows=2, low_memory=False)
            df = rename_columns(df)
        except Exception as e:
            log(f"  ⚠ CSV 읽기 실패: {rec['path'].name} ({e})")
            continue

        sid   = rec["sid"]
        label = rec["cond"]
        data_cols = [c for c in channels if c in df.columns]
        if len(data_cols) < len(channels):
            missing = set(channels) - set(data_cols)
            log(f"  ⚠ {rec['path'].name}: {len(missing)}개 채널 누락, 스킵")
            continue
        data_np = df[channels].values.astype(np.float32)

        file_steps = 0
        for side in ("LT", "RT"):
            try:
                # v8.1 Type 4: Shank ML Gyro + Foot AP Accel
                ml_gyro  = extract_ml_gyro(df, side)
                ap_accel = extract_ap_accel(df, side)

                if ml_gyro is None or ap_accel is None:
                    # 폴백: Foot Acc Norm (Type 4 채널 없을 때)
                    norm = compute_foot_acc_norm(df, side)
                    norm_f = bandpass_filter(norm)
                    # 구버전 피크 검출
                    sigma = float(np.std(norm_f[~np.isnan(norm_f)])) if np.any(~np.isnan(norm_f)) else 1.0
                    peaks, _ = find_peaks(norm_f, height=np.nanmean(norm_f) + 0.5 * sigma,
                                          distance=config.HS_MIN_STRIDE_SAM)
                    steps = [(int(peaks[j]), int(peaks[j+1]))
                             for j in range(len(peaks)-1)
                             if config.HS_MIN_STRIDE_SAM <= peaks[j+1]-peaks[j] <= config.HS_MAX_STRIDE_SAM]
                else:
                    steps = detect_steps(
                        ml_gyro.copy(), ap_accel.copy(),
                        terrain_params, label,
                    )

                for (start, end) in steps:
                    raw_len = end - start
                    if raw_len < config.MIN_STEP_LEN:
                        continue
                    seg_256 = resample_step(data_np[start:end])
                    subj_bufs[sid]["X"].append(seg_256)
                    subj_bufs[sid]["y"].append(label)
                    file_steps += 1
                    new_steps += 1
                    raw_lens.append(raw_len)

                    if len(step_log) < 2000:
                        step_log.append({
                            "file": rec["path"].name,
                            "side": side,
                            "start": start, "end": end,
                            "raw_len": raw_len,
                        })
            except (KeyError, ValueError):
                continue

        del df, data_np; gc.collect()
        if (i + 1) % 20 == 0 or (i + 1) == len(new_records):
            log(f"    {i+1}/{len(new_records)} 파일"
                f"  이 파일: {file_steps} 스텝"
                f"  신규 누적: {new_steps}")

    log(f"\n  [Pass 2] 스텝 검출 완료: {new_steps}스텝  ({time.time()-t1:.1f}s)")

    # ── Subject-group HDF5 쓰기 (전체 복사 없음!) ──
    if force and config.H5_PATH.exists():
        config.H5_PATH.unlink()

    t2 = time.time()
    with h5py.File(config.H5_PATH, "a") as hf:
        # subjects 그룹 보장
        if "subjects" not in hf:
            hf.create_group("subjects")

        total_steps = existing_n
        for sid, buf in sorted(subj_bufs.items()):
            if not buf["X"]:
                continue
            grp_name = f"subjects/S{sid:04d}"
            X_arr = np.stack(buf["X"], axis=0).astype(np.float32)
            y_arr = np.array(buf["y"], dtype=np.int64)
            n_new = len(X_arr)

            if grp_name in hf:
                # 기존 subject 확장 (동일 피험자의 추가 trial)
                grp = hf[grp_name]
                old_n = grp["X"].shape[0]
                new_n = old_n + n_new
                grp["X"].resize(new_n, axis=0)
                grp["y"].resize(new_n, axis=0)
                grp["X"][old_n:new_n] = X_arr
                grp["y"][old_n:new_n] = y_arr
                log(f"    S{sid:04d}: 확장 {old_n}→{new_n}스텝")
            else:
                # 신규 subject 생성
                grp = hf.create_group(grp_name)
                grp.create_dataset(
                    "X", data=X_arr,
                    maxshape=(None, config.TS, n_ch),
                    chunks=(min(64, n_new), config.TS, n_ch),
                )
                grp.create_dataset(
                    "y", data=y_arr, maxshape=(None,),
                )
                log(f"    S{sid:04d}: 신규 {n_new}스텝")

            total_steps += n_new
            del X_arr, y_arr

        # 메타데이터 저장
        if "channels" in hf:
            del hf["channels"]
        hf.create_dataset("channels", data=np.array(channels, dtype="S"))
        hf.attrs["segmentation"] = "heel_strike"
        hf.attrs["sample_rate"]  = config.SAMPLE_RATE
        hf.attrs["target_ts"]    = config.TS
        hf.attrs["n_classes"]    = config.NUM_CLASSES
        hf.attrs["format"]       = "subject_group_v8"

    log(f"  HDF5 쓰기: {time.time()-t2:.1f}s  (전체 복사 없음)")

    # ── 결과 출력 ──
    size_mb = config.H5_PATH.stat().st_size / 1024**2
    log(f"\n  ✅ HDF5: {config.H5_PATH}")
    log(f"     기존: {existing_n}스텝 + 신규: {new_steps}스텝 = 총: {total_steps}스텝")
    log(f"     파일 크기: {size_mb:.1f} MB")

    if total_steps > 0:
        with h5py.File(config.H5_PATH, "r") as f:
            all_sids = sorted(int(k[1:]) for k in f["subjects"])
            label_counts: dict[int, int] = defaultdict(int)
            for skey in f["subjects"]:
                for lbl in f[f"subjects/{skey}/y"][:]:
                    label_counts[int(lbl)] += 1
            log(f"     라벨 분포: {dict(sorted(label_counts.items()))}")
            log(f"     피험자: {all_sids} ({len(all_sids)}명)")
        if raw_lens:
            log(f"     신규 스텝 길이: min={min(raw_lens)} max={max(raw_lens)}"
                f"  mean={np.mean(raw_lens):.0f}"
                f" ({np.mean(raw_lens)/config.SAMPLE_RATE*1000:.0f}ms)")
    else:
        log("  ⚠ 검출된 스텝이 없습니다!")

    log_path = config.BATCH_DIR / "step_log.json"
    log_path.write_text(json.dumps(step_log[:2000], indent=1, ensure_ascii=False, default=int))
    log(f"  스텝 로그 → {log_path}")
    log(f"\n  총 소요: {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()