"""
step_segmentation.py — 힐스트라이크 스텝 분할 (v8.0 증분 모드)
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
# 4. Foot Acc norm + 대역통과 필터
# ─────────────────────────────────────────────

def compute_foot_acc_norm(df: pd.DataFrame, side: str = "LT") -> np.ndarray:
    """발 가속도 센서의 3축 norm을 계산한다.

    컬럼명이 정확히 일치하지 않아도 패턴 매칭으로 자동 탐색.
    """
    cols = config.resolve_foot_acc_cols(df.columns.tolist(), side)
    ax = df[cols["x"]].values.astype(np.float64)
    ay = df[cols["y"]].values.astype(np.float64)
    az = df[cols["z"]].values.astype(np.float64)
    return np.sqrt(ax**2 + ay**2 + az**2)


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
        lambda: {"means": [], "stds": [], "peak_heights": []}
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
                norm = compute_foot_acc_norm(df, side)
                norm_f = bandpass_filter(norm)

                # NaN 보간: 시간축 유지한 채 선형 보간 → 연속 신호 복원
                nan_mask = np.isnan(norm_f)
                n_valid = int(np.sum(~nan_mask))
                if n_valid < config.HS_MIN_STRIDE_SAM * 2:
                    continue

                if nan_mask.any():
                    x = np.arange(len(norm_f))
                    norm_f = norm_f.copy()
                    norm_f[nan_mask] = np.interp(
                        x[nan_mask], x[~nan_mask], norm_f[~nan_mask]
                    )

                mu  = float(np.mean(norm_f))
                std = float(np.std(norm_f))
                terrain_stats[cond]["means"].append(mu)
                terrain_stats[cond]["stds"].append(std)

                # 보간된 연속 신호에서 피크 검출 → 간격 통계 정확
                peaks, props = find_peaks(
                    norm_f, height=mu + 0.5 * std,
                    distance=config.HS_MIN_STRIDE_SAM,
                )
                if len(peaks) > 0:
                    terrain_stats[cond]["peak_heights"].extend(
                        props["peak_heights"].tolist()
                    )
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
            }
            continue
        avg_mean = float(np.mean(s["means"]))
        avg_std  = float(np.mean(s["stds"]))
        avg_peak = (float(np.mean(s["peak_heights"]))
                    if s["peak_heights"] else avg_mean + avg_std)
        snr = (avg_peak - avg_mean) / avg_std if avg_std > 0 else 0
        alpha = max(0.5, min(2.0, snr * 0.5))
        terrain_params[cond] = {
            "alpha": round(alpha, 3),
            "min_dist": config.HS_MIN_STRIDE_SAM,
            "min_peak_ratio": config.HS_PEAK_QUALITY_RATIO,
            "avg_mean": round(avg_mean, 1),
            "avg_std": round(avg_std, 1),
            "avg_peak": round(avg_peak, 1),
        }
        log(f"    C{cond}: α={alpha:.2f}  μ={avg_mean:.0f}"
            f"  σ={avg_std:.0f}  peak={avg_peak:.0f}")

    return terrain_params


# ─────────────────────────────────────────────
# 6. 힐스트라이크 검출 (3단계 검증)
# ─────────────────────────────────────────────

def detect_steps(
    norm_signal: np.ndarray,
    terrain_params: dict[int, dict],
    cond: int,
) -> list[tuple[int, int]]:
    """3단계 힐스트라이크 스텝을 검출한다.

    NaN은 시간축을 유지한 채 선형 보간 후 피크를 검출하고,
    원본 신호의 NaN 비율로 최종 QC를 수행한다.
    좌·우 발 이벤트는 독립적으로 검출되며, 동일 시도 내에서도
    시간 구간이 달라 중복 샘플로 간주하지 않는다.
    """
    params = terrain_params.get(cond, {
        "alpha": 1.0,
        "min_dist": config.HS_MIN_STRIDE_SAM,
        "min_peak_ratio": config.HS_PEAK_QUALITY_RATIO,
    })
    alpha          = params["alpha"]
    min_dist       = params["min_dist"]
    min_peak_ratio = params["min_peak_ratio"]

    # 유효 샘플 수 체크 (시간축 유지)
    nan_mask = np.isnan(norm_signal)
    n_valid = int(np.sum(~nan_mask))
    if n_valid < min_dist * 3:
        return []

    # NaN 보간: 시간축 유지한 채 선형 보간 → 연속 신호 복원
    sig = norm_signal.copy()
    if nan_mask.any():
        x = np.arange(len(sig))
        sig[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], sig[~nan_mask])

    file_mean = float(np.mean(sig))
    file_std  = float(np.std(sig))
    if file_std == 0:
        return []
    threshold = file_mean + alpha * file_std

    peaks, props = find_peaks(
        sig,
        height=threshold,
        distance=min_dist,
        prominence=file_std * config.HS_PROMINENCE_COEFF,
    )
    if len(peaks) < 2:
        return []

    avg_peak_h = float(np.mean(props["peak_heights"]))
    quality_threshold = avg_peak_h * min_peak_ratio

    valid_steps: list[tuple[int, int]] = []
    for i in range(len(peaks) - 1):
        start, end = int(peaks[i]), int(peaks[i + 1])
        length = end - start
        if length < config.HS_MIN_STRIDE_SAM or length > config.HS_MAX_STRIDE_SAM:
            continue
        segment_max = float(np.max(sig[start:end]))
        if segment_max < quality_threshold:
            continue
        # QC: 원본 신호의 NaN 비율로 데이터 품질 검증
        orig_segment = norm_signal[start:end]
        nan_ratio = np.sum(np.isnan(orig_segment)) / len(orig_segment)
        if nan_ratio > config.HS_NAN_THRESHOLD:
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
    log(f"  step_segmentation.py v8.0 ({'전체 재생성' if force else '증분 모드'})")
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
        channels = find_common_channels(all_records)
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
                acc_cols = config.resolve_foot_acc_cols(df.columns.tolist(), side)
                contact_col = config.resolve_foot_contact_col(df.columns.tolist(), side)
                acc_data = df[acc_cols].values.astype(np.float64)
                contact  = df[contact_col].values.astype(np.float64)
                norm_acc = np.sqrt((acc_data**2).sum(axis=1))

                tp = terrain_params.get(str(label))
                if tp is None:
                    tp = terrain_params.get(label)
                if tp is None:
                    continue

                steps = detect_steps(
                    norm_acc, contact,
                    sample_rate=config.SAMPLE_RATE,
                    terrain_params=tp,
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
    log_path.write_text(json.dumps(step_log[:2000], indent=1, ensure_ascii=False))
    log(f"  스텝 로그 → {log_path}")
    log(f"\n  총 소요: {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
