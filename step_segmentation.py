"""
step_segmentation.py — 힐스트라이크 스텝 분할 (v8 Final)
═══════════════════════════════════════════════════════
3-패스 아키텍처:
    Pass 0 : 전 파일 공통 채널 교집합
    Pass 1 : 지면별(C1-C6) 적응적 임계값 자동 튜닝
    Pass 2 : 3단계 힐스트라이크 검출 -> HDF5 청크 쓰기

v7->v8: 진행률 추정, CSV 읽기 재시도, NaN 통계, 검증 강화
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys, time, re, json, gc
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
    """타임스탬프 포함 로그."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────
# 컬럼명 통일
# ─────────────────────────────────────────────
RENAME_MAP: dict[str, str] = {
    "Noraxon MyoMotion-Joints-Knee LT-Rotation Ext (deg)":
        "Knee Rotation Ext LT (deg)",
    "Noraxon MyoMotion-Joints-Knee RT-Rotation Ext (deg)":
        "Knee Rotation Ext RT (deg)",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Noraxon 버전 간 컬럼명 차이를 통일한다."""
    renames = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    if renames:
        df = df.rename(columns=renames)
    return df


def _safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    """CSV를 안전하게 읽는다. 실패 시 None 반환."""
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        log(f"    [WARN] CSV 읽기 실패: {path.name} ({type(e).__name__}: {e})")
        return None


# ─────────────────────────────────────────────
# 1. 파일 탐색
# ─────────────────────────────────────────────

def parse_filename(fname: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """파일명에서 피험자/지형/시행 번호를 추출한다.

    Returns
    -------
    tuple
        (sid, cond, trial) 또는 (None, None, None).
    """
    m = re.search(r"S(\d+)C(\d+)T(\d+)", fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"S(\d+)C(\d+)", fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), 1
    return None, None, None


def discover_csvs(data_dir: Path, n_subjects: int) -> list[dict]:
    """유효한 CSV 레코드를 수집한다.

    Raises
    ------
    FileNotFoundError
        data_dir 미존재 시.
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
# 2. 공통 채널 교집합
# ─────────────────────────────────────────────

def find_common_channels(records: list[dict]) -> list[str]:
    """모든 CSV의 컬럼 교집합 (첫 파일 순서 보존).

    Raises
    ------
    ValueError
        공통 채널 0개 시.
    """
    log("  [Pass 0] 전 파일 공통 채널 계산...")
    drop = set(config.DROP_COLS)
    common: Optional[set[str]] = None
    first_ordered: Optional[list[str]] = None
    scan_failures = 0

    for i, rec in enumerate(records):
        cols_df = _safe_read_csv(rec["path"], skiprows=2, nrows=0)
        if cols_df is None:
            scan_failures += 1
            continue
        cols_df = rename_columns(cols_df)
        cols = cols_df.columns.tolist()
        data_cols = {c for c in cols if c not in drop}
        if common is None:
            common = data_cols
            first_ordered = [c for c in cols if c not in drop]
        else:
            common = common & data_cols

        if (i + 1) % 100 == 0:
            log(f"    {i+1}/{len(records)} 스캔 (공통: {len(common) if common else 0}ch)")

    if scan_failures > 0:
        log(f"    [WARN] {scan_failures}개 파일 스캔 실패")

    if common is None or first_ordered is None or len(common) == 0:
        raise ValueError("공통 채널이 없습니다. CSV 파일과 컬럼명을 확인하세요.")

    channels = [c for c in first_ordered if c in common]
    log(f"  공통 채널: {len(channels)}개")
    return channels


# ─────────────────────────────────────────────
# 3. Foot Acc norm + 필터
# ─────────────────────────────────────────────

def compute_foot_acc_norm(df: pd.DataFrame, side: str = "LT") -> np.ndarray:
    """발 가속도 3축 norm. 컬럼 없으면 KeyError."""
    cols = config.FOOT_ACC_COLS[side]
    for axis, colname in cols.items():
        if colname not in df.columns:
            raise KeyError(f"Foot Acc 컬럼 없음: {colname}")
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
    """Butterworth 대역통과 필터. NaN은 평균 대체 후 복원."""
    if len(signal) == 0:
        return signal
    nyq = fs / 2
    if low >= nyq or high >= nyq:
        return signal
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
# 4. Pass 1: 지면별 통계
# ─────────────────────────────────────────────

def collect_terrain_stats(records: list[dict]) -> dict[int, dict]:
    """지형 조건별 적응적 alpha를 계산한다."""
    log("  [Pass 1] 지면별 통계 수집...")
    terrain_stats: dict[int, dict[str, list]] = defaultdict(
        lambda: {"means": [], "stds": [], "peak_heights": []}
    )

    for i, rec in enumerate(records):
        df = _safe_read_csv(rec["path"], skiprows=2, low_memory=False)
        if df is None:
            continue
        df = rename_columns(df)
        cond = rec["cond"]

        for side in ["LT", "RT"]:
            try:
                norm = compute_foot_acc_norm(df, side)
                norm_f = bandpass_filter(norm)
                valid = norm_f[~np.isnan(norm_f)]
                if len(valid) < config.HS_MIN_STRIDE_SAM * 2:
                    continue
                mu, std = float(np.mean(valid)), float(np.std(valid))
                if std == 0:
                    continue
                terrain_stats[cond]["means"].append(mu)
                terrain_stats[cond]["stds"].append(std)
                peaks, props = find_peaks(
                    valid, height=mu + config.HS_MIN_PEAK_RATIO_FOR_STATS * std,
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
            log(f"    Pass 1: {i+1}/{len(records)}")

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
        log(f"    C{cond}: a={alpha:.2f}  mu={avg_mean:.0f}"
            f"  sig={avg_std:.0f}  peak={avg_peak:.0f}")

    return terrain_params


# ─────────────────────────────────────────────
# 5. 힐스트라이크 검출 (3단계)
# ─────────────────────────────────────────────

def detect_steps(
    norm_signal: np.ndarray,
    terrain_params: dict[int, dict],
    cond: int,
) -> list[tuple[int, int]]:
    """3단계 힐스트라이크 스텝을 검출한다.

    단계:
        1. 적응적 피크 검출 (mu + alpha * sigma)
        2. 보폭 길이 범위 검증
        3. 세그먼트 품질 (NaN, 피크 비율) 검증
    """
    params = terrain_params.get(cond, {
        "alpha": 1.0,
        "min_dist": config.HS_MIN_STRIDE_SAM,
        "min_peak_ratio": config.HS_PEAK_QUALITY_RATIO,
    })
    alpha          = params["alpha"]
    min_dist       = params["min_dist"]
    min_peak_ratio = params["min_peak_ratio"]

    valid = norm_signal[~np.isnan(norm_signal)]
    if len(valid) < min_dist * 3:
        return []

    file_mean = float(np.nanmean(norm_signal))
    file_std  = float(np.nanstd(norm_signal))
    if file_std <= 0:
        return []

    peaks, props = find_peaks(
        norm_signal,
        height=file_mean + alpha * file_std,
        distance=min_dist,
        prominence=file_std * config.HS_PROMINENCE_COEFF,
    )
    if len(peaks) < 2:
        return []

    avg_peak_h = float(np.mean(props["peak_heights"]))
    quality_thr = avg_peak_h * min_peak_ratio

    valid_steps: list[tuple[int, int]] = []
    for i in range(len(peaks) - 1):
        start, end = int(peaks[i]), int(peaks[i + 1])
        length = end - start
        if not (config.HS_MIN_STRIDE_SAM <= length <= config.HS_MAX_STRIDE_SAM):
            continue
        seg = norm_signal[start:end]
        if float(np.nanmax(seg)) < quality_thr:
            continue
        if np.sum(np.isnan(seg)) / len(seg) > config.HS_NAN_THRESHOLD:
            continue
        valid_steps.append((start, end))

    return valid_steps


# ─────────────────────────────────────────────
# 6. 리샘플링
# ─────────────────────────────────────────────

def resample_step(
    data_segment: np.ndarray, target_length: int = config.TS,
) -> np.ndarray:
    """가변 길이 -> 고정 길이 리샘플링. NaN은 보간 후 처리."""
    if data_segment.ndim != 2:
        raise ValueError(f"resample_step: 2D 배열 예상, got {data_segment.ndim}D")
    L, C = data_segment.shape
    if L == target_length:
        return data_segment.astype(np.float32)
    if L < 2:
        return np.zeros((target_length, C), dtype=np.float32)
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
# 7. 메인
# ─────────────────────────────────────────────

def main() -> None:
    """3-패스 힐스트라이크 분할 파이프라인."""
    config.set_seed()
    log(f"{'='*60}")
    log(f"  step_segmentation.py v8 Final")
    log(f"  N={config.N_SUBJECTS}  TS={config.TS}pt  {config.SAMPLE_RATE}Hz")
    log(f"{'='*60}\n")

    records = discover_csvs(config.DATA_DIR, config.N_SUBJECTS)
    log(f"  CSV 파일: {len(records)}개")
    if not records:
        raise FileNotFoundError(f"유효한 CSV 없음: {config.DATA_DIR}")

    cond_count: dict[int, int] = defaultdict(int)
    subj_set: set[int] = set()
    for r in records:
        cond_count[r["cond"]] += 1
        subj_set.add(r["sid"])
    log(f"  피험자: {sorted(subj_set)} ({len(subj_set)}명)")
    for c in sorted(cond_count):
        log(f"    C{c}: {cond_count[c]}파일")

    t0 = time.time()

    channels = find_common_channels(records)
    n_ch = len(channels)

    terrain_params = collect_terrain_stats(records)
    params_path = config.BATCH_DIR / "terrain_params.json"
    params_path.write_text(
        json.dumps(terrain_params, indent=2, default=str, ensure_ascii=False)
    )
    log(f"  지면 파라미터 -> {params_path}  ({time.time()-t0:.1f}s)\n")

    # Pass 2
    log("  [Pass 2] 스텝 검출 -> HDF5...")
    tmp_h5 = config.H5_PATH.with_suffix(".tmp.h5")
    if tmp_h5.exists():
        tmp_h5.unlink()

    total_steps = 0
    step_log: list[dict] = []
    raw_lens: list[int] = []
    skipped_files = 0
    nan_interp_count = 0

    with h5py.File(tmp_h5, "w") as hf:
        ds_X   = hf.create_dataset("X", shape=(0, config.TS, n_ch),
                                   maxshape=(None, config.TS, n_ch),
                                   dtype="float32", chunks=(64, config.TS, n_ch))
        ds_y   = hf.create_dataset("y", shape=(0,), maxshape=(None,), dtype="int64")
        ds_sid = hf.create_dataset("subject_id", shape=(0,), maxshape=(None,), dtype="int64")

        buf_X: list[np.ndarray] = []
        buf_y: list[int] = []
        buf_sid: list[int] = []

        def flush() -> None:
            nonlocal total_steps
            if not buf_X:
                return
            n = len(buf_X)
            old = ds_X.shape[0]
            new = old + n
            ds_X.resize(new, axis=0)
            ds_y.resize(new, axis=0)
            ds_sid.resize(new, axis=0)
            ds_X[old:new]   = np.stack(buf_X, axis=0)
            ds_y[old:new]   = np.array(buf_y, dtype=np.int64)
            ds_sid[old:new] = np.array(buf_sid, dtype=np.int64)
            total_steps += n
            buf_X.clear(); buf_y.clear(); buf_sid.clear()

        for i, rec in enumerate(records):
            df = _safe_read_csv(rec["path"], skiprows=2, low_memory=False)
            if df is None:
                skipped_files += 1
                continue
            df = rename_columns(df)

            drop = set(config.DROP_COLS)
            use_cols = [c for c in channels if c in df.columns and c not in drop]
            if len(use_cols) != n_ch:
                log(f"    [WARN] 채널 부족: {rec['path'].name} ({len(use_cols)}/{n_ch})")
                skipped_files += 1
                del df; gc.collect()
                continue

            data_np = df[use_cols].values.astype(np.float32)

            if np.isnan(data_np).any():
                nan_interp_count += 1
                df_tmp = pd.DataFrame(data_np)
                df_tmp = df_tmp.interpolate(limit_direction="both").fillna(0)
                data_np = df_tmp.values.astype(np.float32)
                del df_tmp

            file_steps = 0
            for side in ["LT", "RT"]:
                try:
                    raw_norm = compute_foot_acc_norm(df, side)
                    norm_f   = bandpass_filter(raw_norm)
                    steps    = detect_steps(norm_f, terrain_params, rec["cond"])
                    for start, end in steps:
                        seg = resample_step(data_np[start:end])
                        buf_X.append(seg)
                        buf_y.append(rec["label"])
                        buf_sid.append(rec["sid"])
                        file_steps += 1
                        raw_lens.append(end - start)
                        if len(buf_X) >= config.FLUSH_SIZE:
                            flush()
                        if len(step_log) < 2000:
                            step_log.append({
                                "file": rec["path"].name, "side": side,
                                "start": start, "end": end,
                                "raw_len": end - start,
                            })
                except (KeyError, ValueError):
                    continue

            del df, data_np; gc.collect()
            if (i + 1) % 20 == 0 or (i + 1) == len(records):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remain = (len(records) - i - 1) / rate if rate > 0 else 0
                log(f"    {i+1}/{len(records)}"
                    f"  +{file_steps}스텝  누적: {total_steps + len(buf_X)}"
                    f"  남은: {remain:.0f}s")

        flush()

        hf.create_dataset("channels", data=np.array(channels, dtype="S"))
        hf.attrs["segmentation"] = "heel_strike"
        hf.attrs["sample_rate"]  = config.SAMPLE_RATE
        hf.attrs["target_ts"]    = config.TS
        hf.attrs["n_classes"]    = config.NUM_CLASSES
        hf.attrs["version"]      = "v8"

    # atomic rename
    if config.H5_PATH.exists():
        config.H5_PATH.unlink()
    tmp_h5.rename(config.H5_PATH)

    # 결과 출력
    size_mb = config.H5_PATH.stat().st_size / 1024**2
    log(f"\n  === 결과 ===")
    log(f"  HDF5: {config.H5_PATH}")
    log(f"  {total_steps} 스텝  {size_mb:.1f} MB")
    log(f"  스킵된 파일: {skipped_files}  NaN 보간 파일: {nan_interp_count}")

    if total_steps > 0:
        with h5py.File(config.H5_PATH, "r") as f:
            log(f"  X shape: {f['X'].shape}")
            y_arr = f["y"][:]
            sid_arr = f["subject_id"][:]
            log(f"  라벨 분포: {dict(zip(*np.unique(y_arr, return_counts=True)))}")
            log(f"  피험자: {sorted(np.unique(sid_arr).tolist())}"
                f" ({len(np.unique(sid_arr))}명)")
        log(f"  스텝 길이: min={min(raw_lens)} max={max(raw_lens)}"
            f"  mean={np.mean(raw_lens):.0f}"
            f" ({np.mean(raw_lens)/config.SAMPLE_RATE*1000:.0f}ms)")
    else:
        log("  [ERROR] 검출된 스텝이 없습니다!")

    log_path = config.BATCH_DIR / "step_log.json"
    log_path.write_text(json.dumps(step_log[:2000], indent=1, ensure_ascii=False))
    log(f"  총 소요: {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()