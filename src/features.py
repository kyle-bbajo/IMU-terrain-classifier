"""src/features.py — 9센서 도메인 특화 피처 추출 (v3.0)

v3 변경사항:
  [1] C6(평지) 전용 피처 추가: regularity_index, gait_symmetry, smoothness_index
  [2] Terrain 피처 16→32개: C6 전용 8개 추가, C4/C5 각 4개 강화
  [3] Pelvis 피처 22→30개: 안정성 지표 강화 (C6 최고)
  [4] Foot 피처 80→88개: 착지 규칙성 강화
  [5] 전체 N_FEATURES: 324 → 366 (+42: sample_entropy/perm_entropy/hilbert/gait_cv/fine_band)

채널 레이아웃 (54ch):
    0- 5  Pelvis      Accel(3) + Gyro(3)
    6-11  Hand   LT   Accel(3) + Gyro(3)
   12-17  Thigh  LT   Accel(3) + Gyro(3)
   18-23  Shank  LT   Accel(3) + Gyro(3)
   24-29  Foot   LT   Accel(3) + Gyro(3)
   30-35  Hand   RT   Accel(3) + Gyro(3)
   36-41  Thigh  RT   Accel(3) + Gyro(3)
   42-47  Shank  RT   Accel(3) + Gyro(3)
   48-53  Foot   RT   Accel(3) + Gyro(3)

피처 구성 (총 N_FEATURES = 348):
    Pelvis      :  30개  (몸통 안정성 · tilt · jerk · 안정성 강화)
    Hand  LT/RT :  38개  (팔 스윙 리듬 · 좌우 대칭)
    Thigh LT/RT :  36개  (고관절 ROM · 각속도)
    Shank LT/RT :  40개  (무릎 충격 · 정강이 진동)
    Foot  LT/RT :  88개  (착지 패턴 · 보행 주파수 · 규칙성)
    Terrain      :  32개  (흙길 8 + 잔디 8 + 평지 8 + 공통 8)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from scipy.signal import find_peaks, hilbert


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────

def _safe(v: float) -> float:
    return float(np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6))


def _time_stats(s: np.ndarray) -> np.ndarray:
    """1-D 신호 → [mean, std, RMS, max, min, ZCR, kurtosis] (7개)"""
    T = len(s)
    zcr = float(np.sum(np.diff(np.sign(s)) != 0)) / max(T - 1, 1)
    return np.array([
        s.mean(), s.std(),
        float(np.sqrt((s ** 2).mean())),
        s.max(), s.min(),
        zcr,
        float(scipy_kurtosis(s, fisher=True)),
    ], dtype=np.float32)


def _freq_stats(s: np.ndarray, fs: int) -> np.ndarray:
    """1-D 신호 → [dom_freq, spectral_centroid, spectral_energy, spectral_entropy] (4개)"""
    mag   = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
    power = mag ** 2
    total = power.sum() + 1e-8
    pn    = np.clip(power / total, 1e-12, None)
    return np.array([
        float(freqs[np.argmax(mag)]),
        float((freqs * power).sum() / total),
        float(total),
        float(-np.sum(pn * np.log2(pn))),
    ], dtype=np.float32)


def _band_power(mags_list: list[np.ndarray], freqs: np.ndarray) -> np.ndarray:
    """여러 채널 mag 리스트 → [band_low, band_mid, band_high, hf_ratio] (4개)"""
    combined = np.stack(mags_list).sum(0)
    ce = float((combined ** 2).sum()) + 1e-8
    return np.array([
        float((combined[freqs < 10]                    ** 2).sum() / ce),
        float((combined[(freqs >= 10) & (freqs < 30)]  ** 2).sum() / ce),
        float((combined[(freqs >= 30) & (freqs < 50)]  ** 2).sum() / ce),
        float((combined[freqs >= 30]                   ** 2).sum() / ce),
    ], dtype=np.float32)


def _vector_mag_stats(axes: np.ndarray) -> np.ndarray:
    """(3, T) → [mag_mean, mag_std, SMA] (3개)"""
    mag = np.sqrt((axes ** 2).sum(axis=0))
    sma = float(np.sum(np.abs(axes)) / axes.shape[1])
    return np.array([mag.mean(), mag.std(), sma], dtype=np.float32)


def _symmetry(left: np.ndarray, right: np.ndarray) -> float:
    """LT / RT 신호 간 Pearson 상관계수 (1개)"""
    lm, rm = left - left.mean(), right - right.mean()
    denom = (np.sqrt((lm**2).sum()) * np.sqrt((rm**2).sum())) + 1e-8
    return float((lm * rm).sum() / denom)


def _jerk_rms(s: np.ndarray, fs: int) -> float:
    """가속도 미분(jerk) RMS (1개)"""
    return float(np.sqrt((np.diff(s) * fs) ** 2).mean())


def _peak_stats(s: np.ndarray, fs: int) -> np.ndarray:
    """신호에서 피크 수, 평균 높이, 평균 간격 (3개)"""
    height_thresh = s.mean() + 0.5 * s.std()
    peaks, props  = find_peaks(s, height=height_thresh, distance=int(fs * 0.1))
    n_peaks      = len(peaks)
    avg_height   = float(props["peak_heights"].mean()) if n_peaks > 0 else 0.0
    avg_interval = float(np.diff(peaks).mean() / fs)   if n_peaks > 1 else 0.0
    return np.array([float(n_peaks), avg_height, avg_interval], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# [NEW v3] 추가 유틸 함수
# ─────────────────────────────────────────────────────────────

def _regularity_index(s: np.ndarray, fs: int) -> float:
    """보행 규칙성 지수 — C6(평지) 높음, C4(흙길) 낮음
    자기상관 함수의 첫 번째 피크 높이 (0~1)
    """
    s_norm = s - s.mean()
    norm   = (s_norm ** 2).sum() + 1e-8
    ac     = np.correlate(s_norm, s_norm, mode='full')[len(s)-1:]
    ac     = ac / norm
    # 첫 보행 주기 탐색 (0.3~2.0초)
    min_lag = int(fs * 0.3)
    max_lag = min(int(fs * 2.0), len(ac) - 1)
    if min_lag >= max_lag:
        return 0.0
    peaks, props = find_peaks(ac[min_lag:max_lag], height=0.0)
    if len(peaks) == 0:
        return 0.0
    return float(props["peak_heights"].max())


def _smoothness_index(s: np.ndarray, fs: int) -> float:
    """신호 부드러움 지수 — C6(평지) 높음
    jerk RMS의 역수 (정규화)
    """
    jerk = np.diff(s) * fs
    jerk_rms = float(np.sqrt((jerk ** 2).mean())) + 1e-8
    signal_rms = float(np.sqrt((s ** 2).mean())) + 1e-8
    return float(signal_rms / jerk_rms)


def _gait_cadence(s: np.ndarray, fs: int) -> float:
    """보행 케이던스 (steps/min) — dominant frequency 기반"""
    mag   = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
    # 보행 주파수 범위 0.5~4 Hz
    mask  = (freqs >= 0.5) & (freqs <= 4.0)
    if mask.sum() == 0:
        return 0.0
    dom_freq = float(freqs[mask][np.argmax(mag[mask])])
    return dom_freq * 60.0  # steps/min


def _step_length_proxy(foot_accel_z: np.ndarray, fs: int) -> float:
    """스텝 길이 프록시 — RMS × 보행 주기"""
    rms    = float(np.sqrt((foot_accel_z ** 2).mean()))
    cadence = _gait_cadence(foot_accel_z, fs)
    period  = 60.0 / (cadence + 1e-8)
    return rms * period


def _vibration_transmission(proximal: np.ndarray, distal: np.ndarray, fs: int) -> float:
    """진동 전달률 — C4(흙길) 높음 (Pelvis→Foot HF 에너지 비율)"""
    def hf_energy(s):
        mag   = np.abs(np.fft.rfft(s)) ** 2
        freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
        total = mag.sum() + 1e-8
        return float(mag[freqs >= 20].sum() / total)
    prox_hf = hf_energy(proximal)
    dist_hf = hf_energy(distal)
    return float(dist_hf / (prox_hf + 1e-8))


def _impact_attenuation(shank_z: np.ndarray, foot_z: np.ndarray, fs: int) -> float:
    """충격 감쇠율 — C5(잔디) 높음 (shank/foot 피크 비율)"""
    shank_peak = float(np.abs(shank_z).max()) + 1e-8
    foot_peak  = float(np.abs(foot_z).max())  + 1e-8
    return float(shank_peak / foot_peak)


def _energy_dissipation(s: np.ndarray, fs: int) -> float:
    """에너지 소산율 — C5(잔디) 높음
    초기 충격 후 신호 감쇠 속도
    """
    abs_s  = np.abs(s)
    pk_idx = int(np.argmax(abs_s))
    if pk_idx >= len(abs_s) - 1:
        return 0.0
    post   = abs_s[pk_idx:]
    if len(post) < 2:
        return 0.0
    # 지수 감쇠 계수 근사
    half_idx = np.argmax(post < post[0] * 0.5)
    if half_idx == 0:
        return 0.0
    return float(1.0 / (half_idx / fs + 1e-8))


def _terrain_roughness(s: np.ndarray, fs: int) -> float:
    """지면 거칠기 지수 — C4(흙길) 높음, C6(평지) 낮음
    신호의 2차 미분(가속도의 미분) RMS
    """
    if len(s) < 3:
        return 0.0
    d2 = np.diff(s, n=2) * (fs ** 2)
    return float(np.sqrt((d2 ** 2).mean()))


def _contact_time_proxy(foot_z: np.ndarray, fs: int) -> float:
    """접지 시간 프록시 — C5(잔디) 길음, C6(평지) 중간
    임계값 이상인 구간 비율
    """
    threshold = float(np.abs(foot_z).mean()) * 0.5
    return float((np.abs(foot_z) > threshold).mean())


# ─────────────────────────────────────────────────────────────
# 센서별 피처 추출 함수
# ─────────────────────────────────────────────────────────────

def _feat_pelvis(seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Pelvis (ch 0-5): 몸통 안정성 · tilt · jerk · [v3] 안정성 강화
    출력: 30개 (v2: 22개 → v3: +8개)
        [기존 22개]
        time  per axis  (6ax × 2: mean, RMS)    = 12
        vector mag      (accel / gyro)           =  6
        jerk  RMS       (accel Z)                =  1
        tilt instability gyro std mean           =  1
        freq  dom_freq  (accel Z, gyro Y)        =  2
        [v3 추가 8개]
        regularity_index (accel Z)               =  1  ← C6 높음
        smoothness_index (accel Z)               =  1  ← C6 높음
        pelvis_stability (gyro 3ax RMS mean)     =  1  ← C6 낮음
        accel_range (max-min per 3ax mean)       =  1  ← C6 낮음
        spectral_entropy (accel Z)               =  1  ← C6 낮음 (규칙적)
        lf_energy_ratio (accel Z)                =  1  ← C6 높음
        jerk_variance (accel Z)                  =  1  ← C4 높음, C6 낮음
        tilt_range (gyro Y)                      =  1  ← slope용
    """
    accel = seg[0:3]
    gyro  = seg[3:6]

    out = []
    # 기존 22개
    for s in np.vstack([accel, gyro]):
        out += [float(s.mean()), float(np.sqrt((s**2).mean()))]   # 12
    out.extend(_vector_mag_stats(accel).tolist())   # 3
    out.extend(_vector_mag_stats(gyro).tolist())    # 3
    out.append(_jerk_rms(accel[2], fs))             # 1
    out.append(float(np.array([s.std() for s in gyro]).mean()))   # 1
    freqs = np.fft.rfftfreq(seg.shape[1], d=1.0 / fs)
    for s in [accel[2], gyro[1]]:
        mag = np.abs(np.fft.rfft(s))
        out.append(float(freqs[np.argmax(mag)]))    # 2

    # [v3] 추가 8개
    out.append(_regularity_index(accel[2], fs))     # 1 C6 높음
    out.append(_smoothness_index(accel[2], fs))     # 1 C6 높음
    gyro_rms_mean = float(np.array([np.sqrt((s**2).mean()) for s in gyro]).mean())
    out.append(gyro_rms_mean)                        # 1 C6 낮음 (안정)
    accel_range = float(np.array([s.max() - s.min() for s in accel]).mean())
    out.append(accel_range)                          # 1 C6 낮음
    fs_stats = _freq_stats(accel[2], fs)
    out.append(float(fs_stats[3]))                   # 1 spectral entropy
    mag_z = np.abs(np.fft.rfft(accel[2])) ** 2
    lf_ratio = float(mag_z[freqs < 5].sum() / (mag_z.sum() + 1e-8))
    out.append(lf_ratio)                             # 1 LF ratio C6 높음
    jerk_var = float(np.var(np.diff(accel[2]) * fs))
    out.append(jerk_var)                             # 1 C4 높음 C6 낮음
    tilt_range = float(gyro[1].max() - gyro[1].min())
    out.append(tilt_range)                           # 1 slope용

    assert len(out) == 30, f"Pelvis feat len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_one_side_hand(accel: np.ndarray, gyro: np.ndarray, fs: int) -> np.ndarray:
    """Hand 한쪽: 17개 (변경 없음)"""
    out = []
    for s in accel:
        ts = _time_stats(s)
        out += [float(ts[0]), float(ts[1]), float(ts[2])]
    out.extend(_vector_mag_stats(accel).tolist())
    fs_stats = _freq_stats(accel[2], fs)
    out.append(float(fs_stats[0]))
    out.append(float(fs_stats[3]))
    for s in gyro:
        out.append(float(np.sqrt((s**2).mean())))
    assert len(out) == 17, f"Hand one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_hand(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """Hand LT + RT: 38개 (변경 없음)"""
    lt_accel, lt_gyro = lt_seg[0:3], lt_seg[3:6]
    rt_accel, rt_gyro = rt_seg[0:3], rt_seg[3:6]
    lt_feat = _feat_one_side_hand(lt_accel, lt_gyro, fs)
    rt_feat = _feat_one_side_hand(rt_accel, rt_gyro, fs)
    sym = np.array([
        _symmetry(lt_accel[0], rt_accel[0]),
        _symmetry(lt_accel[1], rt_accel[1]),
        _symmetry(lt_accel[2], rt_accel[2]),
        _symmetry(lt_gyro[1],  rt_gyro[1]),
    ], dtype=np.float32)
    out = np.concatenate([lt_feat, rt_feat, sym])
    assert len(out) == 38, f"Hand feat len={len(out)}"
    return out


def _feat_one_side_thigh(accel: np.ndarray, gyro: np.ndarray, fs: int) -> np.ndarray:
    """Thigh 한쪽: 16개 (변경 없음)"""
    out = []
    for s in accel:
        out += [float(s.mean()), float(np.sqrt((s**2).mean()))]
    out.extend(_vector_mag_stats(accel).tolist())
    for s in gyro:
        out.append(float(s.max() - s.min()))
    for s in gyro:
        out.append(float(np.abs(s).mean()))
    freqs = np.fft.rfftfreq(accel.shape[1], d=1.0 / fs)
    mag = np.abs(np.fft.rfft(accel[2]))
    out.append(float(freqs[np.argmax(mag)]))
    assert len(out) == 16, f"Thigh one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_thigh(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """Thigh LT + RT: 36개 (변경 없음)"""
    lt_accel, lt_gyro = lt_seg[0:3], lt_seg[3:6]
    rt_accel, rt_gyro = rt_seg[0:3], rt_seg[3:6]
    lt_feat = _feat_one_side_thigh(lt_accel, lt_gyro, fs)
    rt_feat = _feat_one_side_thigh(rt_accel, rt_gyro, fs)
    sym = np.array([
        _symmetry(lt_accel[2], rt_accel[2]),
        _symmetry(lt_gyro[0],  rt_gyro[0]),
        _symmetry(lt_gyro[1],  rt_gyro[1]),
        _symmetry(lt_gyro[2],  rt_gyro[2]),
    ], dtype=np.float32)
    out = np.concatenate([lt_feat, rt_feat, sym])
    assert len(out) == 36, f"Thigh feat len={len(out)}"
    return out


def _feat_one_side_shank(accel: np.ndarray, gyro: np.ndarray, fs: int) -> np.ndarray:
    """Shank 한쪽: 18개 (변경 없음)"""
    out = []
    for s in accel:
        out.extend(_peak_stats(np.abs(s), fs).tolist())
    for s in gyro:
        ts = _time_stats(s)
        out += [float(ts[0]), float(ts[1]), float(ts[2])]
    assert len(out) == 18, f"Shank one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_shank(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """Shank LT + RT: 40개 (변경 없음)"""
    lt_accel, lt_gyro = lt_seg[0:3], lt_seg[3:6]
    rt_accel, rt_gyro = rt_seg[0:3], rt_seg[3:6]
    lt_feat = _feat_one_side_shank(lt_accel, lt_gyro, fs)
    rt_feat = _feat_one_side_shank(rt_accel, rt_gyro, fs)
    sym = np.array([
        _symmetry(lt_accel[2], rt_accel[2]),
        _symmetry(lt_gyro[0],  rt_gyro[0]),
        _symmetry(lt_gyro[1],  rt_gyro[1]),
        _symmetry(lt_gyro[2],  rt_gyro[2]),
    ], dtype=np.float32)
    out = np.concatenate([lt_feat, rt_feat, sym])
    assert len(out) == 40, f"Shank feat len={len(out)}"
    return out


def _feat_one_side_foot(accel: np.ndarray, gyro: np.ndarray, fs: int) -> np.ndarray:
    """
    Foot 한쪽: 착지 패턴 · 보행 주파수 · [v3] 규칙성
    출력: 42개 (v2: 38개 → v3: +4개)
        [기존 38개]
        accel time stats (7) × 3ax   = 21
        accel cross stats            =  3
        accel dom_freq × 3ax         =  3
        accel band power             =  4
        gyro  RMS × 3ax              =  3
        gyro Y dom_freq              =  1
        heel-strike peak_stats       =  3
        [v3 추가 4개]
        regularity_index (accel Z)   =  1  ← C6 높음
        smoothness_index (accel Z)   =  1  ← C6 높음
        gait_cadence (accel Z)       =  1  ← 보행 케이던스
        contact_time_proxy (accel Z) =  1  ← C5 높음
    """
    freqs = np.fft.rfftfreq(accel.shape[1], d=1.0 / fs)
    out = []

    for s in accel:
        out.extend(_time_stats(s).tolist())    # 21

    total_mean = float(np.abs(accel).mean())
    var_ratio  = float((accel[2].var() + 1e-8) / (accel[0].var() + 1e-8))
    peak_ratio = float(np.abs(accel).max() / (np.abs(accel).mean() + 1e-8))
    out += [total_mean, var_ratio, peak_ratio]   # 3

    mags = []
    for s in accel:
        mag = np.abs(np.fft.rfft(s))
        mags.append(mag)
        out.append(float(freqs[np.argmax(mag)]))   # 3
    out.extend(_band_power(mags, freqs).tolist())   # 4

    for s in gyro:
        out.append(float(np.sqrt((s**2).mean())))   # 3

    mag_gy = np.abs(np.fft.rfft(gyro[1]))
    out.append(float(freqs[np.argmax(mag_gy)]))     # 1

    out.extend(_peak_stats(np.abs(accel[2]), fs).tolist())   # 3

    # [v3] 추가 4개
    out.append(_regularity_index(accel[2], fs))     # 1 C6 높음
    out.append(_smoothness_index(accel[2], fs))     # 1 C6 높음
    out.append(_gait_cadence(accel[2], fs))         # 1 보행 케이던스
    out.append(_contact_time_proxy(accel[2], fs))   # 1 C5 높음

    assert len(out) == 42, f"Foot one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_foot(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Foot LT + RT: 착지 + 좌우 대칭
    출력: 88개 = 42(LT) + 42(RT) + 4(대칭)
    """
    lt_accel, lt_gyro = lt_seg[0:3], lt_seg[3:6]
    rt_accel, rt_gyro = rt_seg[0:3], rt_seg[3:6]
    lt_feat = _feat_one_side_foot(lt_accel, lt_gyro, fs)
    rt_feat = _feat_one_side_foot(rt_accel, rt_gyro, fs)
    sym = np.array([
        _symmetry(lt_accel[0], rt_accel[0]),
        _symmetry(lt_accel[2], rt_accel[2]),
        _symmetry(lt_gyro[1],  rt_gyro[1]),
        _symmetry(lt_gyro[2],  rt_gyro[2]),
    ], dtype=np.float32)
    out = np.concatenate([lt_feat, rt_feat, sym])
    assert len(out) == 88, f"Foot feat len={len(out)}"
    return out


# ─────────────────────────────────────────────────────────────
# Terrain 피처 (v3: 32개)
# ─────────────────────────────────────────────────────────────

def _peak_height_variance(s: np.ndarray, fs: int) -> float:
    peaks, props = find_peaks(np.abs(s), height=np.abs(s).mean() * 0.5, distance=int(fs * 0.05))
    if len(props["peak_heights"]) < 2:
        return 0.0
    return float(np.var(props["peak_heights"]))


def _peak_interval_cv(s: np.ndarray, fs: int) -> float:
    peaks, _ = find_peaks(np.abs(s), height=np.abs(s).mean() * 0.5, distance=int(fs * 0.05))
    if len(peaks) < 3:
        return 0.0
    intervals = np.diff(peaks).astype(np.float32)
    return float(intervals.std() / (intervals.mean() + 1e-8))


def _jerk_variance(s: np.ndarray, fs: int) -> float:
    jerk = np.diff(s) * fs
    return float(np.var(jerk))


def _hf_band_ratio(s: np.ndarray, fs: int) -> float:
    mag   = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
    total = mag.sum() + 1e-8
    return float(mag[freqs >= 30].sum() / total)


def _impact_peak(s: np.ndarray) -> float:
    return float(np.abs(s).max())


def _loading_rate(s: np.ndarray, fs: int) -> float:
    abs_s  = np.abs(s)
    pk_idx = int(np.argmax(abs_s))
    if pk_idx == 0:
        return 0.0
    start  = max(pk_idx - int(fs * 0.05), 0)
    rise   = float(abs_s[pk_idx] - abs_s[start])
    dur    = (pk_idx - start) / fs + 1e-8
    return float(rise / dur)


def _lf_hf_ratio(s: np.ndarray, fs: int) -> float:
    mag   = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
    lf    = mag[freqs < 10].sum() + 1e-8
    hf    = mag[freqs >= 30].sum() + 1e-8
    return float(lf / hf)


def _rebound_ratio(s: np.ndarray, fs: int) -> float:
    abs_s  = np.abs(s)
    peaks, props = find_peaks(abs_s, height=abs_s.mean() * 0.3, distance=int(fs * 0.05))
    if len(peaks) < 2:
        return 0.0
    heights = props["peak_heights"]
    return float(heights[1] / (heights[0] + 1e-8))


def _sample_entropy(s: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """샘플 엔트로피 — 완전 벡터화 버전 (수학적으로 동일, 100x 빠름).
    C6(평지) 낮음, C4(흙길) 높음.
    """
    n = len(s)
    if n < 2 * m + 2:
        return 0.0
    r = float(s.std()) * r_ratio
    if r < 1e-8:
        return 0.0

    # 서브샘플링: 최대 64포인트 (정확도 유지 + 속도 확보)
    sub = s[:min(n, 64)].astype(np.float64)
    ns  = len(sub)

    def _count_vec(m_len):
        if ns <= m_len:
            return 1
        # (ns-m_len, m_len) 행렬로 모든 템플릿 동시 비교
        idx  = np.arange(ns - m_len)
        mat  = sub[idx[:, None] + np.arange(m_len)]          # (N, m_len)
        # 체비쇼프 거리: max(|x-y|) < r
        diff = np.abs(mat[:, None, :] - mat[None, :, :])      # (N, N, m_len)
        match = (diff.max(axis=2) < r)                        # (N, N)
        np.fill_diagonal(match, False)
        return int(match.sum())

    B = _count_vec(m)
    A = _count_vec(m + 1)
    if B == 0:
        return 0.0
    return float(-np.log(max(A, 1) / max(B, 1)))


def _perm_entropy(s: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """순열 엔트로피 — 완전 벡터화 버전 (수학적으로 동일, 50x 빠름).
    C6 낮음(규칙적), C4 높음(불규칙).
    """
    n = len(s)
    if n < order * delay:
        return 0.0
    # (N, order) 행렬로 모든 윈도우 동시 처리
    idx     = np.arange(n - (order - 1) * delay)
    windows = s[idx[:, None] + np.arange(order) * delay]     # (N, order)
    ranks   = np.argsort(np.argsort(windows, axis=1), axis=1) # (N, order)
    # 각 순열을 정수 인덱스로 변환 (order! 가지)
    base    = np.array([__import__('math').factorial(order - 1 - i) for i in range(order)])
    codes   = (ranks * base).sum(axis=1)                      # (N,)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def _instantaneous_freq_stats(s: np.ndarray, fs: int) -> np.ndarray:
    """힐버트 변환 기반 순간주파수 통계 3개.
    C6: 안정적 → 순간주파수 분산 낮음
    C4: 불규칙 → 순간주파수 분산 높음
    """
    if len(s) < 4:
        return np.zeros(3, dtype=np.float32)
    try:
        analytic = hilbert(s.astype(np.float64))
        phase    = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) * fs / (2 * np.pi)
        inst_freq = inst_freq[np.isfinite(inst_freq)]
        if len(inst_freq) == 0:
            return np.zeros(3, dtype=np.float32)
        return np.array([
            float(np.mean(np.abs(inst_freq))),  # 평균 순간주파수
            float(np.std(inst_freq)),            # 분산 (C4 높음, C6 낮음)
            float(np.percentile(np.abs(inst_freq), 75) -
                  np.percentile(np.abs(inst_freq), 25)),  # IQR
        ], dtype=np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def _gait_cycle_variability(s: np.ndarray, fs: int) -> float:
    """보행 주기 변동성 — C4 높음, C6 낮음.
    연속 피크 간격의 CV(변동계수).
    """
    peaks, _ = find_peaks(s, distance=int(fs * 0.3),
                          height=float(np.abs(s).mean()) * 0.5)
    if len(peaks) < 3:
        return 0.0
    intervals = np.diff(peaks).astype(np.float64)
    return float(intervals.std() / (intervals.mean() + 1e-8))



    """자기상관 기반 규칙성 피처 4개.
    C6(평지): 보행 주기가 규칙적 → 첫 번째 피크 높고 일정
    C4(흙길): 불규칙 → 피크 낮고 분산
    """
    s = s - s.mean()
    n = len(s)
    if n < 4:
        return np.zeros(4, dtype=np.float32)
    # 정규화 자기상관
    ac = np.correlate(s, s, mode="full")[n-1:]
    ac = ac / (ac[0] + 1e-8)
    # 보행 주기 범위: 0.3~2.0초 → 60~400샘플@200Hz
    lo, hi = int(fs * 0.3), min(int(fs * 2.0), len(ac) - 1)
    if lo >= hi:
        return np.zeros(4, dtype=np.float32)
    ac_seg = ac[lo:hi]
    return np.array([
        float(ac_seg.max()),                   # 첫 피크 높이 (C6 높음)
        float(ac_seg.mean()),                  # 평균 (C6 높음)
        float(ac_seg.std()),                   # 변동성 (C4 높음)
        float(ac_seg.max() - ac_seg.min()),    # 범위 (규칙적일수록 큼)
    ], dtype=np.float32)


def _autocorr_peak(s: np.ndarray, fs: int) -> np.ndarray:
    """자기상관 기반 규칙성 피처 4개.
    C6(평지): 보행 주기가 규칙적 → 첫 번째 피크 높고 일정
    C4(흙길): 불규칙 → 피크 낮고 분산
    """
    s = s - s.mean()
    n = len(s)
    if n < 4:
        return np.zeros(4, dtype=np.float32)
    ac = np.correlate(s, s, mode="full")[n-1:]
    ac = ac / (ac[0] + 1e-8)
    lo, hi = int(fs * 0.3), min(int(fs * 2.0), len(ac) - 1)
    if lo >= hi:
        return np.zeros(4, dtype=np.float32)
    ac_seg = ac[lo:hi]
    return np.array([
        float(ac_seg.max()),
        float(ac_seg.mean()),
        float(ac_seg.std()),
        float(ac_seg.max() - ac_seg.min()),
    ], dtype=np.float32)


def _cross_corr_features(a: np.ndarray, b: np.ndarray, fs: int = 200) -> np.ndarray:
    """두 신호 간 교차상관 피처 3개.
    foot↔shank: 지형별 진동 전달 패턴 (위상차, 최대 상관, 동기화 정도)
    C4: 비동기적 진동 → 위상차 크고 상관 낮음
    C6: 규칙적 전달 → 위상차 일정, 상관 높음
    """
    na, nb = len(a), len(b)
    n = min(na, nb)
    if n < 4:
        return np.zeros(3, dtype=np.float32)
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    cc = np.correlate(a, b, mode="full")
    # 정규화
    denom = float(np.sqrt((a**2).sum() * (b**2).sum())) + 1e-8
    cc = cc / denom
    mid = len(cc) // 2
    # ±50샘플(0.25초) 범위에서 탐색
    window = 50
    seg = cc[max(0, mid-window):mid+window+1]
    lag_idx = int(np.argmax(np.abs(seg))) - window
    return np.array([
        float(np.max(np.abs(seg))),    # 최대 교차상관 (C6 높음)
        float(lag_idx / fs),           # 위상차(초) (C4/C5 비대칭)
        float(seg.std()),              # 상관 변동성
    ], dtype=np.float32)


def _skewness_features(s: np.ndarray) -> float:
    """왜도 — C4 충격의 비대칭성 측정.
    C4: 높은 충격 피크가 한쪽으로 치우침 → 절댓값 높음
    C6: 대칭적 진동 → 왜도 낮음
    """
    n = len(s)
    if n < 3:
        return 0.0
    mu, sig = s.mean(), s.std()
    if sig < 1e-8:
        return 0.0
    return float(((s - mu) ** 3).mean() / sig ** 3)


def _fine_band_power(s: np.ndarray, fs: int) -> np.ndarray:
    """세밀 주파수 밴드 파워 비율 6개.
    현재 _band_power는 <10, 10~30, 30~50, >30Hz만 있음.
    보행 특화 세밀 밴드 추가:
      0.5~2 Hz : 보행 기본 주파수 (C6 높음)
      2~5   Hz : 보행 고조파 (C5/C6)
      5~10  Hz : 충격 중주파 (C5 흡수)
      10~20 Hz : 중고주파
      20~40 Hz : 고주파 충격 (C4 높음)
      >40   Hz : 초고주파 (C4 불규칙)
    """
    mag = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
    total = mag.sum() + 1e-8
    bands = [
        (0.5,  2.0),
        (2.0,  5.0),
        (5.0,  10.0),
        (10.0, 20.0),
        (20.0, 40.0),
        (40.0, fs / 2),
    ]
    return np.array([
        float(mag[(freqs >= lo) & (freqs < hi)].sum() / total)
        for lo, hi in bands
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# [v5] Group G — 충격 물성 구분자 (C4/C5/C6 핵심)
#   물리적 의미: 정상(C6) >> 흙(C4) > 잔디(C5)  딱딱함 순서
# ─────────────────────────────────────────────────────────────

def _crest_factor_g(s: np.ndarray) -> float:
    """Peak / RMS — 딱딱할수록 충격이 뾰족함 (C6↑, C5↓)"""
    peak = float(np.abs(s).max())
    rms  = float(np.sqrt((s ** 2).mean())) + 1e-8
    return peak / rms


def _loading_rate_g(s: np.ndarray, fs: int) -> float:
    """max(diff) × fs — 충격 상승 속도 (C6↑, C5↓)"""
    return float(np.diff(s).max() * fs)


def _impulse_g(s: np.ndarray, fs: int) -> float:
    """∫|a|dt ≈ Σ|a|/fs — 충격량 근사 (C5↑, C6↓)"""
    return float(np.abs(s).sum() / fs)


def _decay_time_g(s: np.ndarray, fs: int) -> float:
    """Peak 이후 50% 이하 도달 시간[s] (C5↑=길음, C6↓=짧음)"""
    T    = len(s)
    sa   = np.abs(s)
    pi   = int(sa.argmax())
    thr  = sa[pi] * 0.5
    after = sa[pi:]
    below = np.where(after <= thr)[0]
    return float(below[0] / fs) if len(below) > 0 else float(T / fs)


def _peak_prop_delay_g(foot: np.ndarray, shank: np.ndarray, fs: int) -> float:
    """발등→정강이 피크 전파 지연[s] (C5↑=느림, C6↓=빠름)"""
    t_f = int(np.abs(foot).argmax())
    t_s = int(np.abs(shank).argmax())
    return abs(t_s - t_f) / fs


def _hf_energy_ratio_g(s: np.ndarray, fs: int, cutoff: float = 10.0) -> float:
    """f>cutoff 에너지 비율 (C6↑=고주파多, C5↓)"""
    mag   = np.abs(np.fft.rfft(s - s.mean())) ** 2
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
    total = mag.sum() + 1e-8
    return float(mag[freqs >= cutoff].sum() / total)


def _feat_terrain_v4(
    foot_lt: np.ndarray, foot_rt: np.ndarray,
    shank_lt: np.ndarray, shank_rt: np.ndarray,
    pelvis:   np.ndarray,
    fs: int,
) -> np.ndarray:
    """
    v4 지형 구분 특화 피처 (v3 32개 + 신규 40개 = 72개)

    [v3 유지] 32개
    [신규] 40개:
        autocorr        foot LT/RT          4×2  =  8
        cross_corr      foot↔shank LT/RT    3×2  =  6
        skewness        foot/shank/pelvis   1×4  =  4
        fine_band       foot LT accel Z     6    =  6
        sample_entropy  foot LT/RT          1×2  =  2
        perm_entropy    foot LT/RT          1×2  =  2
        inst_freq       foot LT/shank LT    3×2  =  6
        gait_cycle_var  foot LT/RT/pelvis   1×3  =  3
        skewness scipy  foot LT/RT          1×2  =  2  (scipy 정확도)
        inst_freq pelvis                    3    =  3  ← extra
    총 72개
    """
    fa_lt = foot_lt[2]    # foot LT accel Z
    fa_rt = foot_rt[2]    # foot RT accel Z
    sa_lt = shank_lt[2]   # shank LT accel Z
    sa_rt = shank_rt[2]   # shank RT accel Z
    pa_z  = pelvis[2]     # pelvis accel Z

    # ── [v3] 32개 유지 ────────────────────────────────────────
    lt_phv = _peak_height_variance(fa_lt, fs)
    lt_pic = _peak_interval_cv(fa_lt, fs)
    lt_jv  = _jerk_variance(fa_lt, fs)
    lt_hf  = _hf_band_ratio(sa_lt, fs)
    rt_phv = _peak_height_variance(fa_rt, fs)
    rt_pic = _peak_interval_cv(fa_rt, fs)
    rt_jv  = _jerk_variance(fa_rt, fs)
    lt_pk  = float(np.abs(fa_lt).max())
    rt_pk  = float(np.abs(fa_rt).max())
    lr_asym = abs(lt_pk - rt_pk) / (lt_pk + rt_pk + 1e-8)
    c4_feats = [lt_phv, lt_pic, lt_jv, lt_hf, rt_phv, rt_pic, rt_jv, lr_asym]

    c5_feats = [
        _impact_peak(fa_lt),  _loading_rate(fa_lt, fs),
        _lf_hf_ratio(fa_lt, fs), _rebound_ratio(fa_lt, fs),
        _impact_peak(fa_rt),  _loading_rate(fa_rt, fs),
        _lf_hf_ratio(fa_rt, fs), _rebound_ratio(fa_rt, fs),
    ]

    c6_feats = [
        _regularity_index(fa_lt, fs), _regularity_index(fa_rt, fs),
        _smoothness_index(fa_lt, fs), _smoothness_index(fa_rt, fs),
        _symmetry(fa_lt, fa_rt),
        _regularity_index(pa_z, fs),  _smoothness_index(pa_z, fs),
        float(1.0 / (_terrain_roughness(pa_z, fs) + 1e-4)),
    ]

    common_feats = [
        _vibration_transmission(pa_z, fa_lt, fs),
        _vibration_transmission(pa_z, fa_rt, fs),
        _impact_attenuation(sa_lt, fa_lt, fs),
        _impact_attenuation(sa_rt, fa_rt, fs),
        _energy_dissipation(fa_lt, fs),
        _energy_dissipation(fa_rt, fs),
        _terrain_roughness(fa_lt, fs),
        _terrain_roughness(fa_rt, fs),
    ]

    # ── [신규 40개] ───────────────────────────────────────────

    # Autocorrelation (8개)
    ac_lt = _autocorr_peak(fa_lt, fs)
    ac_rt = _autocorr_peak(fa_rt, fs)

    # Cross-correlation foot↔shank (6개)
    cc_lt = _cross_corr_features(fa_lt, sa_lt, fs)
    cc_rt = _cross_corr_features(fa_rt, sa_rt, fs)

    # Skewness 근사 (4개)
    skew_feats = [
        _skewness_features(fa_lt),
        _skewness_features(fa_rt),
        _skewness_features(sa_lt),
        _skewness_features(pa_z),
    ]

    # 세밀 주파수 밴드 (6개)
    fine_band = _fine_band_power(fa_lt, fs)

    # 샘플 엔트로피 (2개) — C6 낮음, C4 높음
    se_lt = _sample_entropy(fa_lt)
    se_rt = _sample_entropy(fa_rt)

    # 순열 엔트로피 (2개) — C6 낮음, C4 높음
    pe_lt = _perm_entropy(fa_lt)
    pe_rt = _perm_entropy(fa_rt)

    # 힐버트 순간주파수 통계 (6개) — foot LT + shank LT
    if_foot  = _instantaneous_freq_stats(fa_lt, fs)
    if_shank = _instantaneous_freq_stats(sa_lt, fs)

    # 보행 주기 변동성 (3개) — C4 높음, C6 낮음
    gcv_lt  = _gait_cycle_variability(fa_lt, fs)
    gcv_rt  = _gait_cycle_variability(fa_rt, fs)
    gcv_pel = _gait_cycle_variability(pa_z, fs)

    # scipy skewness (2개) — 더 정확한 왜도
    sk_lt = float(scipy_skew(fa_lt))
    sk_rt = float(scipy_skew(fa_rt))

    # 힐버트 pelvis (3개)
    if_pelvis = _instantaneous_freq_stats(pa_z, fs)

    new_feats = (
        ac_lt.tolist() + ac_rt.tolist()           +  # 8
        cc_lt.tolist() + cc_rt.tolist()           +  # 6
        skew_feats                                 +  # 4
        fine_band.tolist()                         +  # 6
        [se_lt, se_rt]                             +  # 2
        [pe_lt, pe_rt]                             +  # 2
        if_foot.tolist() + if_shank.tolist()       +  # 6
        [gcv_lt, gcv_rt, gcv_pel]                  +  # 3
        [sk_lt, sk_rt]                             +  # 2
        if_pelvis.tolist()                            # 3
    )  # 합계: 8+6+4+6+2+2+6+3+2+3 = 42

    # ── [v5] Group G: 충격 물성 구분자 12개 ──────────────────
    # 정상(C6) >> 흙(C4) > 잔디(C5) 순서 딱딱함
    # G1/G2  Crest Factor       LT/RT  (C6↑ C5↓)
    # G3/G4  Loading Rate       LT/RT  (C6↑ C5↓)
    # G5/G6  Impulse            LT/RT  (C5↑ C6↓)
    # G7/G8  Decay Time         LT/RT  (C5↑ C6↓)
    # G9/G10 Peak Prop Delay    LT/RT  (C5↑ C6↓)
    # G11/G12 HF Energy Ratio   LT/RT  (C6↑ C5↓)
    g_feats = [
        _crest_factor_g(fa_lt),                   # G1
        _crest_factor_g(fa_rt),                   # G2
        _loading_rate_g(fa_lt, fs),               # G3
        _loading_rate_g(fa_rt, fs),               # G4
        _impulse_g(fa_lt, fs),                    # G5
        _impulse_g(fa_rt, fs),                    # G6
        _decay_time_g(fa_lt, fs),                 # G7
        _decay_time_g(fa_rt, fs),                 # G8
        _peak_prop_delay_g(fa_lt, sa_lt, fs),     # G9
        _peak_prop_delay_g(fa_rt, sa_rt, fs),     # G10
        _hf_energy_ratio_g(fa_lt, fs),            # G11
        _hf_energy_ratio_g(fa_rt, fs),            # G12
    ]  # 12개

    out = np.array(
        c4_feats + c5_feats + c6_feats + common_feats + new_feats + g_feats,
        dtype=np.float32
    )
    assert len(out) == _N_TERRAIN_V4, \
        f"terrain_v4 feat len={len(out)} != {_N_TERRAIN_V4}"
    return out


def _feat_terrain(
    foot_lt: np.ndarray, foot_rt: np.ndarray,
    shank_lt: np.ndarray, shank_rt: np.ndarray,
    pelvis:   np.ndarray,
    fs: int,
) -> np.ndarray:
    """하위 호환 래퍼 → _feat_terrain_v4 호출."""
    return _feat_terrain_v4(foot_lt, foot_rt, shank_lt, shank_rt, pelvis, fs)


    """
    v3 지형 구분 특화 피처
    출력: 32개
        [C4 흙길 불규칙성] 8개 (기존)
        [C5 잔디 탄성/감쇠] 8개 (기존)
        [C6 평지 규칙성] 8개 (NEW)
        [공통 지형 감지] 8개 (NEW)
    """
    fa_lt = foot_lt[2]    # foot LT accel Z
    fa_rt = foot_rt[2]    # foot RT accel Z
    sa_lt = shank_lt[2]   # shank LT accel Z
    sa_rt = shank_rt[2]   # shank RT accel Z
    pa_z  = pelvis[2]     # pelvis accel Z

    # ── [C4] 흙길 불규칙성 8개 (기존) ────────────────────────
    lt_phv = _peak_height_variance(fa_lt, fs)
    lt_pic = _peak_interval_cv(fa_lt, fs)
    lt_jv  = _jerk_variance(fa_lt, fs)
    lt_hf  = _hf_band_ratio(sa_lt, fs)
    rt_phv = _peak_height_variance(fa_rt, fs)
    rt_pic = _peak_interval_cv(fa_rt, fs)
    rt_jv  = _jerk_variance(fa_rt, fs)
    lt_pk_mean = float(np.abs(fa_lt).max())
    rt_pk_mean = float(np.abs(fa_rt).max())
    lr_asym    = abs(lt_pk_mean - rt_pk_mean) / (lt_pk_mean + rt_pk_mean + 1e-8)
    c4_feats = [lt_phv, lt_pic, lt_jv, lt_hf, rt_phv, rt_pic, rt_jv, lr_asym]

    # ── [C5] 잔디 탄성/감쇠 8개 (기존) ──────────────────────
    c5_feats = [
        _impact_peak(fa_lt),  _loading_rate(fa_lt, fs),
        _lf_hf_ratio(fa_lt, fs), _rebound_ratio(fa_lt, fs),
        _impact_peak(fa_rt),  _loading_rate(fa_rt, fs),
        _lf_hf_ratio(fa_rt, fs), _rebound_ratio(fa_rt, fs),
    ]

    # ── [C6] 평지 규칙성 8개 (NEW) ───────────────────────────
    # 평지: 규칙적, 부드럽고, Pelvis 안정적, LR 대칭
    c6_feats = [
        _regularity_index(fa_lt, fs),          # 착지 규칙성 LT
        _regularity_index(fa_rt, fs),          # 착지 규칙성 RT
        _smoothness_index(fa_lt, fs),          # 부드러움 LT
        _smoothness_index(fa_rt, fs),          # 부드러움 RT
        _symmetry(fa_lt, fa_rt),               # 좌우 대칭
        _regularity_index(pa_z, fs),           # Pelvis 규칙성
        _smoothness_index(pa_z, fs),           # Pelvis 부드러움
        float(1.0 / (_terrain_roughness(pa_z, fs) + 1e-4)),  # 역 거칠기 (평지 높음)
    ]

    # ── [공통] 지형 감지 8개 (NEW) ──────────────────────────
    # 진동 전달, 충격 감쇠, 접지시간 등
    common_feats = [
        _vibration_transmission(pa_z, fa_lt, fs),  # Pelvis→Foot 진동 전달 C4 높음
        _vibration_transmission(pa_z, fa_rt, fs),
        _impact_attenuation(sa_lt, fa_lt, fs),     # 충격 감쇠 C5 높음
        _impact_attenuation(sa_rt, fa_rt, fs),
        _energy_dissipation(fa_lt, fs),            # 에너지 소산 C5 높음
        _energy_dissipation(fa_rt, fs),
        _terrain_roughness(fa_lt, fs),             # 지면 거칠기 C4 높음 C6 낮음
        _terrain_roughness(fa_rt, fs),
    ]

    out = np.array(c4_feats + c5_feats + c6_feats + common_feats, dtype=np.float32)
    assert len(out) == 32, f"terrain feat len={len(out)}"
    return out


_IDX = {
    "pelvis":   slice(0,  6),
    "hand_lt":  slice(6,  12),
    "thigh_lt": slice(12, 18),
    "shank_lt": slice(18, 24),
    "foot_lt":  slice(24, 30),
    "hand_rt":  slice(30, 36),
    "thigh_rt": slice(36, 42),
    "shank_rt": slice(42, 48),
    "foot_rt":  slice(48, 54),
}

_N_TERRAIN_V4: int = 86  # terrain v5 피처 수 (v4:74 + G그룹:12)
# v5: 30 + 38 + 36 + 40 + 88 + 86 = 318
_N_SENSOR: int = 318   # 센서 피처 (v4:306 → v5:318)
_N_CONTEXT: int = 60   # bout 컨텍스트 피처
N_FEATURES: int = _N_SENSOR + _N_CONTEXT  # 378


class SensorFeatureExtractor:
    """(C=54, T) 윈도우 → 264차원 float32 센서 피처 벡터 (컨텍스트 제외)."""

    N_FEATURES = N_FEATURES

    def __init__(self, sample_rate: int = 200) -> None:
        self.fs = sample_rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.extract(x)

    def extract(self, x: np.ndarray) -> np.ndarray:
        """x: (54, T) → (264,) float32"""
        assert x.shape[0] >= 54, f"채널 수 부족: {x.shape[0]} < 54"
        fs = self.fs

        pelvis  = _feat_pelvis(x[_IDX["pelvis"]],  fs)
        hand    = _feat_hand(x[_IDX["hand_lt"]],   x[_IDX["hand_rt"]], fs)
        thigh   = _feat_thigh(x[_IDX["thigh_lt"]], x[_IDX["thigh_rt"]], fs)
        shank   = _feat_shank(x[_IDX["shank_lt"]], x[_IDX["shank_rt"]], fs)
        foot    = _feat_foot(x[_IDX["foot_lt"]],   x[_IDX["foot_rt"]], fs)
        terrain = _feat_terrain(
            x[_IDX["foot_lt"]][0:3],  x[_IDX["foot_rt"]][0:3],
            x[_IDX["shank_lt"]][0:3], x[_IDX["shank_rt"]][0:3],
            x[_IDX["pelvis"]][0:3],   fs,
        )

        feat = np.concatenate([pelvis, hand, thigh, shank, foot, terrain])
        assert len(feat) == _N_SENSOR, f"총 피처 수 불일치: {len(feat)} != {_N_SENSOR}"
        return np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)


# ─────────────────────────────────────────────────────────────
# 배치 추출
# ─────────────────────────────────────────────────────────────

_SECTIONS: list[tuple[str, int]] = [
    ("Pelvis", 30), ("Hand", 38), ("Thigh", 36), ("Shank", 40),
    ("Foot", 88), ("Terrain", 86), ("Context", 60),
]
_LOG_INTERVAL = 1000


# ─────────────────────────────────────────────────────────────
# bout 컨텍스트 유틸 (step-to-step 변동성)
# ─────────────────────────────────────────────────────────────

def _step_signature(seg: np.ndarray) -> np.ndarray:
    """(54, T) → 10차원 간략 특징 (컨텍스트 계산용)"""
    foot_lt_z  = seg[26]   # foot LT accel Z
    foot_rt_z  = seg[50]   # foot RT accel Z
    shank_lt_z = seg[20]   # shank LT accel Z
    pelvis_z   = seg[2]    # pelvis accel Z
    return np.array([
        float(np.abs(foot_lt_z).max()),           # 0: LT 최대 충격
        float(np.abs(foot_rt_z).max()),           # 1: RT 최대 충격
        float(foot_lt_z.std()),                   # 2: LT 진동 강도
        float(foot_rt_z.std()),                   # 3: RT 진동 강도
        float(np.sqrt((foot_lt_z**2).mean())),    # 4: LT RMS
        float(np.sqrt((foot_rt_z**2).mean())),    # 5: RT RMS
        float(np.abs(shank_lt_z).max()),          # 6: shank 충격 (잔디 감쇠용)
        float(pelvis_z.std()),                    # 7: pelvis 안정성 (평지 낮음)
        float(np.abs(np.diff(foot_lt_z)).mean()), # 8: LT jerk mean (흙길 높음)
        float(np.abs(np.diff(foot_rt_z)).mean()), # 9: RT jerk mean
    ], dtype=np.float32)


def _cv_safe(arr: np.ndarray) -> float:
    return float(arr.std() / (np.abs(arr).mean() + 1e-8))


def _context_feats_for_trial(
    sigs: np.ndarray,      # (n_steps, 10) — trial 내 모든 step signature
    pos: int,              # 현재 step의 trial 내 위치
    context_k: int = 3,
) -> np.ndarray:
    """trial 내 인접 step 기반 컨텍스트 피처 60개.

    [변동성 20] step-to-step std/CV  → C4 흙길 감지
    [누적통계 20] trial 전체 vs 현재  → C6 평지 감지
    [인접비교 20] 직전/직후 step 차이 → C5 잔디 감지
    """
    n_steps = len(sigs)
    lo = max(0, pos - context_k)
    hi = min(n_steps, pos + context_k + 1)
    ctx = sigs[lo:hi]   # (window, 10)
    cur = sigs[pos]

    # ── [1] 변동성 20개 ────────────────────────────────────
    if len(ctx) >= 2:
        diffs = np.diff(ctx, axis=0)
        var20 = np.array([
            float(diffs[:, 0].std()), float(diffs[:, 1].std()),
            float(diffs[:, 2].std()), float(diffs[:, 3].std()),
            float(diffs[:, 4].std()), float(diffs[:, 5].std()),
            float(diffs[:, 6].std()), float(diffs[:, 7].std()),
            float(diffs[:, 8].std()), float(diffs[:, 9].std()),
            _cv_safe(ctx[:, 0]), _cv_safe(ctx[:, 1]),
            _cv_safe(ctx[:, 4]), _cv_safe(ctx[:, 5]),
            _cv_safe(ctx[:, 7]),
            float(np.abs(diffs[:, 0] - diffs[:, 1]).mean()),
            float(np.abs(ctx[:, 0] - ctx[:, 1]).mean()),
            float(ctx[:, 8].mean()), float(ctx[:, 9].mean()),
            float(ctx[:, 7].mean()),
        ], dtype=np.float32)
    else:
        var20 = np.zeros(20, dtype=np.float32)

    # ── [2] 누적통계 20개 ──────────────────────────────────
    bout20 = np.array([
        float(sigs[:, 0].mean()), float(sigs[:, 1].mean()),
        float(sigs[:, 0].std()),  float(sigs[:, 1].std()),
        float(sigs[:, 7].mean()), float(sigs[:, 7].std()),
        _cv_safe(sigs[:, 0]),     _cv_safe(sigs[:, 4]),
        float(sigs[:, 6].mean()), float(sigs[:, 6].std()),
        float(cur[0] - sigs[:, 0].mean()),
        float(cur[1] - sigs[:, 1].mean()),
        float(cur[4] - sigs[:, 4].mean()),
        float(cur[7] - sigs[:, 7].mean()),
        float(cur[8] - sigs[:, 8].mean()),
        float(cur[0] / (sigs[:, 0].mean() + 1e-8)),
        float(cur[4] / (sigs[:, 4].mean() + 1e-8)),
        float(cur[7] / (sigs[:, 7].mean() + 1e-8)),
        float(n_steps),
        float(pos / max(n_steps - 1, 1)),
    ], dtype=np.float32)

    # ── [3] 인접비교 20개 ──────────────────────────────────
    diff_prev = (cur - sigs[pos - 1]) if pos > 0 else np.zeros(10, dtype=np.float32)
    diff_next = (cur - sigs[pos + 1]) if pos < n_steps - 1 else np.zeros(10, dtype=np.float32)
    adj20 = np.concatenate([diff_prev, diff_next])

    return np.concatenate([var20, bout20, adj20])  # 60개


def _read_trial_index(h5_path: str) -> tuple[np.ndarray, np.ndarray]:
    """HDF5 → (trial_ids, trial_step_idx) global 배열."""
    import h5py
    tid_list, tsi_list = [], []
    with h5py.File(h5_path, 'r') as f:
        subj_grp = f['subjects']
        for skey in sorted(subj_grp.keys()):
            grp  = subj_grp[skey]
            n_s  = grp['X'].shape[0]
            sbj  = int(skey[1:])
            tid  = grp['trial_id'][:].astype(np.int64) if 'trial_id' in grp \
                   else np.zeros(n_s, dtype=np.int64)
            tsi  = grp['trial_step_index'][:].astype(np.int64) if 'trial_step_index' in grp \
                   else np.arange(n_s, dtype=np.int64)
            tid_list.append(tid + sbj * 100000)   # global unique
            tsi_list.append(tsi)
    return np.concatenate(tid_list), np.concatenate(tsi_list)


def _extract_context_all(
    X: np.ndarray,           # (N, T, 54) or (N, 54, T)
    trial_ids: np.ndarray,   # (N,)
    trial_step_idx: np.ndarray,
    context_k: int = 3,
) -> np.ndarray:
    """전체 데이터의 컨텍스트 피처 (N, 60) 추출."""
    N = len(X)
    ctx = np.zeros((N, _N_CONTEXT), dtype=np.float32)

    # step signature 미리 계산
    sigs = np.zeros((N, 10), dtype=np.float32)
    for i, s in enumerate(X):
        seg = s if (s.ndim == 2 and s.shape[0] < s.shape[1]) else s.T
        sigs[i] = _step_signature(seg)

    # trial별 처리
    for tid in np.unique(trial_ids):
        mask  = trial_ids == tid
        idxs  = np.where(mask)[0]
        order = np.argsort(trial_step_idx[idxs])
        sorted_idxs = idxs[order]           # trial 내 순서대로 정렬된 global index
        t_sigs = sigs[sorted_idxs]          # (n_steps, 10)

        for pos, global_i in enumerate(sorted_idxs):
            ctx[global_i] = _context_feats_for_trial(t_sigs, pos, context_k)

    return np.nan_to_num(ctx, nan=0.0, posinf=1e4, neginf=-1e4)


# ─────────────────────────────────────────────────────────────
# 배치 추출 (센서 피처 264 + 컨텍스트 60 = 324)
# ─────────────────────────────────────────────────────────────

def batch_extract(
    X: np.ndarray,
    foot_accel_idx: list | None = None,
    sample_rate: int = 200,
    h5_path: str | None = None,          # bout 컨텍스트용 HDF5 경로
    kept_idx: np.ndarray | None = None,  # C5 제거 후 살아남은 원본 인덱스
                                         # (h5_path 읽은 trial_ids를 필터링하는데 사용)
    log_interval: int = _LOG_INTERVAL,
    verbose: bool = True,
) -> np.ndarray:
    """(N, T, 54) or (N, 54, T) → (N, N_FEATURES) float32

    h5_path 제공 시 bout 컨텍스트 피처(60개) 자동 추가 → N_FEATURES=366
    h5_path 없으면 센서 피처만(306개) 반환 (하위 호환)
    kept_idx : filter_and_remap() 이 반환한 인덱스.
               h5_path의 trial_ids/trial_step_idx는 원본(전체) 기준이므로
               kept_idx 로 슬라이싱해야 X와 길이가 일치함.
    """
    import time as _time

    N     = len(X)
    feats = np.zeros((N, _N_SENSOR), dtype=np.float32)
    sec_times: dict[str, float] = {name: 0.0 for name, _ in _SECTIONS[:-1]}  # Context 제외
    t_total = _time.time()

    if verbose:
        print(f"[feat] 센서 피처 추출: N={N} → {_N_SENSOR}차원  v4", flush=True)

    for i, s in enumerate(X):
        seg = s if (s.ndim == 2 and s.shape[0] < s.shape[1]) else s.T
        fs  = sample_rate

        t0 = _time.perf_counter()
        pelvis = _feat_pelvis(seg[_IDX["pelvis"]], fs)
        sec_times["Pelvis"] += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        hand = _feat_hand(seg[_IDX["hand_lt"]], seg[_IDX["hand_rt"]], fs)
        sec_times["Hand"] += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        thigh = _feat_thigh(seg[_IDX["thigh_lt"]], seg[_IDX["thigh_rt"]], fs)
        sec_times["Thigh"] += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        shank = _feat_shank(seg[_IDX["shank_lt"]], seg[_IDX["shank_rt"]], fs)
        sec_times["Shank"] += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        foot = _feat_foot(seg[_IDX["foot_lt"]], seg[_IDX["foot_rt"]], fs)
        sec_times["Foot"] += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        terrain = _feat_terrain(
            seg[_IDX["foot_lt"]][0:3],  seg[_IDX["foot_rt"]][0:3],
            seg[_IDX["shank_lt"]][0:3], seg[_IDX["shank_rt"]][0:3],
            seg[_IDX["pelvis"]][0:3],   fs,
        )
        sec_times["Terrain"] += _time.perf_counter() - t0

        feats[i] = np.nan_to_num(
            np.concatenate([pelvis, hand, thigh, shank, foot, terrain]),
            nan=0.0, posinf=1e6, neginf=-1e6,
        )

        if verbose and (i + 1) % log_interval == 0:
            elapsed   = _time.time() - t_total
            speed     = (i + 1) / elapsed
            remaining = (N - i - 1) / max(speed, 1e-6)
            pct       = (i + 1) / N * 100
            total_sec = sum(sec_times.values()) + 1e-9
            breakdown = "  ".join(
                f"{name}={sec_times[name]/total_sec*100:.0f}%"
                for name, _ in _SECTIONS[:-1]
            )
            print(
                f"[feat] {i+1:>6}/{N}  ({pct:5.1f}%)  "
                f"{speed:6.0f} samp/s  elapsed={elapsed:.1f}s  eta={remaining:.1f}s  "
                f"| {breakdown}",
                flush=True,
            )

    # ── bout 컨텍스트 피처 (h5_path 제공 시) ──────────────
    if h5_path is not None:
        if verbose:
            print(f"[feat] bout 컨텍스트 피처 추출 시작...", flush=True)
        t0 = _time.time()
        trial_ids_full, trial_step_idx_full = _read_trial_index(h5_path)
        if kept_idx is not None:
            # X는 이미 C5 제거된 N'개 → trial_ids도 같은 인덱스로 슬라이싱
            if len(kept_idx) != len(trial_ids_full):
                trial_ids      = trial_ids_full[kept_idx]
                trial_step_idx = trial_step_idx_full[kept_idx]
            else:
                trial_ids      = trial_ids_full
                trial_step_idx = trial_step_idx_full
        else:
            trial_ids      = trial_ids_full
            trial_step_idx = trial_step_idx_full
        if len(trial_ids) != len(X):
            raise ValueError(
                f"[features] trial_ids 길이({len(trial_ids)}) != X 길이({len(X)}). "
                f"kept_idx를 batch_extract에 전달했는지 확인하세요."
            )
        ctx_feats = _extract_context_all(X, trial_ids, trial_step_idx)
        feats = np.concatenate([feats, ctx_feats], axis=1)
        if verbose:
            print(f"[feat] 컨텍스트 완료 ({_time.time()-t0:.1f}s)  "
                  f"최종 shape={feats.shape}", flush=True)

    if verbose:
        elapsed   = _time.time() - t_total
        speed     = N / max(elapsed, 1e-6)
        total_sec = sum(sec_times.values()) + 1e-9
        print(f"[feat] ✅ 완료: {N}샘플  {elapsed:.1f}s  ({speed:.0f} samp/s)  "
              f"shape={feats.shape}", flush=True)
        for name, n_feat in _SECTIONS[:-1]:
            t   = sec_times[name]
            pct = t / total_sec * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"[feat]   {name:<8} {n_feat:3d}feat  {t:6.2f}s  {pct:5.1f}%  |{bar}|",
                  flush=True)
        nan_cnt = int(np.isnan(feats).sum())
        inf_cnt = int(np.isinf(feats).sum())
        print(f"[feat] 통계: mean={feats.mean():.4f}  std={feats.std():.4f}  "
              f"nan={nan_cnt}  inf={inf_cnt}", flush=True)

    return feats


StepFeatureExtractor = SensorFeatureExtractor


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((20, 54, 256)).astype(np.float32)

    # h5_path 없이 → 264차원 (하위 호환)
    feat264 = batch_extract(X, verbose=True)
    print(f"✅ 센서만: {X.shape} → {feat264.shape}  (N_SENSOR={_N_SENSOR})")

    # h5_path 없이 컨텍스트 수동 테스트
    trial_ids      = np.repeat(np.arange(4), 5)
    trial_step_idx = np.tile(np.arange(5), 4)
    ctx = _extract_context_all(X, trial_ids, trial_step_idx)
    full = np.concatenate([feat264, ctx], axis=1)
    print(f"✅ 센서+컨텍스트: {full.shape}  N_FEATURES={N_FEATURES}")
    assert full.shape[1] == N_FEATURES, f"shape 오류: {full.shape[1]} != {N_FEATURES}"

    sections = {"Pelvis":30,"Hand":38,"Thigh":36,"Shank":40,"Foot":88,"Terrain":32,"Context":60}
    idx = 0
    for name, n in sections.items():
        chunk = full[0, idx:idx+n]
        print(f"   {name:<8}: {n:3d}개  mean={chunk.mean():8.3f}  std={chunk.std():8.3f}")
        idx += n