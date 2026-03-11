"""src/features.py — 9센서 도메인 특화 피처 추출 (v2.0)

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

피처 구성 (총 N_FEATURES = 232):
    Pelvis      :  22개  (몸통 안정성 · tilt · jerk)
    Hand  LT/RT :  38개  (팔 스윙 리듬 · 좌우 대칭)
    Thigh LT/RT :  36개  (고관절 ROM · 각속도)
    Shank LT/RT :  40개  (무릎 충격 · 정강이 진동)
    Foot  LT/RT :  80개  (착지 패턴 · 보행 주파수)
    Terrain      :  16개  (흙길 불규칙성 8 + 잔디 탄성감쇠 8)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.signal import find_peaks


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
# 센서별 피처 추출 함수
# ─────────────────────────────────────────────────────────────

def _feat_pelvis(seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Pelvis (ch 0-5): 몸통 안정성 · tilt · jerk
    출력: 22개
        time  per axis  (6ax × 2: mean, RMS)    = 12
        vector mag      (accel / gyro)           =  6
        jerk  RMS       (accel Z)                =  1
        tilt instability gyro std mean           =  1
        freq  dom_freq  (accel Z, gyro Y)        =  2
    """
    accel = seg[0:3]
    gyro  = seg[3:6]

    out = []
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

    assert len(out) == 22, f"Pelvis feat len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_one_side_hand(accel: np.ndarray, gyro: np.ndarray, fs: int) -> np.ndarray:
    """
    Hand 한쪽 (accel 3ax, gyro 3ax): 팔 스윙 리듬
    출력: 17개
        time stats (mean, std, RMS) × 3ax accel   =  9
        vector mag accel                           =  3
        dominant_freq accel Z                      =  1
        spectral_entropy accel Z                   =  1
        gyro RMS × 3ax                             =  3
    """
    out = []
    for s in accel:
        ts = _time_stats(s)
        out += [float(ts[0]), float(ts[1]), float(ts[2])]   # 9
    out.extend(_vector_mag_stats(accel).tolist())            # 3
    fs_stats = _freq_stats(accel[2], fs)
    out.append(float(fs_stats[0]))   # dominant freq         # 1
    out.append(float(fs_stats[3]))   # spectral entropy      # 1
    for s in gyro:
        out.append(float(np.sqrt((s**2).mean())))            # 3
    assert len(out) == 17, f"Hand one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_hand(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Hand LT + RT: 팔 스윙 + 좌우 대칭
    출력: 38개 = 17(LT) + 17(RT) + 4(대칭)
    """
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
    """
    Thigh 한쪽: 고관절 ROM · 각속도
    출력: 16개
        accel time (mean, RMS) × 3ax    =  6
        accel vector mag                =  3
        gyro  ROM (max-min) × 3ax       =  3
        gyro  mean angular vel × 3ax    =  3
        dominant freq (accel Z)         =  1
    """
    out = []
    for s in accel:
        out += [float(s.mean()), float(np.sqrt((s**2).mean()))]   # 6
    out.extend(_vector_mag_stats(accel).tolist())                  # 3
    for s in gyro:
        out.append(float(s.max() - s.min()))   # 3
    for s in gyro:
        out.append(float(np.abs(s).mean()))    # 3
    freqs = np.fft.rfftfreq(accel.shape[1], d=1.0 / fs)
    mag = np.abs(np.fft.rfft(accel[2]))
    out.append(float(freqs[np.argmax(mag)]))                       # 1
    assert len(out) == 16, f"Thigh one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_thigh(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Thigh LT + RT: 고관절 + 좌우 대칭
    출력: 36개 = 16(LT) + 16(RT) + 4(대칭)
    """
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
    """
    Shank 한쪽: 무릎 충격 · 정강이 진동
    출력: 18개
        accel peak stats × 3ax   = 9
        gyro  time stats × 3ax   = 9
    """
    out = []
    for s in accel:
        out.extend(_peak_stats(np.abs(s), fs).tolist())   # 9
    for s in gyro:
        ts = _time_stats(s)
        out += [float(ts[0]), float(ts[1]), float(ts[2])]  # 9
    assert len(out) == 18, f"Shank one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_shank(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Shank LT + RT: 무릎 충격 + 좌우 대칭
    출력: 40개 = 18(LT) + 18(RT) + 4(대칭)
    """
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
    Foot 한쪽: 착지 패턴 · 보행 주파수
    출력: 38개
        accel time stats (7) × 3ax   = 21
        accel cross stats            =  3
        accel dom_freq × 3ax         =  3
        accel band power             =  4
        gyro  RMS × 3ax              =  3
        gyro Y dom_freq              =  1
        heel-strike peak_stats       =  3
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

    assert len(out) == 38, f"Foot one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_foot(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Foot LT + RT: 착지 + 좌우 대칭
    출력: 80개 = 38(LT) + 38(RT) + 4(대칭)
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
    assert len(out) == 80, f"Foot feat len={len(out)}"
    return out


# ─────────────────────────────────────────────────────────────
# 통합 추출기
# ─────────────────────────────────────────────────────────────


def _peak_height_variance(s: np.ndarray, fs: int) -> float:
    """피크 높이의 분산 — 흙길(불규칙) 지표"""
    peaks, props = find_peaks(np.abs(s), height=np.abs(s).mean() * 0.5, distance=int(fs * 0.05))
    if len(props["peak_heights"]) < 2:
        return 0.0
    return float(np.var(props["peak_heights"]))


def _peak_interval_cv(s: np.ndarray, fs: int) -> float:
    """피크 간격 변동계수(CV) — 흙길(불규칙 보행 타이밍) 지표"""
    peaks, _ = find_peaks(np.abs(s), height=np.abs(s).mean() * 0.5, distance=int(fs * 0.05))
    if len(peaks) < 3:
        return 0.0
    intervals = np.diff(peaks).astype(np.float32)
    return float(intervals.std() / (intervals.mean() + 1e-8))


def _jerk_variance(s: np.ndarray, fs: int) -> float:
    """jerk(가속도 미분) 분산 — 흙길(충격 불규칙성) 지표"""
    jerk = np.diff(s) * fs
    return float(np.var(jerk))


def _hf_band_ratio(s: np.ndarray, fs: int) -> float:
    """고주파(>30 Hz) 에너지 비율 — 흙길(진동) 지표"""
    mag   = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
    total = mag.sum() + 1e-8
    return float(mag[freqs >= 30].sum() / total)


def _impact_peak(s: np.ndarray) -> float:
    """최대 충격 피크 — 잔디(낮음) vs 아스팔트(높음) 지표"""
    return float(np.abs(s).max())


def _loading_rate(s: np.ndarray, fs: int) -> float:
    """초기 피크까지의 상승 기울기(loading rate) — 잔디(완만) 지표"""
    abs_s  = np.abs(s)
    pk_idx = int(np.argmax(abs_s))
    if pk_idx == 0:
        return 0.0
    start  = max(pk_idx - int(fs * 0.05), 0)
    rise   = float(abs_s[pk_idx] - abs_s[start])
    dur    = (pk_idx - start) / fs + 1e-8
    return float(rise / dur)


def _lf_hf_ratio(s: np.ndarray, fs: int) -> float:
    """저주파(<10 Hz) / 고주파(>30 Hz) 에너지 비율 — 잔디(높음) 지표"""
    mag   = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
    lf    = mag[freqs < 10].sum() + 1e-8
    hf    = mag[freqs >= 30].sum() + 1e-8
    return float(lf / hf)


def _rebound_ratio(s: np.ndarray, fs: int) -> float:
    """첫 번째 피크 대비 두 번째 피크 비율 — 잔디(낮은 반발) 지표"""
    abs_s  = np.abs(s)
    peaks, props = find_peaks(abs_s, height=abs_s.mean() * 0.3, distance=int(fs * 0.05))
    if len(peaks) < 2:
        return 0.0
    heights = props["peak_heights"]
    return float(heights[1] / (heights[0] + 1e-8))


def _feat_terrain(
    foot_lt: np.ndarray, foot_rt: np.ndarray,
    shank_lt: np.ndarray, shank_rt: np.ndarray,
    fs: int,
) -> np.ndarray:
    """
    C4(흙길) / C5(잔디) 구분 특화 피처
    출력: 16개
        [흙길 = 불규칙성] 8개:
            foot_lt  peak_height_var       1
            foot_lt  peak_interval_cv      1
            foot_lt  jerk_variance         1
            shank_lt hf_band_ratio         1
            foot_rt  peak_height_var       1
            foot_rt  peak_interval_cv      1
            foot_rt  jerk_variance         1
            LR peak height asymmetry       1

        [잔디 = 탄성/감쇠] 8개:
            foot_lt  impact_peak           1
            foot_lt  loading_rate          1
            foot_lt  lf_hf_ratio           1
            foot_lt  rebound_ratio         1
            foot_rt  impact_peak           1
            foot_rt  loading_rate          1
            foot_rt  lf_hf_ratio           1
            foot_rt  rebound_ratio         1
    """
    fa_lt = foot_lt[2]   # foot LT accel Z
    fa_rt = foot_rt[2]   # foot RT accel Z
    sa_lt = shank_lt[2]  # shank LT accel Z
    sa_rt = shank_rt[2]  # shank RT accel Z

    # ── 흙길 불규칙성 ─────────────────────────────────────────
    lt_phv = _peak_height_variance(fa_lt, fs)
    lt_pic = _peak_interval_cv(fa_lt, fs)
    lt_jv  = _jerk_variance(fa_lt, fs)
    lt_hf  = _hf_band_ratio(sa_lt, fs)

    rt_phv = _peak_height_variance(fa_rt, fs)
    rt_pic = _peak_interval_cv(fa_rt, fs)
    rt_jv  = _jerk_variance(fa_rt, fs)

    # LR 피크 높이 비대칭 (흙길 = 비대칭 큼)
    lt_pk_mean = float(np.abs(fa_lt).max())
    rt_pk_mean = float(np.abs(fa_rt).max())
    lr_asym    = abs(lt_pk_mean - rt_pk_mean) / (lt_pk_mean + rt_pk_mean + 1e-8)

    irreg = [lt_phv, lt_pic, lt_jv, lt_hf, rt_phv, rt_pic, rt_jv, lr_asym]

    # ── 잔디 탄성/감쇠 ────────────────────────────────────────
    comp = [
        _impact_peak(fa_lt),  _loading_rate(fa_lt, fs),
        _lf_hf_ratio(fa_lt, fs), _rebound_ratio(fa_lt, fs),
        _impact_peak(fa_rt),  _loading_rate(fa_rt, fs),
        _lf_hf_ratio(fa_rt, fs), _rebound_ratio(fa_rt, fs),
    ]

    out = np.array(irreg + comp, dtype=np.float32)
    assert len(out) == 16, f"terrain feat len={len(out)}"
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

N_FEATURES: int = 232  # 22 + 38 + 36 + 40 + 80 + 16(terrain)


class SensorFeatureExtractor:
    """(C=54, T) 윈도우 → 216차원 float32 피처 벡터."""

    N_FEATURES = N_FEATURES

    def __init__(self, sample_rate: int = 200) -> None:
        self.fs = sample_rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.extract(x)

    def extract(self, x: np.ndarray) -> np.ndarray:
        """x: (54, T) → (216,) float32"""
        assert x.shape[0] >= 54, f"채널 수 부족: {x.shape[0]} < 54"
        fs = self.fs

        pelvis  = _feat_pelvis(x[_IDX["pelvis"]],  fs)
        hand    = _feat_hand(x[_IDX["hand_lt"]],   x[_IDX["hand_rt"]], fs)
        thigh   = _feat_thigh(x[_IDX["thigh_lt"]], x[_IDX["thigh_rt"]], fs)
        shank   = _feat_shank(x[_IDX["shank_lt"]], x[_IDX["shank_rt"]], fs)
        foot    = _feat_foot(x[_IDX["foot_lt"]],   x[_IDX["foot_rt"]], fs)
        terrain = _feat_terrain(
            x[_IDX["foot_lt"]][0:3],  x[_IDX["foot_rt"]][0:3],
            x[_IDX["shank_lt"]][0:3], x[_IDX["shank_rt"]][0:3], fs,
        )

        feat = np.concatenate([pelvis, hand, thigh, shank, foot, terrain])
        assert len(feat) == N_FEATURES, f"총 피처 수 불일치: {len(feat)} != {N_FEATURES}"
        return np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)


# ─────────────────────────────────────────────────────────────
# 배치 추출
# ─────────────────────────────────────────────────────────────

# 센서 구간 정의 (로그용)
_SECTIONS: list[tuple[str, int]] = [
    ("Pelvis", 22), ("Hand", 38), ("Thigh", 36), ("Shank", 40), ("Foot", 80), ("Terrain", 16),
]
_LOG_INTERVAL = 1000   # 몇 샘플마다 진행률을 출력할지


def batch_extract(
    X: np.ndarray,
    foot_accel_idx: list | None = None,   # 하위 호환 (무시됨)
    sample_rate: int = 200,
    log_interval: int = _LOG_INTERVAL,
    verbose: bool = True,
) -> np.ndarray:
    """(N, C, T) → (N, 216) float32

    Parameters
    ----------
    X            : (N, C, T) 또는 (N, T, C) 배열
    sample_rate  : 샘플링 주파수 (Hz)
    log_interval : 진행 로그 출력 간격 (샘플 수)
    verbose      : False 이면 로그 없이 조용히 실행
    """
    import time as _time

    N   = len(X)
    ext = SensorFeatureExtractor(sample_rate)
    feats = np.zeros((N, N_FEATURES), dtype=np.float32)

    # ── 센서별 누적 시간 측정용 ──
    sec_times: dict[str, float] = {name: 0.0 for name, _ in _SECTIONS}

    t_total = _time.time()
    t_log   = _time.time()   # 마지막 로그 시각

    if verbose:
        print(f"[feat] 피처 추출 시작: N={N}  →  {N_FEATURES}차원  "
              f"(로그 간격: {log_interval}샘플)", flush=True)

    for i, s in enumerate(X):
        seg = s if (s.ndim == 2 and s.shape[0] < s.shape[1]) else s.T
        fs  = sample_rate

        # 센서별 시간 측정하며 추출
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
            seg[_IDX["shank_lt"]][0:3], seg[_IDX["shank_rt"]][0:3], fs,
        )
        sec_times["Terrain"] += _time.perf_counter() - t0

        feats[i] = np.nan_to_num(
            np.concatenate([pelvis, hand, thigh, shank, foot, terrain]),
            nan=0.0, posinf=1e6, neginf=-1e6,
        )

        # ── 진행률 로그 ──
        if verbose and (i + 1) % log_interval == 0:
            elapsed   = _time.time() - t_total
            speed     = (i + 1) / elapsed          # 샘플/초
            remaining = (N - i - 1) / max(speed, 1e-6)
            pct       = (i + 1) / N * 100

            # 센서별 비율
            total_sec = sum(sec_times.values()) + 1e-9
            breakdown = "  ".join(
                f"{name}={sec_times[name]/total_sec*100:.0f}%"
                for name, _ in _SECTIONS
            )

            print(
                f"[feat] {i+1:>6}/{N}  ({pct:5.1f}%)  "
                f"{speed:6.0f} samp/s  "
                f"elapsed={elapsed:.1f}s  eta={remaining:.1f}s  "
                f"| {breakdown}",
                flush=True,
            )
            t_log = _time.time()

    # ── 최종 요약 ──
    if verbose:
        elapsed   = _time.time() - t_total
        speed     = N / max(elapsed, 1e-6)
        total_sec = sum(sec_times.values()) + 1e-9

        print(f"[feat] ✅ 완료: {N}샘플  {elapsed:.1f}s  ({speed:.0f} samp/s)", flush=True)
        print(f"[feat] 센서별 누적 시간 (전체 {total_sec:.1f}s):", flush=True)
        for name, n_feat in _SECTIONS:
            t   = sec_times[name]
            pct = t / total_sec * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"[feat]   {name:<8} {n_feat:3d}feat  {t:6.2f}s  {pct:5.1f}%  |{bar}|",
                  flush=True)

        nan_cnt = int(np.isnan(feats).sum())
        inf_cnt = int(np.isinf(feats).sum())
        print(
            f"[feat] 통계: mean={feats.mean():.4f}  std={feats.std():.4f}"
            f"  nan={nan_cnt}  inf={inf_cnt}",
            flush=True,
        )

    return feats


# 하위 호환
StepFeatureExtractor = SensorFeatureExtractor


# ─────────────────────────────────────────────────────────────
# 빠른 검증
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng  = np.random.default_rng(0)
    X    = rng.standard_normal((8, 54, 256)).astype(np.float32)
    feat = batch_extract(X)
    print(f"✅ batch_extract: {X.shape} → {feat.shape}")
    print(f"   mean={feat.mean():.4f}  std={feat.std():.4f}  "
          f"nan={np.isnan(feat).sum()}  inf={np.isinf(feat).sum()}")

    ext = SensorFeatureExtractor()
    f   = ext.extract(X[0])
    sections = {"Pelvis": 22, "Hand": 38, "Thigh": 36, "Shank": 40, "Foot": 80, "Terrain": 16}
    idx = 0
    for name, n in sections.items():
        chunk = f[idx:idx+n]
        print(f"   {name:<8}: {n:3d}개  mean={chunk.mean():7.3f}  std={chunk.std():7.3f}")
        idx += n