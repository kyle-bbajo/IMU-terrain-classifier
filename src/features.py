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

피처 구성 (총 N_FEATURES = 200):
    Pelvis      :  22개  (몸통 안정성 · tilt · jerk)
    Hand  LT/RT :  38개  (팔 스윙 리듬 · 좌우 대칭)
    Thigh LT/RT :  36개  (고관절 ROM · 각속도)
    Shank LT/RT :  40개  (무릎 충격 · 정강이 진동)
    Foot  LT/RT :  64개  (착지 패턴 · 보행 주파수)
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
        float((combined[freqs < 10]             ** 2).sum() / ce),
        float((combined[(freqs >= 10) & (freqs < 30)] ** 2).sum() / ce),
        float((combined[(freqs >= 30) & (freqs < 50)] ** 2).sum() / ce),
        float((combined[freqs >= 30]            ** 2).sum() / ce),
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
    n_peaks   = len(peaks)
    avg_height = float(props["peak_heights"].mean()) if n_peaks > 0 else 0.0
    avg_interval = float(np.diff(peaks).mean() / fs) if n_peaks > 1 else 0.0
    return np.array([float(n_peaks), avg_height, avg_interval], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 센서별 피처 추출 함수
# ─────────────────────────────────────────────────────────────

def _feat_pelvis(seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Pelvis (ch 0-5): 몸통 안정성 · tilt · jerk
    출력: 22개
        time  per axis  (6ax × 2: mean, RMS)         = 12
        vector mag      (accel / gyro)                =  6
        jerk  RMS       (accel Z)                     =  1
        freq  dom_freq  (accel Z, gyro Y)             =  2
        band power      (accel 3ax)                   =  1  → hf_ratio만
    """
    accel = seg[0:3]   # X/Y/Z
    gyro  = seg[3:6]

    out = []
    # time stats (mean + RMS) × 6축
    for s in np.vstack([accel, gyro]):
        out += [float(s.mean()), float(np.sqrt((s**2).mean()))]

    # vector mag
    out.extend(_vector_mag_stats(accel).tolist())
    out.extend(_vector_mag_stats(gyro).tolist())

    # jerk (accel Z)
    out.append(_jerk_rms(accel[2], fs))

    # tilt instability: gyro std mean (몸통 흔들림)
    out.append(float(np.array([s.std() for s in gyro]).mean()))

    # dominant freq (accel Z, gyro Y)
    freqs = np.fft.rfftfreq(seg.shape[1], d=1.0 / fs)
    for s in [accel[2], gyro[1]]:
        mag = np.abs(np.fft.rfft(s))
        out.append(float(freqs[np.argmax(mag)]))

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
        out += [float(ts[0]), float(ts[1]), float(ts[2])]   # mean, std, RMS
    out.extend(_vector_mag_stats(accel).tolist())
    fs_stats = _freq_stats(accel[2], fs)
    out.append(float(fs_stats[0]))   # dominant freq
    out.append(float(fs_stats[3]))   # spectral entropy
    for s in gyro:
        out.append(float(np.sqrt((s**2).mean())))  # RMS
    assert len(out) == 17, f"Hand one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_hand(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Hand LT + RT: 팔 스윙 + 좌우 대칭
    출력: 38개 = 17 + 17 + 4(대칭)
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
        out += [float(s.mean()), float(np.sqrt((s**2).mean()))]
    out.extend(_vector_mag_stats(accel).tolist())
    for s in gyro:
        out.append(float(s.max() - s.min()))   # ROM
    for s in gyro:
        out.append(float(np.abs(s).mean()))    # mean angular vel
    freqs = np.fft.rfftfreq(accel.shape[1], d=1.0 / fs)
    mag = np.abs(np.fft.rfft(accel[2]))
    out.append(float(freqs[np.argmax(mag)]))
    assert len(out) == 16, f"Thigh one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_thigh(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Thigh LT + RT: 고관절 + 좌우 대칭
    출력: 36개 = 16 + 16 + 4(대칭)
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
        accel peak stats × 3ax (peak_count, avg_height, avg_interval) = 9
        gyro  time (mean, std, RMS) × 3ax                             = 9
    """
    out = []
    for s in accel:
        out.extend(_peak_stats(np.abs(s), fs).tolist())
    for s in gyro:
        ts = _time_stats(s)
        out += [float(ts[0]), float(ts[1]), float(ts[2])]
    assert len(out) == 18, f"Shank one side len={len(out)}"
    return np.array(out, dtype=np.float32)


def _feat_shank(lt_seg: np.ndarray, rt_seg: np.ndarray, fs: int) -> np.ndarray:
    """
    Shank LT + RT: 무릎 충격 + 좌우 대칭
    출력: 40개 = 18 + 18 + 4(대칭)
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
    Foot 한쪽: 착지 패턴 · 보행 주파수 (기존 44개 방식 확장)
    출력: 30개
        accel time stats (7개) × 3ax            = 21
        accel cross: SMA, var_ratio, peak_ratio  =  3
        accel freq  dom_freq × 3ax               =  3
        accel band power                         =  4  (hf_ratio 포함)
        gyro  RMS × 3ax                          =  3
        gyro  dominant_freq accel Z proxy        =  1  (gyro Y - sagittal)
        heel-strike proxy: peak count accel Z    =  1  → 3개 peak_stats
    """
    freqs = np.fft.rfftfreq(accel.shape[1], d=1.0 / fs)
    out = []

    # accel 시간 통계
    for s in accel:
        out.extend(_time_stats(s).tolist())    # 7×3 = 21

    # accel 교차축
    total_mean = float(np.abs(accel).mean())
    var_ratio  = float((accel[2].var() + 1e-8) / (accel[0].var() + 1e-8))
    peak_ratio = float(np.abs(accel).max() / (np.abs(accel).mean() + 1e-8))
    out += [total_mean, var_ratio, peak_ratio]  # 3

    # accel 주파수
    mags = []
    for s in accel:
        mag = np.abs(np.fft.rfft(s))
        mags.append(mag)
        out.append(float(freqs[np.argmax(mag)]))  # dom_freq ×3
    out.extend(_band_power(mags, freqs).tolist())  # 4

    # gyro RMS ×3
    for s in gyro:
        out.append(float(np.sqrt((s**2).mean())))

    # gyro Y dominant freq (시상면 각속도 리듬)
    mag_gy = np.abs(np.fft.rfft(gyro[1]))
    out.append(float(freqs[np.argmax(mag_gy)]))

    # heel-strike proxy (accel Z 피크 통계 3개)
    out.extend(_peak_stats(np.abs(accel[2]), fs).tolist())

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

# 채널 인덱스 (고정 레이아웃)
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

N_FEATURES = 216  # 22+38+36+40+80


class SensorFeatureExtractor:
    """(C=54, T) 윈도우 → 216차원 float32 피처 벡터."""

    N_FEATURES = N_FEATURES

    def __init__(self, sample_rate: int = 200) -> None:
        self.fs = sample_rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.extract(x)

    def extract(self, x: np.ndarray) -> np.ndarray:
        """
        x: (54, T) float32
        return: (216,) float32
        """
        assert x.shape[0] >= 54, f"채널 수 부족: {x.shape[0]} < 54"
        fs = self.fs

        pelvis   = _feat_pelvis(x[_IDX["pelvis"]],  fs)             # 22
        hand     = _feat_hand(x[_IDX["hand_lt"]],
                              x[_IDX["hand_rt"]], fs)                # 38
        thigh    = _feat_thigh(x[_IDX["thigh_lt"]],
                               x[_IDX["thigh_rt"]], fs)              # 36
        shank    = _feat_shank(x[_IDX["shank_lt"]],
                               x[_IDX["shank_rt"]], fs)              # 40
        foot     = _feat_foot(x[_IDX["foot_lt"]],
                              x[_IDX["foot_rt"]], fs)                # 80

        feat = np.concatenate([pelvis, hand, thigh, shank, foot])
        assert len(feat) == N_FEATURES, f"총 피처 수 불일치: {len(feat)}"
        return np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)


# ─────────────────────────────────────────────────────────────
# 배치 추출 (하위 호환 유지)
# ─────────────────────────────────────────────────────────────

def batch_extract(
    X: np.ndarray,
    foot_accel_idx: list | None = None,   # 하위 호환 (무시됨)
    sample_rate: int = 200,
) -> np.ndarray:
    """
    (N, C, T) → (N, 216) float32
    foot_accel_idx 는 하위 호환을 위해 남겨두되 무시합니다.
    """
    ext   = SensorFeatureExtractor(sample_rate)
    feats = np.zeros((len(X), N_FEATURES), dtype=np.float32)
    for i, s in enumerate(X):
        # (C, T) or (T, C) → (C, T) 보장
        if s.ndim == 2 and s.shape[0] < s.shape[1]:
            seg = s
        else:
            seg = s.T
        feats[i] = ext.extract(seg)
    return feats


# 하위 호환: 기존 StepFeatureExtractor 이름 유지
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

    # 센서별 피처 범위 확인
    ext = SensorFeatureExtractor()
    f   = ext.extract(X[0])
    sections = {"Pelvis":22, "Hand":38, "Thigh":36, "Shank":40, "Foot":64}
    idx = 0
    for name, n in sections.items():
        chunk = f[idx:idx+n]
        print(f"   {name:<8}: {n:3d}개  "
              f"mean={chunk.mean():7.3f}  std={chunk.std():7.3f}")
        idx += n