"""
config.py — 전역 설정 (v8.2)
═══════════════════════════════════════════════════════
★ 모든 하드코딩 값 중앙 관리
★ 스마트 듀얼 Preload: M1(항상) / M2–M6(RAM 여유 시)
★ AMP: GPU cc ≥ 8.0 → bfloat16, 그 외 → float16
★ apply_overrides(): CLI 인자로 런타임 변경
★ snapshot(): 실험 결과에 config + git hash 자동 저장
★ v8.2: Niswander et al. (2021) — Foot IMU 기반 HS 검출
         AP Accel 극대값(mean+1.8σ) → ML Gyro 검증(0.5σ)
         Contact 신호 제거 — Gyro + Accel 만 사용
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import json
import subprocess
import numpy as np
import torch
from pathlib import Path
import re as _re

# ─────────────────────────────────────────────
# 0. JSON 직렬화 유틸 (numpy 타입 → Python 기본 타입)
# ─────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """numpy 타입을 JSON 직렬화 가능한 Python 기본 타입으로 변환한다."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)


# ─────────────────────────────────────────────
# 1. 디바이스 자동 감지 (Multi-GPU 지원)
# ─────────────────────────────────────────────
if torch.cuda.is_available():
    N_GPU: int              = torch.cuda.device_count()
    DEVICE: torch.device    = torch.device("cuda")
    DEVICE_NAME: str        = torch.cuda.get_device_name(0)
    GPU_MEM_GB: float       = torch.cuda.get_device_properties(0).total_memory / 1024**3
    GPU_TOTAL_MEM_GB: float = GPU_MEM_GB * N_GPU
    USE_GPU: bool           = True
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
else:
    N_GPU            = 0
    DEVICE           = torch.device("cpu")
    DEVICE_NAME      = "CPU"
    GPU_MEM_GB       = 0.0
    GPU_TOTAL_MEM_GB = 0.0
    USE_GPU          = False

# ─────────────────────────────────────────────
# 2. 하드웨어
# ─────────────────────────────────────────────
VCPU: int = os.cpu_count() or 4
try:
    RAM_GIB: int = round(
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    )
except (ValueError, OSError):
    RAM_GIB = 16

# ─────────────────────────────────────────────
# 3. 데이터 로딩 전략
# ─────────────────────────────────────────────
USE_PRELOAD: bool    = True
USE_PRELOAD_M1: bool = True


def can_preload_branch(n_samples: int, n_channels: int, ts: int = 256) -> bool:
    """fp16 기준으로 Preload 가능 여부를 동적 판단한다."""
    needed_gib = n_samples * n_channels * ts * 2 / 1024**3
    return needed_gib < RAM_GIB * 0.80


# ─────────────────────────────────────────────
# 4. 워커 설정
# ─────────────────────────────────────────────
if USE_GPU:
    TORCH_THREADS: int   = max(2, min(4, VCPU // (N_GPU or 1)))
    _workers_per_gpu     = max(2, VCPU // max(N_GPU, 1) // 2)
    LOADER_WORKERS: int  = 0 if not USE_PRELOAD else min(8, _workers_per_gpu)
    PREPROC_WORKERS: int = max(2, VCPU - 1)
else:
    TORCH_THREADS        = max(4, VCPU - 4)
    LOADER_WORKERS       = 0 if not USE_PRELOAD else min(8, max(2, VCPU // 4))
    PREPROC_WORKERS      = max(4, VCPU - 2)

os.environ["OMP_NUM_THREADS"]      = str(TORCH_THREADS)
os.environ["MKL_NUM_THREADS"]      = str(TORCH_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(TORCH_THREADS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────
# 5. Mixed Precision
# ─────────────────────────────────────────────
USE_AMP: bool = USE_GPU
if USE_GPU:
    _cc = torch.cuda.get_device_capability(0)
    AMP_DTYPE: torch.dtype = torch.bfloat16 if _cc >= (8, 0) else torch.float16
else:
    AMP_DTYPE = torch.float32

# ─────────────────────────────────────────────
# 6. 실험 파라미터
# ─────────────────────────────────────────────
N_SUBJECTS: int  = 40
NUM_CLASSES: int = 6
TS: int          = 256
PCA_CH: int      = 32
IPCA_CHUNK: int  = 5000
SAMPLE_RATE: int = 200
SEED: int        = 42

# ─────────────────────────────────────────────
# 7. 힐스트라이크 검출 파라미터 (Niswander et al., 2021)
# ─────────────────────────────────────────────
#
# ★ v8.2: Foot IMU 전용 (Gyro + Accel 만 사용, Contact 신호 없음)
#
#   [Step 1] 전처리   : BPF(1–20Hz)
#   [Step 2] HS 후보  : Foot AP Accel 극대값 > mean + 1.8σ
#   [Step 3] 발 회전  : Foot ML Gyro window_max > 0.5σ → HS 확정
#   [Step 4] 시점 정제: window 내 AP Accel argmax로 보정
#   [Step 5] 보폭 제약: stride_min ~ stride_max
#
#   Foot Accel X = 전후방향(AP) 충격축
#   Foot Gyro  Y = 내외측(ML) 회전축
# ─────────────────────────────────────────────
HS_MIN_STRIDE_MS: int        = 400
HS_MAX_STRIDE_MS: int        = 1800
HS_MIN_STRIDE_SAM: int       = int(HS_MIN_STRIDE_MS  / 1000 * SAMPLE_RATE)
HS_MAX_STRIDE_SAM: int       = int(HS_MAX_STRIDE_MS  / 1000 * SAMPLE_RATE)
HS_NAN_THRESHOLD: float      = 0.1
HS_PROMINENCE_COEFF: float   = 0.3
HS_PEAK_QUALITY_RATIO: float = 0.6
MIN_STEP_LEN: int            = HS_MIN_STRIDE_SAM

# 센서 소스 (Foot IMU)
HS_GYRO_SENSOR: str  = "Foot"
HS_GYRO_AXIS: str    = "y"      # ML축 회전
HS_ACCEL_SENSOR: str = "Foot"
HS_ACCEL_AXIS: str   = "x"      # AP 충격축

# Niswander 임계값
HS_ACCEL_PEAK_SIGMA: float = 1.8   # mean + 1.8σ  (논문 Eq.1)
HS_GYRO_VALID_SIGMA: float = 0.5   # 0.5σ 검증

# 검증 윈도우
HS_FUSION_WINDOW_MS: int  = 125
HS_FUSION_WINDOW_SAM: int = int(HS_FUSION_WINDOW_MS / 1000 * SAMPLE_RATE)

# 하위 호환 (직접 사용 금지)
HS_GYRO_PROMINENCE: float = 0.4
HS_ACCEL_THRESHOLD: float = 0.8
HS_TRUSTED_SWING: float   = 0.2

# ─────────────────────────────────────────────
# 7-a. 발 가속도 컬럼 정의 (Contact 제거)
# ─────────────────────────────────────────────
FOOT_ACC_COLS: dict[str, dict[str, str]] = {
    "LT": {
        "x": "Foot Accel Sensor X LT (mG)",
        "y": "Foot Accel Sensor Y LT (mG)",
        "z": "Foot Accel Sensor Z LT (mG)",
    },
    "RT": {
        "x": "Foot Accel Sensor X RT (mG)",
        "y": "Foot Accel Sensor Y RT (mG)",
        "z": "Foot Accel Sensor Z RT (mG)",
    },
}
DROP_COLS: list[str] = ["time", "Activity", "Marker"]

# ─────────────────────────────────────────────
# 7-b. 컬럼명 유연 매칭 시스템
# ─────────────────────────────────────────────
_FOOT_ACC_PATTERNS: dict[str, dict[str, _re.Pattern]] = {
    "LT": {
        "x": _re.compile(r"(?i)foot.*accel.*[_\s]?x.*lt"),
        "y": _re.compile(r"(?i)foot.*accel.*[_\s]?y.*lt"),
        "z": _re.compile(r"(?i)foot.*accel.*[_\s]?z.*lt"),
    },
    "RT": {
        "x": _re.compile(r"(?i)foot.*accel.*[_\s]?x.*rt"),
        "y": _re.compile(r"(?i)foot.*accel.*[_\s]?y.*rt"),
        "z": _re.compile(r"(?i)foot.*accel.*[_\s]?z.*rt"),
    },
}

_DROP_PATTERNS: list[_re.Pattern] = [
    _re.compile(r"(?i)^time$"),
    _re.compile(r"(?i)^activity$"),
    _re.compile(r"(?i)^marker"),
]


def resolve_column(
    columns: list[str],
    exact_name: str,
    pattern: _re.Pattern | None = None,
) -> str:
    """컬럼명을 정확 매칭 → 패턴 매칭 → 부분 매칭 순서로 찾는다."""
    if exact_name in columns:
        return exact_name
    if pattern is not None:
        for c in columns:
            if pattern.search(c):
                return c
    exact_lower = exact_name.lower()
    candidates = [c for c in columns if exact_lower in c.lower()]
    if candidates:
        return candidates[0]
    raise KeyError(
        f"컬럼 '{exact_name}' 없음. "
        f"후보: {[c for c in columns if any(w in c.lower() for w in exact_lower.split())][:5]}"
    )


def resolve_foot_acc_cols(columns: list[str], side: str) -> dict[str, str]:
    """발 가속도 3축 컬럼명을 찾는다 (x/y/z)."""
    return {
        axis: resolve_column(
            columns,
            FOOT_ACC_COLS[side][axis],
            _FOOT_ACC_PATTERNS[side][axis],
        )
        for axis in ("x", "y", "z")
    }


def resolve_drop_cols(columns: list[str]) -> list[str]:
    """드롭할 컬럼명 리스트를 반환한다."""
    drop: list[str] = []
    for c in columns:
        for pat in _DROP_PATTERNS:
            if pat.search(c):
                drop.append(c)
                break
    return drop


# ─────────────────────────────────────────────
# 8. 필터 파라미터
# ─────────────────────────────────────────────
BANDPASS_LOW: float  = 1.0
BANDPASS_HIGH: float = 20.0
BANDPASS_ORDER: int  = 4

# ─────────────────────────────────────────────
# 9. 학습 파라미터
# ─────────────────────────────────────────────
KFOLD: int      = 5
EPOCHS: int     = 200
EARLY_STOP: int = 20
LR: float       = 3e-4
MIN_LR: float   = 1e-6

if USE_GPU and GPU_TOTAL_MEM_GB >= 32:
    BATCH = 256
elif USE_GPU and GPU_TOTAL_MEM_GB >= 16:
    BATCH = 192
elif USE_GPU:
    BATCH = 128
else:
    BATCH = 128

WEIGHT_DECAY: float       = 1e-3
DROPOUT_CLF: float        = 0.5
DROPOUT_FEAT: float       = 0.3
LABEL_SMOOTH: float       = 0.1
MIXUP_ALPHA: float        = 0.2
USE_FOCAL_LOSS: bool      = True
FOCAL_GAMMA: float        = 2.0
USE_FFT_BRANCH: bool      = True
FFT_SOURCE_GROUP: str     = "Foot"
USE_BALANCED_SAMPLER: bool = True
USE_TTA: bool             = True
TTA_ROUNDS: int           = 5
GRAD_CLIP_NORM: float     = 1.0
GRAD_ACCUM_STEPS: int     = 1
USE_COMPILE: bool         = hasattr(torch, "compile") and USE_GPU
AUG_NOISE: float          = 0.03
AUG_SCALE: float          = 0.15
AUG_SHIFT: int            = 15
AUG_MASK_RATIO: float     = 0.05

# ─────────────────────────────────────────────
# 10. 모델 파라미터
# ─────────────────────────────────────────────
FEAT_DIM: int        = 256
SE_REDUCTION: int    = 16
CROSS_N_HEADS: int   = 4
CROSS_DROPOUT: float = 0.1

# ─────────────────────────────────────────────
# 11. 경로
# ─────────────────────────────────────────────
ROOT: Path      = Path(__file__).resolve().parent
DATA_DIR: Path  = Path("/home/ubuntu/project/data/raw_csv")
BATCH_DIR: Path = ROOT / "batches"
H5_PATH: Path   = BATCH_DIR / "dataset.h5"

OUT_DIR: Path      = ROOT / f"out_N{N_SUBJECTS}"
RESULT_KFOLD: Path = OUT_DIR / "kfold"
RESULT_LOSO: Path  = OUT_DIR / "loso"

for _d in [BATCH_DIR, OUT_DIR, RESULT_KFOLD, RESULT_LOSO]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 12. CLI 오버라이드
# ─────────────────────────────────────────────

def apply_overrides(
    n_subjects: int | None = None,
    seed: int | None = None,
    batch: int | None = None,
    epochs: int | None = None,
    focal: bool | None = None,
    fft: bool | None = None,
    balanced: bool | None = None,
    tta: bool | None = None,
) -> None:
    """CLI 인자로 전역 설정을 런타임에 변경한다."""
    import config as _cfg
    if n_subjects is not None:
        _cfg.N_SUBJECTS   = n_subjects
        _cfg.OUT_DIR      = _cfg.ROOT / f"out_N{n_subjects}"
        _cfg.RESULT_KFOLD = _cfg.OUT_DIR / "kfold"
        _cfg.RESULT_LOSO  = _cfg.OUT_DIR / "loso"
        for _d in [_cfg.OUT_DIR, _cfg.RESULT_KFOLD, _cfg.RESULT_LOSO]:
            _d.mkdir(parents=True, exist_ok=True)
    if seed     is not None: _cfg.SEED              = seed
    if batch    is not None: _cfg.BATCH             = batch
    if epochs   is not None: _cfg.EPOCHS            = epochs
    if focal    is not None: _cfg.USE_FOCAL_LOSS    = focal
    if fft      is not None: _cfg.USE_FFT_BRANCH    = fft
    if balanced is not None: _cfg.USE_BALANCED_SAMPLER = balanced
    if tta      is not None: _cfg.USE_TTA           = tta


# ─────────────────────────────────────────────
# 13. Git commit hash
# ─────────────────────────────────────────────

def _get_git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────
# 14. Config 스냅샷 저장
# ─────────────────────────────────────────────

def snapshot(out_dir: Path | None = None) -> dict:
    """현재 config 상태를 dict로 반환하고, out_dir이 주어지면 JSON 저장."""
    import config as _cfg
    snap = {
        "git_hash":          _get_git_hash(),
        "device":            str(_cfg.DEVICE),
        "device_name":       _cfg.DEVICE_NAME,
        "n_gpu":             _cfg.N_GPU,
        "gpu_mem_gb":        round(_cfg.GPU_MEM_GB, 1),
        "gpu_total_mem_gb":  round(_cfg.GPU_TOTAL_MEM_GB, 1),
        "ram_gib":           _cfg.RAM_GIB,
        "vcpu":              _cfg.VCPU,
        "use_preload":       _cfg.USE_PRELOAD,
        "use_preload_m1":    _cfg.USE_PRELOAD_M1,
        "use_amp":           _cfg.USE_AMP,
        "amp_dtype":         str(_cfg.AMP_DTYPE),
        "loader_workers":    _cfg.LOADER_WORKERS,
        "n_subjects":        _cfg.N_SUBJECTS,
        "num_classes":       _cfg.NUM_CLASSES,
        "ts":                _cfg.TS,
        "pca_ch":            _cfg.PCA_CH,
        "sample_rate":       _cfg.SAMPLE_RATE,
        "seed":              _cfg.SEED,
        "min_step_len":      _cfg.MIN_STEP_LEN,
        # HS 검출 (Niswander v8.2)
        "hs_algorithm":         "Niswander_2021_Foot_IMU",
        "hs_gyro_sensor":       _cfg.HS_GYRO_SENSOR,
        "hs_gyro_axis":         _cfg.HS_GYRO_AXIS,
        "hs_accel_sensor":      _cfg.HS_ACCEL_SENSOR,
        "hs_accel_axis":        _cfg.HS_ACCEL_AXIS,
        "hs_accel_peak_sigma":  _cfg.HS_ACCEL_PEAK_SIGMA,
        "hs_gyro_valid_sigma":  _cfg.HS_GYRO_VALID_SIGMA,
        "hs_fusion_window_ms":  _cfg.HS_FUSION_WINDOW_MS,
        # 학습
        "kfold":                _cfg.KFOLD,
        "epochs":               _cfg.EPOCHS,
        "early_stop":           _cfg.EARLY_STOP,
        "batch":                _cfg.BATCH,
        "lr":                   _cfg.LR,
        "min_lr":               _cfg.MIN_LR,
        "weight_decay":         _cfg.WEIGHT_DECAY,
        "label_smooth":         _cfg.LABEL_SMOOTH,
        "mixup_alpha":          _cfg.MIXUP_ALPHA,
        "grad_clip_norm":       _cfg.GRAD_CLIP_NORM,
        "grad_accum_steps":     _cfg.GRAD_ACCUM_STEPS,
        "use_compile":          _cfg.USE_COMPILE,
        "use_focal_loss":       _cfg.USE_FOCAL_LOSS,
        "focal_gamma":          _cfg.FOCAL_GAMMA,
        "use_fft_branch":       _cfg.USE_FFT_BRANCH,
        "use_balanced_sampler": _cfg.USE_BALANCED_SAMPLER,
        "use_tta":              _cfg.USE_TTA,
        "tta_rounds":           _cfg.TTA_ROUNDS,
        "dropout_clf":          _cfg.DROPOUT_CLF,
        "dropout_feat":         _cfg.DROPOUT_FEAT,
        "aug_noise":            _cfg.AUG_NOISE,
        "aug_scale":            _cfg.AUG_SCALE,
        "aug_shift":            _cfg.AUG_SHIFT,
        "aug_mask_ratio":       _cfg.AUG_MASK_RATIO,
        "feat_dim":             _cfg.FEAT_DIM,
        "se_reduction":         _cfg.SE_REDUCTION,
        "cross_n_heads":        _cfg.CROSS_N_HEADS,
        "cross_dropout":        _cfg.CROSS_DROPOUT,
        "data_dir":             str(_cfg.DATA_DIR),
        "h5_path":              str(_cfg.H5_PATH),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_snapshot.json").write_text(
            json.dumps(snap, indent=2, cls=NumpyEncoder, ensure_ascii=False)
        )
    return snap


# ─────────────────────────────────────────────
# 15. 설정 출력
# ─────────────────────────────────────────────

def print_config() -> None:
    """현재 설정을 콘솔에 출력한다."""
    h5_gb    = H5_PATH.stat().st_size / 1024**3 if H5_PATH.exists() else 0
    amp_name = "BF16" if AMP_DTYPE == torch.bfloat16 else ("FP16" if USE_AMP else "OFF")
    print(f"{'='*60}")
    print(f"  Config v8.2 — {'GPU' if USE_GPU else 'CPU'} 모드  Git:{_get_git_hash()}")
    print(f"{'='*60}")
    print(f"  Device : {DEVICE} ({DEVICE_NAME})")
    if USE_GPU:
        _cc = torch.cuda.get_device_capability(0)
        print(f"  GPU    : {GPU_MEM_GB:.0f}GB×{N_GPU}  AMP={amp_name}  cc={_cc[0]}.{_cc[1]}")
    print(f"  vCPU={VCPU}  RAM={RAM_GIB}GiB  HDF5={h5_gb:.1f}GB")
    print(f"  N={N_SUBJECTS}  Classes={NUM_CLASSES}  TS={TS}  Rate={SAMPLE_RATE}Hz")
    print(f"  Batch={BATCH}  Epochs={EPOCHS}  ES={EARLY_STOP}  LR={LR}→{MIN_LR}")
    print(f"  HS [Niswander2021]: "
          f"Foot-Accel{HS_ACCEL_AXIS}(mean+{HS_ACCEL_PEAK_SIGMA}σ) "
          f"+ Foot-Gyro{HS_GYRO_AXIS}(>{HS_GYRO_VALID_SIGMA}σ) "
          f"window=±{HS_FUSION_WINDOW_MS}ms")
    print(f"  Focal={'ON γ='+str(FOCAL_GAMMA) if USE_FOCAL_LOSS else 'OFF'}"
          f"  FFT={'ON('+FFT_SOURCE_GROUP+')' if USE_FFT_BRANCH else 'OFF'}"
          f"  TTA={'ON×'+str(TTA_ROUNDS) if USE_TTA else 'OFF'}")
    print(f"{'='*60}\n")