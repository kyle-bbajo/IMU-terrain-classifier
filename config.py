"""
config.py — 전역 설정 (v8 Final)
═══════════════════════════════════════════════════════
v7→v8 변경사항
  ★ import 시 자동 검증 (validate_config)
  ★ 재현성: SEED 중앙 관리 + set_seed()
  ★ Gradient Accumulation → 저 VRAM에서 effective batch 유지
  ★ 모든 상수에 유효 범위 검증
  ★ 불변 설정은 Final 표시
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import sys
import torch
from pathlib import Path
from typing import Final

# ─────────────────────────────────────────────
# 1. 재현성
# ─────────────────────────────────────────────
SEED: Final[int] = 42

# ─────────────────────────────────────────────
# 2. 디바이스 자동 감지
# ─────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE: torch.device = torch.device("cuda")
    DEVICE_NAME: str     = torch.cuda.get_device_name(0)
    GPU_MEM_GB: float    = torch.cuda.get_device_properties(0).total_memory / 1024**3
    USE_GPU: bool        = True
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
else:
    DEVICE      = torch.device("cpu")
    DEVICE_NAME = "CPU"
    GPU_MEM_GB  = 0.0
    USE_GPU     = False

# ─────────────────────────────────────────────
# 3. 하드웨어
# ─────────────────────────────────────────────
VCPU: int = os.cpu_count() or 4
try:
    RAM_GIB: int = round(
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    )
except (ValueError, OSError, AttributeError):
    RAM_GIB = 16

# ─────────────────────────────────────────────
# 4. 데이터 로딩 전략
# ─────────────────────────────────────────────
PRELOAD_RAM_THRESHOLD: Final[int] = 48
USE_PRELOAD: bool = RAM_GIB >= PRELOAD_RAM_THRESHOLD

# ─────────────────────────────────────────────
# 5. 워커
# ─────────────────────────────────────────────
if USE_GPU:
    TORCH_THREADS: int   = 2
    LOADER_WORKERS: int  = 0 if not USE_PRELOAD else min(2, VCPU)
    PREPROC_WORKERS: int = max(2, VCPU - 1)
else:
    TORCH_THREADS   = max(4, VCPU - 4)
    LOADER_WORKERS  = 0 if not USE_PRELOAD else min(4, max(2, VCPU // 4))
    PREPROC_WORKERS = max(4, VCPU - 2)

os.environ["OMP_NUM_THREADS"]      = str(TORCH_THREADS)
os.environ["MKL_NUM_THREADS"]      = str(TORCH_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(TORCH_THREADS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────
# 6. Mixed Precision
# ─────────────────────────────────────────────
USE_AMP: bool          = USE_GPU
AMP_DTYPE: torch.dtype = torch.float16 if USE_GPU else torch.float32

# ─────────────────────────────────────────────
# 7. 실험 파라미터
# ─────────────────────────────────────────────
N_SUBJECTS: int        = 40
NUM_CLASSES: Final[int] = 6
TS: Final[int]         = 256
PCA_CH: int            = 64
SAMPLE_RATE: Final[int] = 200

# ─────────────────────────────────────────────
# 8. 힐스트라이크 검출
# ─────────────────────────────────────────────
HS_MIN_STRIDE_MS: Final[int]     = 400
HS_MAX_STRIDE_MS: Final[int]     = 1800
HS_MIN_STRIDE_SAM: int           = int(HS_MIN_STRIDE_MS / 1000 * SAMPLE_RATE)
HS_MAX_STRIDE_SAM: int           = int(HS_MAX_STRIDE_MS / 1000 * SAMPLE_RATE)
HS_NAN_THRESHOLD: float          = 0.1
HS_PROMINENCE_COEFF: float       = 0.3
HS_PEAK_QUALITY_RATIO: float     = 0.6
HS_MIN_PEAK_RATIO_FOR_STATS: float = 0.5

FOOT_ACC_COLS: Final[dict[str, dict[str, str]]] = {
    "LT": {"x": "Foot Accel Sensor X LT (mG)",
            "y": "Foot Accel Sensor Y LT (mG)",
            "z": "Foot Accel Sensor Z LT (mG)"},
    "RT": {"x": "Foot Accel Sensor X RT (mG)",
            "y": "Foot Accel Sensor Y RT (mG)",
            "z": "Foot Accel Sensor Z RT (mG)"},
}
FOOT_CONTACT_COLS: Final[dict[str, str]] = {
    "LT": "Noraxon MyoMotion-Segments-Foot LT-Contact",
    "RT": "Noraxon MyoMotion-Segments-Foot RT-Contact",
}
DROP_COLS: Final[list[str]] = ["time", "Activity", "Marker"]

# ─────────────────────────────────────────────
# 9. 전처리 청크 크기
# ─────────────────────────────────────────────
H5_READ_CHUNK: int  = 2000
IPCA_CHUNK: int     = 5000
FLUSH_SIZE: int     = 100
DS_CHUNK: int       = 5000

BANDPASS_LOW: float  = 1.0
BANDPASS_HIGH: float = 50.0
BANDPASS_ORDER: int  = 4

# ─────────────────────────────────────────────
# 10. 모델 아키텍처
# ─────────────────────────────────────────────
FEAT_DIM: int        = 128
SE_REDUCTION: int    = 8
CROSS_N_HEADS: int   = 4
CROSS_DROPOUT: float = 0.1

# ─────────────────────────────────────────────
# 11. 학습 하이퍼파라미터
# ─────────────────────────────────────────────
KFOLD: int      = 5
EPOCHS: int     = 50
EARLY_STOP: int = 7
LR: float       = 1e-3
MIN_LR: float   = 1e-6

if USE_GPU:
    if GPU_MEM_GB >= 40:
        BATCH: int = 512
    elif GPU_MEM_GB >= 20:
        BATCH = 128
    else:
        BATCH = 64
else:
    BATCH = 64

GRAD_ACCUM_STEPS: int = 1

WEIGHT_DECAY: float   = 1e-3
DROPOUT_CLF: float    = 0.5
DROPOUT_FEAT: float   = 0.3
LABEL_SMOOTH: float   = 0.1
MIXUP_ALPHA: float    = 0.2
GRAD_CLIP_NORM: float = 1.0
USE_COMPILE: bool     = False

AUG_NOISE: float      = 0.03
AUG_SCALE: float      = 0.15
AUG_SHIFT: int        = 15
AUG_MASK_RATIO: float = 0.05

# ─────────────────────────────────────────────
# 12. 경로
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
# 13. 검증
# ─────────────────────────────────────────────
def validate_config() -> None:
    """설정값 범위와 일관성을 검증한다."""
    errors: list[str] = []

    for name, val, lo, hi in [
        ("N_SUBJECTS",       N_SUBJECTS,       1,    500),
        ("NUM_CLASSES",      NUM_CLASSES,       2,    100),
        ("TS",               TS,               16,   4096),
        ("PCA_CH",           PCA_CH,            4,    1024),
        ("SAMPLE_RATE",      SAMPLE_RATE,      50,   10000),
        ("BATCH",            BATCH,             1,    4096),
        ("EPOCHS",           EPOCHS,            1,    1000),
        ("EARLY_STOP",       EARLY_STOP,        1,    EPOCHS),
        ("KFOLD",            KFOLD,             2,    100),
        ("H5_READ_CHUNK",    H5_READ_CHUNK,     1,    100000),
        ("IPCA_CHUNK",       IPCA_CHUNK,       100,   100000),
        ("FLUSH_SIZE",       FLUSH_SIZE,         1,   10000),
        ("FEAT_DIM",         FEAT_DIM,           8,   2048),
        ("SE_REDUCTION",     SE_REDUCTION,       1,   64),
        ("GRAD_ACCUM_STEPS", GRAD_ACCUM_STEPS,   1,  128),
        ("BANDPASS_ORDER",   BANDPASS_ORDER,      1,  10),
    ]:
        if not (lo <= val <= hi):
            errors.append(f"{name}={val} 범위 밖 [{lo}, {hi}]")

    for name, val in [
        ("DROPOUT_CLF",        DROPOUT_CLF),
        ("DROPOUT_FEAT",       DROPOUT_FEAT),
        ("LABEL_SMOOTH",       LABEL_SMOOTH),
        ("HS_NAN_THRESHOLD",   HS_NAN_THRESHOLD),
        ("HS_PEAK_QUALITY_RATIO", HS_PEAK_QUALITY_RATIO),
        ("AUG_MASK_RATIO",     AUG_MASK_RATIO),
        ("CROSS_DROPOUT",      CROSS_DROPOUT),
    ]:
        if not (0.0 <= val <= 1.0):
            errors.append(f"{name}={val} 범위 밖 [0.0, 1.0]")

    for name, val in [
        ("LR", LR), ("MIN_LR", MIN_LR),
        ("WEIGHT_DECAY", WEIGHT_DECAY), ("GRAD_CLIP_NORM", GRAD_CLIP_NORM),
    ]:
        if val <= 0:
            errors.append(f"{name}={val} 양수여야 합니다")

    if LR <= MIN_LR:
        errors.append(f"LR({LR}) > MIN_LR({MIN_LR})")
    if HS_MIN_STRIDE_SAM >= HS_MAX_STRIDE_SAM:
        errors.append(f"HS_MIN({HS_MIN_STRIDE_SAM}) < HS_MAX({HS_MAX_STRIDE_SAM})")
    if BANDPASS_LOW >= BANDPASS_HIGH:
        errors.append(f"BANDPASS_LOW({BANDPASS_LOW}) < HIGH({BANDPASS_HIGH})")
    if BANDPASS_HIGH >= SAMPLE_RATE / 2:
        errors.append(f"BANDPASS_HIGH({BANDPASS_HIGH}) < Nyquist({SAMPLE_RATE/2})")

    if errors:
        raise ValueError("Config 검증 실패:\n  " + "\n  ".join(errors))


def set_seed(seed: int = SEED) -> None:
    """전역 랜덤 시드를 설정한다."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_config() -> None:
    """현재 설정을 콘솔에 출력한다."""
    h5_gb = H5_PATH.stat().st_size / 1024**3 if H5_PATH.exists() else 0
    strategy = "Preload" if USE_PRELOAD else "OTF"
    eff = BATCH * GRAD_ACCUM_STEPS
    print(f"{'='*60}")
    print(f"  Config v8 Final — {'GPU' if USE_GPU else 'CPU'}")
    print(f"{'='*60}")
    print(f"  Device:    {DEVICE}  ({DEVICE_NAME})")
    if USE_GPU:
        print(f"  GPU Mem:   {GPU_MEM_GB:.0f} GB  |  AMP={'FP16' if USE_AMP else 'OFF'}")
    print(f"  vCPU={VCPU}  RAM={RAM_GIB}GiB  HDF5={h5_gb:.1f}GB")
    print(f"  Strategy:  {strategy}  |  Seed: {SEED}")
    print(f"  N={N_SUBJECTS}  C={NUM_CLASSES}  TS={TS}  PCA={PCA_CH}")
    print(f"  Batch={BATCH}x{GRAD_ACCUM_STEPS}={eff}  Epochs={EPOCHS}  ES={EARLY_STOP}")
    print(f"  LR={LR}->{MIN_LR}  WD={WEIGHT_DECAY}  Clip={GRAD_CLIP_NORM}")
    print(f"  Drop: clf={DROPOUT_CLF} feat={DROPOUT_FEAT}")
    print(f"  LabelSmooth={LABEL_SMOOTH}  Mixup={MIXUP_ALPHA}")
    print(f"  Compile={USE_COMPILE}")
    print(f"{'='*60}\n")


# import 시 자동 검증
try:
    validate_config()
except ValueError as e:
    print(f"[FATAL] {e}", file=sys.stderr)
    sys.exit(1)