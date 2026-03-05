"""
config.py — 전역 설정 (v8.0)
═══════════════════════════════════════════════════════
★ 모든 하드코딩 값 중앙 관리
★ 스마트 듀얼 Preload: M1(항상) / M2–M6(RAM 여유 시)
★ AMP: GPU cc ≥ 8.0 → bfloat16, 그 외 → float16
★ apply_overrides(): CLI 인자로 런타임 변경
★ snapshot(): 실험 결과에 config + git hash 자동 저장
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import json
import subprocess
import torch
from pathlib import Path

# ─────────────────────────────────────────────
# 1. 디바이스 자동 감지 (Multi-GPU 지원)
# ─────────────────────────────────────────────
if torch.cuda.is_available():
    N_GPU: int                = torch.cuda.device_count()
    DEVICE: torch.device      = torch.device("cuda")
    DEVICE_NAME: str          = torch.cuda.get_device_name(0)
    GPU_MEM_GB: float         = torch.cuda.get_device_properties(0).total_memory / 1024**3
    GPU_TOTAL_MEM_GB: float   = GPU_MEM_GB * N_GPU     # 전체 GPU 메모리 합산
    USE_GPU: bool             = True
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
else:
    N_GPU           = 0
    DEVICE          = torch.device("cpu")
    DEVICE_NAME     = "CPU"
    GPU_MEM_GB      = 0.0
    GPU_TOTAL_MEM_GB = 0.0
    USE_GPU         = False

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
# 3. 데이터 로딩 전략 (스마트 듀얼 모드)
#    M1(PCA→64ch): ~2.5GB → 항상 Preload 가능
#    M2–M6(305ch): fp16 기준 ~11GB → RAM ≥ 24GB 시 Preload
#    can_preload_branch()로 동적 메모리 추정
# ─────────────────────────────────────────────
USE_PRELOAD: bool    = RAM_GIB >= 24       # M2–M6 Branch 전용
USE_PRELOAD_M1: bool = True                # M1 PCA 결과는 항상 RAM 적재 (~2.5GB)


def can_preload_branch(
    n_samples: int, n_channels: int, ts: int = 256,
) -> bool:
    """실제 fold 크기 기반으로 Preload 가능 여부를 동적 판단한다.

    fp16 저장 기준으로 추정하며, 시스템 여유 RAM의 80%까지 허용.

    Parameters
    ----------
    n_samples : int
        train + test 합산 샘플 수.
    n_channels : int
        채널 수 (보통 305).
    ts : int
        타임스텝 수 (기본 256).
    """
    bytes_fp16 = n_samples * n_channels * ts * 2   # fp16 = 2 bytes
    needed_gib = bytes_fp16 / 1024**3
    available  = RAM_GIB * 0.80                    # 80% 여유분
    return needed_gib < available

# ─────────────────────────────────────────────
# 4. 워커 설정 (CPU 코어 수에 비례 스케일링)
#    OTF + h5py → num_workers 반드시 0
#    Preload → 멀티워커 가능, GPU당 워커 배분
# ─────────────────────────────────────────────
if USE_GPU:
    TORCH_THREADS: int   = max(2, min(4, VCPU // (N_GPU or 1)))
    # GPU당 워커: Preload 모드에서 GPU 개수 고려
    _workers_per_gpu     = max(2, VCPU // max(N_GPU, 1) // 2)
    LOADER_WORKERS: int  = 0 if not USE_PRELOAD else min(8, _workers_per_gpu)
    PREPROC_WORKERS: int = max(2, VCPU - 1)
else:
    TORCH_THREADS   = max(4, VCPU - 4)
    LOADER_WORKERS  = 0 if not USE_PRELOAD else min(8, max(2, VCPU // 4))
    PREPROC_WORKERS = max(4, VCPU - 2)

os.environ["OMP_NUM_THREADS"]      = str(TORCH_THREADS)
os.environ["MKL_NUM_THREADS"]      = str(TORCH_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(TORCH_THREADS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────
# 5. Mixed Precision — GPU compute capability 기반 자동 선택
#    cc >= 8.0 (Ampere+): bfloat16 (수렴 안정성 + 동일 속도)
#    cc < 8.0: float16
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
NUM_CLASSES: int = 6    # ★ C1~C6만 사용. C7/C8 있으면 여기 수정 필요 (모델 출력층도 변경됨)
TS: int          = 256
PCA_CH: int      = 64
SAMPLE_RATE: int = 200
SEED: int        = 42

# ─────────────────────────────────────────────
# 7. 힐스트라이크 검출 파라미터
# ─────────────────────────────────────────────
HS_MIN_STRIDE_MS: int      = 400
HS_MAX_STRIDE_MS: int      = 1800
HS_MIN_STRIDE_SAM: int     = int(HS_MIN_STRIDE_MS / 1000 * SAMPLE_RATE)
HS_MAX_STRIDE_SAM: int     = int(HS_MAX_STRIDE_MS / 1000 * SAMPLE_RATE)
HS_NAN_THRESHOLD: float    = 0.1
HS_PROMINENCE_COEFF: float = 0.3
HS_PEAK_QUALITY_RATIO: float = 0.6
MIN_STEP_LEN: int = HS_MIN_STRIDE_SAM  # 최소 스텝 길이 (80 samples = 400ms)

FOOT_ACC_COLS: dict[str, dict[str, str]] = {
    "LT": {"x": "Foot Accel Sensor X LT (mG)",
            "y": "Foot Accel Sensor Y LT (mG)",
            "z": "Foot Accel Sensor Z LT (mG)"},
    "RT": {"x": "Foot Accel Sensor X RT (mG)",
            "y": "Foot Accel Sensor Y RT (mG)",
            "z": "Foot Accel Sensor Z RT (mG)"},
}
FOOT_CONTACT_COLS: dict[str, str] = {
    "LT": "Noraxon MyoMotion-Segments-Foot LT-Contact",
    "RT": "Noraxon MyoMotion-Segments-Foot RT-Contact",
}
DROP_COLS: list[str] = ["time", "Activity", "Marker"]

# ─────────────────────────────────────────────
# 7-b. 컬럼명 유연 매칭 시스템
#      Noraxon 버전, 내보내기 설정, 단위 표기 차이 대응
#      예: "Foot Accel Sensor X LT (mG)" ↔ "Foot Acceleration X LT(mG)"
# ─────────────────────────────────────────────
import re as _re

# 발 가속도 컬럼 패턴 (x/y/z × LT/RT = 6개)
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

# 발 접지 컬럼 패턴 (LT/RT = 2개)
_FOOT_CONTACT_PATTERNS: dict[str, _re.Pattern] = {
    "LT": _re.compile(r"(?i)(foot|ft).*lt.*contact|contact.*lt.*(foot|ft)"),
    "RT": _re.compile(r"(?i)(foot|ft).*rt.*contact|contact.*rt.*(foot|ft)"),
}

# 드롭 컬럼 패턴
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
    """컬럼명을 정확 매칭 → 패턴 매칭 → 부분 매칭 순서로 찾는다.

    Parameters
    ----------
    columns : list[str]
        CSV 컬럼 리스트.
    exact_name : str
        먼저 시도할 정확한 컬럼명.
    pattern : re.Pattern, optional
        정확 매칭 실패 시 사용할 정규식.

    Returns
    -------
    str
        매칭된 컬럼명.

    Raises
    ------
    KeyError
        매칭 컬럼을 찾지 못했을 때.
    """
    # 1차: 정확 매칭
    if exact_name in columns:
        return exact_name

    # 2차: 정규식 패턴 매칭
    if pattern is not None:
        matches = [c for c in columns if pattern.search(c)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # 가장 짧은 이름 = 가장 정확한 매칭
            return min(matches, key=len)

    # 3차: 핵심 키워드 부분 매칭 (공백/특수문자 무시)
    norm = _re.sub(r"[^a-z0-9]", "", exact_name.lower())
    for c in columns:
        if _re.sub(r"[^a-z0-9]", "", c.lower()) == norm:
            return c

    raise KeyError(
        f"컬럼 '{exact_name}' 매칭 실패. "
        f"사용 가능: {[c for c in columns if any(k in c.lower() for k in exact_name.lower().split()[:2])]}"
    )


def resolve_foot_acc_cols(
    columns: list[str], side: str,
) -> dict[str, str]:
    """발 가속도 x/y/z 컬럼명을 자동 매칭한다."""
    exact = FOOT_ACC_COLS[side]
    patterns = _FOOT_ACC_PATTERNS[side]
    return {
        axis: resolve_column(columns, exact[axis], patterns[axis])
        for axis in ("x", "y", "z")
    }


def resolve_foot_contact_col(
    columns: list[str], side: str,
) -> str:
    """발 접지 컬럼명을 자동 매칭한다."""
    return resolve_column(
        columns, FOOT_CONTACT_COLS[side], _FOOT_CONTACT_PATTERNS[side],
    )


def resolve_drop_cols(columns: list[str]) -> list[str]:
    """드롭할 컬럼명을 유연하게 매칭한다."""
    drops: list[str] = []
    for c in columns:
        # 정확 매칭
        if c in DROP_COLS:
            drops.append(c)
            continue
        # 패턴 매칭
        for pat in _DROP_PATTERNS:
            if pat.search(c):
                drops.append(c)
                break
    return drops

# ─────────────────────────────────────────────
# 8. 전처리 청크 크기
# ─────────────────────────────────────────────
H5_READ_CHUNK: int = 2000
IPCA_CHUNK: int    = 5000
FLUSH_SIZE: int    = 100
DS_CHUNK: int      = 5000

BANDPASS_LOW: float  = 1.0
BANDPASS_HIGH: float = 50.0
BANDPASS_ORDER: int  = 4

# ─────────────────────────────────────────────
# 9. 모델 아키텍처 파라미터
# ─────────────────────────────────────────────
FEAT_DIM: int        = 128
SE_REDUCTION: int    = 8
CROSS_N_HEADS: int   = 4
CROSS_DROPOUT: float = 0.1

# ─────────────────────────────────────────────
# 10. 학습 하이퍼파라미터
# ─────────────────────────────────────────────
KFOLD: int      = 5
EPOCHS: int     = 50
EARLY_STOP: int = 7
LR: float       = 1e-3
MIN_LR: float   = 1e-6

if USE_GPU:
    # 단일 GPU 기준 배치 → N_GPU 배수로 스케일링
    _base_batch: int = 1024 if GPU_MEM_GB >= 40 else (256 if GPU_MEM_GB >= 20 else 64)
    BATCH: int = _base_batch * max(1, N_GPU)
else:
    BATCH = 64

WEIGHT_DECAY: float    = 1e-3
DROPOUT_CLF: float     = 0.5
DROPOUT_FEAT: float    = 0.3
LABEL_SMOOTH: float    = 0.1
MIXUP_ALPHA: float     = 0.2

# Focal Loss: 어려운 샘플(미끄러운↔정상 등)에 집중
USE_FOCAL_LOSS: bool   = True
FOCAL_GAMMA: float     = 2.0     # 높을수록 어려운 샘플에 집중 (0이면 CE와 동일)

# 주파수 Branch: 표면 질감 차이 포착 (Foot FFT)
USE_FFT_BRANCH: bool   = True
FFT_SOURCE_GROUP: str  = "Foot"  # FFT를 적용할 소스 그룹

# 클래스 균형 샘플링: 소수 클래스(C0/C5) 오버샘플링
USE_BALANCED_SAMPLER: bool = True

# TTA (Test Time Augmentation): 평가 시 여러 변형으로 예측 후 평균
USE_TTA: bool     = True
TTA_ROUNDS: int   = 5       # TTA 횟수 (1이면 TTA 없음과 동일)
GRAD_CLIP_NORM: float  = 1.0
GRAD_ACCUM_STEPS: int  = 1
# PyTorch 2.x: torch.compile로 커널 퓨전 → 10~30% 속도 향상
USE_COMPILE: bool      = hasattr(torch, "compile") and USE_GPU

AUG_NOISE: float      = 0.03
AUG_SCALE: float      = 0.15
AUG_SHIFT: int        = 15
AUG_MASK_RATIO: float = 0.05

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
    """CLI 인자로 전역 설정을 런타임에 변경한다.

    Parameters
    ----------
    focal : bool, optional
        Focal Loss ON/OFF. None이면 기존 값 유지.
    fft : bool, optional
        FFT Branch ON/OFF.
    balanced : bool, optional
        클래스 균형 샘플링 ON/OFF.
    tta : bool, optional
        Test Time Augmentation ON/OFF.
    """
    import config as _cfg
    if n_subjects is not None:
        _cfg.N_SUBJECTS = n_subjects
        _cfg.OUT_DIR      = _cfg.ROOT / f"out_N{n_subjects}"
        _cfg.RESULT_KFOLD = _cfg.OUT_DIR / "kfold"
        _cfg.RESULT_LOSO  = _cfg.OUT_DIR / "loso"
        for _d in [_cfg.OUT_DIR, _cfg.RESULT_KFOLD, _cfg.RESULT_LOSO]:
            _d.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        _cfg.SEED = seed
    if batch is not None:
        _cfg.BATCH = batch
    if epochs is not None:
        _cfg.EPOCHS = epochs
    if focal is not None:
        _cfg.USE_FOCAL_LOSS = focal
    if fft is not None:
        _cfg.USE_FFT_BRANCH = fft
    if balanced is not None:
        _cfg.USE_BALANCED_SAMPLER = balanced
    if tta is not None:
        _cfg.USE_TTA = tta


# ─────────────────────────────────────────────
# 13. Git commit hash
# ─────────────────────────────────────────────

def _get_git_hash() -> str:
    """현재 git commit hash를 반환한다."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────
# 14. Config 스냅샷 저장
# ─────────────────────────────────────────────

def snapshot(out_dir: Path | None = None) -> dict:
    """현재 config 상태를 dict로 반환하고, out_dir이 주어지면 JSON 저장."""
    import config as _cfg
    snap = {
        # 환경
        "git_hash": _get_git_hash(),
        "device": str(_cfg.DEVICE), "device_name": _cfg.DEVICE_NAME,
        "n_gpu": _cfg.N_GPU,
        "gpu_mem_gb": round(_cfg.GPU_MEM_GB, 1),
        "gpu_total_mem_gb": round(_cfg.GPU_TOTAL_MEM_GB, 1),
        "ram_gib": _cfg.RAM_GIB, "vcpu": _cfg.VCPU,
        # 전략
        "use_preload": _cfg.USE_PRELOAD, "use_preload_m1": _cfg.USE_PRELOAD_M1,
        "use_amp": _cfg.USE_AMP, "amp_dtype": str(_cfg.AMP_DTYPE),
        "loader_workers": _cfg.LOADER_WORKERS,
        # 실험
        "n_subjects": _cfg.N_SUBJECTS, "num_classes": _cfg.NUM_CLASSES,
        "ts": _cfg.TS, "pca_ch": _cfg.PCA_CH,
        "sample_rate": _cfg.SAMPLE_RATE, "seed": _cfg.SEED,
        "min_step_len": _cfg.MIN_STEP_LEN,
        # 학습
        "kfold": _cfg.KFOLD, "epochs": _cfg.EPOCHS, "early_stop": _cfg.EARLY_STOP,
        "batch": _cfg.BATCH, "lr": _cfg.LR, "min_lr": _cfg.MIN_LR,
        "weight_decay": _cfg.WEIGHT_DECAY, "label_smooth": _cfg.LABEL_SMOOTH,
        "mixup_alpha": _cfg.MIXUP_ALPHA, "grad_clip_norm": _cfg.GRAD_CLIP_NORM,
        "grad_accum_steps": _cfg.GRAD_ACCUM_STEPS, "use_compile": _cfg.USE_COMPILE,
        "use_focal_loss": _cfg.USE_FOCAL_LOSS, "focal_gamma": _cfg.FOCAL_GAMMA,
        "use_fft_branch": _cfg.USE_FFT_BRANCH,
        "use_balanced_sampler": _cfg.USE_BALANCED_SAMPLER,
        "use_tta": _cfg.USE_TTA, "tta_rounds": _cfg.TTA_ROUNDS,
        # 정규화
        "dropout_clf": _cfg.DROPOUT_CLF, "dropout_feat": _cfg.DROPOUT_FEAT,
        # 증강
        "aug_noise": _cfg.AUG_NOISE, "aug_scale": _cfg.AUG_SCALE,
        "aug_shift": _cfg.AUG_SHIFT, "aug_mask_ratio": _cfg.AUG_MASK_RATIO,
        # 모델
        "feat_dim": _cfg.FEAT_DIM, "se_reduction": _cfg.SE_REDUCTION,
        "cross_n_heads": _cfg.CROSS_N_HEADS, "cross_dropout": _cfg.CROSS_DROPOUT,
        # 경로
        "data_dir": str(_cfg.DATA_DIR), "h5_path": str(_cfg.H5_PATH),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_snapshot.json").write_text(
            json.dumps(snap, indent=2, ensure_ascii=False))
    return snap


# ─────────────────────────────────────────────
# 15. 설정 출력
# ─────────────────────────────────────────────

def print_config() -> None:
    """현재 설정을 콘솔에 출력한다."""
    h5_gb = H5_PATH.stat().st_size / 1024**3 if H5_PATH.exists() else 0
    amp_name = "BF16" if AMP_DTYPE == torch.bfloat16 else ("FP16" if USE_AMP else "OFF")
    strategy_m1 = "Preload (항상)" if USE_PRELOAD_M1 else "OTF"
    strategy_br = "Preload" if USE_PRELOAD else "OTF (auto)"
    print(f"{'='*60}")
    print(f"  Config v8.0 — {'GPU' if USE_GPU else 'CPU'} 모드")
    print(f"  Git: {_get_git_hash()}")
    print(f"{'='*60}")
    print(f"  Device:    {DEVICE}  ({DEVICE_NAME})")
    if USE_GPU:
        _cc = torch.cuda.get_device_capability(0)
        gpu_info = f"{GPU_MEM_GB:.0f}GB×{N_GPU}" if N_GPU > 1 else f"{GPU_MEM_GB:.0f}GB"
        print(f"  GPU:       {gpu_info}  |  AMP={amp_name}  cc={_cc[0]}.{_cc[1]}")
        if N_GPU > 1:
            print(f"  Multi-GPU: DataParallel ({N_GPU} GPUs, total {GPU_TOTAL_MEM_GB:.0f}GB)")
    print(f"  vCPU={VCPU}  RAM={RAM_GIB}GiB  HDF5={h5_gb:.1f}GB")
    print(f"  Strategy:  M1={strategy_m1}  Branch={strategy_br}  Workers={LOADER_WORKERS}")
    print(f"  Subjects:  N={N_SUBJECTS}  Classes={NUM_CLASSES}  Seed={SEED}")
    print(f"  Rate={SAMPLE_RATE}Hz  TS={TS}pt  PCA={PCA_CH}ch")
    print(f"  Batch={BATCH}  Epochs={EPOCHS}  ES={EARLY_STOP}")
    print(f"  LR={LR}→{MIN_LR}  WD={WEIGHT_DECAY}")
    print(f"  LabelSmooth={LABEL_SMOOTH}  Mixup={MIXUP_ALPHA}")
    print(f"  Dropout: clf={DROPOUT_CLF}  feat={DROPOUT_FEAT}")
    print(f"  Aug: noise={AUG_NOISE}  scale={AUG_SCALE}  shift={AUG_SHIFT}")
    print(f"  Compile={USE_COMPILE}")
    print(f"  FocalLoss={'ON γ='+str(FOCAL_GAMMA) if USE_FOCAL_LOSS else 'OFF'}")
    print(f"  FFT Branch={'ON ('+FFT_SOURCE_GROUP+')' if USE_FFT_BRANCH else 'OFF'}")
    print(f"  BalancedSampler={'ON' if USE_BALANCED_SAMPLER else 'OFF'}")
    print(f"  TTA={'ON ×'+str(TTA_ROUNDS) if USE_TTA else 'OFF'}")
    print(f"{'='*60}\n")