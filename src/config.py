"""
config.py — 전역 설정 (v9.1)
═══════════════════════════════════════════════════════
★ 모든 하드코딩 값 중앙 관리
★ dataclass 기반 ConfigState → 멀티프로세싱 안전
★ 스마트 듀얼 Preload 도구 제공: can_preload_branch() 구현됨
  └─ 실제 적용은 각 파이프라인 모듈에서 fold마다 호출해야 함
     (예: cfg.use_preload = can_preload_branch(n, ch, ts))
★ AMP: GPU cc ≥ 8.0 → bfloat16, 그 외 → float16
★ apply_overrides(): CLI 인자로 런타임 변경 (객체 기반, thread-safe)
★ snapshot(): 실험 결과에 config + git hash 자동 저장
★ 라벨 체계 명시: 0-based class label (C1→0, C2→1, ...)
★ 컬럼 다중 매칭 시 경고 로그 추가
★ DROP_COLS / _DROP_PATTERNS 단일 소스 통합
★ validate(): 설정 유효성 검사 내장 (확장)
★ 모듈 수준 상수 Deprecated 경고 추가
═══════════════════════════════════════════════════════

변경 이력 (v8.2 → v9.0 → v9.1)
──────────────────────────────────────────────────────
[v9.0 - FIX]  apply_overrides() — 전역 변수 직접 수정 → CFG 객체 속성 수정
              멀티프로세싱 spawn/fork 환경에서 설정 전파 보장
[v9.0 - FIX]  FOOT_Z_ACCEL_IDX / SHANK_Z_ACCEL_IDX — 의도 분리,
              주석으로 센서별 채널 인덱스 출처 명시
[v9.0 - FIX]  resolve_column() — 다중 매칭 시 조용히 최단 선택 → WARNING 로그
[v9.0 - FIX]  DROP_COLS + _DROP_PATTERNS 이중 관리 → _DROP_SPEC 단일 소스
[v9.0 - ADD]  ConfigState dataclass — 전체 설정을 타입 안전 객체로 관리
[v9.0 - ADD]  validate() — 설정 유효성 검사 (학습 시작 전 호출 권장)
[v9.0 - ADD]  get_label_map() — 라벨 매핑 dict 헬퍼
[v9.1 - FIX]  apply_overrides() 모듈 래퍼에 use_preload 파라미터 누락 수정
[v9.1 - FIX]  batch: int → Optional[int]; __post_init__ 에서 명시값 존중
              (ConfigState(batch=64) 시 64 유지, None 이면 _auto_batch() 적용)
[v9.1 - FIX]  validate() 누락 항목 추가: kfold, batch, epochs, grad_accum_steps,
              tta_rounds, label_base, dropout_feat, aug_mask_ratio, mixup_alpha,
              foot/shank z_accel_idx 범위 검사
[v9.1 - FIX]  모듈 수준 상수(BATCH 등)에 DeprecationWarning 추가
              → 신규 코드는 CFG.<attr> 사용 권장, v10에서 상수 제거 예정
[v9.1 - FIX]  docstring 수정: can_preload_branch() 연동 방식 명확화
[v9.1 - CLEAN] 미사용 asdict import 제거
[KEEP] 모든 기존 공개 인터페이스 (하위 호환)
"""
from __future__ import annotations

import logging
import os
import json
import re as _re
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
# 0. 로거 기본 설정 (호출자가 설정하지 않은 경우 대비)
# ══════════════════════════════════════════════════════
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ══════════════════════════════════════════════════════
# 1. 하드웨어 자동 감지 (모듈 수준 — 변경 불가 상수)
# ══════════════════════════════════════════════════════
if torch.cuda.is_available():
    N_GPU: int = torch.cuda.device_count()
    DEVICE: torch.device = torch.device("cuda")
    DEVICE_NAME: str = torch.cuda.get_device_name(0)
    GPU_MEM_GB: float = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    GPU_TOTAL_MEM_GB: float = GPU_MEM_GB * N_GPU
    USE_GPU: bool = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    N_GPU = 0
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"
    GPU_MEM_GB = 0.0
    GPU_TOTAL_MEM_GB = 0.0
    USE_GPU = False

# ══════════════════════════════════════════════════════
# 2. 시스템 리소스
# ══════════════════════════════════════════════════════
VCPU: int = os.cpu_count() or 4
try:
    RAM_GIB: int = round(
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    )
except (ValueError, OSError, AttributeError):
    RAM_GIB = 16

# ══════════════════════════════════════════════════════
# 3. Mixed Precision (모듈 수준 상수)
# ══════════════════════════════════════════════════════
USE_AMP: bool = USE_GPU
if USE_GPU:
    _cc = torch.cuda.get_device_capability(0)
    AMP_DTYPE: torch.dtype = torch.bfloat16 if _cc >= (8, 0) else torch.float16
else:
    AMP_DTYPE = torch.float32


# ══════════════════════════════════════════════════════
# 4. Preload 가능 여부 동적 판단
# ══════════════════════════════════════════════════════
def can_preload_branch(
    n_samples: int,
    n_channels: int,
    ts: int = 256,
    ram_gib: int = RAM_GIB,
) -> bool:
    """실제 fold 크기 기반으로 Preload 가능 여부를 동적 판단한다.

    fp16 저장 기준으로 추정하며, 시스템 여유 RAM의 80%까지 허용.

    Parameters
    ----------
    n_samples  : train + test 합산 샘플 수
    n_channels : 채널 수
    ts         : 타임스텝 수 (기본 256)
    ram_gib    : 판단 기준 RAM (기본값: 현재 시스템)
    """
    bytes_fp16 = n_samples * n_channels * ts * 2
    needed_gib = bytes_fp16 / 1024 ** 3
    available = ram_gib * 0.80
    feasible = needed_gib < available
    if not feasible:
        logger.warning(
            "Preload 불가: 필요 %.2fGiB > 허용 %.2fGiB (RAM %dGiB×0.8) "
            "→ OTF 모드로 자동 전환",
            needed_gib, available, ram_gib,
        )
    return feasible


# ══════════════════════════════════════════════════════
# 5. 워커 수 계산 헬퍼
# ══════════════════════════════════════════════════════
def _calc_workers(use_preload: bool) -> tuple[int, int, int]:
    """(TORCH_THREADS, LOADER_WORKERS, PREPROC_WORKERS) 반환."""
    if USE_GPU:
        n = max(1, N_GPU)
        torch_threads = max(2, min(4, VCPU // n))
        loader = 0 if not use_preload else min(8, max(2, VCPU // n // 2))
        preproc = max(2, VCPU - 1)
    else:
        torch_threads = max(4, VCPU - 4)
        loader = 0 if not use_preload else min(8, max(2, VCPU // 4))
        preproc = max(4, VCPU - 2)
    return torch_threads, loader, preproc


# ══════════════════════════════════════════════════════
# 6. 컬럼명 유연 매칭 — 단일 소스 정의
# ══════════════════════════════════════════════════════

# DROP 대상: (정확한 이름, 정규식 패턴) 한 곳에서 관리
_DROP_SPEC: list[tuple[str, _re.Pattern]] = [
    ("time",     _re.compile(r"(?i)^time$")),
    ("Activity", _re.compile(r"(?i)^activity$")),
    ("Marker",   _re.compile(r"(?i)^marker")),
]

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

FOOT_CONTACT_COLS: dict[str, str] = {
    "LT": "Noraxon MyoMotion-Segments-Foot LT-Contact",
    "RT": "Noraxon MyoMotion-Segments-Foot RT-Contact",
}

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

_FOOT_CONTACT_PATTERNS: dict[str, _re.Pattern] = {
    "LT": _re.compile(r"(?i)(foot|ft).*lt.*contact|contact.*lt.*(foot|ft)"),
    "RT": _re.compile(r"(?i)(foot|ft).*rt.*contact|contact.*rt.*(foot|ft)"),
}


def resolve_column(
    columns: list[str],
    exact_name: str,
    pattern: _re.Pattern | None = None,
) -> str:
    """컬럼명을 정확 매칭 → 패턴 매칭 → 정규화 부분 매칭 순서로 찾는다.

    다중 패턴 매칭 시 WARNING을 남기고 최단 이름을 선택한다.
    """
    # ① 정확 매칭
    if exact_name in columns:
        return exact_name

    # ② 정규식 패턴 매칭
    if pattern is not None:
        matches = [c for c in columns if pattern.search(c)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            chosen = min(matches, key=len)
            logger.warning(
                "컬럼 '%s': 패턴 매칭 후보 %d개 발견 %s → '%s' 선택 (최단 이름). "
                "의도한 컬럼인지 확인하세요.",
                exact_name, len(matches), matches, chosen,
            )
            return chosen

    # ③ 정규화 부분 매칭 (영숫자만 비교)
    norm = _re.sub(r"[^a-z0-9]", "", exact_name.lower())
    for c in columns:
        if _re.sub(r"[^a-z0-9]", "", c.lower()) == norm:
            return c

    candidates = [
        c for c in columns
        if any(k in c.lower() for k in exact_name.lower().split()[:2])
    ]
    raise KeyError(
        f"컬럼 '{exact_name}' 매칭 실패. "
        f"유사 후보: {candidates}"
    )


def resolve_foot_acc_cols(columns: list[str], side: str) -> dict[str, str]:
    """발 가속도 x/y/z 컬럼명을 자동 매칭한다."""
    exact = FOOT_ACC_COLS[side]
    patterns = _FOOT_ACC_PATTERNS[side]
    return {
        axis: resolve_column(columns, exact[axis], patterns[axis])
        for axis in ("x", "y", "z")
    }


def resolve_foot_contact_col(columns: list[str], side: str) -> str:
    """발 접지 컬럼명을 자동 매칭한다."""
    return resolve_column(
        columns, FOOT_CONTACT_COLS[side], _FOOT_CONTACT_PATTERNS[side],
    )


def resolve_drop_cols(columns: list[str]) -> list[str]:
    """드롭할 컬럼명을 단일 소스(_DROP_SPEC) 기반으로 유연하게 매칭한다."""
    drops: list[str] = []
    for c in columns:
        for exact_name, pat in _DROP_SPEC:
            if c == exact_name or pat.search(c):
                drops.append(c)
                break
    return drops


# 하위 호환: DROP_COLS 는 _DROP_SPEC 에서 파생
DROP_COLS: list[str] = [exact for exact, _ in _DROP_SPEC]


# ══════════════════════════════════════════════════════
# 7. ConfigState — 타입 안전 설정 객체 (멀티프로세싱 안전)
# ══════════════════════════════════════════════════════
@dataclass
class ConfigState:
    """실험 설정 전체를 담는 dataclass.

    모듈 수준 전역 변수 대신 이 객체를 worker에 넘기면
    spawn/fork 모두에서 설정이 올바르게 전파된다.
    """

    # ── 실험 파라미터 ──────────────────────────────────
    n_subjects: int = 50
    num_classes: int = 6
    ts: int = 256
    pca_ch: int = 32
    sample_rate: int = 200
    seed: int = 42

    # ── 라벨 체계 ──────────────────────────────────────
    label_base: int = 0
    label_semantics: str = "terrain_condition"

    # ── Preload 전략 ───────────────────────────────────
    # v9: USE_PRELOAD 를 하드코딩하지 않고 생성 시 동적 결정 가능
    use_preload: bool = True      # 전체 Preload 활성 여부
    use_preload_m1: bool = True   # M1(메인 모달) 항상 Preload

    # ── 힐스트라이크 검출 ──────────────────────────────
    hs_min_stride_ms: int = 400
    hs_max_stride_ms: int = 1800
    hs_nan_threshold: float = 0.1
    hs_prominence_coeff: float = 0.3
    hs_peak_quality_ratio: float = 0.6

    # Type 4 Gyro/Accel 융합
    hs_gyro_sensor: str = "Shank"
    hs_gyro_axis: str = "y"
    hs_accel_sensor: str = "Foot"
    hs_accel_axis: str = "x"
    hs_fusion_window_ms: int = 125
    hs_gyro_prominence: float = 0.4
    hs_accel_threshold: float = 0.8
    hs_trusted_swing: float = 0.2

    # ── 전처리 청크 ────────────────────────────────────
    h5_read_chunk: int = 2000
    ipca_chunk: int = 5000
    flush_size: int = 100
    ds_chunk: int = 5000
    bandpass_low: float = 1.0
    bandpass_high: float = 50.0
    bandpass_order: int = 4

    # ── 모델 아키텍처 ──────────────────────────────────
    feat_dim: int = 128
    se_reduction: int = 8
    cross_n_heads: int = 4
    cross_dropout: float = 0.1

    # 센서별 Z축 가속도 채널 인덱스
    # Foot: PCA 32ch 중 발 가속도 Z 채널 (0-based)
    foot_z_accel_idx: list[int] = field(default_factory=lambda: [2, 8])
    # Shank: PCA 32ch 중 정강이 가속도 Z 채널 (0-based)
    # ※ v8.2 에서 foot 과 동일값([2,8]) 이었으나 센서 위치가 다르므로 분리
    #   실제 PCA 채널 확인 후 수정 필요
    shank_z_accel_idx: list[int] = field(default_factory=lambda: [3, 9])

    # ── 학습 하이퍼파라미터 ────────────────────────────
    kfold: int = 5
    epochs: int = 50
    early_stop: int = 7
    lr: float = 1e-3
    min_lr: float = 1e-6
    batch: Optional[int] = None  # None이면 __post_init__에서 _auto_batch() 자동 결정
    weight_decay: float = 1e-3
    dropout_clf: float = 0.5
    dropout_feat: float = 0.3
    label_smooth: float = 0.1
    mixup_alpha: float = 0.2

    # ── 학습 기법 플래그 ───────────────────────────────
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    use_fft_branch: bool = True
    fft_source_group: str = "Foot"
    use_balanced_sampler: bool = True
    use_tta: bool = True
    tta_rounds: int = 5
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    use_compile: bool = field(default_factory=lambda: bool(
        hasattr(torch, "compile") and USE_GPU
    ))

    # ── 데이터 증강 ────────────────────────────────────
    aug_noise: float = 0.03
    aug_scale: float = 0.15
    aug_shift: int = 15
    aug_mask_ratio: float = 0.05

    # ── 경로 (문자열로 저장, Path 프로퍼티로 접근) ──────
    _root_str: str = field(default="", repr=False)
    _repo_str: str = field(default="", repr=False)
    _project_str: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        # 경로 자동 설정
        if not self._root_str:
            self._root_str = str(Path(__file__).resolve().parent)
        if not self._repo_str:
            self._repo_str = str(Path(self._root_str).parent)
        if not self._project_str:
            self._project_str = str(Path(self._repo_str).parent)

        # 배치 자동 결정 — 사용자가 명시한 경우 그 값을 존중
        if self.batch is None:
            self.batch = _auto_batch()

        # 파생 HS 파라미터
        self._refresh_hs()

        # 워커 수 결정
        self._refresh_workers()

        # 디렉터리 생성
        self._ensure_dirs()

    # ── 파생 파라미터 갱신 ─────────────────────────────
    def _refresh_hs(self) -> None:
        self.hs_min_stride_sam: int = int(
            self.hs_min_stride_ms / 1000 * self.sample_rate
        )
        self.hs_max_stride_sam: int = int(
            self.hs_max_stride_ms / 1000 * self.sample_rate
        )
        self.min_step_len: int = self.hs_min_stride_sam
        self.hs_fusion_window_sam: int = int(
            self.hs_fusion_window_ms / 1000 * self.sample_rate
        )

    def _refresh_workers(self) -> None:
        t, lw, pw = _calc_workers(self.use_preload)
        self.torch_threads: int = t
        self.loader_workers: int = lw
        self.preproc_workers: int = pw
        # 환경 변수 업데이트
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(t)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    def _ensure_dirs(self) -> None:
        for d in [
            self.batch_dir,
            self.out_dir,
            self.result_kfold,
            self.result_loso,
            self.log_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ── 경로 프로퍼티 ─────────────────────────────────
    @property
    def root(self) -> Path:
        return Path(self._root_str)

    @property
    def repo_dir(self) -> Path:
        return Path(self._repo_str)

    @property
    def project_dir(self) -> Path:
        return Path(self._project_str)

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data" / "raw_csv"

    @property
    def batch_dir(self) -> Path:
        return self.project_dir / "data" / "processed" / "batches"

    @property
    def h5_path(self) -> Path:
        return self.batch_dir / "dataset.h5"

    @property
    def out_dir(self) -> Path:
        return self.repo_dir / f"out_N{self.n_subjects}"

    @property
    def result_kfold(self) -> Path:
        return self.out_dir / "kfold"

    @property
    def result_loso(self) -> Path:
        return self.out_dir / "loso"

    @property
    def log_dir(self) -> Path:
        return self.repo_dir / "logs"

    # ── 헬퍼 ──────────────────────────────────────────
    def get_label_map(self) -> dict[str, int]:
        """{'C1': 0, 'C2': 1, ...} 형태의 라벨 매핑을 반환한다."""
        return {
            f"C{i + 1}": i + self.label_base
            for i in range(self.num_classes)
        }

    def validate(self) -> None:
        """설정 유효성을 검사한다. 이상 시 ValueError를 발생시킨다."""
        errors: list[str] = []

        # ── 기본 파라미터 ─────────────────────────────
        if self.num_classes < 2:
            errors.append(f"num_classes={self.num_classes} < 2")
        if self.ts <= 0:
            errors.append(f"ts={self.ts} ≤ 0")
        if self.pca_ch <= 0:
            errors.append(f"pca_ch={self.pca_ch} ≤ 0")
        if self.label_base < 0:
            errors.append(f"label_base={self.label_base} < 0")

        # ── 학습 루프 ─────────────────────────────────
        if self.kfold < 2:
            errors.append(f"kfold={self.kfold} < 2")
        if self.batch is None or self.batch <= 0:
            errors.append(f"batch={self.batch} ≤ 0")
        if self.epochs <= 0:
            errors.append(f"epochs={self.epochs} ≤ 0")
        if self.early_stop >= self.epochs:
            errors.append(
                f"early_stop={self.early_stop} ≥ epochs={self.epochs} "
                "(학습이 즉시 중단될 수 있음)"
            )
        if self.grad_accum_steps <= 0:
            errors.append(f"grad_accum_steps={self.grad_accum_steps} ≤ 0")

        # ── 학습률 ────────────────────────────────────
        if not (0.0 < self.lr):
            errors.append(f"lr={self.lr} ≤ 0")
        if self.min_lr >= self.lr:
            errors.append(f"min_lr={self.min_lr} ≥ lr={self.lr}")

        # ── 필터 ─────────────────────────────────────
        if self.bandpass_low >= self.bandpass_high:
            errors.append(
                f"bandpass_low={self.bandpass_low} ≥ bandpass_high={self.bandpass_high}"
            )
        if self.bandpass_high > self.sample_rate / 2:
            errors.append(
                f"bandpass_high={self.bandpass_high} > Nyquist={self.sample_rate / 2}"
            )

        # ── 정규화 / 증강 ─────────────────────────────
        if self.label_smooth < 0 or self.label_smooth >= 1:
            errors.append(f"label_smooth={self.label_smooth} ∉ [0, 1)")
        if self.dropout_clf < 0 or self.dropout_clf >= 1:
            errors.append(f"dropout_clf={self.dropout_clf} ∉ [0, 1)")
        if self.dropout_feat < 0 or self.dropout_feat >= 1:
            errors.append(f"dropout_feat={self.dropout_feat} ∉ [0, 1)")
        if self.aug_mask_ratio < 0 or self.aug_mask_ratio >= 1:
            errors.append(f"aug_mask_ratio={self.aug_mask_ratio} ∉ [0, 1)")
        if self.mixup_alpha < 0:
            errors.append(f"mixup_alpha={self.mixup_alpha} < 0")

        # ── TTA ───────────────────────────────────────
        if self.use_tta and self.tta_rounds <= 0:
            errors.append(f"tta_rounds={self.tta_rounds} ≤ 0 (use_tta=True)")

        # ── 센서 채널 인덱스 범위 ──────────────────────
        if any(i < 0 or i >= self.pca_ch for i in self.foot_z_accel_idx):
            errors.append(
                f"foot_z_accel_idx={self.foot_z_accel_idx} 가 "
                f"pca_ch={self.pca_ch} 범위를 벗어남 (0-based, 최대 {self.pca_ch - 1})"
            )
        if any(i < 0 or i >= self.pca_ch for i in self.shank_z_accel_idx):
            errors.append(
                f"shank_z_accel_idx={self.shank_z_accel_idx} 가 "
                f"pca_ch={self.pca_ch} 범위를 벗어남 (0-based, 최대 {self.pca_ch - 1})"
            )

        if errors:
            raise ValueError("Config 유효성 오류:\n" + "\n".join(f"  · {e}" for e in errors))

        logger.info("Config 유효성 검사 통과 ✓")

    def apply_overrides(
        self,
        n_subjects: Optional[int] = None,
        seed: Optional[int] = None,
        batch: Optional[int] = None,
        epochs: Optional[int] = None,
        focal: Optional[bool] = None,
        fft: Optional[bool] = None,
        balanced: Optional[bool] = None,
        tta: Optional[bool] = None,
        use_preload: Optional[bool] = None,
    ) -> None:
        """CLI 인자로 설정을 런타임에 변경한다.

        CFG 객체 속성을 직접 수정하므로 멀티프로세싱 spawn 환경에서도
        worker에 CFG를 전달하면 변경된 값이 반영된다.
        """
        changed: list[str] = []

        if n_subjects is not None:
            self.n_subjects = n_subjects
            self._ensure_dirs()
            changed.append(f"n_subjects={n_subjects}")

        if seed is not None:
            self.seed = seed
            changed.append(f"seed={seed}")

        if batch is not None:
            self.batch = batch
            changed.append(f"batch={batch}")

        if epochs is not None:
            self.epochs = epochs
            changed.append(f"epochs={epochs}")

        if focal is not None:
            self.use_focal_loss = focal
            changed.append(f"use_focal_loss={focal}")

        if fft is not None:
            self.use_fft_branch = fft
            changed.append(f"use_fft_branch={fft}")

        if balanced is not None:
            self.use_balanced_sampler = balanced
            changed.append(f"use_balanced_sampler={balanced}")

        if tta is not None:
            self.use_tta = tta
            changed.append(f"use_tta={tta}")

        if use_preload is not None:
            self.use_preload = use_preload
            self._refresh_workers()   # 워커 수 재계산
            changed.append(f"use_preload={use_preload}")

        if changed:
            logger.info("Config 오버라이드 적용: %s", ", ".join(changed))

    def snapshot(self, out_dir: Path | None = None) -> dict:
        """현재 config 상태를 dict로 반환하고, out_dir이 주어지면 JSON 저장."""
        snap = {
            # 환경
            "git_hash": _get_git_hash(self.repo_dir),
            "device": str(DEVICE),
            "device_name": DEVICE_NAME,
            "n_gpu": N_GPU,
            "gpu_mem_gb": round(GPU_MEM_GB, 1),
            "gpu_total_mem_gb": round(GPU_TOTAL_MEM_GB, 1),
            "ram_gib": RAM_GIB,
            "vcpu": VCPU,

            # 전략
            "use_preload": self.use_preload,
            "use_preload_m1": self.use_preload_m1,
            "use_amp": USE_AMP,
            "amp_dtype": str(AMP_DTYPE),
            "loader_workers": self.loader_workers,

            # 실험
            "n_subjects": self.n_subjects,
            "num_classes": self.num_classes,
            "ts": self.ts,
            "pca_ch": self.pca_ch,
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "min_step_len": self.min_step_len,

            # 라벨
            "label_base": self.label_base,
            "label_semantics": self.label_semantics,
            "label_mapping": self.get_label_map(),

            # HS 검출
            "hs_gyro_sensor": self.hs_gyro_sensor,
            "hs_gyro_axis": self.hs_gyro_axis,
            "hs_accel_sensor": self.hs_accel_sensor,
            "hs_accel_axis": self.hs_accel_axis,
            "hs_fusion_window_ms": self.hs_fusion_window_ms,
            "hs_gyro_prominence": self.hs_gyro_prominence,

            # 학습
            "kfold": self.kfold,
            "epochs": self.epochs,
            "early_stop": self.early_stop,
            "batch": self.batch,
            "lr": self.lr,
            "min_lr": self.min_lr,
            "weight_decay": self.weight_decay,
            "label_smooth": self.label_smooth,
            "mixup_alpha": self.mixup_alpha,
            "grad_clip_norm": self.grad_clip_norm,
            "grad_accum_steps": self.grad_accum_steps,
            "use_compile": self.use_compile,
            "use_focal_loss": self.use_focal_loss,
            "focal_gamma": self.focal_gamma,
            "use_fft_branch": self.use_fft_branch,
            "use_balanced_sampler": self.use_balanced_sampler,
            "use_tta": self.use_tta,
            "tta_rounds": self.tta_rounds,

            # 정규화
            "dropout_clf": self.dropout_clf,
            "dropout_feat": self.dropout_feat,

            # 증강
            "aug_noise": self.aug_noise,
            "aug_scale": self.aug_scale,
            "aug_shift": self.aug_shift,
            "aug_mask_ratio": self.aug_mask_ratio,

            # 모델
            "feat_dim": self.feat_dim,
            "se_reduction": self.se_reduction,
            "cross_n_heads": self.cross_n_heads,
            "cross_dropout": self.cross_dropout,
            "foot_z_accel_idx": self.foot_z_accel_idx,
            "shank_z_accel_idx": self.shank_z_accel_idx,

            # 경로
            "data_dir": str(self.data_dir),
            "h5_path": str(self.h5_path),
        }

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "config_snapshot.json").write_text(
                json.dumps(snap, indent=2, ensure_ascii=False)
            )
            logger.info("Config 스냅샷 저장: %s/config_snapshot.json", out_dir)

        return snap

    def print_config(self) -> None:
        """현재 설정을 콘솔에 출력한다."""
        h5_gb = self.h5_path.stat().st_size / 1024 ** 3 if self.h5_path.exists() else 0
        amp_name = (
            "BF16" if AMP_DTYPE == torch.bfloat16
            else ("FP16" if USE_AMP else "OFF")
        )
        strategy_m1 = "Preload (항상)" if self.use_preload_m1 else "OTF"
        strategy_br = "Preload" if self.use_preload else "OTF (auto)"

        lines = [
            "=" * 60,
            f"  Config v9.1 — {'GPU' if USE_GPU else 'CPU'} 모드",
            f"  Git: {_get_git_hash(self.repo_dir)}",
            "=" * 60,
            f"  Device:    {DEVICE}  ({DEVICE_NAME})",
        ]
        if USE_GPU:
            _cc2 = torch.cuda.get_device_capability(0)
            gpu_info = (
                f"{GPU_MEM_GB:.0f}GB×{N_GPU}" if N_GPU > 1 else f"{GPU_MEM_GB:.0f}GB"
            )
            lines.append(f"  GPU:       {gpu_info}  |  AMP={amp_name}  cc={_cc2[0]}.{_cc2[1]}")
            if N_GPU > 1:
                lines.append(
                    f"  Multi-GPU: DataParallel ({N_GPU} GPUs, total {GPU_TOTAL_MEM_GB:.0f}GB)"
                )
        lines += [
            f"  vCPU={VCPU}  RAM={RAM_GIB}GiB  HDF5={h5_gb:.1f}GB",
            f"  Strategy:  M1={strategy_m1}  Branch={strategy_br}  Workers={self.loader_workers}",
            f"  Subjects:  N={self.n_subjects}  Classes={self.num_classes}  Seed={self.seed}",
            f"  Rate={self.sample_rate}Hz  TS={self.ts}pt  PCA={self.pca_ch}ch",
            f"  Labels:    base={self.label_base}  semantics={self.label_semantics}",
            "  Mapping:   " + ", ".join(
                [f"C{i+1}→{i + self.label_base}" for i in range(self.num_classes)]
            ),
            f"  Batch={self.batch}  Epochs={self.epochs}  ES={self.early_stop}",
            f"  LR={self.lr}→{self.min_lr}  WD={self.weight_decay}",
            f"  LabelSmooth={self.label_smooth}  Mixup={self.mixup_alpha}",
            f"  Dropout: clf={self.dropout_clf}  feat={self.dropout_feat}",
            f"  Aug: noise={self.aug_noise}  scale={self.aug_scale}  shift={self.aug_shift}",
            f"  HS: Type4 {self.hs_gyro_sensor}-Gyro{self.hs_gyro_axis} + "
            f"{self.hs_accel_sensor}-Accel{self.hs_accel_axis}"
            f"  fusion={self.hs_fusion_window_ms}ms  prom={self.hs_gyro_prominence}σ",
            f"  Foot Z-idx={self.foot_z_accel_idx}  Shank Z-idx={self.shank_z_accel_idx}",
            f"  Compile={self.use_compile}",
            f"  FocalLoss={'ON γ='+str(self.focal_gamma) if self.use_focal_loss else 'OFF'}",
            f"  FFT Branch={'ON ('+self.fft_source_group+')' if self.use_fft_branch else 'OFF'}",
            f"  BalancedSampler={'ON' if self.use_balanced_sampler else 'OFF'}",
            f"  TTA={'ON ×'+str(self.tta_rounds) if self.use_tta else 'OFF'}",
            "=" * 60,
        ]
        print("\n".join(lines))


# ══════════════════════════════════════════════════════
# 8. 유틸리티 함수
# ══════════════════════════════════════════════════════
def _auto_batch() -> int:
    """GPU 메모리 기반 배치 크기를 자동 결정한다."""
    if USE_GPU:
        base = 4096 if GPU_MEM_GB >= 40 else (4096 if GPU_MEM_GB >= 20 else 256)
        return base * max(1, N_GPU)
    return 128


def _get_git_hash(repo_dir: Path | None = None) -> str:
    """현재 git commit hash를 반환한다."""
    cwd = str(repo_dir) if repo_dir else "."
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════
# 9. 전역 CFG 싱글톤
# ══════════════════════════════════════════════════════
CFG = ConfigState()


# ══════════════════════════════════════════════════════
# 10. 하위 호환 — 모듈 수준 변수 (기존 코드가 config.BATCH 등으로 접근 시)
# ══════════════════════════════════════════════════════
# ⚠ DEPRECATED (v9.1): 이 상수들은 import 시점의 초기값만 반영합니다.
#   apply_overrides() 또는 CFG.xxx = ... 로 변경한 이후에는
#   이 상수들이 갱신되지 않습니다.
#
#   신규 코드는 반드시 CFG.<attr> 를 직접 참조하세요.
#   예) config.CFG.batch  (O)    config.BATCH  (X — stale 위험)
#
#   v10 에서 이 상수 블록을 제거할 예정입니다.
# ══════════════════════════════════════════════════════
warnings.warn(
    "\n[config] 모듈 수준 상수(BATCH, EPOCHS 등)는 import 시점 초기값만 반영합니다.\n"
    "  apply_overrides() 호출 후에는 config.CFG.<attr> 를 사용하세요.\n"
    "  이 상수들은 v10에서 제거될 예정입니다.",
    DeprecationWarning,
    stacklevel=1,
)
N_SUBJECTS = CFG.n_subjects
NUM_CLASSES = CFG.num_classes
TS = CFG.ts
PCA_CH = CFG.pca_ch
SAMPLE_RATE = CFG.sample_rate
SEED = CFG.seed
LABEL_BASE = CFG.label_base
LABEL_SEMANTICS = CFG.label_semantics
KFOLD = CFG.kfold
EPOCHS = CFG.epochs
EARLY_STOP = CFG.early_stop
LR = CFG.lr
MIN_LR = CFG.min_lr
BATCH = CFG.batch
WEIGHT_DECAY = CFG.weight_decay
DROPOUT_CLF = CFG.dropout_clf
DROPOUT_FEAT = CFG.dropout_feat
LABEL_SMOOTH = CFG.label_smooth
MIXUP_ALPHA = CFG.mixup_alpha
USE_FOCAL_LOSS = CFG.use_focal_loss
FOCAL_GAMMA = CFG.focal_gamma
USE_FFT_BRANCH = CFG.use_fft_branch
FFT_SOURCE_GROUP = CFG.fft_source_group
USE_BALANCED_SAMPLER = CFG.use_balanced_sampler
USE_TTA = CFG.use_tta
TTA_ROUNDS = CFG.tta_rounds
GRAD_CLIP_NORM = CFG.grad_clip_norm
GRAD_ACCUM_STEPS = CFG.grad_accum_steps
USE_COMPILE = CFG.use_compile
AUG_NOISE = CFG.aug_noise
AUG_SCALE = CFG.aug_scale
AUG_SHIFT = CFG.aug_shift
AUG_MASK_RATIO = CFG.aug_mask_ratio
FEAT_DIM = CFG.feat_dim
SE_REDUCTION = CFG.se_reduction
CROSS_N_HEADS = CFG.cross_n_heads
CROSS_DROPOUT = CFG.cross_dropout
FOOT_Z_ACCEL_IDX = CFG.foot_z_accel_idx
SHANK_Z_ACCEL_IDX = CFG.shank_z_accel_idx
USE_PRELOAD = CFG.use_preload
USE_PRELOAD_M1 = CFG.use_preload_m1
TORCH_THREADS = CFG.torch_threads
LOADER_WORKERS = CFG.loader_workers
PREPROC_WORKERS = CFG.preproc_workers
MIN_STEP_LEN = CFG.min_step_len
HS_MIN_STRIDE_MS = CFG.hs_min_stride_ms
HS_MAX_STRIDE_MS = CFG.hs_max_stride_ms
HS_MIN_STRIDE_SAM = CFG.hs_min_stride_sam
HS_MAX_STRIDE_SAM = CFG.hs_max_stride_sam
HS_NAN_THRESHOLD = CFG.hs_nan_threshold
HS_PROMINENCE_COEFF = CFG.hs_prominence_coeff
HS_PEAK_QUALITY_RATIO = CFG.hs_peak_quality_ratio
HS_GYRO_SENSOR = CFG.hs_gyro_sensor
HS_GYRO_AXIS = CFG.hs_gyro_axis
HS_ACCEL_SENSOR = CFG.hs_accel_sensor
HS_ACCEL_AXIS = CFG.hs_accel_axis
HS_FUSION_WINDOW_MS = CFG.hs_fusion_window_ms
HS_FUSION_WINDOW_SAM = CFG.hs_fusion_window_sam
HS_GYRO_PROMINENCE = CFG.hs_gyro_prominence
HS_ACCEL_THRESHOLD = CFG.hs_accel_threshold
HS_TRUSTED_SWING = CFG.hs_trusted_swing
BANDPASS_LOW = CFG.bandpass_low
BANDPASS_HIGH = CFG.bandpass_high
BANDPASS_ORDER = CFG.bandpass_order
H5_READ_CHUNK = CFG.h5_read_chunk
IPCA_CHUNK = CFG.ipca_chunk
FLUSH_SIZE = CFG.flush_size
DS_CHUNK = CFG.ds_chunk
ROOT = CFG.root
REPO_DIR = CFG.repo_dir
PROJECT_DIR = CFG.project_dir
DATA_DIR = CFG.data_dir
BATCH_DIR = CFG.batch_dir
H5_PATH = CFG.h5_path
OUT_DIR = CFG.out_dir
RESULT_KFOLD = CFG.result_kfold
RESULT_LOSO = CFG.result_loso
LOG_DIR = CFG.log_dir


# ══════════════════════════════════════════════════════
# 11. 하위 호환 — 전역 함수 래퍼
# ══════════════════════════════════════════════════════
def apply_overrides(
    n_subjects: Optional[int] = None,
    seed: Optional[int] = None,
    batch: Optional[int] = None,
    epochs: Optional[int] = None,
    focal: Optional[bool] = None,
    fft: Optional[bool] = None,
    balanced: Optional[bool] = None,
    tta: Optional[bool] = None,
    use_preload: Optional[bool] = None,   # v9.1: 누락 수정
) -> None:
    """하위 호환 래퍼. 내부적으로 CFG.apply_overrides() 를 호출한다."""
    CFG.apply_overrides(
        n_subjects=n_subjects,
        seed=seed,
        batch=batch,
        epochs=epochs,
        focal=focal,
        fft=fft,
        balanced=balanced,
        tta=tta,
        use_preload=use_preload,
    )


def snapshot(out_dir: Path | None = None) -> dict:
    """하위 호환 래퍼. 내부적으로 CFG.snapshot() 을 호출한다."""
    return CFG.snapshot(out_dir=out_dir)


def print_config() -> None:
    """하위 호환 래퍼. 내부적으로 CFG.print_config() 를 호출한다."""
    CFG.print_config()