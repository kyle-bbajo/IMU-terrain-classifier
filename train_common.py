"""
train_common.py — K-Fold / LOSO 공용 유틸 (v7.4)
═══════════════════════════════════════════════════════
★ 스마트 Preload: M1(PCA→64ch)은 항상 RAM 적재 (~2.5GB)
★ M2–M6: RAM 여유 시 Preload, 부족 시 OTF 자동 전환
★ Preload fp16 저장 + 직접 반환 (AMP autocast 위임)
★ IncrementalPCA + chunked StandardScaler
★ persistent_workers (Preload 모드) / num_workers=0 (OTF 모드)
★ 재현성 보장 (seed_everything)
★ NaN 손실 감지 + CUDA OOM 자동 복구
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys
import time
import gc
import os
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, accuracy_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE: torch.device = config.DEVICE


# ═══════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════

def log(msg: str) -> None:
    """타임스탬프 포함 로그 출력."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int = config.SEED) -> None:
    """모든 난수 생성기 시드를 고정하여 재현성을 보장한다.

    Parameters
    ----------
    seed : int
        시드 값 (config.SEED).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _mem_str() -> str:
    """현재 RAM 사용량을 문자열로 반환한다."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024:.0f}MB"
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return "?MB"


def _gpu_mem_str() -> str:
    """현재 GPU 메모리 사용량을 문자열로 반환한다."""
    if not config.USE_GPU:
        return ""
    alloc = torch.cuda.memory_allocated() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return f"  GPU={alloc:.0f}/{total:.0f}MB"


# ═══════════════════════════════════════════════
# 1. HDF5 데이터 핸들러
# ═══════════════════════════════════════════════

class H5Data:
    """HDF5 데이터셋 핸들러 (X는 lazy access, y/subj는 즉시 로드).

    컨텍스트 매니저 프로토콜을 지원하여 ``with`` 문에서 사용 가능.

    Parameters
    ----------
    h5_path : Path | str
        HDF5 파일 경로.

    Raises
    ------
    FileNotFoundError
        파일이 존재하지 않을 때.
    KeyError
        필수 데이터셋이 없을 때.

    Examples
    --------
    >>> with H5Data("dataset.h5") as h5:
    ...     print(h5.N, h5.T, h5.C)
    """

    REQUIRED_KEYS = ("X", "y", "subject_id", "channels")

    def __init__(self, h5_path: Path | str) -> None:
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 없음: {h5_path}")

        t0 = time.time()
        log(f"  HDF5 로드... ({h5_path.name})")

        self.h5f = h5py.File(h5_path, "r")

        missing = [k for k in self.REQUIRED_KEYS if k not in self.h5f]
        if missing:
            self.h5f.close()
            raise KeyError(f"HDF5에 필수 키 없음: {missing}")

        self.X_ds = self.h5f["X"]
        self.N: int
        self.T: int
        self.C: int
        self.N, self.T, self.C = self.X_ds.shape

        self.y_raw: np.ndarray   = self.h5f["y"][:].astype(np.int64)
        self.subj_id: np.ndarray = self.h5f["subject_id"][:].astype(np.int64)
        self.channels: list[str] = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in self.h5f["channels"][:]
        ]

        # 기본 검증
        assert len(self.y_raw) == self.N, (
            f"y 길이({len(self.y_raw)}) ≠ X 행({self.N})"
        )
        assert len(self.subj_id) == self.N, (
            f"subject_id 길이({len(self.subj_id)}) ≠ X 행({self.N})"
        )

        h5_gb = h5_path.stat().st_size / 1024**3
        log(f"  ✅ shape={self.X_ds.shape}  file={h5_gb:.1f}GB"
            f"  ({time.time()-t0:.1f}s)")

    def read_X(self, indices: np.ndarray) -> np.ndarray:
        """여러 인덱스의 X 데이터를 순차 청크로 읽는다.

        HDF5는 순차 접근이 빠르므로 인덱스를 정렬 후 읽고 복원한다.

        Parameters
        ----------
        indices : np.ndarray
            읽을 샘플 인덱스 배열.

        Returns
        -------
        np.ndarray
            ``(len(indices), T, C)`` float32 배열.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return np.empty((0, self.T, self.C), dtype=np.float32)

        sort_order = np.argsort(indices)
        sorted_idx = indices[sort_order]
        chunk_size = config.H5_READ_CHUNK

        parts: list[np.ndarray] = []
        for s in range(0, len(sorted_idx), chunk_size):
            chunk_idx = sorted_idx[s : s + chunk_size]
            parts.append(self.X_ds[chunk_idx.tolist()])

        data_sorted = np.concatenate(parts, axis=0)
        return data_sorted[np.argsort(sort_order)].astype(np.float32)

    def read_X_chunk(
        self, indices: np.ndarray, start: int, size: int
    ) -> np.ndarray:
        """인덱스 배열의 ``[start:start+size]`` 슬라이스를 읽는다."""
        return self.read_X(indices[start : start + size])

    def read_single(self, idx: int) -> np.ndarray:
        """단일 샘플 1개를 읽는다. ``(T, C)`` float32."""
        idx = int(idx)
        if idx < 0 or idx >= self.N:
            raise IndexError(f"인덱스 {idx} 범위 초과 (N={self.N})")
        return self.X_ds[idx].astype(np.float32)

    def close(self) -> None:
        """HDF5 파일 핸들을 닫는다."""
        if hasattr(self, "h5f") and self.h5f.id.valid:
            self.h5f.close()

    def __enter__(self) -> H5Data:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ═══════════════════════════════════════════════
# 2. Scaler / PCA fit (chunked partial_fit)
# ═══════════════════════════════════════════════

def fit_pca_on_train(
    h5data: H5Data, tr_idx: np.ndarray
) -> tuple[StandardScaler, IncrementalPCA]:
    """Train 데이터에 대해 StandardScaler + IncrementalPCA를 fit한다.

    ``IPCA_CHUNK`` 단위로 ``partial_fit`` 하여 OOM을 방지한다.

    Parameters
    ----------
    h5data : H5Data
    tr_idx : np.ndarray
        학습 데이터 인덱스.

    Returns
    -------
    tuple[StandardScaler, IncrementalPCA]
        fit 완료된 (scaler, ipca) 쌍.
    """
    t0 = time.time()
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    n, T, C = len(tr_idx), h5data.T, h5data.C
    chunk = config.IPCA_CHUNK

    # Phase 1: Scaler fit
    sc = StandardScaler()
    for s in range(0, n, chunk):
        X = h5data.read_X_chunk(tr_idx, s, chunk)
        sc.partial_fit(X.reshape(X.shape[0] * T, C))
        del X; gc.collect()

    # Phase 2: IPCA fit
    n_components = min(config.PCA_CH, C, n * T)
    ipca = IncrementalPCA(n_components=n_components)
    for s in range(0, n, chunk):
        X = h5data.read_X_chunk(tr_idx, s, chunk)
        flat = X.reshape(X.shape[0] * T, C)
        scaled = sc.transform(flat)
        # NaN 방어
        if np.isnan(scaled).any():
            np.nan_to_num(scaled, copy=False, nan=0.0)
        ipca.partial_fit(scaled)
        del X, flat, scaled; gc.collect()

    evr = ipca.explained_variance_ratio_.sum()
    log(f"    PCA fit: {C}→{n_components}  n={n}"
        f"  EVR={evr:.3f}  ({time.time()-t0:.1f}s)")
    return sc, ipca


def fit_bsc_on_train(h5data: H5Data, tr_idx: np.ndarray) -> StandardScaler:
    """Train 데이터에 대해 Branch용 StandardScaler를 fit한다.

    Parameters
    ----------
    h5data : H5Data
    tr_idx : np.ndarray
        학습 데이터 인덱스.

    Returns
    -------
    StandardScaler
        fit 완료된 스케일러.
    """
    t0 = time.time()
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    n, T, C = len(tr_idx), h5data.T, h5data.C
    chunk = config.IPCA_CHUNK

    bsc = StandardScaler()
    for s in range(0, n, chunk):
        X = h5data.read_X_chunk(tr_idx, s, chunk)
        bsc.partial_fit(X.reshape(X.shape[0] * T, C))
        del X; gc.collect()

    log(f"    BSC fit: n={n}  ({time.time()-t0:.1f}s)")
    return bsc


# ═══════════════════════════════════════════════
# 3. Dataset — 듀얼 모드 (Preload / OTF)
# ═══════════════════════════════════════════════

# ── 3a. Preload (RAM ≥ 24GB) ──

class FlatDatasetPreload(Dataset):
    """M1용: 전체 PCA 변환 데이터를 RAM에 적재.

    float16으로 저장 + 반환하여 RAM/CPU 오버헤드를 최소화.
    GPU AMP가 dtype 변환을 처리하므로 별도 fp32 변환 불필요.
    """

    def __init__(
        self, h5data: H5Data, y: np.ndarray, indices: np.ndarray,
        sc: StandardScaler, pca: IncrementalPCA,
    ) -> None:
        t0 = time.time()
        indices = np.asarray(indices, dtype=np.int64)
        self.y_arr = y[indices].astype(np.int64)
        T = h5data.T
        chunk = config.DS_CHUNK
        parts: list[np.ndarray] = []

        for s in range(0, len(indices), chunk):
            X = h5data.read_X_chunk(indices, s, chunk)
            n_c = X.shape[0]
            flat = X.reshape(n_c * T, h5data.C)
            scaled = sc.transform(flat)
            if np.isnan(scaled).any():
                np.nan_to_num(scaled, copy=False, nan=0.0)
            pca_out = pca.transform(scaled).reshape(n_c, T, config.PCA_CH)
            parts.append(
                np.ascontiguousarray(pca_out.transpose(0, 2, 1))
            )
            del X, flat, scaled, pca_out; gc.collect()

        # float16 저장 → RAM 절반 (32GB 서버 대응)
        self.data = np.concatenate(parts, axis=0).astype(np.float16)
        mb = self.data.nbytes / 1024**2
        log(f"      FlatDS(Preload/fp16): {self.data.shape}"
            f"  ({mb:.0f}MB)  ({time.time()-t0:.1f}s)")

    def __len__(self) -> int:
        return len(self.y_arr)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # fp16 텐서 직접 반환 → CPU astype 오버헤드 제거
        # GPU AMP autocast가 필요 시 자동 변환
        return (
            torch.from_numpy(self.data[i].copy()),
            torch.tensor(int(self.y_arr[i]), dtype=torch.long),
        )


class BranchDatasetPreload(Dataset):
    """M2–M6용: 전체 변환 데이터를 RAM에 적재.

    float16으로 저장하여 RAM 사용량을 절반으로 줄인다.
    32GB RAM에서도 안전하게 동작 (22GB → 11GB).
    """

    def __init__(
        self, h5data: H5Data, y: np.ndarray, indices: np.ndarray,
        bsc: StandardScaler, branch_idx: dict[str, list[int]],
    ) -> None:
        t0 = time.time()
        indices = np.asarray(indices, dtype=np.int64)
        self.y_arr = y[indices].astype(np.int64)
        T = h5data.T
        chunk = config.DS_CHUNK
        group_parts: dict[str, list[np.ndarray]] = {
            nm: [] for nm in branch_idx
        }

        for s in range(0, len(indices), chunk):
            X = h5data.read_X_chunk(indices, s, chunk)
            n_c = X.shape[0]
            flat = X.reshape(n_c * T, h5data.C)
            scaled = bsc.transform(flat)
            if np.isnan(scaled).any():
                np.nan_to_num(scaled, copy=False, nan=0.0)
            scaled = scaled.reshape(n_c, T, h5data.C)
            for nm, il in branch_idx.items():
                group_parts[nm].append(
                    np.ascontiguousarray(scaled[:, :, il].transpose(0, 2, 1))
                )
            del X, flat, scaled; gc.collect()

        # float16 저장 → RAM 절반 (32GB 서버 대응)
        self.groups: dict[str, np.ndarray] = {
            nm: np.concatenate(p, axis=0).astype(np.float16)
            for nm, p in group_parts.items()
        }
        del group_parts; gc.collect()
        total_mb = sum(v.nbytes for v in self.groups.values()) / 1024**2
        log(f"      BranchDS(Preload/fp16): n={len(indices)}  {len(branch_idx)}그룹"
            f"  ({total_mb:.0f}MB)  ({time.time()-t0:.1f}s)")

    def __len__(self) -> int:
        return len(self.y_arr)

    def __getitem__(self, i: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # fp16 텐서 직접 반환 → CPU astype 오버헤드 제거
        bi = {
            nm: torch.from_numpy(arr[i].copy())
            for nm, arr in self.groups.items()
        }
        return bi, torch.tensor(int(self.y_arr[i]), dtype=torch.long)


# ── 3b. On-the-fly (RAM < 24GB) ──

class FlatDatasetOTF(Dataset):
    """M1용: 매 ``__getitem__`` 에서 HDF5 → scale → PCA. (RAM ≈ 0)"""

    def __init__(
        self, h5data: H5Data, y: np.ndarray, indices: np.ndarray,
        sc: StandardScaler, pca: IncrementalPCA,
    ) -> None:
        self.h5data = h5data
        self.y      = y
        self.idx    = np.asarray(indices, dtype=np.int64)
        self.sc     = sc
        self.pca    = pca
        log(f"      FlatDS(OTF): n={len(self.idx)}  (RAM ≈ 0MB)")

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.h5data.read_single(self.idx[i])       # (T, C)
        scaled  = self.sc.transform(raw)                  # (T, C)
        pca_out = self.pca.transform(scaled)              # (T, PCA_CH)
        x = np.ascontiguousarray(pca_out.T.astype(np.float32))
        return (
            torch.from_numpy(x),
            torch.tensor(int(self.y[self.idx[i]]), dtype=torch.long),
        )


class BranchDatasetOTF(Dataset):
    """M2–M6용: 매 ``__getitem__`` 에서 HDF5 → scale → 그룹 분리. (RAM ≈ 0)"""

    def __init__(
        self, h5data: H5Data, y: np.ndarray, indices: np.ndarray,
        bsc: StandardScaler, branch_idx: dict[str, list[int]],
    ) -> None:
        self.h5data     = h5data
        self.y          = y
        self.idx        = np.asarray(indices, dtype=np.int64)
        self.bsc        = bsc
        self.branch_idx = branch_idx
        log(f"      BranchDS(OTF): n={len(self.idx)}  {len(branch_idx)}그룹"
            f"  (RAM ≈ 0MB)")

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        raw = self.h5data.read_single(self.idx[i])        # (T, C)
        scaled = self.bsc.transform(raw)                   # (T, C)
        bi: dict[str, torch.Tensor] = {}
        for nm, il in self.branch_idx.items():
            arr = np.ascontiguousarray(scaled[:, il].T.astype(np.float32))
            bi[nm] = torch.from_numpy(arr)
        return bi, torch.tensor(int(self.y[self.idx[i]]), dtype=torch.long)


# ── 3c. 팩토리 함수 (스마트 Preload) ──

def make_flat_dataset(
    h5data: H5Data, y: np.ndarray, indices: np.ndarray,
    sc: StandardScaler, pca: IncrementalPCA,
) -> Dataset:
    """M1용 FlatDataset 생성.

    M1 PCA 결과는 항상 Preload (64ch × 256T × 2B ≈ 2.5GB).
    OTF 대비 학습 속도 10배+ 향상.
    """
    if config.USE_PRELOAD_M1:
        return FlatDatasetPreload(h5data, y, indices, sc, pca)
    return FlatDatasetOTF(h5data, y, indices, sc, pca)


def make_branch_dataset(
    h5data: H5Data, y: np.ndarray, indices: np.ndarray,
    bsc: StandardScaler, branch_idx: dict[str, list[int]],
    total_samples: int = 0,
) -> Dataset:
    """M2–M6용 BranchDataset 생성.

    RAM 여유가 충분하면 Preload, 부족하면 OTF로 자동 전환.

    Parameters
    ----------
    total_samples : int
        train + test 합산 샘플 수 (Preload 가능 여부 판단용).
        0이면 indices 길이만으로 추정.
    """
    if config.USE_PRELOAD:
        return BranchDatasetPreload(h5data, y, indices, bsc, branch_idx)

    # 스마트 판단: 전체 데이터가 RAM에 들어가는지 동적 추정
    n_total = total_samples if total_samples > 0 else len(indices) * 2
    if config.can_preload_branch(n_total, h5data.C, h5data.T):
        log(f"      ★ 스마트 Preload: {n_total}샘플 × {h5data.C}ch → RAM 가능")
        return BranchDatasetPreload(h5data, y, indices, bsc, branch_idx)

    return BranchDatasetOTF(h5data, y, indices, bsc, branch_idx)


def collate_branch(
    batch: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """BranchDataset용 custom collate 함수."""
    bis, ys = zip(*batch)
    return (
        {k: torch.stack([b[k] for b in bis]) for k in bis[0]},
        torch.stack(ys),
    )


# ═══════════════════════════════════════════════
# 4. Mixup
# ═══════════════════════════════════════════════

def mixup_data(
    x: torch.Tensor | dict[str, torch.Tensor],
    y: torch.Tensor,
    alpha: float = config.MIXUP_ALPHA,
) -> tuple:
    """Mixup 데이터 증강.

    Returns
    -------
    tuple
        ``(mixed_x, y_a, y_b, lambda)``.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(max(np.random.beta(alpha, alpha), 0.5))
    idx = torch.randperm(y.size(0), device=y.device)
    if isinstance(x, dict):
        mixed = {k: lam * v + (1 - lam) * v[idx] for k, v in x.items()}
    else:
        mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(
    crit: nn.Module, pred: torch.Tensor,
    y_a: torch.Tensor, y_b: torch.Tensor, lam: float,
) -> torch.Tensor:
    """Mixup 손실 함수."""
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)


# ═══════════════════════════════════════════════
# 5. 학습 루프
# ═══════════════════════════════════════════════

def _run(
    model: nn.Module, loader: DataLoader, crit: nn.Module,
    opt: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    branch: bool = False, use_mixup: bool = False,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """1 에포크 학습 또는 평가를 실행한다.

    NaN 손실 감지 시 해당 배치를 스킵하고 경고를 출력한다.

    Returns
    -------
    tuple[float, float, np.ndarray, np.ndarray]
        ``(avg_loss, accuracy, predictions, labels)``.
    """
    train = opt is not None
    model.train() if train else model.eval()
    total_loss: float = 0.0
    correct = total = 0
    nan_batches = 0
    preds_all: list[int] = []
    labels_all: list[int] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    step_idx = 0
    with ctx:
        if train:
            opt.zero_grad(set_to_none=True)
        for batch in loader:
            # 데이터 → 디바이스
            if branch:
                bi, yb = batch
                bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
                yb = yb.to(DEVICE, non_blocking=True)
                # AMP 미사용 시 fp16→fp32 변환 (CPU 모드 안전장치)
                if not config.USE_AMP:
                    bi = {k: v.float() for k, v in bi.items()}
                inp = bi
            else:
                xb, yb = batch
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                if not config.USE_AMP:
                    xb = xb.float()
                inp = xb

            # Mixup
            if train and use_mixup and config.MIXUP_ALPHA > 0:
                inp, ya, yb_mix, lam = mixup_data(inp, yb)
            else:
                ya = yb_mix = yb
                lam = 1.0

            # Forward
            with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                logits = model(inp)
                loss = (
                    mixup_criterion(crit, logits, ya, yb_mix, lam)
                    if lam < 1.0 else crit(logits, yb)
                )

            # NaN 손실 감지
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                if nan_batches <= 3:
                    log(f"  ⚠ NaN/Inf 손실 감지 (배치 스킵)")
                continue

            # Backward (Gradient Accumulation 지원)
            if train:
                loss_scaled = loss / config.GRAD_ACCUM_STEPS
                if scaler is not None:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                step_idx += 1
                if step_idx % config.GRAD_ACCUM_STEPS == 0:
                    if scaler is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.GRAD_CLIP_NORM)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.GRAD_CLIP_NORM)
                        opt.step()
                    opt.zero_grad(set_to_none=True)

            # 통계
            bs = len(yb)
            total_loss += loss.item() * bs
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total   += bs
            preds_all.extend(pred.cpu().numpy().tolist())
            labels_all.extend(yb.cpu().numpy().tolist())

    if nan_batches > 0:
        log(f"  ⚠ NaN/Inf 배치: {nan_batches}개 스킵됨")

    # Gradient Accumulation 잔여분 flush
    if train and step_idx % config.GRAD_ACCUM_STEPS != 0:
        if scaler is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.GRAD_CLIP_NORM)
            opt.step()
        opt.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, np.array(preds_all), np.array(labels_all)


def train_model(
    model: nn.Module, tr_dl: DataLoader, te_dl: DataLoader,
    branch: bool = False, tag: str = "", use_mixup: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """모델을 학습하고 최적 가중치로 평가한다.

    CUDA OOM 발생 시 에러 메시지와 함께 현재까지의 결과를 반환한다.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        ``(test_predictions, test_labels, training_history)``.
    """
    crit = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTH)
    opt  = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    scaler = GradScaler(enabled=config.USE_AMP)

    hist: dict[str, list[float]] = {"tl": [], "ta": [], "vl": [], "va": []}
    best_vl = float("inf")
    best_state: Optional[dict] = None
    patience = 0
    t0 = time.time()

    for ep in range(1, config.EPOCHS + 1):
        try:
            tl, ta, _, _ = _run(
                model, tr_dl, crit, opt, scaler, branch, use_mixup)
            vl, va, _, _ = _run(model, te_dl, crit, branch=branch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log(f"  ✗ CUDA OOM at ep{ep}! 배치 크기를 줄이세요."
                    f" (현재 BATCH={config.BATCH})")
                if config.USE_GPU:
                    torch.cuda.empty_cache()
                break
            raise

        sch.step()

        hist["tl"].append(tl); hist["ta"].append(ta)
        hist["vl"].append(vl); hist["va"].append(va)

        log(f"  {tag} ep{ep:02d}/{config.EPOCHS}"
            f"  loss={tl:.4f}/{ta:.4f}"
            f"  val={vl:.4f}/{va:.4f}"
            f"  lr={opt.param_groups[0]['lr']:.1e}"
            f"  ({time.time()-t0:.0f}s)"
            f"  RAM={_mem_str()}{_gpu_mem_str()}")

        if vl < best_vl:
            best_vl = vl
            # DataParallel 래핑 시 .module로 원본 접근
            _m = _unwrap(model)
            best_state = {
                k: v.cpu().clone() for k, v in _m.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP:
                log(f"  {tag} ★ EarlyStop ep{ep}")
                break

    if best_state:
        _unwrap(model).load_state_dict(best_state)
        model.to(DEVICE)

    # 최종 평가
    try:
        _, _, preds, labels = _run(model, te_dl, crit, branch=branch)
    except RuntimeError:
        log("  ✗ 최종 평가 중 OOM")
        preds = np.array([], dtype=np.int64)
        labels = np.array([], dtype=np.int64)

    return preds, labels, hist


# ═══════════════════════════════════════════════
# 6. DataLoader 생성
# ═══════════════════════════════════════════════

def make_loader(ds: Dataset, shuffle: bool, branch: bool = False) -> DataLoader:
    """Dataset → DataLoader.

    OTF 모드: ``num_workers=0`` (h5py 호환).
    Preload 모드: ``config.LOADER_WORKERS`` + ``persistent_workers``.
    """
    # OTF 데이터셋은 h5py 핸들을 공유하므로 반드시 num_workers=0
    is_otf = isinstance(ds, (FlatDatasetOTF, BranchDatasetOTF))
    workers = 0 if is_otf else config.LOADER_WORKERS

    kw: dict = dict(
        batch_size=config.BATCH,
        num_workers=workers,
        pin_memory=config.USE_GPU and not is_otf,
        shuffle=shuffle,
        drop_last=False,
    )
    # Preload + 멀티워커: persistent_workers로 epoch 간 재생성 방지
    if workers > 0:
        kw["persistent_workers"] = True
    if branch:
        kw["collate_fn"] = collate_branch
    return DataLoader(ds, **kw)


# ═══════════════════════════════════════════════
# 7. 모델별 실행 함수
# ═══════════════════════════════════════════════

def _maybe_compile(model: nn.Module) -> nn.Module:
    """``config.USE_COMPILE`` 이 True이면 ``torch.compile`` 을 시도한다."""
    if config.USE_COMPILE and hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode="reduce-overhead")
        except Exception:
            return model
    return model


def _maybe_parallel(model: nn.Module) -> nn.Module:
    """Multi-GPU 환경에서 ``DataParallel`` 래핑.

    N_GPU > 1이면 자동 적용.
    ``model.module`` 로 원본 접근 가능.
    """
    if config.N_GPU > 1:
        model = nn.DataParallel(model)
        log(f"     ★ DataParallel: {config.N_GPU} GPUs")
    return model


def _unwrap(model: nn.Module) -> nn.Module:
    """DataParallel 래핑을 제거하고 원본 모델을 반환한다."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _prepare_model(model: nn.Module) -> nn.Module:
    """모델을 DEVICE 전송 → compile → parallel 순서로 준비한다."""
    model = model.to(DEVICE)
    model = _maybe_compile(model)
    model = _maybe_parallel(model)
    return model


def run_M1(
    h5data: H5Data, y: np.ndarray,
    tr_idx: np.ndarray, te_idx: np.ndarray,
    sc: StandardScaler, pca: IncrementalPCA,
    fold_tag: str | int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, dict]]:
    """M1 (PCA + Flat CNN) 을 학습·평가한다.

    Returns
    -------
    tuple
        ``(예측 dict, labels, history dict)``.
    """
    from models import M1_FlatCNN, count_parameters
    log(f"── M1  tr={len(tr_idx)}  te={len(te_idx)}")

    tr_ds = make_flat_dataset(h5data, y, tr_idx, sc, pca)
    te_ds = make_flat_dataset(h5data, y, te_idx, sc, pca)
    model = _prepare_model(M1_FlatCNN())
    log(f"     params={count_parameters(_unwrap(model)):,}")

    p, lb, h = train_model(
        model, make_loader(tr_ds, True), make_loader(te_ds, False),
        False, f"[F{fold_tag}][M1]",
    )
    acc = accuracy_score(lb, p) if len(lb) > 0 else 0.0
    log(f"── M1 Acc={acc:.4f}")

    del model, tr_ds, te_ds; gc.collect()
    if config.USE_GPU:
        torch.cuda.empty_cache()
    return {"M1_CNN": p}, lb, {"M1_CNN": h}


def run_branch(
    h5data: H5Data, y: np.ndarray,
    tr_idx: np.ndarray, te_idx: np.ndarray,
    branch_idx: dict[str, list[int]], branch_ch: dict[str, int],
    bsc: StandardScaler, model_fn, mname: str,
    fold_tag: str | int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, dict]]:
    """Branch 모델 (M2–M6) 을 학습·평가한다.

    Returns
    -------
    tuple
        ``(예측 dict, labels, history dict)``.
    """
    from models import count_parameters
    log(f"── {mname}  tr={len(tr_idx)}  te={len(te_idx)}")

    tr_ds = make_branch_dataset(h5data, y, tr_idx, bsc, branch_idx)
    te_ds = make_branch_dataset(h5data, y, te_idx, bsc, branch_idx)
    model = _prepare_model(model_fn(branch_ch))
    log(f"     params={count_parameters(_unwrap(model)):,}")

    p, lb, h = train_model(
        model, make_loader(tr_ds, True, True), make_loader(te_ds, False, True),
        True, f"[F{fold_tag}][{mname}]",
    )
    acc = accuracy_score(lb, p) if len(lb) > 0 else 0.0
    log(f"── {mname} Acc={acc:.4f}")

    del model, tr_ds, te_ds; gc.collect()
    if config.USE_GPU:
        torch.cuda.empty_cache()
    return {f"{mname}_CNN": p}, lb, {f"{mname}_CNN": h}


# ═══════════════════════════════════════════════
# 8. 결과 저장
# ═══════════════════════════════════════════════

def save_report(
    preds: np.ndarray, labels: np.ndarray,
    le: LabelEncoder, tag: str, out_dir: Path,
) -> tuple[float, float]:
    """분류 리포트를 파일로 저장한다.

    Returns
    -------
    tuple[float, float]
        ``(accuracy, macro_f1)``.
    """
    if len(preds) == 0 or len(labels) == 0:
        log(f"  RESULT {tag:<30} 데이터 없음")
        return 0.0, 0.0

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    names = [f"C{c}" for c in le.classes_]
    rep = classification_report(
        labels, preds, target_names=names, digits=4, zero_division=0)

    log(f"  RESULT {tag:<30} Acc={acc:.4f}  F1={f1:.4f}")
    (out_dir / f"report_{tag}.txt").write_text(
        f"{tag}\nAcc={acc:.4f}  F1={f1:.4f}\n\n{rep}")
    return acc, f1


def save_cm(
    preds: np.ndarray, labels: np.ndarray,
    le: LabelEncoder, tag: str, out_dir: Path,
) -> None:
    """Confusion Matrix를 이미지로 저장한다."""
    if len(preds) == 0:
        return

    cm = confusion_matrix(labels, preds)
    names = [f"C{c}" for c in le.classes_]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(names)), yticks=range(len(names)),
        xticklabels=names, yticklabels=names,
        xlabel="Predicted", ylabel="True",
        title=f"{tag}  Acc={accuracy_score(labels, preds):.4f}",
    )
    thr = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thr else "black", fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(out_dir / f"cm_{tag}.png", dpi=150)
    plt.close()


def save_history(all_hist: dict[str, list[dict]], out_dir: Path) -> None:
    """학습 히스토리를 이미지로 저장한다."""
    for mname, fold_hists in all_hist.items():
        if not fold_hists:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for h in fold_hists:
            if h.get("tl"):
                axes[0].plot(h["tl"], alpha=0.3, color="C0")
            if h.get("vl"):
                axes[0].plot(h["vl"], alpha=0.3, color="C1")
            if h.get("ta"):
                axes[1].plot(h["ta"], alpha=0.3, color="C0")
            if h.get("va"):
                axes[1].plot(h["va"], alpha=0.3, color="C1")
        axes[0].set(title=f"{mname} Loss", xlabel="Epoch", ylabel="Loss")
        axes[0].legend(["Train", "Val"])
        axes[1].set(title=f"{mname} Acc", xlabel="Epoch", ylabel="Accuracy")
        axes[1].legend(["Train", "Val"])
        plt.tight_layout()
        plt.savefig(out_dir / f"history_{mname}.png", dpi=150)
        plt.close()
