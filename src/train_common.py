"""
train_common.py — K-Fold / LOSO 공용 유틸 (v8.7)
═══════════════════════════════════════════════════════
v8.1: 54ch Raw IMU (9센서 Accel+Gyro) / 5그룹 Branch
★ 스마트 Preload: M1(PCA→32ch) / M2–M6(54ch) 항상 RAM 적재
★ Preload fp16 저장 + 직접 반환 (AMP autocast 위임)
★ IncrementalPCA + chunked StandardScaler
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
    """HDF5 데이터셋 핸들러 — Subject-group v8 + v7 flat 호환.

    v8 형식: /subjects/S{sid}/X, /subjects/S{sid}/y, /channels
    v7 형식: /X, /y, /subject_id, /channels (자동 감지)

    글로벌 인덱스 → (subject_key, local_idx) 매핑을 빌드하여
    기존 K-Fold/LOSO 코드와 완전 호환.
    """

    def __init__(self, h5_path: Path | str) -> None:
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 없음: {h5_path}")

        t0 = time.time()
        log(f"  HDF5 로드... ({h5_path.name})")

        self.h5f = h5py.File(h5_path, "r")
        self._is_v8 = "subjects" in self.h5f

        if self._is_v8:
            self._init_v8()
        else:
            self._init_v7()

        h5_gb = h5_path.stat().st_size / 1024**3
        log(f"  ✅ N={self.N}  T={self.T}  C={self.C}  "
            f"{'v8-subj' if self._is_v8 else 'v7-flat'}  "
            f"file={h5_gb:.1f}GB  ({time.time()-t0:.1f}s)")

    def _init_v8(self) -> None:
        """Subject-group 형식 초기화. 글로벌 인덱스 매핑 빌드."""
        self.channels: list[str] = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in self.h5f["channels"][:]
        ]
        subj_grp = self.h5f["subjects"]
        skeys = sorted(subj_grp.keys())  # S0001, S0002, ...

        # 글로벌 인덱스 → (subject_key, local_idx) 테이블
        self._idx_map: list[tuple[str, int]] = []
        y_parts: list[np.ndarray] = []
        sid_parts: list[np.ndarray] = []

        for skey in skeys:
            grp = subj_grp[skey]
            n_s = grp["X"].shape[0]
            sid = int(skey[1:])
            for local_i in range(n_s):
                self._idx_map.append((skey, local_i))
            y_parts.append(grp["y"][:].astype(np.int64))
            sid_parts.append(np.full(n_s, sid, dtype=np.int64))

        self.y_raw   = np.concatenate(y_parts)   if y_parts  else np.array([], dtype=np.int64)
        self.subj_id = np.concatenate(sid_parts)  if sid_parts else np.array([], dtype=np.int64)
        self.N = len(self._idx_map)
        if self.N > 0:
            sample_key = skeys[0]
            self.T = subj_grp[sample_key]["X"].shape[1]
            self.C = subj_grp[sample_key]["X"].shape[2]
        else:
            self.T, self.C = 0, 0

    def _init_v7(self) -> None:
        """v7 flat 형식 초기화 (하위 호환)."""
        for key in ("X", "y", "subject_id", "channels"):
            if key not in self.h5f:
                self.h5f.close()
                raise KeyError(f"HDF5에 필수 키 없음: {key}")

        self.X_ds = self.h5f["X"]
        self.N, self.T, self.C = self.X_ds.shape
        self.y_raw   = self.h5f["y"][:].astype(np.int64)
        self.subj_id = self.h5f["subject_id"][:].astype(np.int64)
        self.channels = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in self.h5f["channels"][:]
        ]

    def read_X(self, indices: np.ndarray) -> np.ndarray:
        """여러 인덱스의 X 데이터를 읽는다. → (len, T, C) float32."""
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return np.empty((0, self.T, self.C), dtype=np.float32)

        if not self._is_v8:
            # v7: 기존 로직
            sort_order = np.argsort(indices)
            sorted_idx = indices[sort_order]
            chunk_size = config.H5_READ_CHUNK
            parts: list[np.ndarray] = []
            for s in range(0, len(sorted_idx), chunk_size):
                chunk_idx = sorted_idx[s : s + chunk_size]
                parts.append(self.X_ds[chunk_idx.tolist()])
            data_sorted = np.concatenate(parts, axis=0)
            return data_sorted[np.argsort(sort_order)].astype(np.float32)

        # v8: subject-group 읽기 (배치 최적화)
        subj_grp = self.h5f["subjects"]
        out = np.empty((len(indices), self.T, self.C), dtype=np.float32)

        # subject별로 묶어서 한번에 읽기 (I/O 최적화)
        from collections import defaultdict as _ddict
        skey_batches: dict[str, list[tuple[int, int]]] = _ddict(list)
        for out_i, global_i in enumerate(indices):
            skey, local_i = self._idx_map[global_i]
            skey_batches[skey].append((out_i, local_i))

        for skey, pairs in skey_batches.items():
            ds = subj_grp[f"{skey}/X"]
            out_idx  = np.array([p[0] for p in pairs])
            loc_idx  = np.array([p[1] for p in pairs])
            # 정렬 후 일괄 읽기 → 벡터화 할당
            order = np.argsort(loc_idx)
            chunk = ds[loc_idx[order].tolist()]
            out[out_idx[order]] = chunk

        return out

    def read_X_chunk(
        self, indices: np.ndarray, start: int, size: int
    ) -> np.ndarray:
        """인덱스 배열의 [start:start+size] 슬라이스를 읽는다."""
        return self.read_X(indices[start : start + size])

    def read_single(self, idx: int) -> np.ndarray:
        """단일 샘플 1개를 읽는다. (T, C) float32."""
        idx = int(idx)
        if idx < 0 or idx >= self.N:
            raise IndexError(f"인덱스 {idx} 범위 초과 (N={self.N})")
        if not self._is_v8:
            return self.X_ds[idx].astype(np.float32)
        skey, local_i = self._idx_map[idx]
        return self.h5f[f"subjects/{skey}/X"][local_i].astype(np.float32)

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
# 2b. PCA/BSC 디스크 캐시 (v8: fold별 변환 결과 재사용)
# ═══════════════════════════════════════════════

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "cache"))


def _cache_path(fold_tag: str | int, split: str, kind: str) -> Path:
    """캐시 파일 경로 생성. ex: cache/F1_train_flat.npy"""
    return CACHE_DIR / f"F{fold_tag}_{split}_{kind}.npy"


def _cache_y_path(fold_tag: str | int, split: str) -> Path:
    return CACHE_DIR / f"F{fold_tag}_{split}_y.npy"


def cache_flat_transform(
    h5data: H5Data, y: np.ndarray, indices: np.ndarray,
    sc: StandardScaler, pca: IncrementalPCA,
    fold_tag: str | int, split: str,
) -> Path:
    """PCA 변환 결과를 디스크 캐시로 저장한다. fp16.

    Returns
    -------
    Path
        캐시 파일 경로 (.npy, shape=(N, PCA_CH, T), fp16).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_f = _cache_path(fold_tag, split, "flat")
    y_f     = _cache_y_path(fold_tag, split)

    if cache_f.exists() and y_f.exists():
        log(f"      ★ 캐시 히트: {cache_f.name}")
        return cache_f

    t0 = time.time()
    indices = np.asarray(indices, dtype=np.int64)
    T = h5data.T
    chunk = config.DS_CHUNK
    n = len(indices)

    # memmap 생성
    out = np.lib.format.open_memmap(
        str(cache_f), mode="w+",
        dtype=np.float16, shape=(n, config.PCA_CH, T),
    )

    pos = 0
    for s in range(0, n, chunk):
        X = h5data.read_X_chunk(indices, s, chunk)
        n_c = X.shape[0]
        flat = X.reshape(n_c * T, h5data.C)
        scaled = sc.transform(flat)
        if np.isnan(scaled).any():
            np.nan_to_num(scaled, copy=False, nan=0.0)
        pca_out = pca.transform(scaled).reshape(n_c, T, config.PCA_CH)
        out[pos:pos+n_c] = pca_out.transpose(0, 2, 1).astype(np.float16)
        pos += n_c
        del X, flat, scaled, pca_out; gc.collect()

    out.flush()
    np.save(str(y_f), y[indices].astype(np.int64))
    mb = cache_f.stat().st_size / 1024**2
    log(f"      캐시 저장: {cache_f.name}  ({mb:.0f}MB)  ({time.time()-t0:.1f}s)")
    return cache_f


def cache_branch_transform(
    h5data: H5Data, y: np.ndarray, indices: np.ndarray,
    bsc: StandardScaler, branch_idx: dict[str, list[int]],
    fold_tag: str | int, split: str,
) -> dict[str, Path]:
    """Branch 변환 결과를 그룹별 디스크 캐시로 저장한다. fp16.

    Returns
    -------
    dict[str, Path]
        그룹명 → 캐시 파일 경로.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    y_f = _cache_y_path(fold_tag, split)
    cache_paths: dict[str, Path] = {}
    all_hit = True

    for nm in branch_idx:
        p = _cache_path(fold_tag, split, f"br_{nm}")
        cache_paths[nm] = p
        if not p.exists():
            all_hit = False

    if all_hit and y_f.exists():
        log(f"      ★ Branch 캐시 히트: F{fold_tag}_{split}")
        return cache_paths

    t0 = time.time()
    indices = np.asarray(indices, dtype=np.int64)
    T = h5data.T
    chunk = config.DS_CHUNK
    n = len(indices)

    # 그룹별 memmap 생성
    mms: dict[str, np.ndarray] = {}
    for nm, il in branch_idx.items():
        mms[nm] = np.lib.format.open_memmap(
            str(cache_paths[nm]), mode="w+",
            dtype=np.float16, shape=(n, len(il), T),
        )

    pos = 0
    for s in range(0, n, chunk):
        X = h5data.read_X_chunk(indices, s, chunk)
        n_c = X.shape[0]
        flat = X.reshape(n_c * T, h5data.C)
        scaled = bsc.transform(flat)
        if np.isnan(scaled).any():
            np.nan_to_num(scaled, copy=False, nan=0.0)
        scaled = scaled.reshape(n_c, T, h5data.C)
        for nm, il in branch_idx.items():
            mms[nm][pos:pos+n_c] = scaled[:, :, il].transpose(0, 2, 1).astype(np.float16)
        pos += n_c
        del X, flat, scaled; gc.collect()

    for mm in mms.values():
        mm.flush()
    np.save(str(y_f), y[indices].astype(np.int64))

    total_mb = sum(p.stat().st_size for p in cache_paths.values()) / 1024**2
    log(f"      Branch 캐시 저장: {len(branch_idx)}그룹  ({total_mb:.0f}MB)"
        f"  ({time.time()-t0:.1f}s)")
    return cache_paths


def clear_fold_cache(fold_tag: str | int) -> None:
    """특정 fold의 캐시를 삭제한다."""
    if not CACHE_DIR.exists():
        return
    for f in CACHE_DIR.glob(f"F{fold_tag}_*"):
        f.unlink()


# ═══════════════════════════════════════════════
# 3. Dataset — 캐시 우선 + Preload/OTF 폴백
# ═══════════════════════════════════════════════

# ── 3a. 캐시 Dataset (v8 기본) ──

class CachedFlatDataset(Dataset):
    """M1용: 디스크 캐시(memmap)에서 직접 읽는 Dataset.

    PCA 변환이 이미 완료된 상태라 __getitem__ 비용 ≈ 0.
    """

    def __init__(self, cache_path: Path, y_path: Path) -> None:
        self.data = np.load(str(cache_path), mmap_mode="r")  # (N, PCA_CH, T) fp16
        self.y_arr = np.load(str(y_path))                     # (N,) int64
        mb = cache_path.stat().st_size / 1024**2
        log(f"      FlatDS(Cache): {self.data.shape}  ({mb:.0f}MB)")

    def __len__(self) -> int:
        return len(self.y_arr)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.data[i].copy()),
            torch.tensor(int(self.y_arr[i]), dtype=torch.long),
        )


class CachedBranchDataset(Dataset):
    """M2–M6용: 그룹별 디스크 캐시에서 직접 읽는 Dataset."""

    def __init__(
        self, cache_paths: dict[str, Path], y_path: Path,
    ) -> None:
        self.groups: dict[str, np.ndarray] = {
            nm: np.load(str(p), mmap_mode="r")
            for nm, p in cache_paths.items()
        }
        self.y_arr = np.load(str(y_path))
        total_mb = sum(
            p.stat().st_size for p in cache_paths.values()
        ) / 1024**2
        log(f"      BranchDS(Cache): n={len(self.y_arr)}  "
            f"{len(cache_paths)}그룹  ({total_mb:.0f}MB)")

    def __len__(self) -> int:
        return len(self.y_arr)

    def __getitem__(self, i: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        bi = {
            nm: torch.from_numpy(arr[i].copy())
            for nm, arr in self.groups.items()
        }
        return bi, torch.tensor(int(self.y_arr[i]), dtype=torch.long)


# ── 3b. Preload (RAM ≥ 24GB, 캐시 없을 때 폴백) ──

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
    fold_tag: str | int = "", split: str = "",
) -> Dataset:
    """M1용 FlatDataset 생성.

    v8: 캐시 우선 → Preload 폴백 → OTF 폴백.
    fold_tag/split 지정 시 디스크 캐시 사용.
    """
    # 캐시 모드 (fold_tag 있으면)
    if fold_tag:
        cache_f = cache_flat_transform(
            h5data, y, indices, sc, pca, fold_tag, split)
        y_f = _cache_y_path(fold_tag, split)
        return CachedFlatDataset(cache_f, y_f)

    # 폴백: 기존 Preload/OTF
    if config.USE_PRELOAD_M1:
        return FlatDatasetPreload(h5data, y, indices, sc, pca)
    return FlatDatasetOTF(h5data, y, indices, sc, pca)


def make_branch_dataset(
    h5data: H5Data, y: np.ndarray, indices: np.ndarray,
    bsc: StandardScaler, branch_idx: dict[str, list[int]],
    total_samples: int = 0,
    fold_tag: str | int = "", split: str = "",
) -> Dataset:
    """M2–M6용 BranchDataset 생성.

    v8: 캐시 우선 → Preload 폴백 → OTF 폴백.
    fold_tag/split 지정 시 디스크 캐시 사용.
    """
    # 캐시 모드 (fold_tag 있으면)
    if fold_tag:
        cache_paths = cache_branch_transform(
            h5data, y, indices, bsc, branch_idx, fold_tag, split)
        y_f = _cache_y_path(fold_tag, split)
        return CachedBranchDataset(cache_paths, y_f)

    # 폴백: 기존 Preload/OTF
    if config.USE_PRELOAD:
        return BranchDatasetPreload(h5data, y, indices, bsc, branch_idx)

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

    ctx = torch.enable_grad() if train else torch.inference_mode()
    step_idx = 0
    # v8: GPU에서 예측 누적 후 한 번에 전송
    preds_gpu: list[torch.Tensor] = []
    labels_gpu: list[torch.Tensor] = []

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

            # 통계 — GPU에서 누적, CPU 전송 최소화
            bs = len(yb)
            total_loss += loss.item() * bs
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total   += bs
            preds_gpu.append(pred.detach())
            labels_gpu.append(yb.detach())

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
    # GPU → CPU 한 번에 전송
    preds_all = torch.cat(preds_gpu).cpu().numpy() if preds_gpu else np.array([], dtype=np.int64)
    labels_all = torch.cat(labels_gpu).cpu().numpy() if labels_gpu else np.array([], dtype=np.int64)
    return avg_loss, acc, preds_all, labels_all


# ═══════════════════════════════════════════════
# 5b. Focal Loss — 어려운 샘플에 집중
# ═══════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss: 쉬운 샘플의 가중치를 줄여 어려운 샘플에 집중.

    gamma=0이면 CrossEntropyLoss와 동일.
    gamma=2가 기본값 (Lin et al., 2017).
    label_smoothing도 지원.
    """

    def __init__(
        self, gamma: float = 2.0, label_smoothing: float = 0.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)  # 정답 확률
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


def _make_criterion() -> nn.Module:
    """config에 따라 Focal Loss 또는 CrossEntropy를 생성한다."""
    if config.USE_FOCAL_LOSS:
        log(f"    Loss: FocalLoss(gamma={config.FOCAL_GAMMA}, smooth={config.LABEL_SMOOTH})")
        return FocalLoss(
            gamma=config.FOCAL_GAMMA,
            label_smoothing=config.LABEL_SMOOTH,
        )
    return nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTH)


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
        history에 meta 키로 fold별 상세 메타데이터 포함.
    """
    crit = _make_criterion()
    opt  = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    # bfloat16은 FP32와 같은 exponent range → GradScaler 불필요
    use_scaler = config.USE_AMP and config.AMP_DTYPE == torch.float16
    scaler = GradScaler(enabled=use_scaler)

    hist: dict[str, list[float]] = {"tl": [], "ta": [], "vl": [], "va": []}
    best_vl = float("inf")
    best_state: Optional[dict] = None
    patience = 0
    t0 = time.time()

    # v8: fold별 메타데이터 추적
    meta: dict = {
        "tag": tag,
        "train_samples": len(tr_dl.dataset),
        "test_samples": len(te_dl.dataset),
        "batch_size": config.BATCH,
        "oom_events": 0,
        "nan_events": 0,
        "early_stopped": False,
        "early_stop_epoch": 0,
        "total_epochs": 0,
        "best_val_loss": float("inf"),
        "train_time_sec": 0.0,
        "errors": [],
    }

    for ep in range(1, config.EPOCHS + 1):
        try:
            tl, ta, _, _ = _run(
                model, tr_dl, crit, opt, scaler, branch, use_mixup)
            vl, va, _, _ = _run(model, te_dl, crit, branch=branch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                meta["oom_events"] += 1
                meta["errors"].append(f"CUDA OOM at ep{ep}")
                log(f"  ✗ CUDA OOM at ep{ep}! 배치 크기를 줄이세요."
                    f" (현재 BATCH={config.BATCH})")
                if config.USE_GPU:
                    torch.cuda.empty_cache()
                break
            meta["errors"].append(f"RuntimeError at ep{ep}: {str(e)[:100]}")
            raise

        # NaN/Inf 감지 로깅
        if np.isnan(tl) or np.isinf(tl):
            meta["nan_events"] += 1
            meta["errors"].append(f"NaN/Inf loss at ep{ep}")

        sch.step()

        hist["tl"].append(tl); hist["ta"].append(ta)
        hist["vl"].append(vl); hist["va"].append(va)
        meta["total_epochs"] = ep

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
                meta["early_stopped"] = True
                meta["early_stop_epoch"] = ep
                log(f"  {tag} ★ EarlyStop ep{ep}")
                break

    meta["best_val_loss"] = round(best_vl, 6) if best_vl != float("inf") else None
    meta["train_time_sec"] = round(time.time() - t0, 1)

    if best_state:
        _unwrap(model).load_state_dict(best_state)
        model.to(DEVICE)

    # 최종 평가 (TTA 지원)
    try:
        if config.USE_TTA and config.TTA_ROUNDS > 1:
            _, _, preds, labels = _run_tta(
                model, te_dl, crit, branch=branch,
                n_rounds=config.TTA_ROUNDS)
            meta["tta_rounds"] = config.TTA_ROUNDS
        else:
            _, _, preds, labels = _run(model, te_dl, crit, branch=branch)
    except RuntimeError as e:
        meta["oom_events"] += 1
        meta["errors"].append(f"OOM during final eval: {str(e)[:100]}")
        log("  ✗ 최종 평가 중 OOM")
        preds = np.array([], dtype=np.int64)
        labels = np.array([], dtype=np.int64)

    hist["meta"] = meta
    return preds, labels, hist


# ═══════════════════════════════════════════════
# 6. DataLoader 생성
# ═══════════════════════════════════════════════

def make_loader(ds: Dataset, shuffle: bool, branch: bool = False) -> DataLoader:
    """Dataset → DataLoader.

    OTF 모드: ``num_workers=0`` (h5py 호환).
    Preload/Cache 모드: ``config.LOADER_WORKERS`` + ``persistent_workers``.
    Train 모드: ``drop_last=True`` (BatchNorm 안정성 + GPU 활용률).
    클래스 균형 샘플링: shuffle=True + USE_BALANCED_SAMPLER 시 자동 적용.
    """
    from torch.utils.data import WeightedRandomSampler

    is_otf = isinstance(ds, (FlatDatasetOTF, BranchDatasetOTF))
    is_cached = isinstance(ds, (CachedFlatDataset, CachedBranchDataset))
    # OTF: h5py 호환 필수 0, Cached: memmap이라 워커 불필요
    workers = 0 if (is_otf or is_cached) else config.LOADER_WORKERS

    # 클래스 균형 샘플링 (train만)
    sampler = None
    use_shuffle = shuffle
    if shuffle and config.USE_BALANCED_SAMPLER:
        y_arr = _get_dataset_labels(ds)
        if y_arr is not None and len(y_arr) > 0:
            classes, counts = np.unique(y_arr, return_counts=True)
            class_weights = 1.0 / counts.astype(np.float64)
            sample_weights = class_weights[y_arr]
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(y_arr), replacement=True)
            use_shuffle = False  # sampler와 shuffle 동시 불가
            log(f"      ★ 균형 샘플링: {dict(zip(classes.tolist(), counts.tolist()))}")

    kw: dict = dict(
        batch_size=config.BATCH,
        num_workers=workers,
        pin_memory=config.USE_GPU and not is_otf and not is_cached,
        shuffle=use_shuffle,
        drop_last=shuffle and len(ds) > config.BATCH,
    )
    if sampler is not None:
        kw["sampler"] = sampler
        kw["shuffle"] = False
    if workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    if branch:
        kw["collate_fn"] = collate_branch
    return DataLoader(ds, **kw)


def _get_dataset_labels(ds: Dataset) -> Optional[np.ndarray]:
    """Dataset에서 y 라벨 배열을 추출한다."""
    if hasattr(ds, "y_arr"):
        return np.asarray(ds.y_arr)
    if hasattr(ds, "y") and hasattr(ds, "idx"):
        return np.asarray(ds.y[ds.idx])
    return None


# ═══════════════════════════════════════════════
# 6b. TTA (Test Time Augmentation)
# ═══════════════════════════════════════════════

def _run_tta(
    model: nn.Module, loader: DataLoader, crit: nn.Module,
    branch: bool = False, n_rounds: int = 5,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """TTA: 원본 1회 + augmented (n_rounds-1)회 → logits 평균 → argmax.

    test loader는 shuffle=False, drop_last=False 전제.
    """
    from models import augment

    model.eval()

    def _forward_pass(use_aug: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """1회 forward pass → (logits, labels) concat."""
        logit_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in loader:
                if branch:
                    bi, yb = batch
                    bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
                    yb = yb.to(DEVICE, non_blocking=True)
                    if not config.USE_AMP:
                        bi = {k: v.float() for k, v in bi.items()}
                    inp = {k: augment(v, True) for k, v in bi.items()} if use_aug else bi
                else:
                    xb, yb = batch
                    xb = xb.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True)
                    if not config.USE_AMP:
                        xb = xb.float()
                    inp = augment(xb, True) if use_aug else xb

                with autocast(enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model(inp)
                logit_parts.append(logits.detach())
                label_parts.append(yb.detach())

        return torch.cat(logit_parts, dim=0), torch.cat(label_parts, dim=0)

    # Round 0: 원본
    all_logits_sum, labels = _forward_pass(use_aug=False)

    # Round 1~N-1: augmented
    for _ in range(1, n_rounds):
        round_logits, _ = _forward_pass(use_aug=True)
        all_logits_sum = all_logits_sum + round_logits

    # 평균 → 예측
    avg_logits = all_logits_sum / n_rounds
    preds = avg_logits.argmax(dim=1)
    loss = crit(avg_logits, labels).item()
    acc = (preds == labels).float().mean().item()

    return loss, acc, preds.cpu().numpy(), labels.cpu().numpy()


# ═══════════════════════════════════════════════
# 7. 모델별 실행 함수
# ═══════════════════════════════════════════════

def _maybe_compile(model: nn.Module) -> nn.Module:
    """``config.USE_COMPILE`` 이 True이면 ``torch.compile`` 을 시도한다.

    mode="default": 커널 퓨전 최적화 (안전). reduce-overhead는 eval 시
    dynamic batch size에서 CUDA graph 문제 가능.
    """
    if config.USE_COMPILE and hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode="default")
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

    tr_ds = make_flat_dataset(h5data, y, tr_idx, sc, pca,
                              fold_tag=fold_tag, split="train")
    te_ds = make_flat_dataset(h5data, y, te_idx, sc, pca,
                              fold_tag=fold_tag, split="test")
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

    tr_ds = make_branch_dataset(h5data, y, tr_idx, bsc, branch_idx,
                                fold_tag=fold_tag, split="train")
    te_ds = make_branch_dataset(h5data, y, te_idx, bsc, branch_idx,
                                fold_tag=fold_tag, split="test")
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