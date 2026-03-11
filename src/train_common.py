"""
train_common.py — K-Fold / LOSO 공용 유틸 (v8.8)
═══════════════════════════════════════════════════════
v8.8 변경:
  - _run_tta._forward_pass: dict output(M7_Attr/M8) logits 추출 버그 수정
  - threshold_search() 추가 (softmax multiplier grid → macro F1 최적화)
  - compute_macro_f1_with_thresholds() 추가
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from wandb_init import wandb_log_epoch as _wandb_log_epoch
try:
    import wandb as _wandb
except ImportError:
    _wandb = None
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
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
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class Timer:
    def __enter__(self):
        self._start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
    def __str__(self):
        return f"{self.elapsed:.1f}s"


def _mem_str() -> str:
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
    if not config.USE_GPU:
        return ""
    alloc = torch.cuda.memory_allocated() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return f"  GPU={alloc:.0f}/{total:.0f}MB"


# ═══════════════════════════════════════════════
# 1. HDF5 데이터 핸들러
# ═══════════════════════════════════════════════

class H5Data:
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
        self.channels: list[str] = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in self.h5f["channels"][:]
        ]
        subj_grp = self.h5f["subjects"]
        skeys = sorted(subj_grp.keys())

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

        self.y_raw   = np.concatenate(y_parts)  if y_parts  else np.array([], dtype=np.int64)
        self.subj_id = np.concatenate(sid_parts) if sid_parts else np.array([], dtype=np.int64)
        self.N = len(self._idx_map)
        if self.N > 0:
            sample_key = skeys[0]
            self.T = subj_grp[sample_key]["X"].shape[1]
            self.C = subj_grp[sample_key]["X"].shape[2]
        else:
            self.T, self.C = 0, 0

    def _init_v7(self) -> None:
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

    @property
    def X(self) -> np.ndarray:
        return self.read_X(np.arange(self.N, dtype=np.int64))

    def read_X(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return np.empty((0, self.T, self.C), dtype=np.float32)

        if not self._is_v8:
            sort_order  = np.argsort(indices)
            sorted_idx  = indices[sort_order]
            chunk_size  = config.H5_READ_CHUNK
            parts: list[np.ndarray] = []
            for s in range(0, len(sorted_idx), chunk_size):
                chunk_idx = sorted_idx[s : s + chunk_size]
                parts.append(self.X_ds[chunk_idx.tolist()])
            data_sorted = np.concatenate(parts, axis=0)
            return data_sorted[np.argsort(sort_order)].astype(np.float32)

        subj_grp = self.h5f["subjects"]
        out = np.empty((len(indices), self.T, self.C), dtype=np.float32)

        from collections import defaultdict as _ddict
        skey_batches: dict[str, list[tuple[int, int]]] = _ddict(list)
        for out_i, global_i in enumerate(indices):
            skey, local_i = self._idx_map[global_i]
            skey_batches[skey].append((out_i, local_i))

        for skey, pairs in skey_batches.items():
            ds      = subj_grp[f"{skey}/X"]
            out_idx = np.array([p[0] for p in pairs])
            loc_idx = np.array([p[1] for p in pairs])
            order   = np.argsort(loc_idx)
            chunk   = ds[loc_idx[order].tolist()]
            out[out_idx[order]] = chunk

        return out

    def read_X_chunk(self, indices: np.ndarray, start: int, size: int) -> np.ndarray:
        return self.read_X(indices[start : start + size])

    def read_single(self, idx: int) -> np.ndarray:
        idx = int(idx)
        if idx < 0 or idx >= self.N:
            raise IndexError(f"인덱스 {idx} 범위 초과 (N={self.N})")
        if not self._is_v8:
            return self.X_ds[idx].astype(np.float32)
        skey, local_i = self._idx_map[idx]
        return self.h5f[f"subjects/{skey}/X"][local_i].astype(np.float32)

    def close(self) -> None:
        if hasattr(self, "h5f") and self.h5f.id.valid:
            self.h5f.close()

    def __enter__(self) -> H5Data:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ═══════════════════════════════════════════════
# 2. Scaler / PCA fit
# ═══════════════════════════════════════════════

def fit_pca_on_train(
    h5data: H5Data, tr_idx: np.ndarray,
) -> tuple[StandardScaler, IncrementalPCA]:
    t0     = time.time()
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    n, T, C = len(tr_idx), h5data.T, h5data.C
    chunk   = config.IPCA_CHUNK

    sc = StandardScaler()
    for s in range(0, n, chunk):
        X = h5data.read_X_chunk(tr_idx, s, chunk)
        sc.partial_fit(X.reshape(X.shape[0] * T, C))
        del X; gc.collect()

    n_components = min(config.PCA_CH, C, n * T)
    ipca = IncrementalPCA(n_components=n_components)
    for s in range(0, n, chunk):
        X      = h5data.read_X_chunk(tr_idx, s, chunk)
        flat   = X.reshape(X.shape[0] * T, C)
        scaled = sc.transform(flat)
        if np.isnan(scaled).any():
            np.nan_to_num(scaled, copy=False, nan=0.0)
        ipca.partial_fit(scaled)
        del X, flat, scaled; gc.collect()

    evr = ipca.explained_variance_ratio_.sum()
    log(f"    PCA fit: {C}→{n_components}  n={n}  EVR={evr:.3f}  ({time.time()-t0:.1f}s)")
    return sc, ipca


def fit_bsc_on_train(h5data: H5Data, tr_idx: np.ndarray) -> StandardScaler:
    t0     = time.time()
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    n, T, C = len(tr_idx), h5data.T, h5data.C
    chunk   = config.IPCA_CHUNK

    bsc = StandardScaler()
    for s in range(0, n, chunk):
        X = h5data.read_X_chunk(tr_idx, s, chunk)
        bsc.partial_fit(X.reshape(X.shape[0] * T, C))
        del X; gc.collect()

    log(f"    BSC fit: n={n}  ({time.time()-t0:.1f}s)")
    return bsc


# ═══════════════════════════════════════════════
# 2b. PCA/BSC 디스크 캐시
# ═══════════════════════════════════════════════

CACHE_DIR = Path(os.environ.get(
    "CACHE_DIR", str(Path(__file__).resolve().parent.parent / "cache")
))


def _cache_path(fold_tag: str | int, split: str, kind: str) -> Path:
    return CACHE_DIR / f"F{fold_tag}_{split}_{kind}.npy"


def _cache_y_path(fold_tag: str | int, split: str) -> Path:
    return CACHE_DIR / f"F{fold_tag}_{split}_y.npy"


def cache_flat_transform(
    h5data: H5Data, y: np.ndarray, indices: np.ndarray,
    sc: StandardScaler, pca: IncrementalPCA,
    fold_tag: str | int, split: str,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_f = _cache_path(fold_tag, split, "flat")
    y_f     = _cache_y_path(fold_tag, split)

    if cache_f.exists() and y_f.exists():
        log(f"      ★ 캐시 히트: {cache_f.name}")
        return cache_f

    t0      = time.time()
    indices = np.asarray(indices, dtype=np.int64)
    T       = h5data.T
    chunk   = config.DS_CHUNK
    n       = len(indices)

    out = np.lib.format.open_memmap(
        str(cache_f), mode="w+", dtype=np.float16,
        shape=(n, config.PCA_CH, T),
    )
    pos = 0
    for s in range(0, n, chunk):
        X      = h5data.read_X_chunk(indices, s, chunk)
        n_c    = X.shape[0]
        flat   = X.reshape(n_c * T, h5data.C)
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    y_f         = _cache_y_path(fold_tag, split)
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

    t0      = time.time()
    indices = np.asarray(indices, dtype=np.int64)
    T       = h5data.T
    chunk   = config.DS_CHUNK
    n       = len(indices)

    mms: dict[str, np.ndarray] = {}
    for nm, il in branch_idx.items():
        mms[nm] = np.lib.format.open_memmap(
            str(cache_paths[nm]), mode="w+", dtype=np.float16,
            shape=(n, len(il), T),
        )

    pos = 0
    for s in range(0, n, chunk):
        X      = h5data.read_X_chunk(indices, s, chunk)
        n_c    = X.shape[0]
        flat   = X.reshape(n_c * T, h5data.C)
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
    log(f"      Branch 캐시 저장: {len(branch_idx)}그룹  ({total_mb:.0f}MB)  ({time.time()-t0:.1f}s)")
    return cache_paths


def clear_fold_cache(fold_tag: str | int) -> None:
    if not CACHE_DIR.exists():
        return
    for f in CACHE_DIR.glob(f"F{fold_tag}_*"):
        f.unlink()


# ═══════════════════════════════════════════════
# 3. Dataset
# ═══════════════════════════════════════════════

class CachedFlatDataset(Dataset):
    def __init__(self, cache_path: Path, y_path: Path) -> None:
        self.data  = np.load(str(cache_path), mmap_mode="r")
        self.y_arr = np.load(str(y_path))
        mb = cache_path.stat().st_size / 1024**2
        log(f"      FlatDS(Cache): {self.data.shape}  ({mb:.0f}MB)")

    def __len__(self) -> int: return len(self.y_arr)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.data[i].copy()),
            torch.tensor(int(self.y_arr[i]), dtype=torch.long),
        )


class CachedBranchDataset(Dataset):
    def __init__(self, cache_paths: dict[str, Path], y_path: Path) -> None:
        self.groups = {nm: np.load(str(p), mmap_mode="r") for nm, p in cache_paths.items()}
        self.y_arr  = np.load(str(y_path))
        total_mb = sum(p.stat().st_size for p in cache_paths.values()) / 1024**2
        log(f"      BranchDS(Cache): n={len(self.y_arr)}  {len(cache_paths)}그룹  ({total_mb:.0f}MB)")

    def __len__(self) -> int: return len(self.y_arr)

    def __getitem__(self, i: int):
        bi = {nm: torch.from_numpy(arr[i].copy()) for nm, arr in self.groups.items()}
        return bi, torch.tensor(int(self.y_arr[i]), dtype=torch.long)


class FlatDatasetPreload(Dataset):
    def __init__(self, h5data, y, indices, sc, pca):
        t0 = time.time(); indices = np.asarray(indices, dtype=np.int64)
        self.y_arr = y[indices].astype(np.int64)
        T = h5data.T; chunk = config.DS_CHUNK; parts = []
        for s in range(0, len(indices), chunk):
            X = h5data.read_X_chunk(indices, s, chunk); n_c = X.shape[0]
            flat = X.reshape(n_c * T, h5data.C); scaled = sc.transform(flat)
            if np.isnan(scaled).any(): np.nan_to_num(scaled, copy=False, nan=0.0)
            pca_out = pca.transform(scaled).reshape(n_c, T, config.PCA_CH)
            parts.append(np.ascontiguousarray(pca_out.transpose(0, 2, 1)))
            del X, flat, scaled, pca_out; gc.collect()
        self.data = np.concatenate(parts, axis=0).astype(np.float16)
        log(f"      FlatDS(Preload/fp16): {self.data.shape}  ({self.data.nbytes/1024**2:.0f}MB)  ({time.time()-t0:.1f}s)")

    def __len__(self): return len(self.y_arr)
    def __getitem__(self, i):
        return torch.from_numpy(self.data[i].copy()), torch.tensor(int(self.y_arr[i]), dtype=torch.long)


class BranchDatasetPreload(Dataset):
    def __init__(self, h5data, y, indices, bsc, branch_idx):
        t0 = time.time(); indices = np.asarray(indices, dtype=np.int64)
        self.y_arr = y[indices].astype(np.int64)
        T = h5data.T; chunk = config.DS_CHUNK
        group_parts = {nm: [] for nm in branch_idx}
        for s in range(0, len(indices), chunk):
            X = h5data.read_X_chunk(indices, s, chunk); n_c = X.shape[0]
            flat = X.reshape(n_c * T, h5data.C); scaled = bsc.transform(flat)
            if np.isnan(scaled).any(): np.nan_to_num(scaled, copy=False, nan=0.0)
            scaled = scaled.reshape(n_c, T, h5data.C)
            for nm, il in branch_idx.items():
                group_parts[nm].append(np.ascontiguousarray(scaled[:, :, il].transpose(0, 2, 1)))
            del X, flat, scaled; gc.collect()
        self.groups = {nm: np.concatenate(p, 0).astype(np.float16) for nm, p in group_parts.items()}
        del group_parts; gc.collect()
        total_mb = sum(v.nbytes for v in self.groups.values()) / 1024**2
        log(f"      BranchDS(Preload/fp16): n={len(indices)}  {len(branch_idx)}그룹  ({total_mb:.0f}MB)  ({time.time()-t0:.1f}s)")

    def __len__(self): return len(self.y_arr)
    def __getitem__(self, i):
        bi = {nm: torch.from_numpy(arr[i].copy()) for nm, arr in self.groups.items()}
        return bi, torch.tensor(int(self.y_arr[i]), dtype=torch.long)


class FlatDatasetOTF(Dataset):
    def __init__(self, h5data, y, indices, sc, pca):
        self.h5data = h5data; self.y = y
        self.idx = np.asarray(indices, dtype=np.int64)
        self.sc = sc; self.pca = pca
        log(f"      FlatDS(OTF): n={len(self.idx)}  (RAM ≈ 0MB)")

    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        raw = self.h5data.read_single(self.idx[i])
        scaled = self.sc.transform(raw); pca_out = self.pca.transform(scaled)
        x = np.ascontiguousarray(pca_out.T.astype(np.float32))
        return torch.from_numpy(x), torch.tensor(int(self.y[self.idx[i]]), dtype=torch.long)


class BranchDatasetOTF(Dataset):
    def __init__(self, h5data, y, indices, bsc, branch_idx):
        self.h5data = h5data; self.y = y
        self.idx = np.asarray(indices, dtype=np.int64)
        self.bsc = bsc; self.branch_idx = branch_idx
        log(f"      BranchDS(OTF): n={len(self.idx)}  {len(branch_idx)}그룹  (RAM ≈ 0MB)")

    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        raw = self.h5data.read_single(self.idx[i]); scaled = self.bsc.transform(raw)
        bi = {}
        for nm, il in self.branch_idx.items():
            arr = np.ascontiguousarray(scaled[:, il].T.astype(np.float32))
            bi[nm] = torch.from_numpy(arr)
        return bi, torch.tensor(int(self.y[self.idx[i]]), dtype=torch.long)


def make_flat_dataset(h5data, y, indices, sc, pca, fold_tag="", split=""):
    if fold_tag:
        cache_f = cache_flat_transform(h5data, y, indices, sc, pca, fold_tag, split)
        y_f = _cache_y_path(fold_tag, split)
        return CachedFlatDataset(cache_f, y_f)
    if config.USE_PRELOAD_M1:
        return FlatDatasetPreload(h5data, y, indices, sc, pca)
    return FlatDatasetOTF(h5data, y, indices, sc, pca)


def make_branch_dataset(h5data, y, indices, bsc, branch_idx, total_samples=0, fold_tag="", split=""):
    if fold_tag:
        cache_paths = cache_branch_transform(h5data, y, indices, bsc, branch_idx, fold_tag, split)
        y_f = _cache_y_path(fold_tag, split)
        return CachedBranchDataset(cache_paths, y_f)
    if config.USE_PRELOAD:
        return BranchDatasetPreload(h5data, y, indices, bsc, branch_idx)
    n_total = total_samples if total_samples > 0 else len(indices) * 2
    if config.can_preload_branch(n_total, h5data.C, h5data.T):
        log(f"      ★ 스마트 Preload: {n_total}샘플 × {h5data.C}ch → RAM 가능")
        return BranchDatasetPreload(h5data, y, indices, bsc, branch_idx)
    return BranchDatasetOTF(h5data, y, indices, bsc, branch_idx)


def collate_branch(batch):
    bis, ys = zip(*batch)
    return (
        {k: torch.stack([b[k] for b in bis]) for k in bis[0]},
        torch.stack(ys),
    )


# ═══════════════════════════════════════════════
# 4. Mixup
# ═══════════════════════════════════════════════

def mixup_data(x, y, alpha=config.MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(max(np.random.beta(alpha, alpha), 0.5))
    idx = torch.randperm(y.size(0), device=y.device)
    if isinstance(x, tuple):
        bi, feat = x
        mixed_bi   = {k: lam * v + (1 - lam) * v[idx] for k, v in bi.items()}
        mixed_feat = lam * feat + (1 - lam) * feat[idx]
        mixed = (mixed_bi, mixed_feat)
    elif isinstance(x, dict):
        mixed = {k: lam * v + (1 - lam) * v[idx] for k, v in x.items()}
    else:
        mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)


# ═══════════════════════════════════════════════
# 5. 학습 루프
# ═══════════════════════════════════════════════

def _run(model, loader, crit, opt=None, scaler=None, branch=False, use_mixup=False):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss = 0.0; correct = total = 0; nan_batches = 0

    ctx = torch.enable_grad() if train else torch.inference_mode()
    step_idx = 0
    preds_gpu = []; labels_gpu = []

    with ctx:
        if train:
            opt.zero_grad(set_to_none=True)
        for batch in loader:
            if branch:
                if len(batch) == 3:
                    bi, feat, yb = batch
                    feat = feat.to(DEVICE, non_blocking=True)
                    if not config.USE_AMP: feat = feat.float()
                else:
                    bi, yb = batch; feat = None
                bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
                yb = yb.to(DEVICE, non_blocking=True)
                if not config.USE_AMP: bi = {k: v.float() for k, v in bi.items()}
                inp = (bi, feat) if feat is not None else bi
            else:
                xb, yb = batch
                xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
                if not config.USE_AMP: xb = xb.float()
                inp = xb

            if train and use_mixup and config.MIXUP_ALPHA > 0:
                inp, ya, yb_mix, lam = mixup_data(inp, yb)
            else:
                ya = yb_mix = yb; lam = 1.0

            with autocast("cuda", enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                out = model(*inp) if isinstance(inp, tuple) else model(inp)
                if isinstance(out, dict):
                    logits = out["final_logits"]
                    loss   = crit(out, yb) if lam >= 1.0 else (
                        lam * crit(out, ya) + (1 - lam) * crit(out, yb_mix))
                else:
                    logits = out
                    loss   = (mixup_criterion(crit, logits, ya, yb_mix, lam)
                              if lam < 1.0 else crit(logits, yb))

            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                if nan_batches <= 3: log("  ⚠ NaN/Inf 손실 감지 (배치 스킵)")
                continue

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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                        scaler.step(opt); scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                        opt.step()
                    opt.zero_grad(set_to_none=True)

            bs = len(yb)
            total_loss += loss.item() * bs
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item(); total += bs
            preds_gpu.append(pred.detach()); labels_gpu.append(yb.detach())

    if nan_batches > 0:
        log(f"  ⚠ NaN/Inf 배치: {nan_batches}개 스킵됨")

    if train and step_idx % config.GRAD_ACCUM_STEPS != 0:
        if scaler is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            opt.step()
        opt.zero_grad(set_to_none=True)

    avg_loss   = total_loss / max(total, 1)
    acc        = correct / max(total, 1)
    preds_all  = torch.cat(preds_gpu).cpu().numpy()  if preds_gpu  else np.array([], dtype=np.int64)
    labels_all = torch.cat(labels_gpu).cpu().numpy() if labels_gpu else np.array([], dtype=np.int64)
    return avg_loss, acc, preds_all, labels_all


def _build_attr_targets(y):
    import torch.nn.functional as _F
    slip  = (y == 0).float()
    slope = torch.ones_like(y)
    slope = torch.where(y == 1, torch.full_like(y, 2), slope)
    slope = torch.where(y == 2, torch.zeros_like(y),   slope)
    return slip, slope, (y==3).float(), (y==4).float(), (y==5).float()


class AttributeMultiTaskLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smooth=0.1):
        super().__init__()
        self.ce_final = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        self.ce_slope = nn.CrossEntropyLoss()
        self.ws = dict(final=1., slip=.5, slope=.5, irreg=.4, comp=.4, flat=1.2)

    def forward(self, out, y):
        import torch.nn.functional as F
        if not isinstance(out, dict):
            return self.ce_final(out, y)
        slip, slope, irreg, comp, flat = _build_attr_targets(y)
        return (self.ws["final"] * self.ce_final(out["final_logits"], y)
              + self.ws["slip"]  * F.binary_cross_entropy_with_logits(out["slip_logit"],  slip)
              + self.ws["slope"] * self.ce_slope(out["slope_logits"], slope)
              + self.ws["irreg"] * F.binary_cross_entropy_with_logits(out["irreg_logit"], irreg)
              + self.ws["comp"]  * F.binary_cross_entropy_with_logits(out["comp_logit"],  comp)
              + self.ws["flat"]  * F.binary_cross_entropy_with_logits(out["flat_logit"],  flat))


# ═══════════════════════════════════════════════
# 5b. Focal Loss
# ═══════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def _make_criterion(model=None):
    if model is not None and getattr(model, "IS_HYBRID", False) and hasattr(model, "slip_head"):
        log("    Loss: AttributeMultiTaskLoss (label_smooth=0.0)")
        return AttributeMultiTaskLoss(gamma=config.FOCAL_GAMMA, label_smooth=0.0)
    if config.USE_FOCAL_LOSS:
        log(f"    Loss: FocalLoss(gamma={config.FOCAL_GAMMA}, smooth={config.LABEL_SMOOTH})")
        return FocalLoss(gamma=config.FOCAL_GAMMA, label_smoothing=config.LABEL_SMOOTH)
    return nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTH)


def train_model(model, tr_dl, te_dl, branch=False, tag="", use_mixup=True):
    crit = _make_criterion(model)
    # M7_Attr: 보조 헤드가 많아 LR 낮춤
    base_lr = config.LR * 0.5 if (hasattr(model, "slip_head")) else config.LR
    opt  = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=config.WEIGHT_DECAY)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    use_scaler = config.USE_AMP and config.AMP_DTYPE == torch.float16
    scaler = GradScaler("cuda", enabled=use_scaler)

    hist = {"tl": [], "ta": [], "vl": [], "va": []}
    best_vl = float("inf"); best_state = None; patience = 0; t0 = time.time()

    meta = {
        "tag": tag, "train_samples": len(tr_dl.dataset), "test_samples": len(te_dl.dataset),
        "batch_size": config.BATCH, "oom_events": 0, "nan_events": 0,
        "early_stopped": False, "early_stop_epoch": 0,
        "total_epochs": 0, "best_val_loss": float("inf"),
        "train_time_sec": 0.0, "errors": [],
    }

    for ep in range(1, config.EPOCHS + 1):
        try:
            tl, ta, _, _ = _run(model, tr_dl, crit, opt, scaler, branch, use_mixup)
            vl, va, _, _ = _run(model, te_dl, crit, branch=branch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                meta["oom_events"] += 1; meta["errors"].append(f"CUDA OOM at ep{ep}")
                log(f"  ✗ CUDA OOM at ep{ep}!")
                if config.USE_GPU: torch.cuda.empty_cache()
                break
            meta["errors"].append(f"RuntimeError at ep{ep}: {str(e)[:100]}"); raise

        if np.isnan(tl) or np.isinf(tl):
            meta["nan_events"] += 1; meta["errors"].append(f"NaN/Inf loss at ep{ep}")

        sch.step()
        hist["tl"].append(tl); hist["ta"].append(ta)
        hist["vl"].append(vl); hist["va"].append(va)
        meta["total_epochs"] = ep

        log(f"  {tag} ep{ep:02d}/{config.EPOCHS}"
            f"  loss={tl:.4f}/{ta:.4f}  val={vl:.4f}/{va:.4f}"
            f"  lr={opt.param_groups[0]['lr']:.1e}"
            f"  ({time.time()-t0:.0f}s)  RAM={_mem_str()}{_gpu_mem_str()}")

        _wandb_log_epoch(tag, ep, tl, ta, vl, va, opt.param_groups[0]["lr"])

        if vl < best_vl:
            best_vl = vl
            _m = _unwrap(model)
            best_state = {k: v.cpu().clone() for k, v in _m.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP:
                meta["early_stopped"] = True; meta["early_stop_epoch"] = ep
                log(f"  {tag} ★ EarlyStop ep{ep}"); break

    meta["best_val_loss"]  = round(best_vl, 6) if best_vl != float("inf") else None
    meta["train_time_sec"] = round(time.time() - t0, 1)

    if best_state:
        _unwrap(model).load_state_dict(best_state); model.to(DEVICE)

    try:
        if config.USE_TTA and config.TTA_ROUNDS > 1:
            _, _, preds, labels = _run_tta(model, te_dl, crit, branch=branch, n_rounds=config.TTA_ROUNDS)
            meta["tta_rounds"] = config.TTA_ROUNDS
        else:
            _, _, preds, labels = _run(model, te_dl, crit, branch=branch)
    except RuntimeError as e:
        meta["oom_events"] += 1; meta["errors"].append(f"OOM during final eval: {str(e)[:100]}")
        log("  ✗ 최종 평가 중 OOM")
        preds = labels = np.array([], dtype=np.int64)

    hist["meta"] = meta
    return preds, labels, hist


# ═══════════════════════════════════════════════
# 6. DataLoader
# ═══════════════════════════════════════════════

def make_loader(ds, shuffle, branch=False):
    from torch.utils.data import WeightedRandomSampler

    is_otf    = isinstance(ds, (FlatDatasetOTF, BranchDatasetOTF))
    is_cached = isinstance(ds, (CachedFlatDataset, CachedBranchDataset))
    workers   = 0 if (is_otf or is_cached) else config.LOADER_WORKERS

    sampler = None; use_shuffle = shuffle
    if shuffle and config.USE_BALANCED_SAMPLER:
        y_arr = _get_dataset_labels(ds)
        if y_arr is not None and len(y_arr) > 0:
            classes, counts = np.unique(y_arr, return_counts=True)
            class_weights   = 1.0 / counts.astype(np.float64)
            # C4/C5/C6 (index 3,4,5) 추가 부스트 × 2.0
            boost = np.array([2.0 if int(c) in (3,4,5) else 1.0 for c in classes])
            class_weights   = class_weights * boost
            sample_weights  = class_weights[np.searchsorted(classes, y_arr)]
            sampler         = WeightedRandomSampler(sample_weights, len(y_arr), replacement=True)
            use_shuffle     = False
            log(f"      ★ 균형 샘플링(C4/C5/C6 ×2): {dict(zip(classes.tolist(), counts.tolist()))}")

    kw = dict(
        batch_size=config.BATCH, num_workers=workers,
        pin_memory=config.USE_GPU and not is_otf and not is_cached,
        shuffle=use_shuffle, drop_last=shuffle and len(ds) > config.BATCH,
    )
    if sampler is not None:
        kw["sampler"] = sampler; kw["shuffle"] = False
    if workers > 0:
        kw["persistent_workers"] = True; kw["prefetch_factor"] = 2
    if branch:
        kw["collate_fn"] = collate_branch
    return DataLoader(ds, **kw)


def _get_dataset_labels(ds):
    if hasattr(ds, "y_arr"): return np.asarray(ds.y_arr)
    if hasattr(ds, "y") and hasattr(ds, "idx"): return np.asarray(ds.y[ds.idx])
    return None


# ═══════════════════════════════════════════════
# 6b. TTA  ← v8.8 버그 수정: dict logits 처리
# ═══════════════════════════════════════════════

def _run_tta(model, loader, crit, branch=False, n_rounds=5):
    from models import augment

    model.eval()

    def _forward_pass(use_aug: bool):
        logit_parts = []; label_parts = []
        with torch.inference_mode():
            for batch in loader:
                if branch:
                    if len(batch) == 3:
                        bi, feat, yb = batch
                        feat = feat.to(DEVICE, non_blocking=True)
                        if not config.USE_AMP: feat = feat.float()
                    else:
                        bi, yb = batch; feat = None
                    bi = {k: v.to(DEVICE, non_blocking=True) for k, v in bi.items()}
                    yb = yb.to(DEVICE, non_blocking=True)
                    if not config.USE_AMP: bi = {k: v.float() for k, v in bi.items()}
                    if use_aug:
                        bi_aug = {k: augment(v, True) for k, v in bi.items()}
                        inp = (bi_aug, feat) if feat is not None else bi_aug
                    else:
                        inp = (bi, feat) if feat is not None else bi
                else:
                    xb, yb = batch
                    xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
                    if not config.USE_AMP: xb = xb.float()
                    inp = augment(xb, True) if use_aug else xb
                with autocast("cuda", enabled=config.USE_AMP, dtype=config.AMP_DTYPE):
                    logits = model(*inp) if isinstance(inp, tuple) else model(inp)
                # ── v8.8 버그 수정: dict output (M7_Attr, M8 등) 처리 ──
                if isinstance(logits, dict):
                    logits = logits["final_logits"]
                logit_parts.append(logits.detach())
                label_parts.append(yb.detach())
        return torch.cat(logit_parts, dim=0), torch.cat(label_parts, dim=0)

    all_logits_sum, labels = _forward_pass(use_aug=False)
    for _ in range(1, n_rounds):
        round_logits, _ = _forward_pass(use_aug=True)
        all_logits_sum  = all_logits_sum + round_logits

    avg_logits = all_logits_sum / n_rounds
    preds = avg_logits.argmax(dim=1)
    loss  = crit(avg_logits, labels).item()
    acc   = (preds == labels).float().mean().item()
    return loss, acc, preds.cpu().numpy(), labels.cpu().numpy()


# ═══════════════════════════════════════════════
# 7. 모델별 실행 함수
# ═══════════════════════════════════════════════

def _maybe_compile(model):
    if config.USE_COMPILE and hasattr(torch, "compile"):
        try: return torch.compile(model, mode="default")
        except Exception: return model
    return model


def _maybe_parallel(model):
    if config.N_GPU > 1:
        model = nn.DataParallel(model)
        log(f"     ★ DataParallel: {config.N_GPU} GPUs")
    return model


def _unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _prepare_model(model):
    model = model.to(DEVICE)
    model = _maybe_compile(model)
    model = _maybe_parallel(model)
    return model


def run_M1(h5data, y, tr_idx, te_idx, sc, pca, fold_tag):
    from models import M1_FlatCNN, count_parameters
    log(f"── M1  tr={len(tr_idx)}  te={len(te_idx)}")
    tr_ds = make_flat_dataset(h5data, y, tr_idx, sc, pca, fold_tag=fold_tag, split="train")
    te_ds = make_flat_dataset(h5data, y, te_idx, sc, pca, fold_tag=fold_tag, split="test")
    model = _prepare_model(M1_FlatCNN())
    log(f"     params={count_parameters(_unwrap(model)):,}")
    p, lb, h = train_model(model, make_loader(tr_ds, True), make_loader(te_ds, False), False, f"[F{fold_tag}][M1]")
    acc = accuracy_score(lb, p) if len(lb) > 0 else 0.0
    log(f"── M1 Acc={acc:.4f}")
    del model, tr_ds, te_ds; gc.collect()
    if config.USE_GPU: torch.cuda.empty_cache()
    return {"M1_CNN": p}, lb, {"M1_CNN": h}


def run_branch(h5data, y, tr_idx, te_idx, branch_idx, branch_ch, bsc, model_fn, mname, fold_tag):
    from models import count_parameters
    log(f"── {mname}  tr={len(tr_idx)}  te={len(te_idx)}")
    tr_ds = make_branch_dataset(h5data, y, tr_idx, bsc, branch_idx, fold_tag=fold_tag, split="train")
    te_ds = make_branch_dataset(h5data, y, te_idx, bsc, branch_idx, fold_tag=fold_tag, split="test")
    model = _prepare_model(model_fn(branch_ch))
    log(f"     params={count_parameters(_unwrap(model)):,}")
    p, lb, h = train_model(model, make_loader(tr_ds, True, True), make_loader(te_ds, False, True), True, f"[F{fold_tag}][{mname}]")
    acc = accuracy_score(lb, p) if len(lb) > 0 else 0.0
    log(f"── {mname} Acc={acc:.4f}")
    del model, tr_ds, te_ds; gc.collect()
    if config.USE_GPU: torch.cuda.empty_cache()
    return {f"{mname}_CNN": p}, lb, {f"{mname}_CNN": h}


# ═══════════════════════════════════════════════
# 8. 결과 저장
# ═══════════════════════════════════════════════

def save_report(preds, labels, le, tag, out_dir):
    if len(preds) == 0 or len(labels) == 0:
        log(f"  RESULT {tag:<30} 데이터 없음"); return 0.0, 0.0
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, average="macro", zero_division=0)
    names = [str(c) for c in le.classes_]
    rep   = classification_report(labels, preds, target_names=names, digits=4, zero_division=0)
    log(f"  RESULT {tag:<30} Acc={acc:.4f}  F1={f1:.4f}")
    (out_dir / f"report_{tag}.txt").write_text(f"{tag}\nAcc={acc:.4f}  F1={f1:.4f}\n\n{rep}")
    return acc, f1


def save_cm(preds, labels, le, tag, out_dir):
    if len(preds) == 0: return
    cm = confusion_matrix(labels, preds); names = [str(c) for c in le.classes_]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=names, yticklabels=names,
           xlabel="Predicted", ylabel="True",
           title=f"{tag}  Acc={accuracy_score(labels, preds):.4f}")
    thr = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=9)
    plt.tight_layout(); plt.savefig(out_dir / f"cm_{tag}.png", dpi=150); plt.close()


def save_history(all_hist, out_dir):
    for mname, fold_hists in all_hist.items():
        if not fold_hists: continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for h in fold_hists:
            if h.get("tl"): axes[0].plot(h["tl"], alpha=0.3, color="C0")
            if h.get("vl"): axes[0].plot(h["vl"], alpha=0.3, color="C1")
            if h.get("ta"): axes[1].plot(h["ta"], alpha=0.3, color="C0")
            if h.get("va"): axes[1].plot(h["va"], alpha=0.3, color="C1")
        axes[0].set(title=f"{mname} Loss", xlabel="Epoch", ylabel="Loss")
        axes[0].legend(["Train", "Val"])
        axes[1].set(title=f"{mname} Acc", xlabel="Epoch", ylabel="Accuracy")
        axes[1].legend(["Train", "Val"])
        plt.tight_layout(); plt.savefig(out_dir / f"history_{mname}.png", dpi=150); plt.close()


def save_summary_table(results, out_dir):
    import pandas as pd
    df = pd.DataFrame(results)
    out = Path(out_dir) / "summary_table.csv"
    df.to_csv(out, index=False)
    log(f"  📊 Summary table saved → {out}")


# ═══════════════════════════════════════════════
# 9. Threshold / Ensemble 유틸  (v8.8 신규)
# ═══════════════════════════════════════════════

def threshold_search(
    proba: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 6,
    grid: tuple = (0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),
) -> tuple[np.ndarray, float]:
    """softmax multiplier grid search → macro F1 최적화.

    proba  : (N, n_classes) softmax 확률
    labels : (N,) 정수 레이블
    반환   : (mults, best_f1)  mults shape=(n_classes,)
    """
    mults = np.ones(n_classes, dtype=np.float32)
    for ci in range(n_classes):
        best_f1, best_m = -1.0, 1.0
        for m in grid:
            tmp = mults.copy(); tmp[ci] = m
            pred = (proba * tmp).argmax(1)
            score = f1_score(labels, pred, average="macro", zero_division=0)
            if score > best_f1:
                best_f1, best_m = score, m
        mults[ci] = best_m
    final_pred = (proba * mults).argmax(1)
    best_f1 = f1_score(labels, final_pred, average="macro", zero_division=0)
    return mults, best_f1


def compute_macro_f1_with_thresholds(
    proba: np.ndarray,
    labels: np.ndarray,
    mults: np.ndarray,
) -> float:
    """저장된 mults 적용 후 macro F1 계산."""
    pred = (proba * mults).argmax(1)
    return f1_score(labels, pred, average="macro", zero_division=0)


# ── 하위 호환 alias ──
fit_model = train_model