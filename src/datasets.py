"""src/datasets.py — Dataset / DataLoader / H5 로더 / 캐시."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from channel_groups import build_branch_idx


# ─────────────────────────────────────────────
# Dataset 클래스
# ─────────────────────────────────────────────

class FlatDataset(Dataset):
    """PCA 압축된 flat 입력 (M1 전용)."""
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class BranchDataset(Dataset):
    """5-브랜치 분기 입력 (M2–M7, ResNet1D, CNNTCN, ResNetTCN)."""
    def __init__(
        self,
        x: np.ndarray,                          # (N, C, T)
        y: np.ndarray,
        branch_idx: Dict[str, List[int]],
    ) -> None:
        self.x          = x.astype(np.float32)
        self.y          = y.astype(np.int64)
        self.branch_idx = branch_idx

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        xi = self.x[idx]
        bi = {nm: torch.from_numpy(xi[idxs]) for nm, idxs in self.branch_idx.items()}
        return bi, torch.tensor(self.y[idx], dtype=torch.long)


class HierarchicalDataset(Dataset):
    """브랜치 입력 + 44개 핸드크래프트 피처 (M7, HierarchicalFusionNet)."""
    def __init__(
        self,
        x: np.ndarray,                          # (N, C, T)
        feat44: np.ndarray,                     # (N, 44)
        y: np.ndarray,
        branch_idx: Dict[str, List[int]],
    ) -> None:
        assert len(x) == len(feat44) == len(y)
        self.x          = x.astype(np.float32)
        self.feat44     = torch.from_numpy(feat44.astype(np.float32))
        self.y          = y.astype(np.int64)
        self.branch_idx = branch_idx

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        xi = self.x[idx]
        bi = {nm: torch.from_numpy(xi[idxs]) for nm, idxs in self.branch_idx.items()}
        return bi, self.feat44[idx], torch.tensor(self.y[idx], dtype=torch.long)


# ─────────────────────────────────────────────
# H5 데이터 로더
# ─────────────────────────────────────────────

class H5Data:
    """HDF5 데이터셋 래퍼. v7(flat) / v8(subject-group) 양쪽 지원."""

    def __init__(self, path: str) -> None:
        self._f  = h5py.File(path, "r")
        self._load()

    def _load(self) -> None:
        f = self._f
        if "X" in f:
            # v7 flat 형식
            self.X       = f["X"][:]
            self.y_raw   = f["y"][:].astype(str)
            self.subj_id = f["subj_id"][:].astype(int) if "subj_id" in f else np.zeros(len(self.X), int)
            self.channels: List[str] = list(f["channels"][:].astype(str)) if "channels" in f else []
        else:
            # v9 subjects 그룹 형식
            Xs, ys, ss = [], [], []
            subj_grp = f["subjects"] if "subjects" in f else f
            for sid_key in subj_grp.keys():
                grp = subj_grp[sid_key]
                if "X" not in grp or "y" not in grp:
                    continue
                sid_int = int(sid_key.lstrip("S")) if sid_key.startswith("S") else int(sid_key)
                Xs.append(grp["X"][:])
                ys.append(grp["y"][:])
                ss.extend([sid_int] * len(grp["X"]))
            self.X       = np.concatenate(Xs, axis=0)
            self.y_raw   = np.concatenate(ys, axis=0)
            self.subj_id = np.array(ss, dtype=int)
            self.channels = list(f["channels"][:].astype(str)) if "channels" in f else []

        # (N, T, C) → (N, C, T) 자동 변환
        if self.X.ndim == 3 and self.X.shape[1] > self.X.shape[2]:
            self.X = self.X.transpose(0, 2, 1)

    def close(self) -> None:
        self._f.close()


# ─────────────────────────────────────────────
# Sampler & DataLoader 빌더
# ─────────────────────────────────────────────

def make_balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    _, counts = np.unique(y, return_counts=True)
    w = (1.0 / counts)[y]
    return WeightedRandomSampler(torch.from_numpy(w).float(), len(y), replacement=True)


def make_branch_loaders(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    branch_idx: Dict[str, List[int]],
    batch: int = 256,
    balanced: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    tr_ds = BranchDataset(X_tr, y_tr, branch_idx)
    te_ds = BranchDataset(X_te, y_te, branch_idx)
    sampler   = make_balanced_sampler(y_tr) if balanced else None
    tr_loader = DataLoader(tr_ds, batch_size=batch, sampler=sampler,
                           shuffle=(sampler is None), num_workers=num_workers, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=batch * 2, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    return tr_loader, te_loader


def make_hierarchical_loaders(
    X_tr: np.ndarray, feat_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, feat_te: np.ndarray, y_te: np.ndarray,
    branch_idx: Dict[str, List[int]],
    batch: int = 256,
    balanced: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    tr_ds = HierarchicalDataset(X_tr, feat_tr, y_tr, branch_idx)
    te_ds = HierarchicalDataset(X_te, feat_te, y_te, branch_idx)
    sampler   = make_balanced_sampler(y_tr) if balanced else None
    tr_loader = DataLoader(tr_ds, batch_size=batch, sampler=sampler,
                           shuffle=(sampler is None), num_workers=num_workers, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=batch * 2, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    return tr_loader, te_loader


# ─────────────────────────────────────────────
# 캐시 유틸
# ─────────────────────────────────────────────

def save_cache(path: str | Path, arr: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)


def load_cache(path: str | Path) -> np.ndarray:
    return np.load(str(path), mmap_mode="r")