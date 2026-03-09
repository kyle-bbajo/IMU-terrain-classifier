"""src/datasets.py — Dataset / DataLoader / 캐시 유틸."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from channel_groups import build_branch_idx

# H5Data는 train_common의 정교한 구현(v7/v8 호환, chunked I/O)을 사용한다.
# datasets.py 자체에 중복 구현하지 않는다.
from train_common import H5Data  # noqa: F401  (재export용)


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
        x: np.ndarray,                       # (N, C, T)
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
    """브랜치 입력 + N_FEATURES 핸드크래프트 피처 (M7, HierarchicalFusionNet).

    파라미터명 `feat` 사용. `feat44`는 하위 호환 alias로 유지.
    """

    def __init__(
        self,
        x: np.ndarray,                       # (N, C, T)
        feat: np.ndarray,                    # (N, N_FEATURES)  ← 통일된 이름
        y: np.ndarray,
        branch_idx: Dict[str, List[int]],
        feat44: Optional[np.ndarray] = None, # deprecated alias (무시됨)
    ) -> None:
        # feat44가 넘어온 경우 하위 호환 처리
        _feat = feat if feat is not None else feat44
        assert _feat is not None, "feat 또는 feat44 중 하나는 반드시 제공해야 합니다."
        assert len(x) == len(_feat) == len(y), (
            f"길이 불일치: x={len(x)}, feat={len(_feat)}, y={len(y)}"
        )
        self.x          = x.astype(np.float32)
        self.feat       = torch.from_numpy(_feat.astype(np.float32))
        self.feat44     = self.feat  # 하위 호환 alias
        self.y          = y.astype(np.int64)
        self.branch_idx = branch_idx

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        xi = self.x[idx]
        bi = {nm: torch.from_numpy(xi[idxs]) for nm, idxs in self.branch_idx.items()}
        return bi, self.feat[idx], torch.tensor(self.y[idx], dtype=torch.long)


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
    tr_ds   = BranchDataset(X_tr, y_tr, branch_idx)
    te_ds   = BranchDataset(X_te, y_te, branch_idx)
    sampler = make_balanced_sampler(y_tr) if balanced else None
    tr_loader = DataLoader(
        tr_ds, batch_size=batch, sampler=sampler,
        shuffle=(sampler is None), num_workers=num_workers, pin_memory=True,
    )
    te_loader = DataLoader(
        te_ds, batch_size=batch * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return tr_loader, te_loader


def make_hierarchical_loaders(
    X_tr: np.ndarray, feat_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, feat_te: np.ndarray, y_te: np.ndarray,
    branch_idx: Dict[str, List[int]],
    batch: int = 256,
    balanced: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    tr_ds   = HierarchicalDataset(X_tr, feat_tr, y_tr, branch_idx)
    te_ds   = HierarchicalDataset(X_te, feat_te, y_te, branch_idx)
    sampler = make_balanced_sampler(y_tr) if balanced else None
    tr_loader = DataLoader(
        tr_ds, batch_size=batch, sampler=sampler,
        shuffle=(sampler is None), num_workers=num_workers, pin_memory=True,
    )
    te_loader = DataLoader(
        te_ds, batch_size=batch * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
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