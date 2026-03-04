"""
models.py — M1-M6 CNN 모델 (v8 Final)
═══════════════════════════════════════════════════════
v7->v8: 입력 shape 검증, 파라미터 수 출력, 안전한 forward

모델:
    M1 : PCA -> Flat CNN (baseline)
    M2 : 7-Branch CNN (plain)
    M3 : 7-Branch + SE Attention
    M4 : 7-Branch + CBAM Attention
    M5 : M4 + Cross-Group Attention
    M6 : M5 + Online Data Augmentation
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import torch
import torch.nn as nn

N_CLS: int = config.NUM_CLASSES
FEAT: int  = config.FEAT_DIM


def _init_weights(m: nn.Module) -> None:
    """Kaiming 초기화 (Conv1d, Linear) + BatchNorm 초기화."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 총 수를 반환한다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvBNReLU(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pad: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 채널 어텐션.

    Parameters
    ----------
    ch : int
        입력 채널 수.
    r : int
        축소 비율 (config.SE_REDUCTION).
    """
    def __init__(self, ch: int, r: int = config.SE_REDUCTION) -> None:
        super().__init__()
        mid = max(ch // r, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)


class ChannelAttn(nn.Module):
    """CBAM 채널 어텐션 (avg + max pooling)."""
    def __init__(self, ch: int, r: int = config.SE_REDUCTION) -> None:
        super().__init__()
        mid = max(ch // r, 4)
        self.mlp = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(), nn.Linear(mid, ch),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean(-1))
        mx  = self.mlp(x.max(-1).values)
        return x * torch.sigmoid(avg + mx).unsqueeze(-1)


class TemporalAttn(nn.Module):
    """CBAM 시간축 어텐션."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 1, 7, padding=3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(1, keepdim=True)
        mx  = x.max(1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAM(nn.Module):
    """CBAM (채널 + 시간) + 잔차 연결."""
    def __init__(self, ch: int, r: int = config.SE_REDUCTION) -> None:
        super().__init__()
        self.ch_attn = ChannelAttn(ch, r)
        self.t_attn  = TemporalAttn()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.t_attn(self.ch_attn(x)) + x


class CrossGroupAttn(nn.Module):
    """그룹 간 Self-Attention."""
    def __init__(
        self, dim: int,
        n_heads: int = config.CROSS_N_HEADS,
        dropout: float = config.CROSS_DROPOUT,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(out))


class Branch(nn.Module):
    """단일 신체 부위 CNN 브랜치.

    Parameters
    ----------
    in_ch : int
        입력 채널 (해당 부위 센서 수).
    out_dim : int
        출력 feature 차원.
    mode : str
        'plain' | 'se' | 'cbam'.
    """
    def __init__(self, in_ch: int, out_dim: int = FEAT, mode: str = "plain") -> None:
        super().__init__()
        if in_ch <= 0:
            raise ValueError(f"Branch in_ch={in_ch} 은 양수여야 합니다")
        self.c1 = ConvBNReLU(in_ch, 64, 7, 3)
        self.p1 = nn.MaxPool1d(2)
        self.d1 = nn.Dropout(config.DROPOUT_FEAT)
        self.c2 = ConvBNReLU(64, 128, 5, 2)
        self.p2 = nn.MaxPool1d(2)
        self.d2 = nn.Dropout(config.DROPOUT_FEAT)
        self.c3 = ConvBNReLU(128, out_dim, 3, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        if mode == "se":
            self.attn: nn.Module = SEBlock(out_dim)
        elif mode == "cbam":
            self.attn = CBAM(out_dim)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, out_dim)."""
        assert x.ndim == 3, f"Branch 입력은 3D(B,C,T)이어야 합니다. got {x.shape}"
        x = self.d1(self.p1(self.c1(x)))
        x = self.d2(self.p2(self.c2(x)))
        x = self.c3(x)
        x = self.attn(x)
        return self.pool(x).squeeze(-1)


class M1_FlatCNN(nn.Module):
    """M1: PCA 차원축소 후 단일 CNN.

    Parameters
    ----------
    in_ch : int
        PCA 출력 채널 (config.PCA_CH).
    num_classes : int
        분류 클래스 수 (config.NUM_CLASSES).
    """
    def __init__(self, in_ch: int = config.PCA_CH, num_classes: int = N_CLS) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 3), nn.MaxPool1d(2),
            nn.Dropout(config.DROPOUT_FEAT),
            ConvBNReLU(64, 128, 5, 2), nn.MaxPool1d(2),
            nn.Dropout(config.DROPOUT_FEAT),
            ConvBNReLU(128, 256, 3, 1),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, FEAT), nn.ReLU(),
            nn.Dropout(config.DROPOUT_CLF),
        )
        self.head = nn.Linear(FEAT, num_classes)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, PCA_CH, T) -> (B, num_classes)."""
        assert x.ndim == 3, f"M1 입력은 3D(B,C,T)이어야 합니다. got {x.shape}"
        assert x.shape[1] == self.in_ch, (
            f"M1 채널 불일치: 예상 {self.in_ch}, 실제 {x.shape[1]}"
        )
        return self.head(self.net(x))

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """분류 헤드 제외, feature 벡터 반환. (B, FEAT_DIM)."""
        return self.net(x)


class _BranchBase(nn.Module):
    """M2-M6 공통 기반 클래스.

    Parameters
    ----------
    branch_channels : dict[str, int]
        {그룹명: 채널수}.
    mode : str
        'plain' | 'se' | 'cbam'.
    cross : bool
        Cross-Group Attention 사용 여부.
    num_classes : int
        분류 클래스 수.
    """
    def __init__(
        self,
        branch_channels: dict[str, int],
        mode: str = "plain",
        cross: bool = False,
        num_classes: int = N_CLS,
    ) -> None:
        super().__init__()
        if not branch_channels:
            raise ValueError("branch_channels가 비어 있습니다")
        self.names = list(branch_channels.keys())
        self.n     = len(self.names)
        self.cross = cross
        self.branches = nn.ModuleDict(
            {nm: Branch(ch, FEAT, mode) for nm, ch in branch_channels.items()}
        )
        if cross:
            self.cross_attn = CrossGroupAttn(FEAT)
        total = FEAT * self.n
        self.clf = nn.Sequential(
            nn.Linear(total, 256), nn.ReLU(), nn.Dropout(config.DROPOUT_CLF),
            nn.Linear(256, 128),   nn.ReLU(), nn.Dropout(config.DROPOUT_CLF),
            nn.Linear(128, num_classes),
        )
        self.apply(_init_weights)

    def _encode(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        """각 브랜치 feature 추출 -> (선택적) cross attention -> concat."""
        missing = set(self.names) - set(bi.keys())
        if missing:
            raise KeyError(f"Branch 입력에 누락된 그룹: {missing}")
        feats = [self.branches[nm](bi[nm]) for nm in self.names]
        if self.cross:
            x = torch.stack(feats, dim=1)
            x = self.cross_attn(x)
            feats = [x[:, i, :] for i in range(self.n)]
        return torch.cat(feats, dim=1)

    def forward(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.clf(self._encode(bi))

    def extract(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode(bi)


def M2_BranchCNN(bc: dict[str, int]) -> _BranchBase:
    """M2: 7-Branch CNN (no attention)."""
    return _BranchBase(bc, "plain", False)

def M3_BranchSE(bc: dict[str, int]) -> _BranchBase:
    """M3: 7-Branch + SE Attention."""
    return _BranchBase(bc, "se", False)

def M4_BranchCBAM(bc: dict[str, int]) -> _BranchBase:
    """M4: 7-Branch + CBAM."""
    return _BranchBase(bc, "cbam", False)

def M5_BranchCBAMCross(bc: dict[str, int]) -> _BranchBase:
    """M5: 7-Branch + CBAM + Cross-Group Attention."""
    return _BranchBase(bc, "cbam", True)


def augment(x: torch.Tensor) -> torch.Tensor:
    """학습 시 온라인 증강 (eval 시 identity).

    Parameters
    ----------
    x : torch.Tensor
        (B, C, T).

    Returns
    -------
    torch.Tensor
        증강된 텐서.
    """
    if not torch.is_grad_enabled():
        return x
    B, C, T = x.shape
    x = x + torch.randn_like(x) * config.AUG_NOISE
    s = 1.0 + (torch.rand(B, 1, 1, device=x.device) - 0.5) * config.AUG_SCALE
    x = x * s
    if config.AUG_SHIFT > 0:
        sh = torch.randint(-config.AUG_SHIFT, config.AUG_SHIFT + 1, (1,)).item()
        x = torch.roll(x, sh, dims=-1)
    if config.AUG_MASK_RATIO > 0:
        n_mask = max(1, int(C * config.AUG_MASK_RATIO))
        for b in range(B):
            idx = torch.randperm(C, device=x.device)[:n_mask]
            x[b, idx, :] = 0
    return x


class M6_BranchCBAMCrossAug(_BranchBase):
    """M6: M5 + Online Augmentation."""
    def __init__(self, branch_channels: dict[str, int]) -> None:
        super().__init__(branch_channels, "cbam", True)
    def forward(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.clf(self._encode({k: augment(v) for k, v in bi.items()}))
    def extract(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode(bi)