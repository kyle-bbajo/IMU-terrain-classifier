"""
models.py — M1–M6 + ResNet1D + CNNTCN CNN 모델 정의 (v9.0)
═══════════════════════════════════════════════════════════
변경 이력 (v8.1 → v9.0)
──────────────────────────────────────────────────────────
[ADD]  ResNet1D 브랜치: 잔차 연결 + CBAM + Cross-Group Attention
[ADD]  CNNTCN 브랜치: CNN stem + Dilated TCN + CBAM
[ADD]  MODEL_REGISTRY / get_model_factories(): 외부에서 모델 선택 가능
[FIX]  augment(): torch.roll(같은 shift) → 샘플별 독립 shift (vectorized)
[FIX]  Conv1d bias=False (BN 앞에서 bias는 무효)
[FIX]  config.CFG 직접 참조 (_cfg() 헬퍼 제거 — 불필요한 간접 레이어)
[KEEP] M1–M6 모든 기존 인터페이스 (하위 호환)
═══════════════════════════════════════════════════════════
모델 구조:
    M1       : PCA → Flat CNN (baseline)
    M2       : 5-Branch CNN (plain)
    M3       : 5-Branch CNN + SE Attention
    M4       : 5-Branch CNN + CBAM Attention
    M5       : M4 + Cross-Group Attention
    M6       : M5 + Online Data Augmentation  ← 권장
    ResNet1D : Branch ResNet (잔차 + CBAM + Cross)
    CNNTCN   : Branch CNN + Dilated TCN + CBAM + Cross
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import torch
import torch.nn as nn

# config.CFG 싱글톤에서 직접 읽기 (apply_overrides 이후 값도 반영)
_CFG = config.CFG


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수를 반환한다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _init_weights(m: nn.Module) -> None:
    """Kaiming He 초기화 (Conv1d, Linear) + BatchNorm 단위 초기화."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────
# 기본 블록
# ─────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv1d(bias=False) → BatchNorm1d → ReLU(inplace)."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, pad: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 채널 어텐션 (Hu et al., 2018)."""

    def __init__(self, ch: int, r: int | None = None) -> None:
        super().__init__()
        r = r or _CFG.se_reduction
        mid = max(ch // r, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)


class ChannelAttn(nn.Module):
    """CBAM 채널 어텐션: avg + max pooling → 공유 MLP."""

    def __init__(self, ch: int, r: int | None = None) -> None:
        super().__init__()
        r = r or _CFG.se_reduction
        mid = max(ch // r, 4)
        self.mlp = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(inplace=True), nn.Linear(mid, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean(-1))
        mx  = self.mlp(x.max(-1).values)
        return x * torch.sigmoid(avg + mx).unsqueeze(-1)


class TemporalAttn(nn.Module):
    """CBAM 시간축 어텐션: avg + max concat → Conv1d(2→1)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(1, keepdim=True)
        mx  = x.max(1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (채널 → 시간) + 잔차 연결."""

    def __init__(self, ch: int, r: int | None = None) -> None:
        super().__init__()
        self.ch = ChannelAttn(ch, r)
        self.t  = TemporalAttn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.t(self.ch(x)) + x


class CrossGroupAttn(nn.Module):
    """그룹 간 Self-Attention (Pre-Norm + Residual).

    브랜치 피처 시퀀스 (B, n_branches, FEAT) 에 적용.
    """

    def __init__(
        self, dim: int, n_heads: int | None = None, dropout: float | None = None
    ) -> None:
        super().__init__()
        n_heads = n_heads or _CFG.cross_n_heads
        dropout = float(dropout if dropout is not None else _CFG.cross_dropout)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + self.drop(out))


# ─────────────────────────────────────────────
# 브랜치 모듈 3종
# ─────────────────────────────────────────────

class Branch(nn.Module):
    """단일 신체 부위 CNN 브랜치 (plain / SE / CBAM)."""

    def __init__(self, in_ch: int, out_dim: int, mode: str = "plain") -> None:
        super().__init__()
        drop = _CFG.dropout_feat
        self.c1   = ConvBNReLU(in_ch, 64, 7, 3)
        self.p1   = nn.MaxPool1d(2)
        self.d1   = nn.Dropout(drop)
        self.c2   = ConvBNReLU(64, 128, 5, 2)
        self.p2   = nn.MaxPool1d(2)
        self.d2   = nn.Dropout(drop)
        self.c3   = ConvBNReLU(128, out_dim, 3, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.attn: nn.Module = (
            SEBlock(out_dim) if mode == "se" else
            CBAM(out_dim)    if mode == "cbam" else
            nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.d1(self.p1(self.c1(x)))
        x = self.d2(self.p2(self.c2(x)))
        x = self.attn(self.c3(x))
        return self.pool(x).squeeze(-1)


class ResBlock1D(nn.Module):
    """1-D 잔차 블록 (optional CBAM/SE 어텐션)."""

    def __init__(
        self, in_ch: int, out_ch: int,
        stride: int = 1, kernel: int = 7, attn: str = "none",
    ) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.down  = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        self.attn: nn.Module = (
            SEBlock(out_ch) if attn == "se" else
            CBAM(out_ch)    if attn == "cbam" else
            nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ident = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(self.attn(out) + ident)


class DilatedTCNBlock(nn.Module):
    """팽창 인과 TCN 블록 (dilation=1/2/4 로 쌓아 수용장 확장)."""

    def __init__(self, ch: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class ResNetBranch(nn.Module):
    """잔차 블록 3스테이지 브랜치 (CBAM + Cross 적합)."""

    def __init__(self, in_ch: int, out_dim: int, attn: str = "cbam") -> None:
        super().__init__()
        drop = _CFG.dropout_feat
        self.stem   = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock1D(64,  64,  1, 7, attn="none"),
            ResBlock1D(64,  64,  1, 7, attn=attn),
        )
        self.layer2 = nn.Sequential(
            ResBlock1D(64,  128, 2, 5, attn="none"),
            ResBlock1D(128, 128, 1, 5, attn=attn),
        )
        self.layer3 = nn.Sequential(
            ResBlock1D(128, out_dim, 2, 3, attn="none"),
            ResBlock1D(out_dim, out_dim, 1, 3, attn=attn),
        )
        self.drop = nn.Dropout(drop)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.pool(self.drop(self.layer3(x))).squeeze(-1)


class CNNTCNBranch(nn.Module):
    """CNN 스템 + Dilated TCN 브랜치 (장거리 시계열 의존성 포착)."""

    def __init__(self, in_ch: int, out_dim: int, mode: str = "cbam") -> None:
        super().__init__()
        drop = _CFG.dropout_feat
        self.c1   = ConvBNReLU(in_ch, 64,  7, 3)
        self.p1   = nn.MaxPool1d(2)
        self.c2   = ConvBNReLU(64,  128, 5, 2)
        self.p2   = nn.MaxPool1d(2)
        self.proj = ConvBNReLU(128, out_dim, 3, 1)
        self.tcn  = nn.Sequential(
            DilatedTCNBlock(out_dim, 1, drop),
            DilatedTCNBlock(out_dim, 2, drop),
            DilatedTCNBlock(out_dim, 4, drop),
        )
        self.attn: nn.Module = (
            SEBlock(out_dim) if mode == "se" else
            CBAM(out_dim)    if mode == "cbam" else
            nn.Identity()
        )
        self.drop = nn.Dropout(drop)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.p1(self.c1(x))
        x = self.p2(self.c2(x))
        x = self.proj(x)
        return self.pool(self.drop(self.attn(self.tcn(x)))).squeeze(-1)


class FreqBranch(nn.Module):
    """주파수 도메인 브랜치: FFT magnitude → CNN.

    발 가속도의 주파수 성분으로 표면 질감(잔디/흙/아스팔트) 구분.
    """

    def __init__(self, in_ch: int, out_dim: int) -> None:
        super().__init__()
        self.c1   = ConvBNReLU(in_ch, 64,  7, 3)
        self.p1   = nn.MaxPool1d(2)
        self.c2   = ConvBNReLU(64,  out_dim, 5, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.rfft(x, dim=-1).abs()        # (B, C, T//2+1)
        return self.pool(self.c2(self.p1(self.c1(freq)))).squeeze(-1)


# ─────────────────────────────────────────────
# 공통 기반 클래스
# ─────────────────────────────────────────────

class _BranchBase(nn.Module):
    """M2–M6 / ResNet1D / CNNTCN 공통 기반.

    Parameters
    ----------
    branch_channels : dict[str, int]
        그룹명 → 채널 수.
    branch_ctor : callable
        브랜치 생성자 (Branch / ResNetBranch / CNNTCNBranch).
    mode : str
        어텐션 모드 ("plain" / "se" / "cbam").
    cross : bool
        Cross-Group Attention 사용 여부.
    num_classes : int | None
        None이면 config.CFG.num_classes 사용.
    """

    def __init__(
        self,
        branch_channels: dict[str, int],
        branch_ctor,
        mode: str = "plain",
        cross: bool = False,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        feat  = _CFG.feat_dim
        n_cls = int(num_classes if num_classes is not None else _CFG.num_classes)
        drop  = _CFG.dropout_clf

        self.names = list(branch_channels.keys())
        self.n     = len(self.names)
        self.cross = cross
        self.branches = nn.ModuleDict(
            {nm: branch_ctor(ch, feat, mode) for nm, ch in branch_channels.items()}
        )

        # 주파수 브랜치 (config 옵션)
        self.use_fft    = bool(_CFG.use_fft_branch) and _CFG.fft_source_group in branch_channels
        self.fft_source = _CFG.fft_source_group
        if self.use_fft:
            self.freq_branch = FreqBranch(branch_channels[self.fft_source], feat)

        n_feats = self.n + (1 if self.use_fft else 0)
        if cross:
            self.cross_attn = CrossGroupAttn(feat)

        total = feat * n_feats
        self.clf = nn.Sequential(
            nn.Linear(total, 256), nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(256, 128),   nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(128, n_cls),
        )
        self.apply(_init_weights)

    def _encode(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        feats = [self.branches[nm](bi[nm]) for nm in self.names]
        if self.cross:
            x     = self.cross_attn(torch.stack(feats, dim=1))
            feats = [x[:, i, :] for i in range(self.n)]
        if self.use_fft:
            feats.append(self.freq_branch(bi[self.fft_source]))
        return torch.cat(feats, dim=1)

    def forward(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.clf(self._encode(bi))

    def extract(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        """분류 헤드 이전 피처 벡터를 반환한다 (임베딩 추출용)."""
        return self._encode(bi)


# ─────────────────────────────────────────────
# 온라인 데이터 증강
# ─────────────────────────────────────────────

def augment(x: torch.Tensor, training: bool) -> torch.Tensor:
    """학습 전용 온라인 증강 (noise / scale / per-sample shift / channel mask).

    Parameters
    ----------
    x : torch.Tensor
        (B, C, T) 입력.
    training : bool
        model.training 상태. False면 원본 그대로 반환.

    주요 변경 (v9.0):
        - shift: torch.roll(동일 값) → 샘플별 독립 시프트 (gather 방식)
          → 배치 내 데이터 다양성 향상, GPU-friendly
    """
    if not training:
        return x
    b, c, t = x.shape

    # Gaussian noise
    if _CFG.aug_noise > 0:
        x = x + torch.randn_like(x) * _CFG.aug_noise

    # Random scale (배치 단위 독립)
    if _CFG.aug_scale > 0:
        x = x * (1.0 + (torch.rand(b, 1, 1, device=x.device) - 0.5) * _CFG.aug_scale)

    # Per-sample circular shift (v9.0: 샘플마다 다른 shift 적용)
    if _CFG.aug_shift > 0:
        shifts     = torch.randint(-_CFG.aug_shift, _CFG.aug_shift + 1, (b,), device=x.device)
        base       = torch.arange(t, device=x.device).view(1, 1, t).expand(b, c, t)
        gather_idx = (base - shifts.view(b, 1, 1)) % t
        x          = torch.gather(x, 2, gather_idx)

    # Random channel mask (표면 노이즈 강건성)
    if _CFG.aug_mask_ratio > 0:
        n_mask   = max(1, int(c * _CFG.aug_mask_ratio))
        mask_idx = torch.stack(
            [torch.randperm(c, device=x.device)[:n_mask] for _ in range(b)]
        )                                                          # (B, n_mask)
        mask = torch.ones(b, c, 1, device=x.device, dtype=x.dtype)
        mask.scatter_(1, mask_idx.unsqueeze(-1), 0.0)
        x = x * mask

    return x


# ─────────────────────────────────────────────
# M1: PCA + Flat CNN
# ─────────────────────────────────────────────

class M1_FlatCNN(nn.Module):
    """M1: PCA 차원 축소 후 단일 Flat CNN (baseline).

    train_common.run_M1() 에서 사용.
    """

    def __init__(self, in_ch: int | None = None, num_classes: int | None = None) -> None:
        super().__init__()
        in_ch      = int(in_ch if in_ch is not None else _CFG.pca_ch)
        feat       = _CFG.feat_dim
        drop_f     = _CFG.dropout_feat
        drop_c     = _CFG.dropout_clf
        num_classes = int(num_classes if num_classes is not None else _CFG.num_classes)

        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64,  7, 3), nn.MaxPool1d(2), nn.Dropout(drop_f),
            ConvBNReLU(64,  128,  5, 2), nn.MaxPool1d(2), nn.Dropout(drop_f),
            ConvBNReLU(128, 256,  3, 1), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, feat), nn.ReLU(inplace=True), nn.Dropout(drop_c),
        )
        self.head = nn.Linear(feat, num_classes)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# M2–M6 공개 팩토리
# ─────────────────────────────────────────────

def M2_BranchCNN(bc: dict[str, int]) -> _BranchBase:
    """M2: 5-Branch CNN (어텐션 없음)."""
    return _BranchBase(bc, Branch, "plain", False)

def M3_BranchSE(bc: dict[str, int]) -> _BranchBase:
    """M3: 5-Branch CNN + SE Attention."""
    return _BranchBase(bc, Branch, "se", False)

def M4_BranchCBAM(bc: dict[str, int]) -> _BranchBase:
    """M4: 5-Branch CNN + CBAM Attention."""
    return _BranchBase(bc, Branch, "cbam", False)

def M5_BranchCBAMCross(bc: dict[str, int]) -> _BranchBase:
    """M5: 5-Branch CNN + CBAM + Cross-Group Attention."""
    return _BranchBase(bc, Branch, "cbam", True)


class M6_BranchCBAMCrossAug(_BranchBase):
    """M6: M5 + 온라인 Data Augmentation (권장 모델).

    학습 시 배치 단위로 augment()를 적용하고
    평가 시(model.eval())에는 원본 데이터를 사용한다.
    """

    def __init__(self, branch_channels: dict[str, int]) -> None:
        super().__init__(branch_channels, Branch, "cbam", True)

    def forward(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.clf(self._encode(
            {k: augment(v, self.training) for k, v in bi.items()}
        ))

    def extract(self, bi: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode(bi)   # extract는 항상 증강 없이


# ─────────────────────────────────────────────
# ResNet1D / CNNTCN 팩토리
# ─────────────────────────────────────────────

def BranchResNet1D(bc: dict[str, int]) -> _BranchBase:
    """ResNet1D: Branch ResNet + CBAM + Cross-Group Attention."""
    return _BranchBase(bc, ResNetBranch, "cbam", True)

def BranchCNNTCN(bc: dict[str, int]) -> _BranchBase:
    """CNNTCN: Branch CNN + Dilated TCN + CBAM + Cross-Group Attention."""
    return _BranchBase(bc, CNNTCNBranch, "cbam", True)


# ─────────────────────────────────────────────
# MODEL_REGISTRY — 외부에서 이름으로 모델 선택
# ─────────────────────────────────────────────

MODEL_REGISTRY: dict[str, object] = {
    "M2":       M2_BranchCNN,
    "M3":       M3_BranchSE,
    "M4":       M4_BranchCBAM,
    "M5":       M5_BranchCBAMCross,
    "M6":       M6_BranchCBAMCrossAug,
    "ResNet1D": BranchResNet1D,
    "CNNTCN":   BranchCNNTCN,
}

# K-Fold 기본 비교 순서 (M1은 run_M1()이 별도 처리)
DEFAULT_COMPARE_ORDER: list[str] = ["M2", "M4", "M6", "ResNet1D", "CNNTCN"]

# LOSO 기본: 상위 2개만 (시간 절약)
DEFAULT_LOSO_ORDER: list[str] = ["M6", "ResNet1D"]


def get_model_factories(
    selected: list[str] | None = None,
) -> list[tuple[str, object]]:
    """선택된 모델 이름에 대한 (이름, 팩토리) 리스트를 반환한다.

    Parameters
    ----------
    selected : list[str] | None
        None이면 DEFAULT_COMPARE_ORDER 전체 반환.

    Raises
    ------
    KeyError
        알 수 없는 모델 이름 포함 시.
    """
    names = DEFAULT_COMPARE_ORDER if not selected else selected
    bad   = [n for n in names if n not in MODEL_REGISTRY]
    if bad:
        raise KeyError(
            f"알 수 없는 모델: {bad}  사용 가능: {list(MODEL_REGISTRY)}"
        )
    return [(n, MODEL_REGISTRY[n]) for n in names]