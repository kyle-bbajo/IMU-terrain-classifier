"""src/models.py — M1–M7 + ResNet1D + CNNTCN + ResNetTCN + HierarchicalFusionNet.

MODEL_REGISTRY 키:
    baseline    : "M2", "M4", "M6"
    comparison  : "ResNet1D", "CNNTCN", "ResNetTCN"
    hybrid      : "M7"             (CNN + 44-feat, IS_HYBRID=True)
    advanced    : "Hierarchical"   (ResNetTCN + 44-feat, IS_HYBRID=True)
    flat        : "M1"             (PCA + flat CNN, 별도 run_M1)
"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


# ─────────────────────────────────────────────
# 초기화 / 증강
# ─────────────────────────────────────────────

def _init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)


def augment(x: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return x
    b, c, t = x.shape
    if CFG.aug_noise > 0:
        x = x + torch.randn_like(x) * CFG.aug_noise
    if CFG.aug_scale > 0:
        x = x * (1.0 + (torch.rand(b, 1, 1, device=x.device) - 0.5) * CFG.aug_scale)
    if CFG.aug_shift > 0:
        shifts = torch.randint(-CFG.aug_shift, CFG.aug_shift + 1, (b,), device=x.device)
        base   = torch.arange(t, device=x.device).view(1, 1, t).expand(b, c, t)
        x      = torch.gather(x, 2, (base - shifts.view(b, 1, 1)) % t)
    if CFG.aug_mask_ratio > 0:
        n_m  = max(1, int(c * CFG.aug_mask_ratio))
        idx  = torch.rand(b, c, device=x.device).argsort(1)[:, :n_m]
        mask = torch.ones(b, c, 1, device=x.device)
        mask.scatter_(1, idx.unsqueeze(-1), 0.0)
        x = x * mask
    return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 공통 블록
# ─────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, pad, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class ChannelAttn(nn.Module):
    def __init__(self, ch, r=None):
        super().__init__()
        r   = r or CFG.se_reduction
        mid = max(ch // r, 4)
        self.mlp = nn.Sequential(nn.Linear(ch, mid), nn.ReLU(inplace=True), nn.Linear(mid, ch))
    def forward(self, x):
        w = torch.sigmoid(self.mlp(x.mean(-1)) + self.mlp(x.max(-1).values))
        return x * w.unsqueeze(-1)


class TemporalAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        pool = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], 1)
        return x * torch.sigmoid(self.conv(pool))


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ChannelAttn(ch); self.t = TemporalAttn()
    def forward(self, x): return self.t(self.ch(x)) + x


class SEBlock(nn.Module):
    def __init__(self, ch, r=None):
        super().__init__()
        r   = r or CFG.se_reduction
        mid = max(ch // r, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )
    def forward(self, x): return x * self.se(x).unsqueeze(-1)


class CrossGroupAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, CFG.cross_n_heads, batch_first=True,
                                          dropout=CFG.cross_dropout)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(CFG.cross_dropout)
    def forward(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + self.drop(out))


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, k=7, attn="cbam"):
        super().__init__()
        pad  = k // 2
        self.c1   = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False)
        self.b1   = nn.BatchNorm1d(out_ch)
        self.c2   = nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False)
        self.b2   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        self.attn_m: nn.Module = (
            CBAM(out_ch) if attn == "cbam" else
            SEBlock(out_ch) if attn == "se" else nn.Identity()
        )
    def forward(self, x):
        out = self.act(self.b1(self.c1(x)))
        return self.act(self.attn_m(self.b2(self.c2(out))) + self.skip(x))


class DilatedTCN(nn.Module):
    def __init__(self, ch, dilation, drop=None):
        super().__init__()
        drop = drop or CFG.dropout_feat
        pad  = dilation
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.net(x) + x)


class AttnPool1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        return (x * torch.softmax(self.score(x), dim=-1)).sum(-1)


class FreqBranch(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 3), nn.MaxPool1d(2),
            ConvBNReLU(64, out_dim, 5, 2), nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x):
        return self.net(torch.fft.rfft(x, dim=-1).abs()).squeeze(-1)


# ─────────────────────────────────────────────
# 브랜치 인코더
# ─────────────────────────────────────────────

class Branch(nn.Module):
    """CNN 브랜치 (plain / se / cbam 모드)."""
    def __init__(self, in_ch, out_dim, mode="plain"):
        super().__init__()
        d = CFG.dropout_feat
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 3), nn.MaxPool1d(2), nn.Dropout(d),
            ConvBNReLU(64, 128, 5, 2),   nn.MaxPool1d(2), nn.Dropout(d),
            ConvBNReLU(128, out_dim, 3, 1),
        )
        self.attn: nn.Module = (
            SEBlock(out_dim) if mode == "se" else
            CBAM(out_dim)    if mode == "cbam" else nn.Identity()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        return self.pool(self.attn(self.net(x))).squeeze(-1)


class ResNetBranch(nn.Module):
    """ResNet 브랜치 (time-axis preserving)."""
    def __init__(self, in_ch, out_dim):
        super().__init__()
        d = CFG.dropout_feat
        self.stem   = ConvBNReLU(in_ch, 64, 7, 3)
        self.layer1 = nn.Sequential(ResBlock1D(64, 64, 1, 7, "none"), ResBlock1D(64, 64, 1, 7, "cbam"))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, 2, 5, "none"), ResBlock1D(128, 128, 1, 5, "cbam"))
        self.layer3 = nn.Sequential(ResBlock1D(128, out_dim, 2, 3, "none"), ResBlock1D(out_dim, out_dim, 1, 3, "cbam"))
        self.drop   = nn.Dropout(d)
    def forward(self, x):
        return self.drop(self.layer3(self.layer2(self.layer1(self.stem(x)))))  # (B, out_dim, T')


class CNNTCNBranch(nn.Module):
    def __init__(self, in_ch, out_dim, mode="cbam"):
        super().__init__()
        d = CFG.dropout_feat
        self.cnn = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 3), nn.MaxPool1d(2),
            ConvBNReLU(64, 128, 5, 2),   nn.MaxPool1d(2),
            ConvBNReLU(128, out_dim, 3, 1),
        )
        self.tcn  = nn.Sequential(DilatedTCN(out_dim,1,d), DilatedTCN(out_dim,2,d), DilatedTCN(out_dim,4,d))
        self.attn: nn.Module = CBAM(out_dim) if mode == "cbam" else SEBlock(out_dim) if mode == "se" else nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(d)
    def forward(self, x):
        return self.pool(self.drop(self.attn(self.tcn(self.cnn(x))))).squeeze(-1)


# ─────────────────────────────────────────────
# M1 (Flat CNN, PCA input)
# ─────────────────────────────────────────────

class M1_FlatCNN(nn.Module):
    def __init__(self, in_ch=None, num_classes=None):
        super().__init__()
        in_ch = in_ch or CFG.pca_ch
        n_cls = num_classes or CFG.num_classes
        fd, dc = CFG.feat_dim, CFG.dropout_clf
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, 64, 7, 3), nn.MaxPool1d(2), nn.Dropout(CFG.dropout_feat),
            ConvBNReLU(64, 128, 5, 2),   nn.MaxPool1d(2), nn.Dropout(CFG.dropout_feat),
            ConvBNReLU(128, 256, 3, 1),  nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, fd), nn.ReLU(inplace=True), nn.Dropout(dc),
        )
        self.head = nn.Linear(fd, n_cls)
        self.apply(_init)
    def forward(self, x): return self.head(self.net(x))
    def extract(self, x): return self.net(x)


# ─────────────────────────────────────────────
# _BranchBase (M2–M6 공통)
# ─────────────────────────────────────────────

class _BranchBase(nn.Module):
    def __init__(self, branch_channels, branch_ctor, mode="plain", cross=False, num_classes=None):
        super().__init__()
        feat  = CFG.feat_dim
        n_cls = num_classes or CFG.num_classes
        drop  = CFG.dropout_clf
        self.names    = list(branch_channels.keys())
        self.n        = len(self.names)
        self.cross    = cross
        self.branches = nn.ModuleDict({nm: branch_ctor(ch, feat, mode) for nm, ch in branch_channels.items()})
        self.use_fft  = CFG.use_fft_branch and CFG.fft_source_group in branch_channels
        if self.use_fft:
            self.fft_src     = CFG.fft_source_group
            self.freq_branch = FreqBranch(branch_channels[self.fft_src], feat)
        if cross:
            self.cross_attn = CrossGroupAttn(feat)
        n_f   = self.n + (1 if self.use_fft else 0)
        self.clf = nn.Sequential(
            nn.Linear(feat * n_f, 256), nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(256, 128),        nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(128, n_cls),
        )
        self.apply(_init)

    def _encode(self, bi):
        feats = [self.branches[nm](bi[nm]) for nm in self.names]
        if self.cross:
            x     = self.cross_attn(torch.stack(feats, dim=1))
            feats = [x[:, i, :] for i in range(self.n)]
        if self.use_fft:
            feats.append(self.freq_branch(bi[self.fft_src]))
        return torch.cat(feats, dim=1)

    def forward(self, bi): return self.clf(self._encode(bi))
    def extract(self, bi): return self._encode(bi)


# ─────────────────────────────────────────────
# M2–M6 팩토리
# ─────────────────────────────────────────────

def M2_BranchCNN(bc):    return _BranchBase(bc, Branch, "plain", False)
def M3_BranchSE(bc):     return _BranchBase(bc, Branch, "se",    False)
def M4_BranchCBAM(bc):   return _BranchBase(bc, Branch, "cbam",  False)
def M5_BranchCBAMCross(bc): return _BranchBase(bc, Branch, "cbam", True)

class M6_BranchCBAMCrossAug(_BranchBase):
    def __init__(self, bc):
        super().__init__(bc, Branch, "cbam", True)
    def forward(self, bi):
        return self.clf(self._encode({k: augment(v, self.training) for k, v in bi.items()}))


# ─────────────────────────────────────────────
# ResNet1D, CNNTCN (브랜치 래퍼)
# ─────────────────────────────────────────────

def BranchResNet1D(bc):
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            feat = CFG.feat_dim
            self.names    = list(bc.keys())
            self.branches = nn.ModuleDict({nm: ResNetBranch(ch, feat) for nm, ch in bc.items()})
            self.cross    = CrossGroupAttn(feat)
            n_f   = len(self.names)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.clf  = nn.Sequential(
                nn.Linear(feat * n_f, 256), nn.ReLU(inplace=True), nn.Dropout(CFG.dropout_clf),
                nn.Linear(256, 128),        nn.ReLU(inplace=True), nn.Dropout(CFG.dropout_clf),
                nn.Linear(128, CFG.num_classes),
            )
            self.apply(_init)
        def _encode(self, bi):
            feats = [self.pool(self.branches[nm](bi[nm])).squeeze(-1) for nm in self.names]
            x     = self.cross(torch.stack(feats, dim=1))
            return torch.cat([x[:, i, :] for i in range(len(self.names))], dim=1)
        def forward(self, bi): return self.clf(self._encode(bi))
        def extract(self, bi): return self._encode(bi)
    return _M()


def BranchCNNTCN(bc):
    return _BranchBase(bc, CNNTCNBranch, "cbam", True)


# ─────────────────────────────────────────────
# ResNetTCN (시간축 보존 + AttnPool)
# ─────────────────────────────────────────────

def ResNetTCN(bc: dict) -> nn.Module:
    """ResNet 브랜치 → concat → 4단 Dilated TCN → AttnPool."""
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            feat = CFG.feat_dim; d = CFG.dropout_feat; dc = CFG.dropout_clf
            self.names    = list(bc.keys())
            self.branches = nn.ModuleDict({nm: ResNetBranch(ch, feat) for nm, ch in bc.items()})
            total         = feat * len(self.names)
            self.tcn      = nn.Sequential(
                DilatedTCN(total,1,d), DilatedTCN(total,2,d),
                DilatedTCN(total,4,d), DilatedTCN(total,8,d),
            )
            self.pool = AttnPool1D(total)
            self.clf  = nn.Sequential(
                nn.Linear(total, 512), nn.ReLU(inplace=True), nn.Dropout(dc),
                nn.Linear(512, 128),   nn.ReLU(inplace=True), nn.Dropout(dc),
                nn.Linear(128, CFG.num_classes),
            )
            self.apply(_init)
        def _encode(self, bi):
            feats = [self.branches[nm](bi[nm]) for nm in self.names]  # list (B,feat,T')
            return self.pool(self.tcn(torch.cat(feats, dim=1)))         # (B, total)
        def forward(self, bi): return self.clf(self._encode(bi))
        def extract(self, bi): return self._encode(bi)
    return _M()


# ─────────────────────────────────────────────
# FeatureMLP (44-feat 스트림)
# ─────────────────────────────────────────────

class FeatureMLP(nn.Module):
    def __init__(self, n_feat=44, out_dim=None):
        super().__init__()
        out_dim = out_dim or CFG.feat_dim
        d = CFG.dropout_feat
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_feat),
            nn.Linear(n_feat, 128), nn.ReLU(inplace=True), nn.Dropout(d),
            nn.Linear(128, out_dim), nn.ReLU(inplace=True), nn.Dropout(d),
        )
    def forward(self, f): return self.net(f.float())


# ─────────────────────────────────────────────
# M7: CNN(M6) + 44-feat Hybrid
# ─────────────────────────────────────────────

class M7_HybridModel(nn.Module):
    IS_HYBRID = True

    def __init__(self, bc):
        super().__init__()
        feat = CFG.feat_dim; dc = CFG.dropout_clf
        self.names    = list(bc.keys())
        self.n        = len(self.names)
        self.branches = nn.ModuleDict({nm: Branch(ch, feat, "cbam") for nm, ch in bc.items()})
        self.cross    = CrossGroupAttn(feat)
        self.use_fft  = CFG.use_fft_branch and CFG.fft_source_group in bc
        if self.use_fft:
            self.fft_src     = CFG.fft_source_group
            self.freq_branch = FreqBranch(bc[self.fft_src], feat)
        n_f       = self.n + (1 if self.use_fft else 0)
        cnn_dim   = feat * n_f
        self.feat_mlp  = FeatureMLP(44)
        fusion_dim     = cnn_dim + CFG.feat_dim
        self.bn_fuse   = nn.BatchNorm1d(fusion_dim)
        self.clf       = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dc),
            nn.Linear(256, 128),        nn.ReLU(inplace=True), nn.Dropout(dc),
            nn.Linear(128, CFG.num_classes),
        )
        self.apply(_init)

    def _cnn_encode(self, bi):
        feats = [self.branches[nm](bi[nm]) for nm in self.names]
        x     = self.cross(torch.stack(feats, dim=1))
        feats = [x[:, i, :] for i in range(self.n)]
        if self.use_fft:
            feats.append(self.freq_branch(bi[self.fft_src]))
        return torch.cat(feats, dim=1)

    def forward(self, bi, feat44):
        if self.training:
            bi = {k: augment(v, True) for k, v in bi.items()}
        fused = self.bn_fuse(torch.cat([self._cnn_encode(bi), self.feat_mlp(feat44)], dim=1))
        return self.clf(fused)

    def extract(self, bi, feat44):
        return self.bn_fuse(torch.cat([self._cnn_encode(bi), self.feat_mlp(feat44)], dim=1))


def M7_Hybrid(bc): return M7_HybridModel(bc)


# ─────────────────────────────────────────────
# HierarchicalFusionNet: ResNetTCN + 44-feat
# ─────────────────────────────────────────────

class HierarchicalFusionNet(nn.Module):
    IS_HYBRID = True

    def __init__(self, bc):
        super().__init__()
        feat = CFG.feat_dim; dc = CFG.dropout_clf; d = CFG.dropout_feat
        self.names    = list(bc.keys())
        self.branches = nn.ModuleDict({nm: ResNetBranch(ch, feat) for nm, ch in bc.items()})
        total         = feat * len(self.names)
        self.tcn      = nn.Sequential(
            DilatedTCN(total,1,d), DilatedTCN(total,2,d),
            DilatedTCN(total,4,d), DilatedTCN(total,8,d),
        )
        self.pool      = AttnPool1D(total)
        self.feat_mlp  = FeatureMLP(44)
        fusion_dim     = total + CFG.feat_dim
        self.bn_fuse   = nn.BatchNorm1d(fusion_dim)
        self.clf       = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dc),
            nn.Linear(256, 128),        nn.ReLU(inplace=True), nn.Dropout(dc),
            nn.Linear(128, CFG.num_classes),
        )
        self.apply(_init)

    def _deep(self, bi):
        feats = [self.branches[nm](bi[nm]) for nm in self.names]
        return self.pool(self.tcn(torch.cat(feats, dim=1)))

    def forward(self, bi, feat44):
        if self.training:
            bi = {k: augment(v, True) for k, v in bi.items()}
        fused = self.bn_fuse(torch.cat([self._deep(bi), self.feat_mlp(feat44)], dim=1))
        return self.clf(fused)

    def extract(self, bi, feat44):
        return self.bn_fuse(torch.cat([self._deep(bi), self.feat_mlp(feat44)], dim=1))


def Hierarchical(bc): return HierarchicalFusionNet(bc)


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, object] = {
    "M2":          M2_BranchCNN,
    "M3":          M3_BranchSE,
    "M4":          M4_BranchCBAM,
    "M5":          M5_BranchCBAMCross,
    "M6":          M6_BranchCBAMCrossAug,
    "M7":          M7_Hybrid,
    "ResNet1D":    BranchResNet1D,
    "CNNTCN":      BranchCNNTCN,
    "ResNetTCN":   ResNetTCN,
    "Hierarchical": Hierarchical,
}

DEFAULT_COMPARE_ORDER = ["M2", "M4", "M6", "ResNet1D", "CNNTCN", "ResNetTCN", "M7"]
DEFAULT_LOSO_ORDER    = ["M6", "ResNet1D", "ResNetTCN", "M7"]
ENSEMBLE_MODELS       = ["ResNet1D", "CNNTCN", "ResNetTCN", "M6", "M7"]


def get_model_factories(names=None):
    names = names or DEFAULT_COMPARE_ORDER
    bad   = [n for n in names if n not in MODEL_REGISTRY]
    if bad:
        raise KeyError(f"Unknown models: {bad}. Available: {list(MODEL_REGISTRY)}")
    return [(n, MODEL_REGISTRY[n]) for n in names]