"""src/utils.py — 공통 유틸리티."""
from __future__ import annotations

import random
import time
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ─────────────────────────────────────────────
# 재현성
# ─────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# 로거
# ─────────────────────────────────────────────

def get_logger(name: str = "imu", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    sh  = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


_logger = get_logger()

def log(msg: str) -> None:
    _logger.info(msg)


# ─────────────────────────────────────────────
# 타이머
# ─────────────────────────────────────────────

class Timer:
    def __init__(self) -> None:
        self._s = time.time()

    def elapsed(self) -> float:
        return time.time() - self._s

    def elapsed_str(self) -> str:
        e = self.elapsed()
        m, s = divmod(int(e), 60)
        return f"{m}m{s:02d}s" if m else f"{s}s"


# ─────────────────────────────────────────────
# 파일 I/O
# ─────────────────────────────────────────────

def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dir(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 모델 유틸
# ─────────────────────────────────────────────

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def move_bi(bi: dict, device: str) -> dict:
    """브랜치 배치 dict를 device로 이동."""
    return {k: v.to(device, non_blocking=True) for k, v in bi.items()}