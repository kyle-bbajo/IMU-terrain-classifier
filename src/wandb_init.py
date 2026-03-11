"""
wandb_init.py — W&B 공통 초기화 유틸 (전 실험 공유)
═══════════════════════════════════════════════════════
사용법:
    from wandb_init import wandb_start, wandb_log_fold, wandb_finish

    # 실험 시작
    run = wandb_start("kfold", args, cfg_dict={...})

    # fold 결과 기록
    wandb_log_fold(fold=1, metrics={"acc": 0.92, "f1": 0.91})

    # epoch 기록 (train_common.py 에서 자동 호출됨)
    wandb_log_epoch(tag="[F1][ResNet1D]", ep=3, tl=0.4, ta=0.85, vl=0.3, va=0.9, lr=1e-3)

    # 최종 요약 & 종료
    wandb_finish(results=[{"model": "ResNet1D", "acc": 0.92, "f1": 0.91}])
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import datetime
from typing import Any

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    wandb = None
    _WANDB_OK = False

# ── 활성화 여부 판단 ────────────────────────────────────────────────────────
def _enabled() -> bool:
    """WANDB_PROJECT 환경변수가 있고 wandb 설치된 경우에만 True."""
    return _WANDB_OK and bool(os.getenv("WANDB_PROJECT"))


# ── 실험 시작 ───────────────────────────────────────────────────────────────
def wandb_start(
    experiment: str,        # "kfold" | "loso" | "hierarchical"
    args=None,              # argparse Namespace
    cfg_dict: dict | None = None,
    extra_tags: list[str] | None = None,
) -> Any:
    """
    W&B run을 초기화한다.
    WANDB_PROJECT 미설정 또는 wandb 미설치 시 아무것도 하지 않는다.

    Returns
    -------
    wandb.run 또는 None
    """
    if not _enabled():
        return None

    # run 이름
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = getattr(args, "run_name", None) or f"{experiment}_{ts}"

    # 태그
    tags = [experiment]
    if extra_tags:
        tags.extend(extra_tags)

    # config: CFG + CLI args 합산
    config = {}
    if cfg_dict:
        config.update(cfg_dict)
    if args is not None:
        config.update({k: v for k, v in vars(args).items() if v is not None})

    run = wandb.init(
        project = os.getenv("WANDB_PROJECT", "imu-terrain"),
        entity  = os.getenv("WANDB_ENTITY",  None),
        name    = run_name,
        group   = experiment,
        tags    = tags,
        config  = config,
        dir     = os.getenv("WANDB_DIR", None),
        reinit  = True,
    )
    print(f"[W&B] 실험={experiment}  run={run.name}  url={run.url}")
    return run


# ── epoch 단위 로그 ─────────────────────────────────────────────────────────
def wandb_log_epoch(
    tag: str,
    ep: int,
    tl: float, ta: float,
    vl: float, va: float,
    lr: float,
) -> None:
    """
    train_common.py 의 epoch 루프에서 호출.
    tag 예: "[F1][ResNet1D]"  →  W&B key: "F1/ResNet1D/train_loss"
    """
    if not (_WANDB_OK and wandb.run is not None):
        return

    # tag 정리: "[F1][ResNet1D]" → "F1/ResNet1D"
    key = tag.replace("][", "/").replace("[", "").replace("]", "")

    wandb.log({
        f"{key}/train_loss": tl,
        f"{key}/train_acc":  ta,
        f"{key}/val_loss":   vl,
        f"{key}/val_acc":    va,
        f"{key}/lr":         lr,
        "epoch": ep,
    })


# ── fold 단위 결과 ───────────────────────────────────────────────────────────
def wandb_log_fold(fold: int, metrics: dict) -> None:
    """fold 완료 시점에 호출. metrics 예: {"acc": 0.92, "f1": 0.91}"""
    if not (_WANDB_OK and wandb.run is not None):
        return
    wandb.log({f"fold{fold}/{k}": v for k, v in metrics.items()})


# ── 최종 요약 & 종료 ─────────────────────────────────────────────────────────
def wandb_finish(
    results: list[dict] | None = None,
    extra_summary: dict | None = None,
) -> None:
    """
    실험 완료 시 호출.

    Parameters
    ----------
    results : [{"model": "ResNet1D", "acc": 0.92, "f1": 0.91}, ...]
              또는 [{"acc": 0.92, "f1": 0.91}]  (단일 실험)
    extra_summary : 추가로 기록할 dict
    """
    if not (_WANDB_OK and wandb.run is not None):
        return

    summary = {}

    if results:
        for r in results:
            prefix = r.get("model", "final")
            summary[f"{prefix}/acc"] = r.get("acc", 0)
            summary[f"{prefix}/f1"]  = r.get("f1",  0)
        # 전체 평균
        accs = [r["acc"] for r in results if "acc" in r]
        f1s  = [r["f1"]  for r in results if "f1"  in r]
        if accs: summary["mean_acc"] = round(sum(accs) / len(accs), 4)
        if f1s:  summary["mean_f1"]  = round(sum(f1s)  / len(f1s),  4)

    if extra_summary:
        summary.update(extra_summary)

    wandb.summary.update(summary)
    wandb.finish()
    print("[W&B] run 종료 완료")