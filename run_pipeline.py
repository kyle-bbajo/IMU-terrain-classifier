#!/usr/bin/env python3
"""run_pipeline.py — 전체 학습 파이프라인 오케스트레이터

실행 순서:
  [PHASE 0]  step_segmentation.py          (CPU 전용, 1회)
  [PHASE 1]  train_surface   + train_attribute   (GPU 경량, 병렬)
  [PHASE 2]  train_raw                           (GPU 중간)
  [PHASE 3]  train_hierarchical                  (GPU 중간)
  [PHASE 4]  train_kfold                         (GPU 최고, 마지막)
             └─ train_kfold --ensemble_only      (PHASE 1~3 확률 수집 후 앙상블)

GPU 사용량 분류:
  경량  : train_surface (feature MLP), train_attribute (feature MLP)
  중간  : train_raw (CNN+Transformer), train_hierarchical (CBAM+Fusion)
  최고  : train_kfold (M7_Attr, CNN+GRU+Feature 복합 앙상블)

사용법:
  python run_pipeline.py                  # 전체 실행 (기본)
  python run_pipeline.py --bg             # 백그라운드로 실행 (nohup, 로그 자동 저장)
  python run_pipeline.py --skip_seg       # step_segmentation 건너뜀 (H5 이미 있음)
  python run_pipeline.py --force_seg      # HDF5 처음부터 완전 재생성 (기본: 새 CSV만 누적 append)
  python run_pipeline.py --phase 1        # PHASE 1부터 시작
  python run_pipeline.py --only kfold     # kfold만 실행
  python run_pipeline.py --no_cache       # 피처 캐시 무시하고 재추출
  python run_pipeline.py --dry_run        # 실제 실행 없이 명령어만 출력
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 색상 출력 유틸
# ─────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
CYAN  = "\033[96m"
GRAY  = "\033[90m"

def _c(color, msg): return f"{color}{msg}{RESET}"

def log(msg: str, color: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = _c(GRAY, f"[{ts}]")
    print(f"{prefix} {color}{msg}{RESET}", flush=True)

def log_phase(n: int, title: str):
    bar = "─" * 60
    print(f"\n{_c(CYAN+BOLD, bar)}")
    print(_c(CYAN+BOLD, f"  PHASE {n}  {title}"))
    print(f"{_c(CYAN+BOLD, bar)}\n", flush=True)

# ─────────────────────────────────────────────────────────────
# 경로 설정
#
# 레포 구조:
#   repo/
#     src/                  ← config.py, features.py, models.py, train_common.py
#     train_kfold.py
#     train_raw.py
#     train_surface.py
#     train_attribute.py
#     train_hierarchical.py
#     step_segmentation.py
#     run_pipeline.py       ← 이 파일
# ─────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).parent.resolve()   # run_pipeline.py 위치 = repo 루트
LOG_DIR  = REPO_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

SCRIPTS = {
    "seg"         : REPO_DIR / "src" / "step_segmentation.py",
    "surface"     : REPO_DIR / "train_surface.py",
    "attribute"   : REPO_DIR / "train_attribute.py",
    "raw"         : REPO_DIR / "train_raw.py",
    "hierarchical": REPO_DIR / "train_hierarchical.py",
    "kfold"       : REPO_DIR / "train_kfold.py",
}

# ─────────────────────────────────────────────────────────────
# 스크립트 실행 함수
# ─────────────────────────────────────────────────────────────
def run_script(
    name: str,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> tuple[str, int, float]:
    """스크립트를 실행하고 (name, returncode, elapsed) 반환."""
    script = SCRIPTS[name]
    cmd = ["/home/ubuntu/project/.venv/bin/python", str(script)] + (extra_args or [])
    log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    cmd_str = " ".join(str(c) for c in cmd)
    log(_c(YELLOW, f"▶ 시작: {name}"), "")
    log(_c(GRAY,   f"  CMD : {cmd_str}"), "")
    log(_c(GRAY,   f"  LOG : {log_file}"), "")

    if dry_run:
        log(_c(YELLOW, f"  [DRY RUN] 실행 건너뜀"), "")
        return name, 0, 0.0

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(REPO_DIR),
        )
        lf.write(proc.stdout)

    elapsed = time.time() - t0
    rc = proc.returncode

    if rc == 0:
        log(_c(GREEN, f"✅ 완료: {name}  ({elapsed/60:.1f}분)"), "")
    else:
        log(_c(RED, f"❌ 실패: {name}  (code={rc})  로그: {log_file}"), "")
        # 실패 시 마지막 30줄 출력
        tail = proc.stdout.strip().split("\n")[-30:]
        for line in tail:
            print(_c(RED, f"    {line}"), flush=True)

    return name, rc, elapsed


def run_parallel(
    jobs: dict[str, list[str]],  # {name: extra_args}
    dry_run: bool = False,
) -> dict[str, int]:
    """여러 스크립트를 ThreadPoolExecutor로 병렬 실행. {name: returncode} 반환."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {
            ex.submit(run_script, name, args, dry_run): name
            for name, args in jobs.items()
        }
        for fut in as_completed(futures):
            name, rc, _ = fut.result()
            results[name] = rc
    return results


# ─────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="전체 학습 파이프라인 실행기")
    p.add_argument("--bg",        action="store_true",
                   help="백그라운드로 실행 (nohup). 로그: logs/pipeline_TIMESTAMP.log")
    p.add_argument("--skip_seg",  action="store_true",
                   help="step_segmentation 건너뜀 (H5 이미 존재할 때)")
    p.add_argument("--force_seg", action="store_true",
                   help="step_segmentation --force: HDF5 처음부터 완전 재생성 (기본: 새 CSV만 누적 append)")
    p.add_argument("--phase",     type=int, default=0,
                   help="해당 PHASE부터 시작 (0=처음부터, 1=surface+attribute부터, ...)")
    p.add_argument("--only",      type=str, default="",
                   help="특정 스크립트만 실행 (seg/surface/attribute/raw/hierarchical/kfold)")
    p.add_argument("--no_cache",  action="store_true",
                   help="피처 캐시 무시 (train_kfold에 --no-feat-cache 전달)")
    p.add_argument("--dry_run",   action="store_true",
                   help="명령어만 출력, 실제 실행 없음")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--kfold",     type=int, default=5)
    # 에포크 오버라이드 (빠른 테스트용)
    p.add_argument("--surface_epochs",    type=int, default=None)
    p.add_argument("--attribute_epochs",  type=int, default=None)
    p.add_argument("--raw_epochs",        type=int, default=None)
    p.add_argument("--kfold_epochs",      type=int, default=None)
    p.add_argument("--hier_fusion_epochs",type=int, default=None)
    args = p.parse_args()

    # ── 백그라운드 재실행 ──────────────────────────────────────
    if args.bg:
        LOG_DIR.mkdir(exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_log = LOG_DIR / f"pipeline_{ts}.log"
        pid_file = LOG_DIR / f"pipeline_{ts}.pid"

        # --bg 제거한 나머지 인수로 자기 자신을 nohup 실행
        cmd = ["/home/ubuntu/project/.venv/bin/python"] + [a for a in sys.argv if a != "--bg"]
        print(f"[백그라운드 실행]")
        print(f"  CMD : {' '.join(cmd)}")
        print(f"  LOG : {main_log}")

        with open(main_log, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,   # 터미널 종료해도 계속 실행
                cwd=str(REPO_DIR),
            )
        pid_file.write_text(str(proc.pid))
        print(f"  PID : {proc.pid}  (저장: {pid_file})")
        print()
        print(f"  로그 확인: tail -f {main_log}")
        print(f"  중단하기 : kill {proc.pid}")
        sys.exit(0)

    # 스크립트 존재 확인
    for name, path in SCRIPTS.items():
        if not path.exists():
            log(_c(RED, f"스크립트 없음: {path}"), "")
            if name != "seg":  # seg는 없어도 --skip_seg면 OK
                sys.exit(1)

    # ── --only 옵션 ──────────────────────────────────────────
    if args.only:
        name = args.only.strip()
        if name not in SCRIPTS:
            print(f"알 수 없는 스크립트: {name}. 선택지: {list(SCRIPTS.keys())}")
            sys.exit(1)
        extra = _build_extra(name, args)
        _, rc, _ = run_script(name, extra, args.dry_run)
        sys.exit(rc)

    total_start = time.time()
    failed: list[str] = []

    # ══════════════════════════════════════════════════════════
    # PHASE 0 — step_segmentation (CPU, 1회)
    # ══════════════════════════════════════════════════════════
    if not args.skip_seg and args.phase <= 0:
        mode_str = "FULL REBUILD (--force)" if args.force_seg else "누적 append (새 CSV만)"
        log_phase(0, f"스텝 분할  step_segmentation.py  [CPU]  [{mode_str}]")
        seg_extra = ["--force"] if args.force_seg else []
        _, rc, _ = run_script("seg", extra_args=seg_extra, dry_run=args.dry_run)
        if rc != 0:
            log(_c(RED, "PHASE 0 실패. 중단합니다."), "")
            sys.exit(1)
    else:
        log(_c(GRAY, "PHASE 0 건너뜀 (--skip_seg 또는 --phase >= 1)"), "")

    # ══════════════════════════════════════════════════════════
    # PHASE 1 — surface + attribute  (GPU 경량, 병렬)
    # ══════════════════════════════════════════════════════════
    if args.phase <= 1:
        log_phase(1, "surface + attribute  [GPU 경량 — 병렬 실행]")
        jobs = {
            "surface"  : _build_extra("surface",   args),
            "attribute": _build_extra("attribute", args),
        }
        results = run_parallel(jobs, args.dry_run)
        for name, rc in results.items():
            if rc != 0:
                failed.append(name)
                log(_c(YELLOW, f"  ⚠ {name} 실패, 후속 앙상블에서 제외됩니다."), "")

    # ══════════════════════════════════════════════════════════
    # PHASE 2 — train_raw  (GPU 중간)
    # ══════════════════════════════════════════════════════════
    if args.phase <= 2:
        log_phase(2, "train_raw  [GPU 중간 — 단독 실행]")
        _, rc, _ = run_script("raw", _build_extra("raw", args), args.dry_run)
        if rc != 0:
            failed.append("raw")
            log(_c(YELLOW, "  ⚠ train_raw 실패, 후속 앙상블에서 제외됩니다."), "")

    # ══════════════════════════════════════════════════════════
    # PHASE 3 — train_hierarchical  (GPU 중간)
    # ══════════════════════════════════════════════════════════
    if args.phase <= 3:
        log_phase(3, "train_hierarchical  [GPU 중간 — 단독 실행]")
        _, rc, _ = run_script("hierarchical", _build_extra("hierarchical", args), args.dry_run)
        if rc != 0:
            failed.append("hierarchical")
            log(_c(YELLOW, "  ⚠ train_hierarchical 실패, 앙상블에서 제외됩니다."), "")

    # ══════════════════════════════════════════════════════════
    # PHASE 4 — train_kfold  (GPU 최고)
    # ══════════════════════════════════════════════════════════
    if args.phase <= 4:
        log_phase(4, "train_kfold  [GPU 최고 — 단독 실행]")
        kfold_extra = _build_extra("kfold", args)
        if args.no_cache:
            kfold_extra.append("--no-feat-cache")
        _, rc, _ = run_script("kfold", kfold_extra, args.dry_run)
        if rc != 0:
            failed.append("kfold")

    # ══════════════════════════════════════════════════════════
    # 최종 요약
    # ══════════════════════════════════════════════════════════
    elapsed_total = time.time() - total_start
    print()
    print(_c(CYAN+BOLD, "═" * 60))
    print(_c(CYAN+BOLD, "  파이프라인 완료"))
    print(_c(CYAN+BOLD, "═" * 60))
    print(f"  총 소요 시간: {elapsed_total/60:.1f}분  ({elapsed_total/3600:.2f}시간)")
    if failed:
        print(_c(YELLOW, f"  실패한 스크립트: {failed}"))
        print(_c(YELLOW, f"  → logs/ 디렉토리에서 로그 확인하세요"))
    else:
        print(_c(GREEN, "  ✅ 모든 스크립트 성공"))
    print()


# ─────────────────────────────────────────────────────────────
# 스크립트별 추가 인수 빌드
# ─────────────────────────────────────────────────────────────
def _build_extra(name: str, args: argparse.Namespace) -> list[str]:
    extra = []
    seed  = str(args.seed)
    kfold = str(args.kfold)

    if name == "seg":
        pass  # step_segmentation은 config.py로만 제어

    elif name == "surface":
        extra += ["--seed", seed, "--kfold", kfold]
        if args.surface_epochs:
            extra += ["--epochs", str(args.surface_epochs)]

    elif name == "attribute":
        extra += ["--seed", seed, "--kfold", kfold, "--no-wandb"]
        if args.attribute_epochs:
            extra += ["--epochs", str(args.attribute_epochs)]

    elif name == "raw":
        extra += ["--seed", seed, "--kfold", kfold]
        if args.raw_epochs:
            extra += ["--epochs", str(args.raw_epochs)]

    elif name == "hierarchical":
        extra += ["--n_subjects", "999"]   # 전체 피험자 사용
        if args.hier_fusion_epochs:
            extra += ["--fusion_epochs", str(args.hier_fusion_epochs)]

    elif name == "kfold":
        extra += ["--seed", seed, "--n_folds", kfold]
        if args.kfold_epochs:
            extra += ["--epochs", str(args.kfold_epochs)]

    return extra


if __name__ == "__main__":
    main()