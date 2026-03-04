"""
train_kfold.py — M1-M6 K-Fold 교차검증 (v8 Final)
═══════════════════════════════════════════════════════
v7->v8 변경사항
  ★ Fold 단위 에러 복구 (하나 실패해도 나머지 계속)
  ★ 피험자 누수(data leakage) 자동 검증
  ★ seed_everything 으로 재현성 보장
  ★ 모델별 실패 격리 (한 모델 OOM → 나머지 모델 계속)
  ★ 결과 JSON에 실패 기록 포함
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys
import time
import json
import gc
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

from channel_groups import build_branch_idx
from models import (
    M2_BranchCNN, M3_BranchSE,
    M4_BranchCBAM, M5_BranchCBAMCross, M6_BranchCBAMCrossAug,
)
from train_common import (
    log, seed_everything, H5Data,
    fit_pca_on_train, fit_bsc_on_train,
    run_M1, run_branch,
    save_report, save_cm, save_history,
)

# 모델 레지스트리
MODELS: list[tuple[str, Any]] = [
    ("M2", M2_BranchCNN),
    ("M3", M3_BranchSE),
    ("M4", M4_BranchCBAM),
    ("M5", M5_BranchCBAMCross),
    ("M6", M6_BranchCBAMCrossAug),
]


def main() -> None:
    """K-Fold 교차검증 전체 파이프라인을 실행한다."""
    seed_everything()
    config.print_config()
    log(f"  ★ K-Fold {config.KFOLD}-Fold (Subject-wise)\n")

    # HDF5 로드
    with H5Data(config.H5_PATH) as h5data:
        le = LabelEncoder()
        y: np.ndarray = le.fit_transform(h5data.y_raw).astype(np.int64)
        branch_idx, branch_ch = build_branch_idx(h5data.channels)
        groups: np.ndarray = h5data.subj_id

        log(f"  클래스: {le.classes_.tolist()} ({len(le.classes_)}개)")
        log(f"  피험자: {len(np.unique(groups))}명  샘플: {len(y)}")

        out = config.RESULT_KFOLD
        sgkf = StratifiedGroupKFold(
            n_splits=config.KFOLD, shuffle=True, random_state=config.SEED,
        )

        all_preds: dict[str, list[int]] = {}
        all_labels: list[int] = []
        all_hist: dict[str, list[dict]] = {}
        failures: list[dict] = []
        t_total = time.time()

        for fi, (tr_idx, te_idx) in enumerate(
            sgkf.split(np.zeros(len(y)), y, groups=groups), 1
        ):
            tr_s = sorted(set(groups[tr_idx].tolist()))
            te_s = sorted(set(groups[te_idx].tolist()))

            # 피험자 누수 검증 (data leakage)
            overlap = set(tr_s) & set(te_s)
            if overlap:
                msg = f"Fold {fi}: 피험자 누수! {overlap}"
                log(f"  [FATAL] {msg}")
                raise ValueError(msg)

            log(f"\n{'='*55}")
            log(f"  Fold {fi}/{config.KFOLD}"
                f"  tr={len(tr_idx)}({len(tr_s)}명)"
                f"  te={len(te_idx)}({len(te_s)}명)")
            log(f"  Test 피험자: {te_s}")
            log(f"{'='*55}")

            try:
                sc, pca = fit_pca_on_train(h5data, tr_idx)
                bsc     = fit_bsc_on_train(h5data, tr_idx)
            except Exception as e:
                log(f"  [ERROR] Fold {fi} Scaler/PCA 실패: {e}")
                failures.append({"fold": fi, "stage": "pca", "error": str(e)})
                continue

            fold_labels_saved = False

            # M1
            try:
                res, labels, hist = run_M1(
                    h5data, y, tr_idx, te_idx, sc, pca, fi)
                for t, p in res.items():
                    all_preds.setdefault(t, []).extend(p.tolist())
                for t, h in hist.items():
                    all_hist.setdefault(t, []).append(h)
                if not fold_labels_saved:
                    all_labels.extend(labels.tolist())
                    fold_labels_saved = True
            except Exception as e:
                log(f"  [ERROR] Fold {fi} M1 실패: {e}")
                failures.append({"fold": fi, "model": "M1", "error": str(e)})

            # M2-M6 (각각 독립 — 하나 실패해도 나머지 계속)
            for mname, mfn in MODELS:
                try:
                    r2, lb2, h2 = run_branch(
                        h5data, y, tr_idx, te_idx,
                        branch_idx, branch_ch, bsc, mfn, mname, fi,
                    )
                    for t, p in r2.items():
                        all_preds.setdefault(t, []).extend(p.tolist())
                    for t, h in h2.items():
                        all_hist.setdefault(t, []).append(h)
                    if not fold_labels_saved:
                        all_labels.extend(lb2.tolist())
                        fold_labels_saved = True
                except Exception as e:
                    log(f"  [ERROR] Fold {fi} {mname} 실패: {e}")
                    failures.append({
                        "fold": fi, "model": mname, "error": str(e)
                    })
                    if config.USE_GPU:
                        import torch
                        torch.cuda.empty_cache()

            del sc, pca, bsc
            gc.collect()

        # ── 최종 결과 ──
        labels_arr = np.array(all_labels)
        results: dict[str, tuple[float, float]] = {}
        for tag, ps in all_preds.items():
            pa = np.array(ps)
            # label/pred 길이 불일치 방어
            min_len = min(len(pa), len(labels_arr))
            if min_len == 0:
                continue
            results[tag] = save_report(
                pa[:min_len], labels_arr[:min_len], le,
                f"KF{config.KFOLD}_{tag}", out,
            )
            save_cm(
                pa[:min_len], labels_arr[:min_len], le,
                f"KF{config.KFOLD}_{tag}", out,
            )
        if all_hist:
            save_history(all_hist, out)

    # 최종 출력
    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  ★ {config.KFOLD}-Fold  {config.DEVICE_NAME}")
    print(f"  총 소요: {total_min:.1f}분")
    if failures:
        print(f"  실패: {len(failures)}건")
    print(f"{'='*60}")
    for tag, (acc, f1) in results.items():
        print(f"  {tag:<20} Acc={acc:.4f}  F1={f1:.4f}")

    summary = {
        "experiment": "kfold",
        "version": "v8",
        "device": config.DEVICE_NAME,
        "n_subjects": config.N_SUBJECTS,
        "n_classes": config.NUM_CLASSES,
        "kfold": config.KFOLD,
        "seed": config.SEED,
        "split": "StratifiedGroupKFold",
        "strategy": "preload" if config.USE_PRELOAD else "otf",
        "amp": config.USE_AMP,
        "batch": config.BATCH,
        "grad_accum": config.GRAD_ACCUM_STEPS,
        "effective_batch": config.BATCH * config.GRAD_ACCUM_STEPS,
        "total_minutes": round(total_min, 1),
        "results": {
            t: {"acc": round(a, 4), "f1": round(f, 4)}
            for t, (a, f) in results.items()
        },
        "failures": failures,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  ✅ {out / 'summary.json'}")


if __name__ == "__main__":
    main()