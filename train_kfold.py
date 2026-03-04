"""
train_kfold.py — M1–M6 K-Fold 교차검증 (v8.0)
═══════════════════════════════════════════════════════
★ argparse로 N/seed/batch/epochs 런타임 변경
★ config 스냅샷 자동 저장 (재현성)
★ StratifiedGroupKFold (subject-wise, 피험자 누수 방지)
═══════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys, time, json, gc, warnings, argparse
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

from channel_groups import build_branch_idx
from models import (M2_BranchCNN, M3_BranchSE,
                    M4_BranchCBAM, M5_BranchCBAMCross, M6_BranchCBAMCrossAug)
from train_common import (
    log, H5Data,
    fit_pca_on_train, fit_bsc_on_train,
    run_M1, run_branch,
    save_report, save_cm, save_history,
    clear_fold_cache,
)

MODELS: list[tuple[str, type]] = [
    ("M2", M2_BranchCNN),
    ("M3", M3_BranchSE),
    ("M4", M4_BranchCBAM),
    ("M5", M5_BranchCBAMCross),
    ("M6", M6_BranchCBAMCrossAug),
]


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    p = argparse.ArgumentParser(description="K-Fold 교차검증")
    p.add_argument("--n_subjects", type=int, default=None,
                   help="피험자 수 (기본: config.N_SUBJECTS)")
    p.add_argument("--seed", type=int, default=None,
                   help="랜덤 시드 (기본: 42)")
    p.add_argument("--batch", type=int, default=None,
                   help="배치 크기 (기본: 자동)")
    p.add_argument("--epochs", type=int, default=None,
                   help="에포크 수 (기본: 50)")
    return p.parse_args()


def main() -> None:
    """K-Fold 교차검증 전체 파이프라인을 실행한다."""
    args = parse_args()
    config.apply_overrides(
        n_subjects=args.n_subjects,
        seed=args.seed,
        batch=args.batch,
        epochs=args.epochs,
    )
    config.print_config()
    log(f"  ★ K-Fold {config.KFOLD}-Fold (Subject-wise)\n")

    # Config 스냅샷 저장
    config.snapshot(config.RESULT_KFOLD)

    h5data = H5Data(config.H5_PATH)

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

    all_preds: dict[str, list] = {}
    all_labels: list[int] = []
    all_hist: dict[str, list[dict]] = {}
    t_total = time.time()

    for fi, (tr_idx, te_idx) in enumerate(
        sgkf.split(np.zeros(len(y)), y, groups=groups), 1
    ):
        tr_s = sorted(set(groups[tr_idx].tolist()))
        te_s = sorted(set(groups[te_idx].tolist()))

        overlap = set(tr_s) & set(te_s)
        if overlap:
            raise ValueError(f"Fold {fi}: 피험자 누수 발견! {overlap}")

        log(f"\n{'='*55}")
        log(f"  Fold {fi}/{config.KFOLD}"
            f"  tr={len(tr_idx)}({len(tr_s)}명)"
            f"  te={len(te_idx)}({len(te_s)}명)")
        log(f"  Test 피험자: {te_s}")
        log(f"{'='*55}")

        sc, pca = fit_pca_on_train(h5data, tr_idx)
        bsc     = fit_bsc_on_train(h5data, tr_idx)

        res, labels, hist = run_M1(h5data, y, tr_idx, te_idx, sc, pca, fi)
        for t, p in res.items():
            all_preds.setdefault(t, []).extend(p)
        for t, h in hist.items():
            all_hist.setdefault(t, []).append(h)
        all_labels.extend(labels.tolist())

        for mname, mfn in MODELS:
            r2, _, h2 = run_branch(
                h5data, y, tr_idx, te_idx,
                branch_idx, branch_ch, bsc, mfn, mname, fi,
            )
            for t, p in r2.items():
                all_preds.setdefault(t, []).extend(p)
            for t, h in h2.items():
                all_hist.setdefault(t, []).append(h)

        del sc, pca, bsc; gc.collect()
        clear_fold_cache(fi)  # v8: 디스크 캐시 정리

    labels_arr = np.array(all_labels)
    results: dict[str, tuple[float, float]] = {}
    for tag, ps in all_preds.items():
        pa = np.array(ps)
        results[tag] = save_report(pa, labels_arr, le,
                                   f"KF{config.KFOLD}_{tag}", out)
        save_cm(pa, labels_arr, le, f"KF{config.KFOLD}_{tag}", out)
    save_history(all_hist, out)

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  ★ {config.KFOLD}-Fold  {config.DEVICE_NAME}")
    print(f"  총 소요: {total_min:.1f}분")
    print(f"{'='*60}")
    for tag, (acc, f1) in results.items():
        print(f"  {tag:<20} Acc={acc:.4f}  F1={f1:.4f}")

    summary = {
        "experiment": "kfold",
        "config": config.snapshot(),
        "total_minutes": round(total_min, 1),
        "results": {
            t: {"acc": round(a, 4), "f1": round(f, 4)}
            for t, (a, f) in results.items()
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  ✅ {out / 'summary.json'}")
    h5data.close()


if __name__ == "__main__":
    main()
