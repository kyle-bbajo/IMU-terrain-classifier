"""
train_loso.py — M1–M6 LOSO 교차검증 (v7.3 Final)
═══════════════════════════════════════════════════════
★ argparse로 N/seed/batch/epochs 런타임 변경
★ config 스냅샷 자동 저장 (재현성)
★ 체크포인트 (Spot 인스턴스 중단 후 재개)
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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS: list[tuple[str, type]] = [
    ("M2", M2_BranchCNN),
    ("M3", M3_BranchSE),
    ("M4", M4_BranchCBAM),
    ("M5", M5_BranchCBAMCross),
    ("M6", M6_BranchCBAMCrossAug),
]


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    p = argparse.ArgumentParser(description="LOSO 교차검증")
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
    """LOSO 교차검증 전체 파이프라인을 실행한다."""
    args = parse_args()
    config.apply_overrides(
        n_subjects=args.n_subjects,
        seed=args.seed,
        batch=args.batch,
        epochs=args.epochs,
    )
    config.print_config()
    log(f"  ★ LOSO ({config.N_SUBJECTS}-Fold)\n")

    # Config 스냅샷 저장
    config.snapshot(config.RESULT_LOSO)

    h5data = H5Data(config.H5_PATH)

    le = LabelEncoder()
    y: np.ndarray = le.fit_transform(h5data.y_raw).astype(np.int64)
    branch_idx, branch_ch = build_branch_idx(h5data.channels)
    groups: np.ndarray = h5data.subj_id
    unique_subjs: list[int] = sorted(np.unique(groups).tolist())

    log(f"  클래스: {le.classes_.tolist()} ({len(le.classes_)}개)")
    log(f"  피험자: {len(unique_subjs)}명  샘플: {len(y)}")

    out = config.RESULT_LOSO
    logo = LeaveOneGroupOut()
    n_folds: int = logo.get_n_splits(groups=groups)

    # 체크포인트 복원
    ckpt_path = out / "checkpoint.json"
    done_subjects: list[int] = []
    all_preds: dict[str, list[int]] = {}
    all_labels: list[int] = []

    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text())
            done_subjects = ckpt.get("done_subjects", [])
            all_preds = {k: list(v) for k, v in ckpt.get("preds", {}).items()}
            all_labels = list(ckpt.get("labels", []))
            log(f"  ★ 체크포인트 복원: {len(done_subjects)}명 완료")
        except (json.JSONDecodeError, KeyError) as e:
            log(f"  ⚠ 체크포인트 손상, 처음부터 시작 ({e})")
            done_subjects = []
            all_preds = {}
            all_labels = []

    all_hist: dict[str, list[dict]] = {}
    per_subj: dict[int, dict[str, float]] = {}
    t_total = time.time()
    fold_times: list[float] = []

    for fi, (tr_idx, te_idx) in enumerate(
        logo.split(np.zeros(len(y)), y, groups), 1
    ):
        te_subj = int(groups[te_idx[0]])
        if te_subj in done_subjects:
            continue

        t_fold = time.time()
        log(f"\n{'='*55}")
        log(f"  LOSO {fi}/{n_folds}  Test=S{te_subj:03d}"
            f"  ({len(te_idx)}샘플)  Train={len(tr_idx)}")
        log(f"{'='*55}")

        sc, pca = fit_pca_on_train(h5data, tr_idx)
        bsc     = fit_bsc_on_train(h5data, tr_idx)

        fold_preds: dict[str, np.ndarray] = {}
        fold_labels: np.ndarray = np.array([])

        res, labels, hist = run_M1(
            h5data, y, tr_idx, te_idx, sc, pca, f"S{te_subj:03d}")
        for t, p in res.items():
            all_preds.setdefault(t, []).extend(p.tolist())
            fold_preds[t] = p
        for t, h in hist.items():
            all_hist.setdefault(t, []).append(h)
        all_labels.extend(labels.tolist())
        fold_labels = labels

        for mname, mfn in MODELS:
            r2, _, h2 = run_branch(
                h5data, y, tr_idx, te_idx,
                branch_idx, branch_ch, bsc, mfn, mname, f"S{te_subj:03d}",
            )
            for t, p in r2.items():
                all_preds.setdefault(t, []).extend(p.tolist())
                fold_preds[t] = p
            for t, h in h2.items():
                all_hist.setdefault(t, []).append(h)

        per_subj[te_subj] = {
            t: round(accuracy_score(fold_labels, p), 4)
            for t, p in fold_preds.items()
        }
        done_subjects.append(te_subj)

        ckpt_data = {
            "done_subjects": done_subjects,
            "preds": {k: [int(x) for x in v] for k, v in all_preds.items()},
            "labels": [int(x) for x in all_labels],
        }
        ckpt_path.write_text(json.dumps(ckpt_data))

        del sc, pca, bsc; gc.collect()
        clear_fold_cache(f"S{te_subj:03d}")  # v8: 디스크 캐시 정리

        elapsed = (time.time() - t_fold) / 60
        fold_times.append(elapsed)
        remain = np.mean(fold_times) * (n_folds - fi)
        log(f"  S{te_subj:03d} 완료 ({elapsed:.1f}분)  남은: {remain:.0f}분")

    labels_arr = np.array(all_labels)
    results: dict[str, tuple[float, float]] = {}
    for tag, ps in all_preds.items():
        pa = np.array(ps)
        results[tag] = save_report(pa, labels_arr, le, f"LOSO_{tag}", out)
        save_cm(pa, labels_arr, le, f"LOSO_{tag}", out)
    if all_hist:
        save_history(all_hist, out)
    if per_subj:
        _save_per_subject(per_subj, results, out)

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  ★ LOSO  N={config.N_SUBJECTS}  {config.DEVICE_NAME}")
    print(f"  총 소요: {total_min:.1f}분")
    print(f"{'='*60}")
    for tag, (acc, f1) in results.items():
        print(f"  {tag:<20} Acc={acc:.4f}  F1={f1:.4f}")

    summary = {
        "experiment": "loso",
        "config": config.snapshot(),
        "total_minutes": round(total_min, 1),
        "results": {
            t: {"acc": round(a, 4), "f1": round(f, 4)}
            for t, (a, f) in results.items()
        },
        "per_subject": {str(k): v for k, v in per_subj.items()},
    }
    (out / "summary_loso.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    if ckpt_path.exists():
        ckpt_path.unlink()
    h5data.close()


def _save_per_subject(
    per_subj: dict[int, dict[str, float]],
    results: dict[str, tuple[float, float]],
    out_dir: Path,
) -> None:
    """피험자별 정확도를 CSV + 히트맵으로 저장한다."""
    subjs = sorted(per_subj.keys())
    tags  = list(results.keys())

    lines = ["Subject," + ",".join(tags)]
    for s in subjs:
        vals = [str(per_subj[s].get(t, "")) for t in tags]
        lines.append(f"S{s:03d},{','.join(vals)}")
    for name, fn in [("Mean", np.mean), ("Std", np.std)]:
        vals = [
            f"{fn([per_subj[s][t] for s in subjs if t in per_subj[s]]):.4f}"
            for t in tags
        ]
        lines.append(f"{name},{','.join(vals)}")
    (out_dir / "per_subject_accuracy.csv").write_text("\n".join(lines))

    data = np.array([[per_subj[s].get(t, 0) for t in tags] for s in subjs])
    fig, ax = plt.subplots(
        figsize=(max(10, len(tags) * 1.5), max(8, len(subjs) * 0.3)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_yticks(range(len(subjs)))
    ax.set_yticklabels([f"S{s:03d}" for s in subjs], fontsize=7)
    ax.set_title("LOSO Per-Subject Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "per_subject_heatmap.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
