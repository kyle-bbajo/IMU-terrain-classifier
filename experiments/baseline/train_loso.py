"""
train_loso.py — LOSO 교차검증 (v9.0)
═══════════════════════════════════════════════════════════
변경 이력 (v8.1 → v9.0)
──────────────────────────────────────────────────────────
[ADD]  MODEL_REGISTRY 기반 --models CLI 옵션
[ADD]  --skip-m1 플래그
[KEEP] 체크포인트 자동 저장/복원 (Spot 인스턴스 중단 대비)
[KEEP] 앙상블 다수결 투표
[KEEP] 피험자별 정확도 CSV + 히트맵 저장
[KEEP] fold별 클래스 분포 / 오류 메타 로깅
[KEEP] 남은 시간 ETA 표시
[FIX]  --models 기본값 DEFAULT_LOSO_ORDER 사용 (시간 절약)
[FIX]  summary JSON 키 통일
═══════════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys, time, json, gc, warnings, argparse
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

from channel_groups import build_branch_idx
from models import get_model_factories, DEFAULT_LOSO_ORDER
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


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOSO 교차검증 (v9.0)")
    p.add_argument("--n_subjects", type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--batch",      type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument(
        "--models", type=str,
        default=",".join(DEFAULT_LOSO_ORDER),
        help=f"LOSO에 돌릴 모델 목록. 기본: {','.join(DEFAULT_LOSO_ORDER)}",
    )
    p.add_argument("--skip-m1",    action="store_true", help="M1 baseline 생략")
    p.add_argument("--no-focal",   action="store_true", help="Focal Loss OFF")
    p.add_argument("--no-fft",     action="store_true", help="FFT Branch OFF")
    p.add_argument("--no-balanced",action="store_true", help="균형 샘플링 OFF")
    p.add_argument("--no-tta",     action="store_true", help="TTA OFF")
    return p.parse_args()


def _selected_models(spec: str) -> list[tuple[str, object]]:
    names = [s.strip() for s in spec.split(",") if s.strip()]
    return get_model_factories(names)


# ─────────────────────────────────────────────
# 피험자별 결과 저장
# ─────────────────────────────────────────────

def _save_per_subject(
    per_subj: dict[int, dict[str, float]],
    results:  dict[str, tuple[float, float]],
    out_dir:  Path,
) -> None:
    """피험자별 정확도를 CSV + 히트맵으로 저장한다."""
    subjs = sorted(per_subj.keys())
    tags  = list(results.keys())

    lines = ["Subject," + ",".join(tags)]
    for s in subjs:
        vals = [str(per_subj[s].get(t, "")) for t in tags]
        lines.append(f"S{s:03d},{','.join(vals)}")
    # 통계 행
    for stat_name, fn in [("Mean", np.mean), ("Std", np.std)]:
        vals = [
            f"{fn([per_subj[s][t] for s in subjs if t in per_subj[s]]):.4f}"
            for t in tags
        ]
        lines.append(f"{stat_name},{','.join(vals)}")
    (out_dir / "per_subject_accuracy.csv").write_text("\n".join(lines))

    # 히트맵
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


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config.apply_overrides(
        n_subjects=args.n_subjects,
        seed=args.seed,
        batch=args.batch,
        epochs=args.epochs,
        focal=False    if args.no_focal    else None,
        fft=False      if args.no_fft      else None,
        balanced=False if args.no_balanced else None,
        tta=False      if args.no_tta      else None,
    )
    config.print_config()

    models_to_run = _selected_models(args.models)
    log(f"  ★ LOSO 상위 모델: {[m for m, _ in models_to_run]}")
    if args.skip_m1:
        log("  ★ M1 baseline 생략 (--skip-m1)")

    out = config.CFG.result_loso
    config.snapshot(out)

    h5data = H5Data(config.CFG.h5_path)
    le     = LabelEncoder()
    y      = le.fit_transform(h5data.y_raw).astype(np.int64)
    branch_idx, branch_ch = build_branch_idx(h5data.channels)
    groups       = h5data.subj_id
    unique_subjs = sorted(np.unique(groups).tolist())
    logo         = LeaveOneGroupOut()
    n_folds      = logo.get_n_splits(groups=groups)

    log(f"  클래스: {le.classes_.tolist()} ({len(le.classes_)}개)")
    log(f"  피험자: {len(unique_subjs)}명  샘플: {len(y)}")

    # ── 체크포인트 복원 ──
    ckpt_path = out / "checkpoint.json"
    done_subjects: list[int] = []
    all_preds:     dict[str, list[int]] = {}
    all_labels:    list[int] = []

    if ckpt_path.exists():
        try:
            ckpt          = json.loads(ckpt_path.read_text())
            done_subjects = ckpt.get("done_subjects", [])
            all_preds     = {k: list(v) for k, v in ckpt.get("preds", {}).items()}
            all_labels    = list(ckpt.get("labels", []))
            log(f"  ★ 체크포인트 복원: {len(done_subjects)}명 완료 → 이어서 진행")
        except (json.JSONDecodeError, KeyError) as e:
            log(f"  ⚠ 체크포인트 손상, 처음부터 시작 ({e})")
            done_subjects, all_preds, all_labels = [], {}, []

    all_hist:   dict[str, list[dict]] = {}
    per_subj:   dict[int, dict[str, float]] = {}
    fold_meta:  list[dict] = []
    t_total     = time.time()
    fold_times: list[float] = []

    n_done_prev  = len(done_subjects)
    n_remaining  = n_folds - n_done_prev
    done_this_run = 0

    if n_done_prev > 0:
        log(f"  ★ {n_done_prev}명 완료 → 남은 {n_remaining}명 처리")

    for fi, (tr_idx, te_idx) in enumerate(
        logo.split(np.zeros(len(y)), y, groups), 1
    ):
        te_subj = int(groups[te_idx[0]])
        if te_subj in done_subjects:
            continue

        done_this_run += 1
        total_done = n_done_prev + done_this_run
        t_fold     = time.time()

        tr_dist = dict(zip(*np.unique(y[tr_idx], return_counts=True)))
        te_dist = dict(zip(*np.unique(y[te_idx], return_counts=True)))

        log(f"\n{'='*55}")
        log(f"  LOSO  Test=S{te_subj:03d}"
            f"  [{total_done}/{n_folds}]"
            f"  tr={len(tr_idx)}  te={len(te_idx)}")
        log(f"  Train 분포: {tr_dist}  /  Test 분포: {te_dist}")
        log(f"{'='*55}")

        sc, pca = fit_pca_on_train(h5data, tr_idx)
        bsc     = fit_bsc_on_train(h5data, tr_idx)

        fold_preds:       dict[str, np.ndarray] = {}
        fold_errors:      list[str] = []
        fold_models_meta: dict[str, dict] = {}

        # ── M1 ──
        if not args.skip_m1:
            res, labels, hist = run_M1(
                h5data, y, tr_idx, te_idx, sc, pca, f"S{te_subj:03d}")
            for t, p in res.items():
                all_preds.setdefault(t, []).extend(p.tolist())
                fold_preds[t] = p
            for t, h in hist.items():
                all_hist.setdefault(t, []).append(h)
                fold_errors.extend(h.get("meta", {}).get("errors", []))
                fold_models_meta[t] = h.get("meta", {})
            all_labels.extend(labels.tolist())
        else:
            labels = y[te_idx]
            all_labels.extend(labels.tolist())

        # ── Branch 모델들 ──
        for mname, mfn in models_to_run:
            r2, _, h2 = run_branch(
                h5data, y, tr_idx, te_idx,
                branch_idx, branch_ch, bsc, mfn, mname, f"S{te_subj:03d}",
            )
            for t, p in r2.items():
                all_preds.setdefault(t, []).extend(p.tolist())
                fold_preds[t] = p
            for t, h in h2.items():
                all_hist.setdefault(t, []).append(h)
                fold_errors.extend(h.get("meta", {}).get("errors", []))
                fold_models_meta[t] = h.get("meta", {})

        per_subj[te_subj] = {
            t: round(accuracy_score(labels, p), 4)
            for t, p in fold_preds.items()
        }
        done_subjects.append(te_subj)

        # 체크포인트 저장
        ckpt_path.write_text(json.dumps({
            "done_subjects": done_subjects,
            "preds":  {k: [int(x) for x in v] for k, v in all_preds.items()},
            "labels": [int(x) for x in all_labels],
        }))

        del sc, pca, bsc
        gc.collect()
        clear_fold_cache(f"S{te_subj:03d}")

        elapsed = (time.time() - t_fold) / 60
        fold_times.append(elapsed)
        n_left  = n_remaining - done_this_run
        eta     = np.mean(fold_times) * n_left if n_left > 0 else 0.0

        fold_meta.append({
            "fold":          total_done,
            "test_subject":  te_subj,
            "train_samples": int(len(tr_idx)),
            "test_samples":  int(len(te_idx)),
            "train_class_dist": {int(k): int(v) for k, v in tr_dist.items()},
            "test_class_dist":  {int(k): int(v) for k, v in te_dist.items()},
            "fold_time_min": round(elapsed, 1),
            "per_subject_acc": per_subj[te_subj],
            "errors":        fold_errors,
            "models":        fold_models_meta,
        })
        if fold_errors:
            log(f"  ⚠ S{te_subj:03d} 오류 {len(fold_errors)}건")
        log(f"  S{te_subj:03d} 완료 ({elapsed:.1f}분)"
            f"  [{total_done}/{n_folds}]  남은: {n_left}명 ~{eta:.0f}분")

    # ── 전체 결과 저장 ──
    labels_arr = np.array(all_labels)
    results: dict[str, tuple[float, float]] = {}
    for tag, preds in all_preds.items():
        pred_arr = np.array(preds)
        results[tag] = save_report(pred_arr, labels_arr, le, f"LOSO_{tag}", out)
        save_cm(pred_arr, labels_arr, le, f"LOSO_{tag}", out)

    # ── 앙상블 다수결 투표 ──
    branch_tags = [t for t in all_preds if t != "M1_CNN"]
    if len(branch_tags) >= 2 and len(labels_arr) > 0:
        from scipy.stats import mode as _mode
        pred_stack    = np.stack([np.array(all_preds[t]) for t in branch_tags])
        ensemble_pred = _mode(pred_stack, axis=0, keepdims=False).mode
        etag          = "Ensemble_Vote"
        results[etag] = save_report(ensemble_pred, labels_arr, le, f"LOSO_{etag}", out)
        save_cm(ensemble_pred, labels_arr, le, f"LOSO_{etag}", out)
        log(f"  ★ 앙상블 ({', '.join(branch_tags)})"
            f"  Acc={results[etag][0]:.4f}  F1={results[etag][1]:.4f}")

    if all_hist:
        save_history(all_hist, out)
    if per_subj:
        _save_per_subject(per_subj, results, out)

    total_min    = (time.time() - t_total) / 60
    total_errors = sum(len(fm["errors"]) for fm in fold_meta)
    total_ooms   = sum(
        m.get("oom_events", 0)
        for fm in fold_meta for m in fm["models"].values()
    )

    print(f"\n{'='*60}")
    print(f"  ★ LOSO  N={config.CFG.n_subjects}  {config.DEVICE_NAME}")
    print(f"  총 소요: {total_min:.1f}분")
    if total_errors > 0:
        print(f"  ⚠ 총 오류: {total_errors}건 (OOM: {total_ooms}건)")
    print(f"{'='*60}")
    for tag, (acc, f1) in results.items():
        print(f"  {tag:<25} Acc={acc:.4f}  F1={f1:.4f}")

    summary = {
        "experiment":       "loso",
        "version":          "v9.0",
        "compare_order":    [m for m, _ in models_to_run],
        "skip_m1":          args.skip_m1,
        "config":           config.snapshot(),
        "total_minutes":    round(total_min, 1),
        "total_errors":     total_errors,
        "total_oom_events": total_ooms,
        "results": {
            t: {"acc": round(a, 4), "f1": round(f, 4)}
            for t, (a, f) in results.items()
        },
        "per_subject": {str(k): v for k, v in per_subj.items()},
        "fold_meta":   fold_meta,
    }
    (out / "summary_loso.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # 체크포인트 삭제 (정상 완료)
    if ckpt_path.exists():
        ckpt_path.unlink()
        log(f"  체크포인트 삭제 완료")

    log(f"  ✅ {out / 'summary_loso.json'}")
    h5data.close()


if __name__ == "__main__":
    main()