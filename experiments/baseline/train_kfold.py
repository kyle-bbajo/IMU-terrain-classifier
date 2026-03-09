"""
train_kfold.py — K-Fold 교차검증 (v9.0)
═══════════════════════════════════════════════════════════
변경 이력 (v8.1 → v9.0)
──────────────────────────────────────────────────────────
[ADD]  MODEL_REGISTRY 기반 --models CLI 옵션 (외부 선택 가능)
[ADD]  --skip-m1 플래그 (M1 baseline 생략)
[ADD]  ResNet1D / CNNTCN 자동 포함 (DEFAULT_COMPARE_ORDER)
[KEEP] 피험자 누수 감지 (overlap check)
[KEEP] fold별 클래스 분포 로깅 + fold_meta 저장
[KEEP] 앙상블 다수결 투표 (M2–M6 + ResNet1D + CNNTCN)
[KEEP] 체크포인트 없음 (K-Fold는 짧으므로 불필요)
[FIX]  summary JSON 키 통일 (v8 summary.json 호환)
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
from sklearn.model_selection import StratifiedGroupKFold

from channel_groups import build_branch_idx
from models import get_model_factories, DEFAULT_COMPARE_ORDER
from train_common import (
    log, H5Data,
    fit_pca_on_train, fit_bsc_on_train,
    run_M1, run_branch,
    save_report, save_cm, save_history,
    clear_fold_cache,
)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-Fold 교차검증 (v9.0)")
    p.add_argument("--n_subjects", type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--batch",      type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument(
        "--models", type=str,
        default=",".join(DEFAULT_COMPARE_ORDER),
        help=f"쉼표 구분 모델 목록. 기본: {','.join(DEFAULT_COMPARE_ORDER)}",
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
# 메인
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config.apply_overrides(
        n_subjects=args.n_subjects,
        seed=args.seed,
        batch=args.batch,
        epochs=args.epochs,
        focal=False   if args.no_focal    else None,
        fft=False     if args.no_fft      else None,
        balanced=False if args.no_balanced else None,
        tta=False     if args.no_tta      else None,
    )
    config.print_config()

    models_to_run = _selected_models(args.models)
    log(f"  ★ K-Fold {config.CFG.kfold}-Fold  비교 모델: {[m for m, _ in models_to_run]}")
    if args.skip_m1:
        log("  ★ M1 baseline 생략 (--skip-m1)")

    out = config.CFG.result_kfold
    config.snapshot(out)

    h5data = H5Data(config.CFG.h5_path)
    le     = LabelEncoder()
    y      = le.fit_transform(h5data.y_raw).astype(np.int64)
    branch_idx, branch_ch = build_branch_idx(h5data.channels)
    groups = h5data.subj_id

    log(f"  클래스: {le.classes_.tolist()} ({len(le.classes_)}개)")
    log(f"  피험자: {len(np.unique(groups))}명  샘플: {len(y)}")

    sgkf = StratifiedGroupKFold(
        n_splits=config.CFG.kfold, shuffle=True, random_state=config.CFG.seed)

    all_preds: dict[str, list[int]] = {}
    all_labels: list[int] = []
    all_hist:   dict[str, list[dict]] = {}
    fold_meta:  list[dict] = []
    t_total = time.time()

    for fi, (tr_idx, te_idx) in enumerate(
        sgkf.split(np.zeros(len(y)), y, groups), 1
    ):
        t_fold = time.time()
        tr_s = sorted(set(groups[tr_idx].tolist()))
        te_s = sorted(set(groups[te_idx].tolist()))

        # 피험자 누수 검사
        overlap = set(tr_s) & set(te_s)
        if overlap:
            raise ValueError(f"Fold {fi}: 피험자 누수 발견! {overlap}")

        tr_dist = dict(zip(*np.unique(y[tr_idx], return_counts=True)))
        te_dist = dict(zip(*np.unique(y[te_idx], return_counts=True)))

        log(f"\n{'='*55}")
        log(f"  Fold {fi}/{config.CFG.kfold}"
            f"  tr={len(tr_idx)}({len(tr_s)}명)  te={len(te_idx)}({len(te_s)}명)")
        log(f"  Test 피험자: {te_s}")
        log(f"  Train 분포: {tr_dist}  /  Test 분포: {te_dist}")
        log(f"{'='*55}")

        sc, pca = fit_pca_on_train(h5data, tr_idx)
        bsc     = fit_bsc_on_train(h5data, tr_idx)

        fold_errors: list[str] = []
        fold_models_meta: dict[str, dict] = {}

        # ── M1 ──
        if not args.skip_m1:
            res, labels, hist = run_M1(h5data, y, tr_idx, te_idx, sc, pca, fi)
            for t, p in res.items():
                all_preds.setdefault(t, []).extend(p.tolist())
            for t, h in hist.items():
                all_hist.setdefault(t, []).append(h)
                fold_errors.extend(h.get("meta", {}).get("errors", []))
                fold_models_meta[t] = h.get("meta", {})
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(y[te_idx].tolist())
            labels = y[te_idx]

        # ── Branch 모델들 ──
        for mname, mfn in models_to_run:
            r2, _, h2 = run_branch(
                h5data, y, tr_idx, te_idx,
                branch_idx, branch_ch, bsc, mfn, mname, fi,
            )
            for t, p in r2.items():
                all_preds.setdefault(t, []).extend(p.tolist())
            for t, h in h2.items():
                all_hist.setdefault(t, []).append(h)
                fold_errors.extend(h.get("meta", {}).get("errors", []))
                fold_models_meta[t] = h.get("meta", {})

        fold_time = round((time.time() - t_fold) / 60, 1)
        fold_meta.append({
            "fold": fi,
            "train_subjects": tr_s,
            "test_subjects":  te_s,
            "train_samples":  int(len(tr_idx)),
            "test_samples":   int(len(te_idx)),
            "train_class_dist": {int(k): int(v) for k, v in tr_dist.items()},
            "test_class_dist":  {int(k): int(v) for k, v in te_dist.items()},
            "fold_time_min":  fold_time,
            "errors":         fold_errors,
            "models":         fold_models_meta,
        })
        if fold_errors:
            log(f"  ⚠ Fold {fi} 오류 {len(fold_errors)}건: {fold_errors}")

        del sc, pca, bsc
        gc.collect()
        clear_fold_cache(fi)

    # ── 전체 결과 저장 ──
    labels_arr = np.array(all_labels)
    results: dict[str, tuple[float, float]] = {}
    for tag, preds in all_preds.items():
        pred_arr = np.array(preds)
        results[tag] = save_report(
            pred_arr, labels_arr, le, f"KFOLD_{tag}", out)
        save_cm(pred_arr, labels_arr, le, f"KFOLD_{tag}", out)

    # ── 앙상블 다수결 투표 ──
    branch_tags = [t for t in all_preds if t != "M1_CNN"]
    if len(branch_tags) >= 2:
        from scipy.stats import mode as _mode
        pred_stack    = np.stack([np.array(all_preds[t]) for t in branch_tags])
        ensemble_pred = _mode(pred_stack, axis=0, keepdims=False).mode
        etag          = "Ensemble_Vote"
        results[etag] = save_report(ensemble_pred, labels_arr, le, f"KFOLD_{etag}", out)
        save_cm(ensemble_pred, labels_arr, le, f"KFOLD_{etag}", out)
        log(f"  ★ 앙상블 ({', '.join(branch_tags)})"
            f"  Acc={results[etag][0]:.4f}  F1={results[etag][1]:.4f}")

    if all_hist:
        save_history(all_hist, out)

    total_min    = (time.time() - t_total) / 60
    total_errors = sum(len(fm["errors"]) for fm in fold_meta)
    total_ooms   = sum(
        m.get("oom_events", 0)
        for fm in fold_meta for m in fm["models"].values()
    )

    print(f"\n{'='*60}")
    print(f"  ★ {config.CFG.kfold}-Fold  {config.DEVICE_NAME}")
    print(f"  총 소요: {total_min:.1f}분")
    if total_errors > 0:
        print(f"  ⚠ 총 오류: {total_errors}건 (OOM: {total_ooms}건)")
    print(f"{'='*60}")
    for tag, (acc, f1) in results.items():
        print(f"  {tag:<25} Acc={acc:.4f}  F1={f1:.4f}")

    summary = {
        "experiment":      "kfold",
        "version":         "v9.0",
        "compare_order":   [m for m, _ in models_to_run],
        "skip_m1":         args.skip_m1,
        "config":          config.snapshot(),
        "total_minutes":   round(total_min, 1),
        "total_errors":    total_errors,
        "total_oom_events": total_ooms,
        "results": {
            t: {"acc": round(a, 4), "f1": round(f, 4)}
            for t, (a, f) in results.items()
        },
        "fold_meta": fold_meta,
    }
    (out / "summary_kfold.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  ✅ {out / 'summary_kfold.json'}")
    h5data.close()


if __name__ == "__main__":
    main()