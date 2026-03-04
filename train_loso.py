"""
train_loso.py — M1-M6 LOSO 교차검증 (v8 Final)
═══════════════════════════════════════════════════════
v7->v8 변경사항
  ★ 체크포인트 무결성 검증 (손상 시 해당 피험자만 재실행)
  ★ 모델별 실패 격리 (한 모델 OOM → 나머지 계속)
  ★ ETA 표시 + fold-level 소요 시간 통계
  ★ 피험자별 히트맵 + CSV + JSON 결과
  ★ seed_everything 재현성
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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS: list[tuple[str, Any]] = [
    ("M2", M2_BranchCNN),
    ("M3", M3_BranchSE),
    ("M4", M4_BranchCBAM),
    ("M5", M5_BranchCBAMCross),
    ("M6", M6_BranchCBAMCrossAug),
]


def _load_checkpoint(
    ckpt_path: Path,
) -> tuple[list[int], dict[str, list[int]], list[int]]:
    """체크포인트를 안전하게 로드한다. 실패 시 빈 상태 반환."""
    if not ckpt_path.exists():
        return [], {}, []
    try:
        ckpt = json.loads(ckpt_path.read_text())
        done  = ckpt.get("done_subjects", [])
        preds = {k: list(v) for k, v in ckpt.get("preds", {}).items()}
        labels = list(ckpt.get("labels", []))

        # 무결성 검증: 모든 preds의 길이가 labels와 같아야 함
        for k, v in preds.items():
            if len(v) != len(labels):
                log(f"  [WARN] 체크포인트 길이 불일치: {k}({len(v)}) != labels({len(labels)})")
                return [], {}, []

        log(f"  ★ 체크포인트 복원: {len(done)}명 완료 ({ckpt_path})")
        return done, preds, labels
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log(f"  [WARN] 체크포인트 손상 — 처음부터 시작 ({type(e).__name__})")
        return [], {}, []


def _save_checkpoint(
    ckpt_path: Path,
    done_subjects: list[int],
    all_preds: dict[str, list[int]],
    all_labels: list[int],
) -> None:
    """체크포인트를 원자적으로 저장한다."""
    ckpt_data = {
        "done_subjects": done_subjects,
        "preds": {k: [int(x) for x in v] for k, v in all_preds.items()},
        "labels": [int(x) for x in all_labels],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v8",
    }
    tmp = ckpt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt_data))
    tmp.rename(ckpt_path)


def _save_per_subject(
    per_subj: dict[int, dict[str, float]],
    results: dict[str, tuple[float, float]],
    out_dir: Path,
) -> None:
    """피험자별 정확도를 CSV + 히트맵으로 저장한다."""
    subjs = sorted(per_subj.keys())
    tags  = list(results.keys())
    if not subjs or not tags:
        return

    # CSV
    lines = ["Subject," + ",".join(tags)]
    for s in subjs:
        vals = [str(per_subj[s].get(t, "")) for t in tags]
        lines.append(f"S{s:03d},{','.join(vals)}")
    for name, fn in [("Mean", np.mean), ("Std", np.std)]:
        vals = []
        for t in tags:
            scores = [per_subj[s][t] for s in subjs if t in per_subj[s]]
            vals.append(f"{fn(scores):.4f}" if scores else "")
        lines.append(f"{name},{','.join(vals)}")
    (out_dir / "per_subject_accuracy.csv").write_text("\n".join(lines))

    # 히트맵
    data = np.array([[per_subj[s].get(t, 0) for t in tags] for s in subjs])
    fig_h = max(8, len(subjs) * 0.3)
    fig_w = max(10, len(tags) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
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


def main() -> None:
    """LOSO 교차검증 전체 파이프라인을 실행한다."""
    seed_everything()
    config.print_config()
    log(f"  ★ LOSO ({config.N_SUBJECTS}-Fold)\n")

    with H5Data(config.H5_PATH) as h5data:
        le = LabelEncoder()
        y: np.ndarray = le.fit_transform(h5data.y_raw).astype(np.int64)
        branch_idx, branch_ch = build_branch_idx(h5data.channels)
        groups: np.ndarray = h5data.subj_id
        unique_subjs = sorted(np.unique(groups).tolist())

        log(f"  클래스: {le.classes_.tolist()} ({len(le.classes_)}개)")
        log(f"  피험자: {len(unique_subjs)}명  샘플: {len(y)}")

        out = config.RESULT_LOSO
        logo = LeaveOneGroupOut()
        n_folds: int = logo.get_n_splits(groups=groups)

        # 체크포인트
        ckpt_path = out / "checkpoint.json"
        done_subjects, all_preds, all_labels = _load_checkpoint(ckpt_path)

        all_hist: dict[str, list[dict]] = {}
        per_subj: dict[int, dict[str, float]] = {}
        failures: list[dict] = []
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
                f"  ({len(te_idx)} samples)  Train={len(tr_idx)}")
            log(f"{'='*55}")

            try:
                sc, pca = fit_pca_on_train(h5data, tr_idx)
                bsc     = fit_bsc_on_train(h5data, tr_idx)
            except Exception as e:
                log(f"  [ERROR] S{te_subj:03d} Scaler/PCA 실패: {e}")
                failures.append({
                    "subject": te_subj, "stage": "pca", "error": str(e)
                })
                continue

            fold_preds: dict[str, np.ndarray] = {}
            fold_labels: np.ndarray = np.array([], dtype=np.int64)

            # M1
            try:
                res, labels, hist = run_M1(
                    h5data, y, tr_idx, te_idx, sc, pca, f"S{te_subj:03d}")
                for t, p in res.items():
                    fold_preds[t] = p
                    all_preds.setdefault(t, []).extend(p.tolist())
                for t, h in hist.items():
                    all_hist.setdefault(t, []).append(h)
                all_labels.extend(labels.tolist())
                fold_labels = labels
            except Exception as e:
                log(f"  [ERROR] S{te_subj:03d} M1 실패: {e}")
                failures.append({
                    "subject": te_subj, "model": "M1", "error": str(e)
                })

            # M2-M6
            for mname, mfn in MODELS:
                try:
                    r2, lb2, h2 = run_branch(
                        h5data, y, tr_idx, te_idx,
                        branch_idx, branch_ch, bsc, mfn, mname,
                        f"S{te_subj:03d}",
                    )
                    for t, p in r2.items():
                        fold_preds[t] = p
                        all_preds.setdefault(t, []).extend(p.tolist())
                    for t, h in h2.items():
                        all_hist.setdefault(t, []).append(h)
                    if len(fold_labels) == 0:
                        all_labels.extend(lb2.tolist())
                        fold_labels = lb2
                except Exception as e:
                    log(f"  [ERROR] S{te_subj:03d} {mname} 실패: {e}")
                    failures.append({
                        "subject": te_subj, "model": mname, "error": str(e)
                    })
                    if config.USE_GPU:
                        import torch
                        torch.cuda.empty_cache()

            # 피험자별 정확도
            if len(fold_labels) > 0 and fold_preds:
                per_subj[te_subj] = {
                    t: round(float(accuracy_score(fold_labels, p)), 4)
                    for t, p in fold_preds.items()
                    if len(p) == len(fold_labels)
                }

            done_subjects.append(te_subj)
            _save_checkpoint(ckpt_path, done_subjects, all_preds, all_labels)

            del sc, pca, bsc
            gc.collect()

            elapsed = (time.time() - t_fold) / 60
            fold_times.append(elapsed)
            remaining = len(unique_subjs) - len(done_subjects)
            eta_min = np.mean(fold_times) * remaining if fold_times else 0
            log(f"  S{te_subj:03d} 완료 ({elapsed:.1f}분)"
                f"  남은: {remaining}명  ETA: {eta_min:.0f}분")

        # ── 최종 결과 ──
        labels_arr = np.array(all_labels)
        results: dict[str, tuple[float, float]] = {}
        for tag, ps in all_preds.items():
            pa = np.array(ps)
            min_len = min(len(pa), len(labels_arr))
            if min_len == 0:
                continue
            results[tag] = save_report(
                pa[:min_len], labels_arr[:min_len], le,
                f"LOSO_{tag}", out,
            )
            save_cm(
                pa[:min_len], labels_arr[:min_len], le,
                f"LOSO_{tag}", out,
            )
        if all_hist:
            save_history(all_hist, out)
        if per_subj:
            _save_per_subject(per_subj, results, out)

    # 최종 출력
    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  ★ LOSO  N={config.N_SUBJECTS}  {config.DEVICE_NAME}")
    print(f"  총 소요: {total_min:.1f}분")
    if fold_times:
        print(f"  평균 fold: {np.mean(fold_times):.1f}분"
              f"  std: {np.std(fold_times):.1f}분")
    if failures:
        print(f"  실패: {len(failures)}건")
    print(f"{'='*60}")
    for tag, (acc, f1) in results.items():
        print(f"  {tag:<20} Acc={acc:.4f}  F1={f1:.4f}")

    summary = {
        "experiment": "loso",
        "version": "v8",
        "device": config.DEVICE_NAME,
        "n_subjects": config.N_SUBJECTS,
        "n_classes": config.NUM_CLASSES,
        "n_folds": n_folds,
        "seed": config.SEED,
        "split": "LeaveOneGroupOut",
        "strategy": "preload" if config.USE_PRELOAD else "otf",
        "amp": config.USE_AMP,
        "batch": config.BATCH,
        "grad_accum": config.GRAD_ACCUM_STEPS,
        "total_minutes": round(total_min, 1),
        "results": {
            t: {"acc": round(a, 4), "f1": round(f, 4)}
            for t, (a, f) in results.items()
        },
        "per_subject": {str(k): v for k, v in per_subj.items()},
        "failures": failures,
    }
    (out / "summary_loso.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    if ckpt_path.exists() and not failures:
        ckpt_path.unlink()
        log("  체크포인트 삭제 (모두 완료)")

    log(f"  ✅ 완료")


if __name__ == "__main__":
    main()