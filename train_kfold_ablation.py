# -*- coding: utf-8 -*-
"""
train_kfold_ablation.py — 3가지 조건 병렬 K-Fold 학습
═══════════════════════════════════════════════════════════════
조건:
  A — 전체     : 6클래스 (C1~C6)
  B — 흙 제외  : 5클래스 (C4 제거)
  C — 잔디 제외: 5클래스 (C5 제거)  ← 기존 train_kfold 기본값

기존 train_kfold와의 차이:
  - 3가지 조건을 multiprocessing으로 동시 실행
  - 각 조건별 독립 출력 디렉토리
  - 특징 추출은 1회만 (공유)
  - features_v5.py (N_FEATURES=378) 기준

실행:
  python train_kfold_ablation.py
  python train_kfold_ablation.py --conditions A,B,C --no_parallel
  python train_kfold_ablation.py --models M7_Attr --epochs 100
"""
from __future__ import annotations

import sys, gc, argparse, os, copy, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, recall_score,
)

from config import CFG, apply_overrides, print_config
import config as _cfg
from datasets import make_hierarchical_loaders
from train_common import (
    H5Data, train_model, save_report, save_cm,
    save_summary_table, seed_everything, log, ensure_dir,
    save_json, Timer, threshold_search,
)
from channel_groups import build_branch_idx, get_foot_accel_idx
from features import batch_extract, N_FEATURES
from models import MODEL_REGISTRY


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

ALL_CLASS_NAMES = [
    "C1-Slippery", "C2-Uphill", "C3-Downhill",
    "C4-Dirt",     "C5-Grass",  "C6-Flat"
]

# 조건 정의: (이름, 제거할 원본 라벨, out_dir_name)
CONDITIONS = {
    "A": ("A_전체",     [],  "kfold_A_all"),
    "B": ("B_흙제외",   [3], "kfold_B_no_dirt"),
    "C": ("C_잔디제외", [4], "kfold_C_no_grass"),
}


# ══════════════════════════════════════════════════════════════
# 1. 데이터 필터링 & 재매핑
# ══════════════════════════════════════════════════════════════

def filter_condition(X, y, groups, feat, exclude_labels):
    """exclude_labels 제거 후 연속 정수 재매핑."""
    if not exclude_labels:
        names = ALL_CLASS_NAMES
        remap = {i: i for i in range(len(names))}
        return X, y, groups, feat, names, remap

    mask   = ~np.isin(y, exclude_labels)
    X_f    = X[mask]
    y_raw  = y[mask]
    grp_f  = groups[mask]
    feat_f = feat[mask]

    orig_labels = sorted(np.unique(y_raw).tolist())
    remap  = {orig: new for new, orig in enumerate(orig_labels)}
    y_f    = np.array([remap[v] for v in y_raw], dtype=np.int64)
    names  = [ALL_CLASS_NAMES[i] for i in orig_labels]
    return X_f, y_f, grp_f, feat_f, names, remap


# ══════════════════════════════════════════════════════════════
# 2. softmax 확률 추출
# ══════════════════════════════════════════════════════════════

@torch.inference_mode()
def get_probas(model, loader, device):
    model.eval()
    all_p, all_l = [], []
    for batch in loader:
        if len(batch) == 3:
            bi, feat, yb = batch
            feat = feat.to(device)
            bi   = {k: v.to(device) for k, v in bi.items()}
            out  = model(bi, feat)
        else:
            bi, yb = batch
            bi  = {k: v.to(device) for k, v in bi.items()}
            out = model(bi)
        logits = out["final_logits"] if isinstance(out, dict) else out
        all_p.append(torch.softmax(logits.float(), -1).cpu().numpy())
        all_l.append(yb.numpy())
    return np.concatenate(all_p), np.concatenate(all_l)


# ══════════════════════════════════════════════════════════════
# 3. 단일 조건 K-Fold 학습
# ══════════════════════════════════════════════════════════════

def run_condition(task: dict) -> dict:
    """
    하나의 조건(A/B/C)에 대해 K-Fold 전체 학습 수행.
    multiprocessing worker 또는 직접 호출 모두 가능.
    """
    import os
    # GPU 할당
    n_gpu  = task["n_gpu"]
    gpu_id = task["gpu_id"]
    if n_gpu > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # config 재로드 (spawn 프로세스에서 독립)
    import config as cfg2
    from config import CFG as CFG2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cond_name    = task["cond_name"]
    exclude      = task["exclude"]
    class_names_all = task["class_names_all"]
    X_all        = task["X_all"]
    y_all_raw    = task["y_all_raw"]
    groups       = task["groups"]
    feat_all     = task["feat_all"]
    branch_idx   = task["branch_idx"]
    branch_ch    = task["branch_ch"]
    out_root     = Path(task["out_root"])
    model_names  = task["model_names"]
    kfold        = task["kfold"]
    seed         = task["seed"]

    seed_everything(seed + task["cond_idx"])
    log(f"\n{'='*64}")
    log(f"  [조건 {cond_name}]  GPU={gpu_id if n_gpu>1 else 'shared'}  "
        f"제거: {[ALL_CLASS_NAMES[i] for i in exclude] or '없음'}")

    # 데이터 필터링
    X_f, y_f, grp_f, feat_f, class_names, remap = filter_condition(
        X_all, y_all_raw, groups, feat_all, exclude
    )
    n_classes = len(class_names)
    log(f"  샘플: {len(y_f)}  클래스({n_classes}): {class_names}")

    # C6(평지) 재매핑된 인덱스 (subject normalization 기준)
    flat_orig = 5  # C6 원본 라벨
    flat_label = remap.get(flat_orig, n_classes - 1)

    # 출력 디렉토리
    out_dir = ensure_dir(out_root / task["out_dir_name"])
    proba_dir = ensure_dir(out_dir / "probas")

    # 모델 팩토리
    model_fns = {n: MODEL_REGISTRY[n] for n in model_names if n in MODEL_REGISTRY}
    if not model_fns:
        log(f"  ERROR: 모델 없음: {model_names}")
        return {"condition": cond_name, "error": "no model"}

    # K-Fold
    sgkf = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=seed)
    fold_store = {n: {"preds": [], "probas": [], "labels": []} for n in model_fns}

    with Timer() as t:
        for fi, (tr_idx, te_idx) in enumerate(
            sgkf.split(np.zeros(len(y_f)), y_f, grp_f), 1
        ):
            log(f"  [{cond_name}] Fold {fi}/{kfold}  "
                f"tr={len(tr_idx)}  te={len(te_idx)}")

            # Subject-wise 정규화 (C6 평지 기준)
            from sklearn.preprocessing import StandardScaler
            train_mask = np.zeros(len(y_f), dtype=bool)
            train_mask[tr_idx] = True
            feat_tr = _subject_normalize(
                feat_f, grp_f, y_f, flat_label, train_mask
            )[tr_idx]
            feat_te = _subject_normalize(
                feat_f, grp_f, y_f, flat_label, train_mask
            )[te_idx]

            for mname, mfn in model_fns.items():
                log(f"  [{cond_name}][F{fi}] {mname} 학습 시작")

                # 클래스 수에 맞게 모델 생성
                _orig_nc = getattr(CFG2, "num_classes", 6)
                CFG2.num_classes = n_classes
                model = mfn(branch_ch).to(device)
                CFG2.num_classes = _orig_nc

                # ── 조건 B/C: 속성 헤드 제거 → FocalLoss 사용 ──────────
                # M7_Attr의 slip_head/comp_head 존재 시
                # AttributeMultiTaskLoss → comp_logit 요구 → KeyError 발생
                # 클래스 수가 달라진 조건에서는 헤드 제거 후 일반 FocalLoss 사용
                if len(exclude) > 0:
                    removed = []
                    for head_name in ["slip_head", "comp_head", "surf_head",
                                      "reg_head", "attr_head"]:
                        if hasattr(model, head_name):
                            delattr(model, head_name)
                            removed.append(head_name)
                    if removed:
                        log(f"  [{cond_name}] 속성 헤드 제거: {removed} → FocalLoss 사용")
                # ────────────────────────────────────────────────────────

                # 조건 B/C: 속성 헤드 제거 → FocalLoss 강제 전환
                if len(exclude) > 0:
                    for h in ["slip_head","comp_head","surf_head","reg_head","attr_head"]:
                        if hasattr(model, h):
                            delattr(model, h)
                            model.IS_HYBRID = False

                tr_dl, te_dl = make_hierarchical_loaders(
                    X_f[tr_idx], feat_tr, y_f[tr_idx],
                    X_f[te_idx], feat_te, y_f[te_idx],
                    branch_idx,
                    batch    = CFG2.batch,
                    balanced = CFG2.use_balanced_sampler,
                )

                preds, labels, _ = train_model(
                    model, tr_dl, te_dl,
                    branch=True,
                    tag=f"[{cond_name}][F{fi}][{mname}]",
                    use_mixup=True,
                )

                probas, _ = get_probas(model, te_dl, device)

                fold_store[mname]["preds"].extend(preds.tolist())
                fold_store[mname]["probas"].append(probas)
                fold_store[mname]["labels"].extend(labels.tolist())

                fa = accuracy_score(labels, preds)
                ff = f1_score(labels, preds, average="macro", zero_division=0)
                log(f"  [{cond_name}][F{fi}][{mname}] Acc={fa:.4f}  F1={ff:.4f}")

                # fold proba 저장
                np.save(proba_dir / f"{mname}_proba_fold{fi}.npy",  probas)
                np.save(proba_dir / f"{mname}_labels_fold{fi}.npy", labels)

                del model, tr_dl, te_dl
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ── 최종 평가 ────────────────────────────────────────────
    results = []
    for mname, data in fold_store.items():
        pred_arr  = np.array(data["preds"])
        label_arr = np.array(data["labels"])
        proba_arr = np.concatenate(data["probas"])

        acc = accuracy_score(label_arr, pred_arr)
        f1  = f1_score(label_arr, pred_arr, average="macro", zero_division=0)

        # Threshold search
        best_mults, best_f1_t = threshold_search(
            proba_arr, label_arr, n_classes=n_classes
        )
        thresh_preds = (proba_arr * best_mults).argmax(1)
        thresh_acc   = accuracy_score(label_arr, thresh_preds)

        recalls = recall_score(label_arr, thresh_preds,
                               average=None, zero_division=0)
        cm = confusion_matrix(label_arr, thresh_preds)

        log(f"\n  ★ [{cond_name}][{mname}]  "
            f"Acc={thresh_acc:.4f}  F1={best_f1_t:.4f}  ({t})")
        log(f"\n{classification_report(label_arr, thresh_preds, target_names=class_names, digits=4, zero_division=0)}")

        # 클래스별 recall 출력
        for cn, r in zip(class_names, recalls):
            flag = "✅" if r >= 0.85 else ("⚠" if r >= 0.70 else "❌")
            log(f"    {flag} {cn:<16} recall={r*100:.1f}%")

        # 저장
        np.save(out_dir / f"{mname}_proba.npy",  proba_arr)
        np.save(out_dir / f"{mname}_labels.npy", label_arr)
        np.save(out_dir / f"{mname}_mults.npy",  best_mults)

        result = {
            "condition":   cond_name,
            "model":       mname,
            "n_classes":   n_classes,
            "n_samples":   int(len(y_f)),
            "class_names": class_names,
            "acc":         round(thresh_acc, 4),
            "macro_f1":    round(best_f1_t,  4),
            "per_class_recall": {
                cn: round(float(r), 4)
                for cn, r in zip(class_names, recalls)
            },
            "confusion_matrix": cm.tolist(),
            "elapsed": str(t),
        }
        save_json(result, out_dir / f"result_{cond_name}_{mname}.json")
        results.append(result)

    return results[0] if len(results) == 1 else results


def _subject_normalize(feat, groups, y, flat_label, train_mask):
    """C6(평지) 샘플 기준 피험자별 정규화."""
    feat_n = feat.copy().astype(np.float32)
    subjs  = np.unique(groups)

    tr_flat = train_mask & (y == flat_label)
    g_mu  = feat[tr_flat].mean(0) if tr_flat.sum() > 0 else np.zeros(feat.shape[1])
    g_std = feat[tr_flat].std(0)  + 1e-8

    for s in subjs:
        sm = train_mask & (groups == s) & (y == flat_label)
        mu  = feat[sm].mean(0) if sm.sum() >= 5 else g_mu
        std = feat[sm].std(0) + 1e-8 if sm.sum() >= 5 else g_std
        feat_n[groups == s] = (feat[groups == s] - mu) / std
    return feat_n


# ══════════════════════════════════════════════════════════════
# 4. 결과 비교 출력
# ══════════════════════════════════════════════════════════════

def print_comparison(results: list[dict]):
    log(f"\n{'='*72}")
    log("  📊 3가지 조건 비교 — CNN + 특징 융합 모델")
    log(f"{'='*72}")
    log(f"  {'조건':<16}  {'Acc':>7}  {'F1':>7}  "
        f"{'C4-흙':>8}  {'C5-잔디':>9}  {'C6-평지':>9}  {'C1-미끄':>9}")
    log(f"  {'-'*68}")
    for r in results:
        pr = r.get("per_class_recall", {})
        def g(k): return pr.get(k, pr.get(k.split('-')[0], 0))
        c4 = f"{g('C4-Dirt')*100:6.1f}%"
        c5 = f"{g('C5-Grass')*100:6.1f}%"
        c6 = f"{g('C6-Flat')*100:6.1f}%"
        c1 = f"{g('C1-Slippery')*100:6.1f}%"
        log(f"  {r['condition']:<16}  {r['acc']:>6.4f}  "
            f"{r['macro_f1']:>7.4f}  {c4:>8}  {c5:>9}  {c6:>9}  {c1:>9}")
    log(f"  {'-'*68}")
    log(f"{'='*72}")


# ══════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",        type=str, default="M7_Attr")
    p.add_argument("--epochs",        type=int, default=None)
    p.add_argument("--batch",         type=int, default=None)
    p.add_argument("--seed",          type=int, default=None)
    p.add_argument("--kfold",         type=int, default=5)
    p.add_argument("--no_feat_cache", action="store_true")
    p.add_argument("--conditions",    type=str, default="A,B,C",
                   help="A=전체, B=흙제외, C=잔디제외")
    p.add_argument("--no_parallel",   action="store_true",
                   help="순차 실행 (기본: 병렬)")
    return p.parse_args()


def main():
    args = parse_args()
    apply_overrides(epochs=args.epochs, batch=args.batch, seed=args.seed)
    seed_everything(CFG.seed)
    print_config()

    run_conds = set(args.conditions.upper().split(","))
    use_parallel = not args.no_parallel
    model_names  = [m.strip() for m in args.models.split(",") if m.strip()]
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

    log("=" * 72)
    log("  train_kfold_ablation.py — 3가지 조건 병렬 K-Fold")
    log(f"  N_FEATURES={N_FEATURES}  models={model_names}")
    log(f"  kfold={args.kfold}  조건={sorted(run_conds)}  병렬={use_parallel}")
    log(f"  GPU={n_gpu}장")
    log("=" * 72)

    # ── 데이터 로드 ────────────────────────────────────────────
    h5     = H5Data(CFG.h5_path)
    le     = LabelEncoder()
    y_all  = le.fit_transform(h5.y_raw).astype(np.int64)
    groups = h5.subj_id
    X_all  = h5.X

    log(f"  데이터: N={len(y_all)}  피험자={len(np.unique(groups))}명")
    for i, name in enumerate(ALL_CLASS_NAMES):
        log(f"    {name}: {int((y_all == i).sum())}개")

    # ── 특징 추출 (1회 공통) ───────────────────────────────────
    cache_dir  = ensure_dir(CFG.repo_dir / "cache" / f"feat_v5_seed{CFG.seed}")
    cache_path = cache_dir / "all_feat_v5.npy"

    if cache_path.exists() and not args.no_feat_cache:
        log(f"  ★ 특징 캐시 히트 → {cache_path}")
        with Timer() as t:
            feat_all = np.load(cache_path)
        log(f"  로드 완료  shape={feat_all.shape}  ({t})")
    else:
        log(f"  특징 추출 시작 (N_FEATURES={N_FEATURES})...")
        with Timer() as t:
            foot_idx = get_foot_accel_idx(h5.channels)
            feat_all = batch_extract(
                X_all, foot_idx, CFG.sample_rate,
                h5_path=str(CFG.h5_path)
            )
        np.save(cache_path, feat_all)
        log(f"  추출 완료  shape={feat_all.shape}  ({t})")

    assert feat_all.shape == (len(y_all), N_FEATURES), \
        f"특징 shape 불일치: {feat_all.shape} != ({len(y_all)}, {N_FEATURES})"

    h5.close()

    # 브랜치 인덱스
    h5tmp = H5Data(CFG.h5_path)
    branch_idx, branch_ch = build_branch_idx(h5tmp.channels)
    h5tmp.close()

    # ── 태스크 구성 ────────────────────────────────────────────
    out_root = ensure_dir(CFG.repo_dir / "out_kfold_ablation")
    task_list = []

    for ci, cond_key in enumerate(sorted(run_conds)):
        if cond_key not in CONDITIONS:
            continue
        cond_name, exclude, out_dir_name = CONDITIONS[cond_key]
        task_list.append({
            "cond_idx":      ci,
            "cond_key":      cond_key,
            "cond_name":     cond_name,
            "exclude":       exclude,
            "out_dir_name":  out_dir_name,
            "class_names_all": ALL_CLASS_NAMES,
            "X_all":         X_all,
            "y_all_raw":     y_all,
            "groups":        groups,
            "feat_all":      feat_all,
            "branch_idx":    branch_idx,
            "branch_ch":     branch_ch,
            "out_root":      str(out_root),
            "model_names":   model_names,
            "kfold":         args.kfold,
            "seed":          CFG.seed,
            "gpu_id":        ci % max(n_gpu, 1),
            "n_gpu":         n_gpu,
        })

    log(f"\n  실행 조건 {len(task_list)}개: {[t['cond_name'] for t in task_list]}")

    # ── 병렬 / 순차 실행 ──────────────────────────────────────
    all_results = []
    with Timer() as total_t:
        if use_parallel and len(task_list) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            log(f"\n  ★ 병렬 실행 시작 ({len(task_list)}개 프로세스 동시 구동)\n")
            with ctx.Pool(processes=len(task_list)) as pool:
                raw = pool.map(run_condition, task_list)
            # 결과 평탄화
            for r in raw:
                if isinstance(r, list):
                    all_results.extend(r)
                elif r:
                    all_results.append(r)
        else:
            log(f"\n  순차 실행 ({len(task_list)}개 조건)")
            for task in task_list:
                r = run_condition(task)
                if isinstance(r, list):
                    all_results.extend(r)
                elif r:
                    all_results.append(r)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ── 최종 비교 출력 ─────────────────────────────────────────
    order = {"A_전체": 0, "B_흙제외": 1, "C_잔디제외": 2}
    all_results.sort(key=lambda r: order.get(r.get("condition", ""), 9))

    if len(all_results) > 1:
        print_comparison(all_results)

    save_json({
        "experiment":  "kfold_ablation",
        "n_features":  N_FEATURES,
        "models":      model_names,
        "kfold":       args.kfold,
        "parallel":    use_parallel,
        "total_time":  str(total_t),
        "conditions":  all_results,
    }, out_root / "summary_kfold_ablation.json")

    log(f"\n★ 완료  총 소요: {total_t}")
    log(f"  결과 저장: {out_root}")


if __name__ == "__main__":
    main()