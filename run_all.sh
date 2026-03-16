#!/bin/bash
# run_all.sh — 전체 학습 파이프라인
#
# 사용법:
#   cd ~/project/repo
#   bash run_all.sh                # 전체 실행 (4모델 + 앙상블)
#   bash run_all.sh --skip_phase1  # 앙상블만 재실행
# ─────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/experiments/logs"
mkdir -p "$LOG_DIR"

cd "$REPO_DIR"

# ═══════════════════════════════════════════════
# Phase 1 — 4개 모델 동시 백그라운드
# ═══════════════════════════════════════════════
if [[ "$1" != "--skip_phase1" ]]; then
    echo ""
    echo "══════════════════════════════════════"
    echo "  Phase 1 — 모델 학습 (동시 실행)"
    echo "══════════════════════════════════════"

    # 캐시 삭제 (features.py N_FEATURES 변경)
    if [ -d "$REPO_DIR/cache" ]; then
        echo "  [cache] 삭제 중..."
        rm -rf "$REPO_DIR/cache"
        echo "  [cache] 삭제 완료"
    fi

    nohup python train_attribute.py \
        > "$LOG_DIR/attribute.log" 2>&1 &
    PID_ATTR=$!
    echo "  ▶ train_attribute    PID=$PID_ATTR"

    nohup python train_hierarchical.py \
        > "$LOG_DIR/hierarchical.log" 2>&1 &
    PID_HIER=$!
    echo "  ▶ train_hierarchical PID=$PID_HIER"

    nohup python train_surface.py \
        > "$LOG_DIR/surface.log" 2>&1 &
    PID_SURF=$!
    echo "  ▶ train_surface      PID=$PID_SURF"

    nohup python train_raw.py \
        > "$LOG_DIR/raw.log" 2>&1 &
    PID_RAW=$!
    echo "  ▶ train_raw          PID=$PID_RAW"

    echo ""
    echo "  로그 확인:"
    echo "    tail -f $LOG_DIR/attribute.log"
    echo "    tail -f $LOG_DIR/hierarchical.log"
    echo "    tail -f $LOG_DIR/surface.log"
    echo "    tail -f $LOG_DIR/raw.log"
    echo ""
    echo "  Phase 1 완료 대기 중..."

    FAILED=0
    for JOB in "$PID_ATTR:attribute" "$PID_HIER:hierarchical" \
               "$PID_SURF:surface"   "$PID_RAW:raw"; do
        PID="${JOB%%:*}"; NAME="${JOB##*:}"
        if wait "$PID"; then
            echo "  ✅ $NAME 완료 (PID=$PID)"
        else
            echo "  ❌ $NAME 실패 (PID=$PID) — 로그: $LOG_DIR/${NAME}.log"
            FAILED=1
        fi
    done

    if [ $FAILED -eq 1 ]; then
        echo "  ⚠ 일부 모델 실패. 가용 proba로 앙상블 진행합니다."
    fi
fi

# ═══════════════════════════════════════════════
# Phase 2 — 앙상블
# ═══════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════"
echo "  Phase 2 — 앙상블"
echo "══════════════════════════════════════"

python train_kfold.py --ensemble_only \
    2>&1 | tee "$LOG_DIR/ensemble.log"

echo ""
echo "══════════════════════════════════════"
echo "  ★ 전체 완료"
echo "══════════════════════════════════════"

SUMMARY="$REPO_DIR/out_N50/final/ensemble/ensemble_summary.json"
if [ -f "$SUMMARY" ]; then
    python3 -c "
import json
d = json.load(open('$SUMMARY'))
print(f'  최종 Acc={d[\"acc\"]}  F1={d[\"f1\"]}')
print(f'  방법: {d[\"method\"]}')
print(f'  모델: {d.get(\"models_used\", [])}')
"
fi