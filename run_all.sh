#!/bin/bash
# run_all.sh — attribute + kfold 동시 → hierarchical 순차 실행
# 사용: bash run_all.sh
# GPU: L4 22GB 1개 기준

REPO="$HOME/project/repo"
LOG_DIR="$REPO/experiments/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

echo "========================================"
echo " IMU Terrain 학습 시작"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# ── Phase 1: attribute + kfold 동시 실행 ────────────────────
echo ""
echo "[Phase 1] attribute + kfold 동시 시작..."

nohup python train_attribute.py \
    > "$LOG_DIR/att_v4.log" 2>&1 &
ATT_PID=$!
echo "  attribute  PID=$ATT_PID  → $LOG_DIR/att_v4.log"

nohup python train_kfold.py \
    > "$LOG_DIR/kfold_ctx324.log" 2>&1 &
KF_PID=$!
echo "  kfold      PID=$KF_PID  → $LOG_DIR/kfold_ctx324.log"

# 둘 다 완료 대기
wait $ATT_PID; ATT_EXIT=$?
echo "  [Phase 1] attribute 완료 (exit=$ATT_EXIT)  $(date '+%H:%M:%S')"

wait $KF_PID;  KF_EXIT=$?
echo "  [Phase 1] kfold 완료 (exit=$KF_EXIT)  $(date '+%H:%M:%S')"

echo "[Phase 1] 완료 ✓"

# ── Phase 2: hierarchical 단독 실행 ─────────────────────────
echo ""
echo "[Phase 2] hierarchical v14.5 단독 시작..."

nohup python train_hierarchical.py \
    > "$LOG_DIR/hier_v145.log" 2>&1 &
HIER_PID=$!
echo "  hierarchical  PID=$HIER_PID  → $LOG_DIR/hier_v145.log"

wait $HIER_PID; HIER_EXIT=$?
echo "  [Phase 2] hierarchical 완료 (exit=$HIER_EXIT)  $(date '+%H:%M:%S')"

# ── 최종 요약 ────────────────────────────────────────────────
echo ""
echo "========================================"
echo " 전체 완료  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  attribute    exit=$ATT_EXIT"
echo "  kfold        exit=$KF_EXIT"
echo "  hierarchical exit=$HIER_EXIT"
echo "========================================"

echo ""
echo "=== attribute 결과 ==="
grep -E "Mean Acc|Mean F1|★" "$LOG_DIR/att_v4.log"       | tail -5 || true

echo ""
echo "=== kfold 결과 ==="
grep -E "Mean Acc|Mean F1|★" "$LOG_DIR/kfold_ctx324.log"  | tail -5 || true

echo ""
echo "=== hierarchical 결과 ==="
grep -E "Acc=|MacroF1|★" "$LOG_DIR/hier_v145.log"         | tail -10 || true