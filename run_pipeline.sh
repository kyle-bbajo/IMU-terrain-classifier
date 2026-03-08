#!/bin/bash
# ═══════════════════════════════════════════════════════
# run_pipeline.sh — 증분 세그멘테이션 → Hierarchical 학습
#
# 사용법:
#   bash run_pipeline.sh 50    # 50명 (기존 40명에서 증분)
#   bash run_pipeline.sh 60    # 60명 (기존 50명에서 증분)
#   bash run_pipeline.sh 100   # 100명
#
# tmux 세션 'pipeline' 에서 실행 → SSH 끊겨도 유지
#   재접속:  tmux attach -t pipeline
#   로그:    tail -f ~/project/repo/logs/pipeline_N{N}.log
#   중지:    tmux kill-session -t pipeline
# ═══════════════════════════════════════════════════════

N=${1:-50}

PROJECT=/home/ubuntu/project/repo
SRC=/home/ubuntu/project/repo/src
LOG_DIR=${PROJECT}/logs
LOG=${LOG_DIR}/pipeline_N${N}.log
TRAIN=${PROJECT}/experiments/hierarchical/train_hierarchical.py

# ── tmux 백그라운드 진입 ──────────────────────────────
if [ "${2:-fg}" != "_inside_tmux" ]; then
    mkdir -p "${LOG_DIR}"
    tmux kill-session -t pipeline 2>/dev/null || true
    tmux new-session -d -s pipeline \
        "bash $(realpath "$0") ${N} _inside_tmux 2>&1 | tee ${LOG}; echo '=== 완료 ==='; sleep 86400"

    echo "═══════════════════════════════════════════════"
    echo "  ★ 백그라운드 시작  N=${N}  (tmux: pipeline)"
    echo "  재접속:  tmux attach -t pipeline"
    echo "  로그:    tail -f ${LOG}"
    echo "  중지:    tmux kill-session -t pipeline"
    echo "═══════════════════════════════════════════════"
    exit 0
fi

# ── 이하 tmux 내부에서 실행 ──────────────────────────
set -euo pipefail
source /home/ubuntu/project/.venv/bin/activate

echo "═══════════════════════════════════════════════"
echo "  run_pipeline.sh  N=${N}  $(date)"
echo "═══════════════════════════════════════════════"

# ─────────────────────────────────────────────
# Step 1: 증분 세그멘테이션
# ─────────────────────────────────────────────
echo ""
echo "[1] 증분 스텝 분할  N=${N}"
echo "    (기존 완료된 피험자는 자동 스킵)"

cd "${SRC}"
python3 step_segmentation.py --n_subjects "${N}"

# HDF5 상태 출력
python3 - << EOF
import h5py
path = "/home/ubuntu/project/data/processed/batches/dataset.h5"
with h5py.File(path, "r") as f:
    skeys = list(f["subjects"].keys())
    total = sum(f[f"subjects/{k}/X"].shape[0] for k in skeys)
    print(f"  HDF5: {len(skeys)}명  {total}샘플")
EOF

echo "  ✅ 세그멘테이션 완료  $(date)"

# ─────────────────────────────────────────────
# Step 2: Hierarchical 학습
# ─────────────────────────────────────────────
echo ""
echo "[2] Hierarchical 학습 시작  N=${N}  $(date)"

TRAIN_LOG=${LOG_DIR}/hierarchical_N${N}.log
cd "${PROJECT}/experiments/hierarchical"
python3 train_hierarchical.py > "${TRAIN_LOG}" 2>&1

# 결과 요약 출력
echo ""
grep "★ 최종 6cls\|★ Hierarchical" "${TRAIN_LOG}" | tail -10

echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ 완료  N=${N}  $(date)"
echo "  학습 로그: ${TRAIN_LOG}"
echo "═══════════════════════════════════════════════"