#!/bin/bash
# ═══════════════════════════════════════════════════════
# run.sh v8 Final — 힐스트라이크 파이프라인
#
# 사용법:
#   bash run.sh              # 전체 (seg + kfold + loso), N=40
#   bash run.sh 40 kfold     # K-Fold만
#   bash run.sh 40 loso      # LOSO만
#   bash run.sh 40 train     # K-Fold + LOSO
#   bash run.sh 40 seg       # Segmentation만
#
# v7->v8: 에러 처리 강화, GPU 상태 확인, 로그 타임스탬프
# ═══════════════════════════════════════════════════════
set -euo pipefail

N=${1:-40}
MODE=${2:-all}
PROJECT=/home/ubuntu/project/repo
S3=s3://imu-khu-jooho-s1-6
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

source /home/ubuntu/project/venv/bin/activate
cd "${PROJECT}"

echo "====================================================="
echo "  v8 파이프라인  N=${N}  MODE=${MODE}"
echo "  $(date)"
echo "====================================================="

# 환경 확인
python3 -c "
import torch, os, sys
print(f'  Python {sys.version.split()[0]}  PyTorch {torch.__version__}')
if torch.cuda.is_available():
    n = torch.cuda.get_device_name(0)
    m = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU: {n}  ({m:.0f}GB)')
else:
    print(f'  CPU: {os.cpu_count()} vCPU')
import config
print(f'  Strategy: {\"Preload\" if config.USE_PRELOAD else \"OTF\"}')
print(f'  RAM: {config.RAM_GIB}GiB  Batch: {config.BATCH}')
print(f'  GradAccum: {config.GRAD_ACCUM_STEPS}  EffBatch: {config.BATCH * config.GRAD_ACCUM_STEPS}')
" || { echo "[FATAL] Python 환경 오류"; exit 1; }

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used \
               --format=csv,noheader 2>/dev/null || true
fi

# N 값 변경
sed -i "s/^N_SUBJECTS.*=.*/N_SUBJECTS: int        = ${N}/" "${PROJECT}/config.py"

# S3 동기화
echo ""
echo "[S3] CSV 동기화..."
aws s3 sync "${S3}/raw_csv" /home/ubuntu/project/data/raw_csv/ --quiet \
    || { echo "[WARN] S3 동기화 실패 — 로컬 데이터 사용"; }
echo "  ✅ 동기화 완료"

# 1) Segmentation
if [ "$MODE" = "all" ] || [ "$MODE" = "seg" ]; then
    echo ""
    echo "====================================================="
    echo "  [1] 힐스트라이크 스텝 분할"
    echo "====================================================="
    python3 step_segmentation.py \
        || { echo "[ERROR] Segmentation 실패"; exit 1; }
fi

# HDF5 존재 확인
if [ "$MODE" != "seg" ]; then
    if [ ! -f "${PROJECT}/batches/dataset.h5" ]; then
        echo "[FATAL] dataset.h5 없음 — 먼저 seg를 실행하세요"
        exit 1
    fi
fi

# 2) K-Fold
if [ "$MODE" = "all" ] || [ "$MODE" = "kfold" ] || [ "$MODE" = "train" ]; then
    echo ""
    echo "====================================================="
    echo "  [2] K-Fold 교차검증 (M1~M6)"
    echo "====================================================="
    LOGFILE="out_N${N}/kfold/train_kfold_${TIMESTAMP}.log"
    mkdir -p "out_N${N}/kfold"
    python3 train_kfold.py 2>&1 | tee "${LOGFILE}" \
        || echo "[WARN] K-Fold에서 에러 발생 (로그 확인: ${LOGFILE})"
fi

# 3) LOSO
if [ "$MODE" = "all" ] || [ "$MODE" = "loso" ] || [ "$MODE" = "train" ]; then
    echo ""
    echo "====================================================="
    echo "  [3] LOSO 교차검증 (M1~M6)"
    echo "====================================================="
    LOGFILE="out_N${N}/loso/train_loso_${TIMESTAMP}.log"
    mkdir -p "out_N${N}/loso"
    python3 train_loso.py 2>&1 | tee "${LOGFILE}" \
        || echo "[WARN] LOSO에서 에러 발생 (로그 확인: ${LOGFILE})"
fi

# S3 업로드
echo ""
echo "[S3] 결과 업로드..."
aws s3 sync "${PROJECT}/out_N${N}" "${S3}/results/out_N${N}/" --quiet \
    || echo "[WARN] S3 업로드 실패"
aws s3 sync "${PROJECT}/batches" "${S3}/results/batches/" --quiet \
    || echo "[WARN] S3 업로드 실패"
echo "  ✅ ${S3}/results/out_N${N}/"

echo ""
echo "====================================================="
echo "  ✅ 완료  N=${N}  MODE=${MODE}  $(date)"
echo "====================================================="