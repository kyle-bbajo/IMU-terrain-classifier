#!/bin/bash
# ═══════════════════════════════════════════════════════
# run.sh v7.3 Final — 힐스트라이크 파이프라인
#
# 사용법:
#   bash run.sh              # N=40, 전체
#   bash run.sh 40 all       # N=40, 전체
#   bash run.sh 100 kfold    # N=100, K-Fold만
#   bash run.sh 40 loso      # N=40, LOSO만
#   bash run.sh 40 train     # K-Fold + LOSO
#   bash run.sh 40 seg       # Segmentation만
#
# ★ sed 제거 — argparse로 N 전달
# ★ config 스냅샷 자동 저장
# ═══════════════════════════════════════════════════════
set -euo pipefail

N=${1:-40}
MODE=${2:-all}
PROJECT=/home/ubuntu/project/repo
S3=s3://imu-khu-jooho-s1-6

source /home/ubuntu/project/venv/bin/activate
cd "${PROJECT}"

echo "====================================================="
echo "  v7.3 Final 파이프라인  N=${N}  MODE=${MODE}  $(date)"
echo "  ★ argparse 기반 (sed 없음)"
echo "====================================================="

# 환경 확인
python3 -c "
import torch, os
print(f'  PyTorch {torch.__version__}')
if torch.cuda.is_available():
    n = torch.cuda.get_device_name(0)
    m = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU: {n}  ({m:.0f}GB)')
else:
    print(f'  CPU: {os.cpu_count()} vCPU')
import config
print(f'  Strategy: {\"Preload\" if config.USE_PRELOAD else \"OTF\"}')
print(f'  RAM: {config.RAM_GIB}GiB  Batch: {config.BATCH}  Workers: {config.LOADER_WORKERS}')
"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
fi

# S3 동기화
echo ""
echo "[S3] CSV 동기화..."
aws s3 sync "${S3}/raw_csv" /home/ubuntu/project/data/raw_csv/ --quiet
echo "  ✅ 동기화 완료"

# 1) Segmentation (증분 모드)
if [ "$MODE" = "all" ] || [ "$MODE" = "seg" ]; then
    echo ""
    echo "====================================================="
    echo "  [1] 힐스트라이크 스텝 분할 (증분)"
    echo "====================================================="
    python3 step_segmentation.py --n_subjects "${N}"
fi

# 2) K-Fold
if [ "$MODE" = "all" ] || [ "$MODE" = "kfold" ] || [ "$MODE" = "train" ]; then
    echo ""
    echo "====================================================="
    echo "  [2] K-Fold 교차검증 (M1~M6)"
    echo "====================================================="
    python3 train_kfold.py --n_subjects "${N}" 2>&1 | tee "out_N${N}/kfold/train_kfold.log"
fi

# 3) LOSO
if [ "$MODE" = "all" ] || [ "$MODE" = "loso" ] || [ "$MODE" = "train" ]; then
    echo ""
    echo "====================================================="
    echo "  [3] LOSO 교차검증 (M1~M6)"
    echo "====================================================="
    python3 train_loso.py --n_subjects "${N}" 2>&1 | tee "out_N${N}/loso/train_loso.log"
fi

# S3 업로드
echo ""
echo "[S3] 결과 업로드..."
aws s3 sync "${PROJECT}/out_N${N}" "${S3}/results/out_N${N}/" --quiet
aws s3 sync "${PROJECT}/batches" "${S3}/results/batches/" --quiet
echo "  ✅ ${S3}/results/out_N${N}/"

echo ""
echo "====================================================="
echo "  ✅ 완료  N=${N}  MODE=${MODE}  $(date)"
echo "====================================================="
