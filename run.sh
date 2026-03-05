#!/bin/bash
# ═══════════════════════════════════════════════════════
# run.sh v8.0 — 힐스트라이크 파이프라인 (SSH 분리 지원)
#
# 사용법:
#   bash run.sh              # N=40, 전체 (포그라운드)
#   bash run.sh 40 all       # N=40, 전체
#   bash run.sh 100 kfold    # N=100, K-Fold만
#   bash run.sh 40 loso      # N=40, LOSO만
#   bash run.sh 40 train     # K-Fold + LOSO
#   bash run.sh 40 seg       # Segmentation만
#   bash run.sh 40 all bg    # ★ 백그라운드 (SSH 끊어도 유지)
#
# 백그라운드 모드:
#   tmux 세션 'train' 에서 실행, SSH 종료 후에도 지속
#   재접속:  tmux attach -t train
#   로그:    tail -f /home/ubuntu/project/repo/out_N40/pipeline.log
#   중지:    tmux kill-session -t train
#
# ★ sed 제거 — argparse로 N 전달
# ★ config 스냅샷 자동 저장
# ═══════════════════════════════════════════════════════
set -euo pipefail

N=${1:-40}
MODE=${2:-all}
BG=${3:-fg}
PROJECT=/home/ubuntu/project/repo
S3=s3://imu-khu-jooho-s1-6
LOGFILE="${PROJECT}/out_N${N}/pipeline.log"

# ─────────────────────────────────────────────
# 백그라운드 모드: tmux 세션에서 자동 재실행
# ─────────────────────────────────────────────
if [ "$BG" = "bg" ]; then
    # tmux 설치 확인
    if ! command -v tmux &>/dev/null; then
        echo "tmux 설치 중..."
        sudo apt-get install -y tmux 2>/dev/null || sudo yum install -y tmux 2>/dev/null
    fi

    # 기존 세션 종료
    tmux kill-session -t train 2>/dev/null || true

    # 로그 디렉토리 생성
    mkdir -p "${PROJECT}/out_N${N}"

    # tmux 세션에서 포그라운드 모드로 재실행
    echo "═══════════════════════════════════════════════"
    echo "  ★ 백그라운드 모드 시작"
    echo "  세션 확인:  tmux attach -t train"
    echo "  로그 확인:  tail -f ${LOGFILE}"
    echo "  중지:       tmux kill-session -t train"
    echo "═══════════════════════════════════════════════"
    tmux new-session -d -s train \
        "bash ${PROJECT}/run.sh ${N} ${MODE} fg 2>&1 | tee ${LOGFILE}; echo '=== 완료 ==='; sleep 86400"
    echo "  ✅ tmux 세션 'train' 시작됨. SSH 종료해도 안전합니다."
    exit 0
fi

# ─────────────────────────────────────────────
# 포그라운드 실행 (일반 모드)
# ─────────────────────────────────────────────
source /home/ubuntu/project/venv/bin/activate
cd "${PROJECT}"

echo "====================================================="
echo "  v8.0 파이프라인  N=${N}  MODE=${MODE}  $(date)"
echo "  ★ argparse 기반 (sed 없음)"
echo "====================================================="

# 환경 확인
python3 -c "
import torch, os
print(f'  PyTorch {torch.__version__}')
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    n = torch.cuda.get_device_name(0)
    m = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU: {n}  ({m:.0f}GB) x{n_gpu}')
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
    mkdir -p "out_N${N}/kfold"
    python3 train_kfold.py --n_subjects "${N}" 2>&1 | tee "out_N${N}/kfold/train_kfold.log"
fi

# 3) LOSO
if [ "$MODE" = "all" ] || [ "$MODE" = "loso" ] || [ "$MODE" = "train" ]; then
    echo ""
    echo "====================================================="
    echo "  [3] LOSO 교차검증 (M1~M6)"
    echo "====================================================="
    mkdir -p "out_N${N}/loso"
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