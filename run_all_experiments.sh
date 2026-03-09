#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# run_all_experiments.sh — IMU 지형 분류 전체 실험 파이프라인
# ══════════════════════════════════════════════════════════════
# 사용법:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh              # 전체 (kfold → loso → hierarchical)
#   ./run_all_experiments.sh kfold
#   ./run_all_experiments.sh loso
#   ./run_all_experiments.sh hierarchical
# ══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/experiments/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)

run_kfold() {
    echo "════════════════════════════════════════"
    echo "  [1/3] K-Fold 비교 실험"
    echo "════════════════════════════════════════"
    LOG="${LOG_DIR}/kfold_${TS}.log"
    python "${SCRIPT_DIR}/train_kfold.py" \
        --models "M2,M4,M6,ResNet1D,CNNTCN,ResNetTCN,M7" \
        2>&1 | tee "${LOG}"
    echo "  ✅ K-Fold 완료 → ${LOG}"
}

run_loso() {
    echo "════════════════════════════════════════"
    echo "  [2/3] LOSO 실험"
    echo "════════════════════════════════════════"
    LOG="${LOG_DIR}/loso_${TS}.log"
    python "${SCRIPT_DIR}/train_loso.py" \
        --models "M6,ResNet1D,ResNetTCN,M7" \
        2>&1 | tee "${LOG}"
    echo "  ✅ LOSO 완료 → ${LOG}"
}

run_hierarchical() {
    echo "════════════════════════════════════════"
    echo "  [3/3] HierarchicalFusionNet 실험"
    echo "════════════════════════════════════════"
    LOG="${LOG_DIR}/hierarchical_${TS}.log"
    python "${SCRIPT_DIR}/train_hierarchical.py" \
        2>&1 | tee "${LOG}"
    echo "  ✅ Hierarchical 완료 → ${LOG}"
}

case "${1:-all}" in
    kfold)         run_kfold ;;
    loso)          run_loso ;;
    hierarchical)  run_hierarchical ;;
    all)
        run_kfold
        run_loso
        run_hierarchical
        echo ""
        echo "════════════════════════════════════════"
        echo "  ★ 전체 실험 완료"
        echo "  결과: ${SCRIPT_DIR}/experiments/"
        echo "════════════════════════════════════════"
        ;;
    *)
        echo "Usage: $0 [kfold|loso|hierarchical|all]"
        exit 1
        ;;
esac