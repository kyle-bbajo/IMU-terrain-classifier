#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# run_all_experiments.sh — IMU 지형 분류 전체 실험 파이프라인
# ══════════════════════════════════════════════════════════════
# 사용법:
#   ./run_all_experiments.sh kfold
#   ./run_all_experiments.sh kfold --bg
#   ./run_all_experiments.sh kfold,hierarchical       # 동시 백그라운드
#   ./run_all_experiments.sh all                      # 순차
#
# W&B 비활성화: WANDB_PROJECT="" ./run_all_experiments.sh kfold
# ══════════════════════════════════════════════════════════════

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# ── W&B 설정 ─────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT:-imu-terrain}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_DIR="${SCRIPT_DIR}/experiments/wandb"
mkdir -p "${WANDB_DIR}"
[ -n "${WANDB_PROJECT}" ] \
    && echo "  [W&B] project=${WANDB_PROJECT}" \
    || echo "  [W&B] 비활성화"

LOG_DIR="${SCRIPT_DIR}/experiments/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)

TARGETS="${1:-all}"
BG=0
[ "${2}" = "--bg" ] && BG=1

# ── 실험별 명령 (배열) ─────────────────────────────────────────
_cmd_kfold=(
    python "${SCRIPT_DIR}/train_kfold.py"
    --models "M2,M4,M6,ResNet1D,CNNTCN,ResNetTCN,M7"
    --run_name "kfold_${TS}"
)
_cmd_loso=(
    python "${SCRIPT_DIR}/train_loso.py"
    --models "M6,ResNet1D,ResNetTCN,M7"
    --run_name "loso_${TS}"
)
_cmd_hierarchical=(
    python "${SCRIPT_DIR}/train_hierarchical.py"
    --run_name "hierarchical_${TS}"
)

# ── 실행 헬퍼 ─────────────────────────────────────────────────
_run() {
    local name="$1"; shift
    local log="${LOG_DIR}/${name}_${TS}.log"
    if [ $BG -eq 1 ]; then
        nohup "$@" > "${log}" 2>&1 &
        local pid=$!
        echo "${pid}" > "${LOG_DIR}/${name}.pid"
        echo "  ▶ ${name} 백그라운드 시작  PID=${pid}"
        echo "    로그: tail -f ${log}"
        echo "    종료: kill ${pid}"
    else
        echo "════════════════════════════════════════"
        echo "  ${name} 실행 중..."
        echo "════════════════════════════════════════"
        "$@" 2>&1 | tee "${log}"
        echo "  ✅ ${name} 완료 → ${log}"
    fi
}

# ── 타겟 실행 ─────────────────────────────────────────────────
_dispatch() {
    local name="$1"
    case "$name" in
        kfold)         _run kfold        "${_cmd_kfold[@]}" ;;
        loso)          _run loso         "${_cmd_loso[@]}" ;;
        hierarchical)  _run hierarchical "${_cmd_hierarchical[@]}" ;;
        *) echo "알 수 없는 실험: $name"; exit 1 ;;
    esac
}

case "$TARGETS" in
    all)
        _run kfold        "${_cmd_kfold[@]}"
        _run loso         "${_cmd_loso[@]}"
        _run hierarchical "${_cmd_hierarchical[@]}"
        echo "★ 전체 실험 완료"
        ;;
    *,*)
        BG=1
        IFS=',' read -ra NAMES <<< "$TARGETS"
        for name in "${NAMES[@]}"; do
            _dispatch "$(echo "$name" | tr -d ' ')"
        done
        ;;
    *)
        _dispatch "$TARGETS"
        ;;
esac