#!/usr/bin/env bash
# 训练前冒烟：单元测试 + 三阶段端到端 + 可选速度/显存基准。
#
# 用法：
#   bash scripts/server/smoke.sh                    # pytest + 三阶段冒烟
#   bash scripts/server/smoke.sh --no-pytest        # 跳过单元测试
#   bash scripts/server/smoke.sh --bench            # 追加 50 step 基准
#   bash scripts/server/smoke.sh --bench --batch 8
#   bash scripts/server/smoke.sh --device cpu       # 无卡环境的功能验证
#
# 冒烟只验证"代码能跑通"，不产生任何精度结论。
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
冒烟与基准

用法: bash scripts/server/smoke.sh [选项]

选项:
  --no-pytest        跳过单元测试
  --bench            追加 50 step 速度/显存基准（阶段 0，不需要已有检查点）
  --batch N          基准 batch（默认 4）
  --imgsz H W        冒烟输入尺寸（默认 320 256）
  --device ...       透传给 Python
  -h, --help
EOF
}

RUN_PYTEST=1
BENCH=0
BENCH_BATCH=4
SMOKE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --no-pytest) RUN_PYTEST=0 ;;
        --bench) BENCH=1 ;;
        --batch) BENCH_BATCH="${2:?--batch 需要数值}"; shift ;;
        --batch=*) BENCH_BATCH="${1#*=}" ;;
        --imgsz) SMOKE_ARGS+=(--imgsz "$2" "$3"); shift 2 ;;
        --device) SMOKE_ARGS+=(--device "$2"); shift ;;
        *) die "未知参数：$1（见 --help）" ;;
    esac
    shift
done

check_python
gpu_info
check_weights
check_data

LOG_FILE="$LOG_DIR/smoke_$(date '+%Y%m%d_%H%M%S').log"
info "设备：$DEVICE  日志：$LOG_FILE"

if [[ $RUN_PYTEST -eq 1 ]]; then
    info "单元测试"
    run_logged "$LOG_FILE" "$PYTHON" -m pytest -q
fi

info "三阶段端到端冒烟"
run_logged "$LOG_FILE" "$PYTHON" -m pcdf.cli.smoke --device "$DEVICE" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}

if [[ $BENCH -eq 1 ]]; then
    BENCH_DIR="$RUNS_DIR/bench_$(date '+%Y%m%d_%H%M%S')"
    info "50 step 基准：$BENCH_DIR（batch=$BENCH_BATCH，阶段 0）"
    start=$SECONDS
    run_logged "$LOG_FILE" "$PYTHON" -m pcdf.cli.train \
        --stage 0 \
        --device "$DEVICE" \
        --epochs 1 \
        --batch "$BENCH_BATCH" \
        --max-batches 50 \
        --val-batches 1 \
        --set data.fraction=0.05 \
        --set train.save_period=0 \
        --set train.save_every_minutes=0 \
        --run-dir "$BENCH_DIR"
    elapsed=$((SECONDS - start))
    info "50 step 用时 ${elapsed}s（含启动与验证），约 $((elapsed / 50))s/step"
    info "估算：1 epoch 秒数 = 每 epoch step 数 x 单 step 秒数"
fi

info "冒烟通过"
