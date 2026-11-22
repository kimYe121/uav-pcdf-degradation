#!/usr/bin/env bash
# 单阶段训练入口：自动续训、自动接上一阶段权重。
#
# 用法：
#   bash scripts/server/train.sh baseline                 # 阶段 0：双流 concat baseline
#   bash scripts/server/train.sh stage1                   # 阶段 1：从 baseline 自动初始化
#   bash scripts/server/train.sh stage2                   # 阶段 2：从 stage1 自动初始化
#   bash scripts/server/train.sh stage1 --fresh           # 旧检查点改名备份后重跑
#   bash scripts/server/train.sh stage1 --dry-run         # 只打印命令
#   bash scripts/server/train.sh resume1                  # 显式续训阶段 1
#   bash scripts/server/train.sh stage1 --profile 48g     # 选配置档位
#   bash scripts/server/train.sh stage1 --batch 8 --epochs 50 --no-amp
#
# 环境变量（均可覆盖）：
#   DEVICE=0  PYTHON=/opt/conda/bin/python  RUNS_DIR=...  DATA_ROOT=...  WEIGHTS_DIR=...
#
# 超参不写在本脚本里：来自 configs/pcdf.yaml + --profile，或直接透传给 Python。
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
单阶段训练入口

用法: bash scripts/server/train.sh <命令> [选项]

命令:
  baseline | stage0   阶段 0：baseline 双流 concat
  stage1              阶段 1：SCPL + CDF（主干冻结，基于 baseline）
  stage2              阶段 2：full pipeline（基于 stage 1）
  resume0/1/2         显式续训对应阶段

选项:
  --fresh             把旧运行目录改名备份（不删除），从头训练
  --dry-run           只打印最终命令
  -h, --help          显示本帮助

其余参数原样传给 pcdf.cli.train，例如 --batch 8 --epochs 50 --profile 48g --no-amp
EOF
}

CMD="${1:-}"
case "$CMD" in
    -h|--help) usage; exit 0 ;;
    baseline|stage0|stage1|stage2|resume0|resume1|resume2) ;;
    "") usage; exit 1 ;;
    *) die "未知命令：$CMD（可选 baseline/stage1/stage2/resume0/resume1/resume2）" ;;
esac
shift

case "$CMD" in
    baseline|stage0) STAGE=0; STAGE_NAME=baseline ;;
    stage1)          STAGE=1; STAGE_NAME=stage1 ;;
    stage2)          STAGE=2; STAGE_NAME=stage2 ;;
    resume0)         STAGE=0; STAGE_NAME=baseline ;;
    resume1)         STAGE=1; STAGE_NAME=stage1 ;;
    resume2)         STAGE=2; STAGE_NAME=stage2 ;;
esac

FRESH=0
DRY=0
EXTRA=()
for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=1 ;;
        --dry-run) DRY=1 ;;
        *) EXTRA+=("$arg") ;;
    esac
done

check_python
gpu_info
check_weights
check_data
require_disk "$RUNS_DIR" 3

RUN_DIR="$(stage_dir "$STAGE")"

# --fresh 只改名备份，绝不删除用户数据。
if [[ $FRESH -eq 1 && -d "$RUN_DIR" ]]; then
    BAK="${RUN_DIR}_bak_$(date '+%Y%m%d_%H%M%S')"
    warn "--fresh：旧目录改名为 $BAK"
    mv "$RUN_DIR" "$BAK"
fi

ARGS=(--stage "$STAGE" --device "$DEVICE" --run-dir "$RUN_DIR")

# 档位优先用命令行透传的 --profile，其次用环境变量 PROFILE。
if [[ -n "$PROFILE" ]] && ! has_flag --profile ${EXTRA[@]+"${EXTRA[@]}"}; then
    ARGS+=(--profile "$PROFILE")
fi

# 已有 last.pt 时自动续训；显式给了 --init/--resume 则尊重用户选择。
if [[ $FRESH -eq 0 && -f "$RUN_DIR/last.pt" ]] \
    && ! has_flag --resume ${EXTRA[@]+"${EXTRA[@]}"} \
    && ! has_flag --init ${EXTRA[@]+"${EXTRA[@]}"}; then
    ARGS+=(--resume auto)
    warn "检测到 $RUN_DIR/last.pt：自动续训（--fresh 可重跑）"
fi

LOG_FILE="$LOG_DIR/train_${STAGE_NAME}_$(date '+%Y%m%d_%H%M%S').log"
if has_flag --profile ${EXTRA[@]+"${EXTRA[@]}"}; then
    PROFILE_DISPLAY="(命令行 --profile)"
else
    PROFILE_DISPLAY="${PROFILE:-base}"
fi
info "阶段：$STAGE（$STAGE_NAME）  目录：$RUN_DIR"
info "设备：$DEVICE  档位：$PROFILE_DISPLAY  日志：$LOG_FILE"

if [[ $DRY -eq 1 ]]; then
    print_cmd "$PYTHON" -m pcdf.cli.train "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
    exit 0
fi

set +e
run_logged "$LOG_FILE" "$PYTHON" -m pcdf.cli.train "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
STATUS=$?
set -e

if [[ $STATUS -ne 0 ]]; then
    warn "训练退出码 $STATUS。查看日志：$LOG_FILE"
    warn "续训：bash scripts/server/train.sh resume${STAGE}"
    exit "$STATUS"
fi

if stage_complete "$STAGE"; then
    info "阶段 $STAGE 已跑满，结果目录：$RUN_DIR"
else
    warn "阶段 $STAGE 未跑满（被中断或手动停止）"
    warn "续训：bash scripts/server/train.sh resume${STAGE}"
fi
