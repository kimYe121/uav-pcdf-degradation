#!/usr/bin/env bash
# 一键跑完三阶段：baseline -> 阶段 1 -> 阶段 2。
#
# 用法：
#   bash scripts/server/train_all.sh                  # 基础配置
#   bash scripts/server/train_all.sh --profile 48g    # 三阶段共用档位
#   bash scripts/server/train_all.sh --batch 8        # 参数透传给三个阶段
#   bash scripts/server/train_all.sh --dry-run        # 只打印三阶段的命令
#   DEVICE=1 bash scripts/server/train_all.sh
#
# 行为：
#   - 每阶段已有 last.pt 会自动续训，跑满才进入下一阶段；
#   - 被抢占/中断不会误开下一阶段，而是提示续训命令；
#   - 阶段 1/2 的初始权重由 Python 侧按运行目录自动解析，脚本不再拼路径。
#
# 建议放 screen（断线不影响）：
#   screen -dmS pcdf bash -lc 'cd <REMOTE> && bash scripts/server/train_all.sh'
#   screen -r pcdf      # 进去看；Ctrl-a d 退出但不停止
#   screen -X -S pcdf quit   # 需要时结束会话
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
三阶段一键训练

用法: bash scripts/server/train_all.sh [选项]

选项:
  --dry-run       只打印三阶段命令
  -h, --help      显示本帮助

其余参数透传给每个阶段（--profile / --batch / --epochs / --device ...）。
不接受 --init / --resume：阶段衔接与续训由脚本自动处理。
EOF
}

DRY=0
EXTRA=()
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY=1 ;;
        --init|--init=*|--resume|--resume=*)
            die "$arg 请单独用 train.sh 指定；train_all 会自动衔接阶段与续训" ;;
        *) EXTRA+=("$arg") ;;
    esac
done

check_python
gpu_info
check_weights
check_data

run_stage() {
    local cmd="$1"
    local -a forward=()
    [[ $DRY -eq 1 ]] && forward+=(--dry-run)
    bash "$SERVER_DIR/train.sh" "$cmd" ${forward[@]+"${forward[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
}

start_all=$SECONDS
for spec in "0:baseline" "1:stage1" "2:stage2"; do
    stage="${spec%%:*}"
    cmd="${spec##*:}"
    info "===== 阶段 $stage（$cmd）====="
    run_stage "$cmd"
    if [[ $DRY -eq 1 ]]; then
        continue
    fi
    if ! stage_complete "$stage"; then
        die "阶段 $stage 未跑满。续训：bash scripts/server/train.sh resume${stage}"
    fi
    info "===== 阶段 $stage 完成，累计用时 $((SECONDS - start_all))s ====="
done

if [[ $DRY -eq 1 ]]; then
    exit 0
fi

info "===== 三阶段全部完成，总用时 $((SECONDS - start_all))s ====="
info "baseline 权重：$(stage_dir 0)/best.pt（对照论文 58.4）"
info "最终权重：$(stage_dir 2)/best.pt（对照论文 63.1）"
info "下一步：bash scripts/server/eval.sh --split test --map50-95"
