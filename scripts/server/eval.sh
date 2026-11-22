#!/usr/bin/env bash
# 评估旋转框 mAP。默认取阶段 2 的 best.pt，在 val 上算 mAP50。
#
# 用法：
#   bash scripts/server/eval.sh                          # val / mAP50
#   bash scripts/server/eval.sh --split test --map50-95   # test / mAP50-95
#   bash scripts/server/eval.sh --checkpoint <路径>
#   bash scripts/server/eval.sh --max-batches 20          # 快速抽查
#   bash scripts/server/eval.sh --split train --max-batches 30   # 判断过拟合
#
# 评估只走视觉引导 F_s，不读取条件标注、不调用 CLIP 文本分支。
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
评估检查点

用法: bash scripts/server/eval.sh [选项]

选项:
  --checkpoint PATH   / --ckpt PATH     指定检查点；默认阶段 2 的 best.pt
  --split train|val|test                默认 val
  --map50-95                            额外计算 mAP50-95
  --batch N / --workers N / --max-batches N
  -h, --help

其余参数透传给 pcdf.cli.evaluate。
EOF
}

CKPT=""
SPLIT="val"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --checkpoint|--ckpt) CKPT="${2:?--checkpoint 需要路径}"; shift ;;
        --checkpoint=*|--ckpt=*) CKPT="${1#*=}" ;;
        --split) SPLIT="${2:?--split 需要取值}"; shift ;;
        --split=*) SPLIT="${1#*=}" ;;
        *) EXTRA+=("$1") ;;
    esac
    shift
done

check_python
gpu_info
check_weights

if [[ -z "$CKPT" ]]; then
    for candidate in "$(stage_dir 2)/best.pt" "$(stage_dir 2)/last.pt"; do
        if [[ -f "$candidate" ]]; then CKPT="$candidate"; break; fi
    done
fi
[[ -n "$CKPT" ]] || die "找不到检查点，请用 --checkpoint 指定"
need_file "$CKPT"

case "$SPLIT" in
    train|val|test) ;;
    *) die "--split 只能是 train / val / test" ;;
esac

ARGS=(--checkpoint "$CKPT" --split "$SPLIT" --device "$DEVICE")
if [[ -n "$PROFILE" ]] && ! has_flag --profile ${EXTRA[@]+"${EXTRA[@]}"}; then
    ARGS+=(--profile "$PROFILE")
fi

LOG_FILE="$LOG_DIR/eval_${SPLIT}_$(date '+%Y%m%d_%H%M%S').log"
info "检查点：$CKPT"
info "划分：$SPLIT  设备：$DEVICE  日志：$LOG_FILE"

run_logged "$LOG_FILE" "$PYTHON" -m pcdf.cli.evaluate "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
info "评估完成；结果 JSON 与检查点同目录（eval_${SPLIT}.json）"
