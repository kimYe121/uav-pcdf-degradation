#!/usr/bin/env bash
# 把原始 ATR-UMOD 转换成训练用的 YOLO-OBB 目录 + condition_index.csv。
#
# 原始目录要求：
#   <raw>/train/{images,images_ir,labels,labels_ir}
#   <raw>/test/ {images,images_ir,labels,labels_ir}
# 其中 labels_ir/*.xml 带 robndbox 与六个条件属性。
#
# 用法：
#   bash scripts/server/prepare_data.sh --raw /root/rivermind-data/ATR-UMOD
#   bash scripts/server/prepare_data.sh --raw ... --out ... --val-frac 0.15
#   bash scripts/server/prepare_data.sh --raw ... --skip-check
#
# 注意：转换会为每张图建立硬链接（同盘）或复制（跨盘），需要与原始数据
# 大致等量的磁盘空间；先看 df -h 再跑。
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
原始数据转换

用法: bash scripts/server/prepare_data.sh --raw <原始数据根> [选项]

选项:
  --raw DIR        原始 ATR-UMOD 根目录（必填）
  --out DIR        输出目录；默认 $DATA_ROOT
  --val-frac F     验证集比例（按场景组留出），默认 0.15
  --limit N        只转换前 N 个训练样本（调试）
  --skip-check     跳过转换前的统计
  -h, --help

环境变量: DATA_ROOT 指定默认输出目录。
EOF
}

RAW=""
OUT="$DATA_ROOT"
PASSTHRU=()
RUN_CHECK=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --raw) RAW="${2:?--raw 需要目录}"; shift ;;
        --raw=*) RAW="${1#*=}" ;;
        --out) OUT="${2:?--out 需要目录}"; shift ;;
        --out=*) OUT="${1#*=}" ;;
        --skip-check) RUN_CHECK=0 ;;
        *) PASSTHRU+=("$1") ;;
    esac
    shift
done

[[ -n "$RAW" ]] || die "--raw 必填（见 --help）"
need_dir "$RAW"
need_dir "$RAW/train/labels_ir"

check_python
info "原始数据：$RAW"
info "输出目录：$OUT"

if [[ $RUN_CHECK -eq 1 ]]; then
    info "统计原始数据规模与类别/条件分布"
    "$PYTHON" -m pcdf.cli.prepare_data check --root "$RAW" --sample 300
fi

info "开始转换"
"$PYTHON" -m pcdf.cli.prepare_data convert --root "$RAW" --out "$OUT" \
    ${PASSTHRU[@]+"${PASSTHRU[@]}"}

info "转换完成，检查数据完整性"
preflight data
info "下一步：bash scripts/server/smoke.sh"
