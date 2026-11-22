#!/usr/bin/env bash
# 环境准备：安装项目、补依赖、下载/校验权重、检查数据、跑自检。
#
# 用法：
#   bash scripts/server/setup.sh                     # 安装 + 下载缺失权重 + 全量检查
#   bash scripts/server/setup.sh --skip-install      # 已装好，只检查/补权重
#   bash scripts/server/setup.sh --check-only        # 只跑自检，不动任何文件
#   bash scripts/server/setup.sh --no-download       # 缺权重时不联网
#   bash scripts/server/setup.sh --force-weights     # 重新下载并校验权重
#   bash scripts/server/setup.sh --skip-data         # 数据还没传完
#
# 权重全部由脚本按配置推导路径，不存在"手放到某个目录"的隐式约定：
#   $WEIGHTS_DIR/<model.encoder_weights>    骨架权重（yolov5su.pt）
#   $WEIGHTS_DIR/<model.text_encoder>/      CLIP 目录（含 text_projection.pt）
#
# 环境变量：PYTHON  DEVICE  DATA_ROOT  WEIGHTS_DIR  CLIP_SOURCE  PCDF_BACKBONE_URL
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
环境准备

用法: bash scripts/server/setup.sh [选项]

选项:
  --skip-install     跳过 pip install -e .
  --check-only       只做检查，不安装、不下载
  --no-download      缺权重时不联网（给出上传指引）
  --force-weights    重新下载骨架权重与 CLIP（覆盖已有文件）
  --skip-data        跳过数据检查
  -h, --help

环境变量:
  PYTHON=/opt/conda/bin/python
  WEIGHTS_DIR=/path/to/weights     默认 ./weights
  DATA_ROOT=/path/to/data          默认 ./data/atr_umod
  CLIP_SOURCE=auto|modelscope|hf-mirror|huggingface
  PCDF_BACKBONE_URL=url1,url2      自定义骨架权重下载源
EOF
}

SKIP_INSTALL=0
CHECK_ONLY=0
DOWNLOAD=1
FORCE_WEIGHTS=0
SKIP_DATA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --skip-install) SKIP_INSTALL=1 ;;
        --check-only) CHECK_ONLY=1; SKIP_INSTALL=1; DOWNLOAD=0 ;;
        --no-download) DOWNLOAD=0 ;;
        --force-weights) FORCE_WEIGHTS=1 ;;
        --skip-data) SKIP_DATA=1 ;;
        *) die "未知参数：$1（见 --help）" ;;
    esac
    shift
done

mkdir -p "$LOG_DIR" "$RUNS_DIR" "$WEIGHTS_DIR"
check_python
gpu_info
info "项目根：$ROOT"
info "权重目录：$WEIGHTS_DIR"
info "数据目录：$DATA_ROOT"

# --------------------------------------------------------------------------- #
# 1) 安装
# --------------------------------------------------------------------------- #
if [[ $SKIP_INSTALL -eq 0 ]]; then
    info "安装项目（editable，不重装 torch）"
    if ! "$PYTHON" -m pip install -e "$ROOT" --no-deps --no-build-isolation; then
        warn "--no-build-isolation 失败，退回默认构建隔离"
        "$PYTHON" -m pip install -e "$ROOT" --no-deps
    fi
fi

if [[ $CHECK_ONLY -eq 0 ]]; then
    info "检查运行依赖"
    missing=()
    for mod in ultralytics transformers cv2 yaml tqdm; do
        "$PYTHON" -c "import $mod" >/dev/null 2>&1 || missing+=("$mod")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        info "缺少 ${missing[*]}，安装 ultralytics transformers opencv-python pyyaml tqdm"
        "$PYTHON" -m pip install ultralytics transformers opencv-python pyyaml tqdm
    else
        info "运行依赖已齐全"
    fi
    "$PYTHON" -c "import pytest" >/dev/null 2>&1 \
        || "$PYTHON" -m pip install pytest >/dev/null 2>&1 \
        || warn "pytest 安装失败，smoke.sh 会跳过单元测试"
fi

# --------------------------------------------------------------------------- #
# 2) 权重（路径全部由配置推导，脚本不写死文件名）
# --------------------------------------------------------------------------- #
if [[ $DOWNLOAD -eq 1 ]]; then
    BACKBONE_ARGS=()
    CLIP_ARGS=()
    [[ $FORCE_WEIGHTS -eq 1 ]] && BACKBONE_ARGS+=(--force) && CLIP_ARGS+=(--force)

    info "骨架权重"
    "$PYTHON" -m pcdf.cli.fetch_backbone ${BACKBONE_ARGS[@]+"${BACKBONE_ARGS[@]}"} \
        || warn "骨架权重未就绪；可手动上传到 $WEIGHTS_DIR 后重跑 --check-only"

    info "CLIP 文本编码器（源：${CLIP_SOURCE:-auto}）"
    "$PYTHON" -m pcdf.cli.fetch_clip ${CLIP_ARGS[@]+"${CLIP_ARGS[@]}"} \
        || warn "CLIP 下载失败；可换源 CLIP_SOURCE=hf-mirror 或手动上传"
else
    warn "已禁用下载：请手动把骨架权重与 clip-vit-base-patch16/ 放到 $WEIGHTS_DIR"
fi

# --------------------------------------------------------------------------- #
# 3) 就绪检查 + 自检
# --------------------------------------------------------------------------- #
READY=1
if ! preflight weights --quiet; then
    READY=0
fi
if [[ $SKIP_DATA -eq 0 ]]; then
    preflight data --quiet || READY=0
fi

if [[ $READY -eq 1 ]]; then
    info "运行完整自检"
    "$PYTHON" -m pcdf.cli.check_env || die "自检未通过，先按上面的提示修复"
    info "环境准备完成"
    info "下一步：bash scripts/server/smoke.sh"
else
    warn "环境已安装，但权重或数据未齐。补齐后重跑：bash scripts/server/setup.sh --check-only"
    [[ $SKIP_DATA -eq 0 ]] && warn "数据可先跑：bash scripts/server/prepare_data.sh --raw <原始数据目录>"
    exit 1
fi
