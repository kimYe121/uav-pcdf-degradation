#!/usr/bin/env bash
# PCDF 服务器端公共环境与工具函数（所有默认值的唯一来源）。
#
# 由 scripts/server/*.sh 通过 source 引入，不单独执行。
#
# 设计约定
# --------
# 1. **超参不落 shell**：batch/lr/epochs/保存周期一律来自 configs/*.yaml 或命令行，
#    这里只提供"运行环境"（解释器、设备、目录）。
# 2. **清单不重复**：权重/数据的完整性检查交给 `python -m pcdf.preflight`，
#    运行目录命名交给 `python -m pcdf.report --stage-dir`，shell 不再各写一份。
# 3. **全部可覆盖**：下面每个变量都可以在命令前用环境变量改写。
set -Eeuo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SERVER_DIR/../.." && pwd)"
cd "$ROOT"

# --------------------------------------------------------------------------- #
# 可覆盖环境变量（全部有默认值，见各脚本的 --help）
# --------------------------------------------------------------------------- #
: "${PYTHON:=python}"           # 解释器，例如 /opt/conda/bin/python
: "${DEVICE:=auto}"             # auto / cpu / 0 / cuda:0
: "${PROFILE:=}"                # 配置档位；留空用 configs/pcdf.yaml
: "${RUN_NAME:=pcdf}"           # 运行名前缀（决定默认目录名）
: "${DATA_ROOT:=$ROOT/data/atr_umod}"
: "${WEIGHTS_DIR:=$ROOT/weights}"
: "${TEE:=1}"                   # 1=日志同时写终端与文件；0=只写终端
: "${OMP_NUM_THREADS:=8}"

# 运行/日志目录：优先数据盘，容器重建或换卡后检查点不丢。
if [[ -z "${RUNS_DIR:-}" || -z "${LOG_DIR:-}" ]]; then
    if [[ -d /root/rivermind-data ]]; then
        : "${RUNS_DIR:=/root/rivermind-data/pcdf/runs}"
        : "${LOG_DIR:=/root/rivermind-data/pcdf/logs}"
    else
        : "${RUNS_DIR:=$ROOT/runs}"
        : "${LOG_DIR:=$ROOT/logs}"
    fi
fi

# 交给 Python 侧统一生效：从任意 cwd 启动都能找到同一份资源。
export PCDF_HOME="$ROOT"
export PCDF_DATA_ROOT="$DATA_ROOT"
export PCDF_WEIGHTS_DIR="$WEIGHTS_DIR"
export PCDF_RUNS_DIR="$RUNS_DIR"
export PCDF_LOGS_DIR="$LOG_DIR"
# 让 `python -m pcdf...` 优先导入**本仓库**的 src，而不是 site-packages 里的旧副本。
# 这样"覆盖代码后直接开跑"不会跑到上一次安装的版本；editable 安装仍然有效。
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS

# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
log() { printf '%s [%-5s] %s\n' "$(date '+%F %T')" "$1" "$2"; }
info() { log INFO "$*"; }
warn() { log WARN "$*"; }
die() { log ERROR "$*"; exit 1; }

# --------------------------------------------------------------------------- #
# 前置检查
# --------------------------------------------------------------------------- #
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"; }
need_file() { [[ -f "$1" ]] || die "缺少文件：$1"; }
need_dir() { [[ -d "$1" ]] || die "缺少目录：$1"; }

check_python() {
    need_cmd "$PYTHON"
    "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
        || die "需要 Python >= 3.10，当前：$("$PYTHON" -V 2>&1)"
    info "Python：$("$PYTHON" -V 2>&1)  (${PYTHON})"
}

gpu_info() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader | sed 's/^/        GPU: /'
    else
        warn "未找到 nvidia-smi；DEVICE=${DEVICE}（cpu 仅用于功能验证）"
    fi
}

# --------------------------------------------------------------------------- #
# Python 入口（统一走 -m，避免各处拼脚本路径）
# --------------------------------------------------------------------------- #
pcdf() { "$PYTHON" -m "$@"; }
preflight() { pcdf pcdf.preflight "$@"; }

check_weights() {
    local out
    if ! out="$(preflight weights --quiet 2>&1)"; then
        printf '%s\n' "$out"
        die "权重不可用。修复：bash scripts/server/setup.sh --skip-install --force-weights"
    fi
    info "权重齐全：$WEIGHTS_DIR"
}

check_data() {
    local out
    if ! out="$(preflight data --quiet 2>&1)"; then
        printf '%s\n' "$out"
        die "数据未就绪：$DATA_ROOT（先把原始数据传上来，再跑 prepare_data.sh）"
    fi
}

# 某阶段的运行目录（命名规则只存在于 pcdf.paths.stage_dir）。
stage_dir() {
    pcdf pcdf.report --stage-dir "$1" --runs-dir "$RUNS_DIR" --run-name "$RUN_NAME"
}

# 某阶段是否跑满（读 metrics/summary 的 JSON，不 import torch）。
stage_complete() {
    pcdf pcdf.report --require-stage "$1" --runs-dir "$RUNS_DIR" --run-name "$RUN_NAME" >/dev/null 2>&1
}

# 判断透传参数里是否已包含某个选项（支持 --key value 与 --key=value）。
has_flag() {
    local needle="$1" item
    shift
    for item in "$@"; do
        [[ "$item" == "$needle" || "$item" == "$needle="* ]] && return 0
    done
    return 1
}

# --------------------------------------------------------------------------- #
# 执行与日志
# --------------------------------------------------------------------------- #
run_logged() {
    local log_file="$1"
    shift
    mkdir -p "$(dirname "$log_file")"
    if [[ "${TEE}" == "1" ]]; then
        "$@" 2>&1 | tee -a "$log_file"
    else
        "$@" 2>&1
    fi
}

# 打印将要执行的命令（--dry-run），保持可直接复制运行。
print_cmd() {
    printf 'DRY-RUN:'
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
}

require_disk() {
    local dir="$1" need_gb="${2:-3}" avail_kb avail_gb
    mkdir -p "$dir"
    avail_kb=$(df -Pk "$dir" 2>/dev/null | awk 'NR==2 {print $4}')
    [[ -n "${avail_kb:-}" ]] || return 0
    avail_gb=$((avail_kb / 1024 / 1024))
    if (( avail_gb < need_gb )); then
        warn "磁盘剩余 ${avail_gb} GB < ${need_gb} GB：$dir"
    else
        info "磁盘余量：${avail_gb} GB  ($dir)"
    fi
}
