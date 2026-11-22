#!/usr/bin/env bash
# 训练状态速览：GPU、磁盘、内存、三阶段进度、最新日志。
#
# 用法：
#   bash scripts/server/status.sh
#   bash scripts/server/status.sh --json     # 机器可读（交给监控/脚本解析）
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
训练状态速览

用法: bash scripts/server/status.sh [--json]

只读 metrics.json / summary.json，不加载 torch，可在无卡跳板机运行。
EOF
}

JSON=0
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --json) JSON=1 ;;
        *) die "未知参数：$arg" ;;
    esac
done

if [[ $JSON -eq 1 ]]; then
    pcdf pcdf.report --runs-dir "$RUNS_DIR" --run-name "$RUN_NAME" --json
    exit 0
fi

echo "================ 时间 ================"
date '+%F %T %Z'
echo "运行目录：$RUNS_DIR    运行名：$RUN_NAME"

echo "================ GPU ================="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu \
        --format=csv
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
else
    echo "未找到 nvidia-smi"
fi

echo "================ 磁盘 ================="
df -h "$ROOT" | awk 'NR==1 || NR==2'

echo "================ 内存 ================="
if command -v free >/dev/null 2>&1; then
    free -h | awk 'NR<=2'
else
    echo "当前系统没有 free 命令"
fi

echo "================ 训练进度 =============="
pcdf pcdf.report --runs-dir "$RUNS_DIR" --run-name "$RUN_NAME"

echo "================ 最新日志 =============="
LATEST_LOG=""
if [[ -d "$LOG_DIR" ]]; then
    LATEST_LOG="$(ls -1t "$LOG_DIR"/*.log 2>/dev/null | head -1 || true)"
fi
if [[ -n "$LATEST_LOG" ]]; then
    echo "文件：$LATEST_LOG"
    tail -n 15 "$LATEST_LOG"
else
    echo "暂无日志"
fi
