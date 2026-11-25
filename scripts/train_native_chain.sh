#!/usr/bin/env bash
# 原生几何三阶段链（2026-11-25 起唯一推荐入口）。
#
# 输入几何：源图 640×512（W×H）= imgsz 512 640（H W），LetterBox ratio=1、无灰边。
# 旧的 2026-11-24 权重是 0.8 内容尺度（512×410 + 115 px 灰边），与当前代码几何
# 不同，只能 `--init` 热启动，不能 `--resume`（checkpoint.py 会直接报错）。
#
# 用法：
#   bash scripts/train_native_chain.sh                 # 服务器 4090，b16
#   BATCH=8 ACCUM=2 bash scripts/train_native_chain.sh # 本地 8 GB 卡
#   中断后重跑同一命令即可（有 last.pt 时自动 --resume auto）
set -u
PY=${PYTHON:-/d/anaconda3/python.exe}
DEVICE=${DEVICE:-0}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-1}
EPOCHS=${EPOCHS:-50}
BASE_RUN=${BASE_RUN:-runs/base_native50}
S1_RUN=${S1_RUN:-runs/pcdf_s1_native}
S2_RUN=${S2_RUN:-runs/pcdf_s2_native}
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

epochs_done() {  # $1 = run dir
  "$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['history']))" \
    "$1/metrics.json" 2>/dev/null || echo 0
}

train_stage() {  # $1=stage $2=run_dir $3...=额外参数
  local stage=$1 run_dir=$2; shift 2
  local done resume=()
  done=$(epochs_done "$run_dir")
  if [ "$done" -ge "$EPOCHS" ]; then
    echo "[native] stage$stage 已完成 $done/$EPOCHS，跳过"
    return 0
  fi
  # 只有存在 last.pt 才续训：--resume auto 在空目录会从零重训。
  [ -f "$run_dir/last.pt" ] && resume=(--resume auto)
  echo "[native] stage$stage 开始/续训（$done/$EPOCHS）batch=$BATCH accum=$ACCUM"
  "$PY" -m pcdf.cli.train --stage "$stage" --device "$DEVICE" --run-dir "$run_dir" \
    --imgsz 512 640 --batch "$BATCH" --grad-accum "$ACCUM" --epochs "$EPOCHS" \
    "${resume[@]}" "$@" >> "logs/native_stage$stage.log" 2>&1
  local rc=$?
  done=$(epochs_done "$run_dir")
  echo "[native] stage$stage 退出码=$rc 完成=$done/$EPOCHS"
  [ "$rc" -eq 0 ] && [ "$done" -ge "$EPOCHS" ]
}

train_stage 0 "$BASE_RUN" || exit 1
train_stage 1 "$S1_RUN" --init "$BASE_RUN/best.pt" || exit 1
# stage2 的 best.pt 可能因监控口径选点靠前；init 用 last.pt（见 STATE §4）
train_stage 2 "$S2_RUN" --init "$S1_RUN/last.pt" || exit 1

echo "[native] 训练完成，评估 test（论文口径）"
for ck in "$BASE_RUN/best.pt" "$S1_RUN/best.pt" "$S2_RUN/best.pt"; do
  echo "[native] eval $ck"
  "$PY" -m pcdf.cli.evaluate --checkpoint "$ck" --split test --device "$DEVICE" \
    >> logs/native_eval.log 2>&1
done
echo "[native] ALL DONE"