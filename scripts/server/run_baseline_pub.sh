#!/usr/bin/env bash
# 论文口径 baseline（Table 4 的 58.4）一次性跑完 + 双口径评估。
#
# 论文原文（§5.1 Implementation Details）：
#   "SGD optimizer with an initial learning rate of 0.01 and decayed exponentially.
#    Momentum and weight decay were set to 0.937 and 0.0005.
#    two trainable processes ... both trained for 50 epochs with an 640 x 512 image size
#    and a batch size of 16."
# 论文原文（§5.2 baseline 定义）：
#   "Our baseline is a one-stage dual-stream detector that integrates two modalities
#    through concatenation fusion. Noted that the multimodal detectors are all trained
#    with IR labels."
#
# 用法（服务器上）：
#   bash scripts/server/run_baseline_pub.sh                # 训练 + 评估
#   bash scripts/server/run_baseline_pub.sh --eval-only    # 只评估已有 best.pt
set -euo pipefail

REPO=${REPO:-/root/rivermind-data/pcdf}
RUN_DIR=${RUN_DIR:-runs/base_native50}
BATCH=${BATCH:-16}          # 论文值；48G 卡想快可给 32/48，但要同步改 lr
WORKERS=${WORKERS:-16}
EVAL_ONLY=${1:-}

cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

if [[ "$EVAL_ONLY" != "--eval-only" ]]; then
  echo "== 训练：50 epoch / batch $BATCH / 640x512 / SGD 0.01 / IR 标签 =="
  screen -L -Logfile "logs/base_native50.log" -dmS bnat50 \
    python -m pcdf.cli.train \
      --stage 0 --device 0 --batch "$BATCH" --workers "$WORKERS" --epochs 50 \
      --lr 0.01 --warmup-epochs 3 \
      --imgsz 512 640 \
      --run-dir "$RUN_DIR" \
      --set train.eval_interval=5
  echo "已启动 screen 会话 b50；跟踪：tail -f logs/base50pub.log"
  echo "完成后执行：bash scripts/server/run_baseline_pub.sh --eval-only"
  exit 0
fi

echo "== 评估：val（选权重）+ test（论文报数口径） =="
python -m pcdf.cli.evaluate \
  --checkpoint "$RUN_DIR/best.pt" --split both --device 0 --batch "$BATCH" \
  --map50-95 2>&1 | tee "$RUN_DIR/eval_both.log"

echo
echo "== 与论文 Table 4 对照 =="
python - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
data = json.loads((run / "eval_both.json").read_text(encoding="utf-8"))["splits"]
test = data["test"]["per_class_ap50"]

# 论文 Table 4 的列序与数值（CR SV VN BS FC TK TT TR CE ER ME）
paper_cols = ["car", "suv", "van", "bus", "freight_car",
              "truck", "tank_truck", "trailer", "crane", "excavator", "motorcycle"]
paper_name = ["CR", "SV", "VN", "BS", "FC", "TK", "TT", "TR", "CE", "ER", "ME"]
rgb = [45.8, 60.7, 57.5, 75.2, 41.6, 52.1, 18.2, 42.3, 68.7, 47.5, 47.4]
ir = [65.8, 51.2, 51.6, 75.3, 53.1, 38.9, 83.3, 46.2, 57.9, 12.0, 42.7]

print(f"{'':<4}{'cls':<13}{'ours(test)':>11}{'YOLOv5s-RGB':>13}{'YOLOv5s-IR':>12}{'Δ(IR)':>9}")
for i, cls in enumerate(paper_cols):
    ours = 100.0 * test[cls]
    print(f"{paper_name[i]:<4}{cls:<13}{ours:>11.2f}{rgb[i]:>13.1f}{ir[i]:>12.1f}{ours - ir[i]:>+9.2f}")
ours_map = 100.0 * data["test"]["map50"]
print("-" * 62)
print(f"{'':<4}{'mAP50':<13}{ours_map:>11.2f}{50.7:>13.1f}{52.5:>12.1f}{ours_map - 52.5:>+9.2f}")
print(f"{'':<4}{'target':<13}{'58.4':>11}{'':>13}{'':>12}{ours_map - 58.4:>+9.2f}"
      "   <- 论文 baseline 58.4")
print(f"\nval={100.0*data['val']['map50']:.2f}  test={ours_map:.2f}")
PY