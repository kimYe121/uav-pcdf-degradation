"""全仓库共享的结构常量。

这些常量描述的是**数据/模型契约**（类别顺序、特征金字塔步长、默认锚框、
每 epoch 指标列），不是可调超参。可调超参一律放在 ``configs/`` 下；这里只放
"改了就同时影响多个模块"的硬约束，避免同一份知识在多个文件里各写一遍。
"""

from __future__ import annotations

from typing import Final

# ATR-UMOD 的 11 个类别，顺序与 ``prepare_data`` 的排序结果一致。
# 它是 ``configs/pcdf.yaml`` 中 ``data.names`` 的默认值；YAML 是运行时唯一来源。
DEFAULT_NAMES: Final[list[str]] = [
    "bus",
    "car",
    "crane",
    "excavator",
    "freight_car",
    "motorcycle",
    "suv",
    "tank_truck",
    "trailer",
    "truck",
    "van",
]

# 特征金字塔的三个输出尺度（P3/8、P4/16、P5/32）。
# 编码器、CDF、neck 与检测头都以它为准。
STRIDES: Final[tuple[int, int, int]] = (8, 16, 32)

# 每个 epoch 写入 log.csv 的列。列顺序固定，追加新指标只能加在末尾，
# 以免破坏已有的曲线脚本与历史日志。
EPOCH_FIELDS: Final[tuple[str, ...]] = (
    "epoch", "lr", "loss", "box", "cls", "dfl", "angle",
    "l_dt", "l_irr", "l_dc", "map50", "kept",
)

# 训练阶段编号 -> 人类可读名称。0 是论文 Table 4 的 baseline，
# 1 是 SCPL+CDF（文本引导），2 是 full pipeline（视觉引导）。
STAGE_NAMES: Final[dict[int, str]] = {
    0: "baseline",
    1: "stage1",
    2: "stage2",
}
