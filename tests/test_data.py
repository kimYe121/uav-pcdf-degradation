"""配对 RGB/IR 数据管线测试。"""

from __future__ import annotations

import torch

from pcdf.conditions import ConditionTable
from pcdf.config import load_config
from pcdf.data import build_dataset


def test_paired_sample() -> None:
    cfg = load_config()
    cfg.data.fraction = 0.001
    cfg.data.workers = 0
    table = ConditionTable.load(cfg.paths.condition_index, cfg.model.condition_spec)
    dataset = build_dataset(
        cfg.data, table, split="train", augment=False,
        root=cfg.paths.data_root,
    )

    sample = dataset[0]
    assert sample["img"].shape[0] == 6
    assert sample["img"].dtype == torch.uint8
    assert sample["cond"].shape == (table.n_attributes,)
    assert sample["cond"].dtype == torch.int64
    assert sample["bboxes"].shape[1] == 5
    # xywh are normalised; the angle is in radians and may exceed 1.
    assert torch.all(sample["bboxes"][:, :4] >= 0)
    assert torch.all(sample["bboxes"][:, :4] <= 1)
    assert torch.all(sample["cls"] < cfg.data.nc)

    batch = dataset.collate_fn([sample, dataset[1]])
    assert batch["img"].shape[0] == 2
    assert batch["img"].shape[1] == 6
    assert batch["cond"].shape == (2, table.n_attributes)
    assert batch["bboxes"].shape[1] == 5
