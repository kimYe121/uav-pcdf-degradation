"""旋转框 AP 统计测试。"""

from __future__ import annotations

import torch

from pcdf.evaluate import _average_precision


def _item(pred_box: torch.Tensor, pred_cls: torch.Tensor, gt_box: torch.Tensor,
          gt_cls: torch.Tensor, conf: torch.Tensor | None = None) -> dict:
    return {
        "pred_cls": pred_cls,
        "pred_conf": torch.ones_like(pred_cls, dtype=torch.float) if conf is None else conf,
        "pred_box": pred_box,
        "gt_cls": gt_cls,
        "gt_box": gt_box,
    }


def test_perfect_prediction() -> None:
    box = torch.tensor([[10.0, 10.0, 20.0, 10.0, 0.3]])
    item = _item(box, torch.tensor([0]), box, torch.tensor([0]))
    ap = _average_precision([item], nc=1, threshold=0.5)
    assert abs(float(ap[0]) - 1.0) < 1e-6


def test_missed_class_counts_as_zero() -> None:
    gt = torch.tensor([[10.0, 10.0, 20.0, 10.0, 0.3]])
    item = _item(torch.empty(0, 5), torch.empty(0, dtype=torch.long), gt, torch.tensor([0]))
    ap = _average_precision([item], nc=1, threshold=0.5)
    assert float(ap[0]) == 0.0


def test_class_without_gt_is_ignored() -> None:
    gt = torch.tensor([[10.0, 10.0, 20.0, 10.0, 0.3]])
    item = _item(torch.empty(0, 5), torch.empty(0, dtype=torch.long), gt, torch.tensor([0]))
    ap = _average_precision([item], nc=2, threshold=0.5)
    assert float(ap[0]) == 0.0
    assert torch.isnan(torch.tensor(ap[1]))
