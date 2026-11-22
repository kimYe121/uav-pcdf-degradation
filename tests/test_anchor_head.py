"""anchor-based OBB 头（论文式 8：L_obj + L_cls + L_reg）的形状与可训练性测试。"""

from __future__ import annotations

import pytest
import torch

from pcdf.config import load_config
from pcdf.losses import make_criterion
from pcdf.models.anchor_head import DEFAULT_ANCHORS, AnchorOBBHead, AnchorOBBLoss


def _head(nc: int = 11, ch=(64, 128, 256)) -> AnchorOBBHead:
    return AnchorOBBHead(nc=nc, ch=ch)


def test_decode_channel_layout() -> None:
    """decoded 必须是 (B, 4+nc+1, N)，通道顺序 xywh/cls/θ，供 rotated NMS 使用。"""
    head = _head().eval()
    feats = [
        torch.randn(2, 64, 32, 40),
        torch.randn(2, 128, 16, 20),
        torch.randn(2, 256, 8, 10),
    ]
    with torch.no_grad():
        decoded = head.decode(head(feats))
    assert decoded.shape == (2, 4 + 11 + 1, 3 * (32 * 40 + 16 * 20 + 8 * 10))
    # 坐标应落在输入尺度内（320×256）
    assert decoded[:, 0].max() <= 320 * 1.1
    assert decoded[:, 1].max() <= 256 * 1.1
    # cls 已乘 objectness，必然 ≤ 1
    assert decoded[:, 4:15].max() <= 1.0 + 1e-5
    # θ 在 (-π/2, π/2)
    assert decoded[:, -1].abs().max() <= torch.pi / 2 + 1e-5


def test_objectness_bias_prior() -> None:
    """obj 先验应为 0.02 左右，避免初期正负样本失衡。"""
    head = _head()
    for conv in head.convs:
        view = conv.bias.detach().view(head.na, head.no)
        assert torch.allclose(
            view[:, 4].sigmoid(),
            torch.full((head.na,), 0.02, dtype=torch.float32),
            atol=1e-3,
        )


def test_anchor_shapes_are_canonical() -> None:
    """锚框宽高必须是短边在前、按尺度递增。"""
    for stride, anchors in DEFAULT_ANCHORS.items():
        areas = [w * h for w, h in anchors]
        assert all(w <= h for w, h in anchors), stride
        assert areas == sorted(areas), stride


def test_loss_is_finite_and_differentiable() -> None:
    head = _head()
    head.train()
    feats = [
        torch.randn(2, 64, 32, 40, requires_grad=True),
        torch.randn(2, 128, 16, 20, requires_grad=True),
        torch.randn(2, 256, 8, 10, requires_grad=True),
    ]
    raw = head(feats)
    criterion = AnchorOBBLoss(head)
    batch = {
        "bboxes": torch.tensor([[0.5, 0.5, 0.10, 0.20, 0.30], [0.3, 0.4, 0.05, 0.05, 1.20]]),
        "cls": torch.tensor([[3], [7]]),
        "batch_idx": torch.tensor([[0], [1]]),
    }
    loss, items = criterion((head.decode(raw), raw), batch)
    assert torch.isfinite(loss).all()
    assert set(items) == {"box_loss", "cls_loss", "obj_loss", "dfl_loss", "angle_loss"}
    loss.sum().backward()
    assert feats[0].grad is not None and torch.isfinite(feats[0].grad).all()


def test_loss_handles_empty_batch() -> None:
    """没有 GT 的 batch 不能产生 NaN（空验证集/背景图会走到这里）。"""
    head = _head().eval()
    criterion = AnchorOBBLoss(head)
    batch = {
        "bboxes": torch.zeros((0, 5)),
        "cls": torch.zeros((0, 1), dtype=torch.long),
        "batch_idx": torch.zeros((0, 1), dtype=torch.long),
    }
    with torch.no_grad():
        raw = head([
            torch.randn(1, 64, 32, 40),
            torch.randn(1, 128, 16, 20),
            torch.randn(1, 256, 8, 10),
        ])
        loss, _ = criterion((head.decode(raw), raw), batch)
    assert torch.isfinite(loss).all()


def test_base_config_builds_anchor_head() -> None:
    """默认配置（论文最可能的 v5s family）必须使用 anchor 头并可分发损失。"""
    from pcdf.models.baseline import BaselineNet

    cfg = load_config()
    assert cfg.model.head_type == "anchor"
    model = BaselineNet(cfg.model, nc=cfg.data.nc)
    assert isinstance(model.head, AnchorOBBHead)
    assert isinstance(make_criterion(model.head, cfg.model), AnchorOBBLoss)


def test_loss_gains_come_from_config() -> None:
    """增益必须来自配置，不能是 head 内部的魔数。"""
    cfg = load_config(overrides={"model.loss.anchor_box": "0.2", "model.loss.anchor_obj": "2.0"})
    head = _head()
    criterion = make_criterion(head, cfg.model)
    assert isinstance(criterion, AnchorOBBLoss)
    assert criterion.box_gain == pytest.approx(0.2)
    assert criterion.obj_gain == pytest.approx(2.0)


def test_packaged_v5_weights_are_scale_s() -> None:
    """仓库里的 yolov5su.pt 必须是 s 规模（首层 32 通道）。

    服务器曾出现权重是 n 档（16 通道）而配置写 s 的情况，导致逐层加载失败；
    这条测试把"打错权重"挡在打包之前。
    """
    from pathlib import Path

    from pcdf.models.encoder import infer_v5_scale

    weights = Path("weights/yolov5su.pt")
    if not weights.exists():
        pytest.skip("未打包 v5 权重")
    assert infer_v5_scale(weights) == "s"
