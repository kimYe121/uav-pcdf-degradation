"""OBB 头路径的冒烟测试。这里盯三件事：训练态返回 loss 需要的 dict、推理态
返回可进 NMS 的张量、损失能反传到 backbone。
"""

from __future__ import annotations

import pytest
import torch

from pcdf.config import load_config
from pcdf.models.baseline import BaselineNet
from pcdf.models.heads import make_criterion


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    model = BaselineNet(cfg.model, nc=cfg.data.nc)
    return cfg, model


def test_train_mode_returns_loss_dict(setup) -> None:
    """训练态必须返回 dict(boxes, scores, angle)，v8OBBLoss 依赖它。"""
    cfg, model = setup
    model.train()
    out = model(torch.rand(2, 6, 128, 128))
    main = out["main"]
    assert isinstance(main, dict)
    assert {"boxes", "scores", "angle", "feats"} <= set(main)
    assert main["scores"].shape[1] == cfg.data.nc
    assert isinstance(main["feats"], list) and len(main["feats"]) == 3


def test_eval_mode_returns_decoded(setup) -> None:
    """推理态返回 (B, 4+nc+1, N)，通道布局与旋转 NMS 一致。"""
    cfg, model = setup
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(2, 6, 128, 128))
    decoded, _ = out["main"]
    assert decoded.shape[0] == 2
    assert decoded.shape[1] == 4 + cfg.data.nc + 1
    assert float(decoded[:, 4 : 4 + cfg.data.nc].max()) <= 1.0 + 1e-4


def test_loss_runs_and_backprops(setup) -> None:
    """v8OBBLoss 能跑通，且梯度回到双流 backbone。"""
    _, model = setup
    model.train()
    criterion = make_criterion(model)
    x = torch.rand(2, 6, 128, 128)
    gts = {
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3], [0.3, 0.4, 0.1, 0.08, -0.6]]),
        "cls": torch.tensor([[3.0], [7.0]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
    }
    out = model(x)
    loss_vec, items = criterion(out["main"], gts)
    assert torch.isfinite(loss_vec).all()
    assert {"box_loss", "cls_loss", "dfl_loss", "angle_loss"} <= set(items)
    loss_vec.sum().backward()
    grad = model.encoder_rgb.layers[0].conv.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_cls_bias_prior_is_applied(setup) -> None:
    """回归用例：分类 bias 必须是 ``log(5/nc/(640/s)²)``，不是 0。

    ultralytics 的 ``bias_init`` 要等 stride 就位后由 ``DetectionModel`` 调用，
    ``OBB.__init__`` 自己不调。漏掉时初始 sigmoid≈0.5，全量训练第一步的
    cls_loss 会到 5500+（正常 5 左右），前几百步全用来压背景。
    """
    import math

    from ultralytics.nn.modules.head import OBB

    cfg, model = setup
    head = model.head
    assert isinstance(head, OBB)
    for i, stride in enumerate(head.stride.tolist()):
        expect = math.log(5 / cfg.data.nc / (640 / stride) ** 2)
        got = float(head.cv3[i][-1].bias[: cfg.data.nc].detach().mean())
        assert abs(got - expect) < 0.05, f"stride {stride}: cls bias {got:.3f} != {expect:.3f}"
        # 框分支的 bias 由 bias_init 设为 2.0（DFL 分布初始偏均匀）
        assert abs(float(head.cv2[i][-1].bias.detach().mean()) - 2.0) < 0.05


def test_head_stride_and_frozen_context(setup) -> None:
    """v8OBBLoss 需要 model.model[-1] 与 model.args，两个属性必须挂上。"""
    _, model = setup
    assert model.model[-1] is model.head
    assert tuple(model.head.stride.tolist()) == (8.0, 16.0, 32.0)
    assert model.args.box > 0 and model.args.dfl > 0