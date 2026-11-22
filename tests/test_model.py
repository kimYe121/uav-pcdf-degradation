"""完整 PCDF 网络的张量形状、梯度与冻结策略测试（v8 anchor-free 头）。"""

from __future__ import annotations

import pytest
import torch

from pcdf.conditions import ConditionTable
from pcdf.config import load_config
from pcdf.models.pcdf import PCDFNet
from pcdf.text_encoder import CLIPTextEncoder


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    # 本文件测 Ultralytics v8 头（anchor-free + DFL）的行为，显式切成 11n 结构。
    # 仓库不附带 11 系权重，因此从 yaml 构造、随机初始化——测的是形状与梯度，
    # 与预训练无关。
    cfg.model.backbone_yaml = "yolo11n-obb.yaml"
    cfg.model.widths = (128, 128, 256)
    cfg.model.head_type = "v8"
    cfg.model.pretrained_backbone = False
    table = ConditionTable.load(cfg.paths.condition_index, cfg.model.condition_spec)
    encoder = CLIPTextEncoder(cfg.model.text_encoder, device="cpu")
    model = PCDFNet(cfg.model, nc=cfg.data.nc, table=table, text_encoder=encoder)
    model.eval()
    return cfg, table, model


def test_train_forward_shapes(setup) -> None:
    cfg, table, model = setup
    model.train()
    x = torch.rand(2, 6, 128, 128)
    cond = torch.tensor([[0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1]])
    out = model(x, cond=cond, guidance="text", return_aux=True, return_ft=True)
    assert out["guidance"].shape == (2, cfg.model.guidance_dim)
    assert out["ft"].shape == (2, 512)
    assert out["gate_mask"].shape == (2, table.n_attributes)
    assert out["fs_gap"].shape == (2, cfg.model.guidance_dim)
    assert out["main"]["boxes"].shape[0] == 2
    assert out["main"]["scores"].shape[1] == cfg.data.nc
    assert out["aux_rgb"]["scores"].shape == out["aux_ir"]["scores"].shape
    # P3/8 + P4/16 + P5/32 anchors at 128x128.
    assert out["main"]["boxes"].shape[-1] == 16 * 16 + 8 * 8 + 4 * 4


def test_soft_gate_is_channel_normalised(setup) -> None:
    cfg, table, model = setup
    model.train()
    cond = torch.tensor([[0, 1, 0, 0, 0, 0]])
    out = model(torch.rand(1, 6, 128, 128), cond=cond, guidance="text")
    for w_ir in out["cdf_ir_weights"]:
        assert torch.all(w_ir >= 0) and torch.all(w_ir <= 1)
        assert torch.allclose(w_ir + (1 - w_ir), torch.ones_like(w_ir))


def test_gate_receives_gradient(setup) -> None:
    cfg, table, model = setup
    model.train()
    model.zero_grad(set_to_none=True)
    cond = torch.tensor([[0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1]])
    out = model(torch.rand(2, 6, 128, 128), cond=cond, guidance="text")
    out["guidance"].sum().backward()
    grad = model.scpl.gate.mlp[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_text_encoder_is_frozen(setup) -> None:
    _, _, model = setup
    assert all(not p.requires_grad for p in model.text_encoder.model.parameters())
    assert not any(p.grad is not None for p in model.text_encoder.model.parameters())


def test_eval_forward_has_no_text_branch(setup) -> None:
    cfg, table, model = setup
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(2, 6, 128, 128), guidance="visual")
    # v8 头在推理时返回 (decoded, raw_dict)，训练时返回 raw_dict；两种都由 head 决定。
    assert isinstance(out["main"], tuple)
    decoded, raw = out["main"]
    assert decoded.shape == (2, 4 + cfg.data.nc + 1, 16 * 16 + 8 * 8 + 4 * 4)
    assert raw["boxes"].shape[0] == 2
    assert "gate_mask" not in out


def test_stage2_freezes_gate_and_soft_path(setup) -> None:
    """阶段 2 冻结门控后，straight-through 路径也要关闭（否则白算 6N 次编码）。"""
    _, _, model = setup
    model.freeze_scpl()
    assert all(not p.requires_grad for p in model.scpl.parameters())
    assert model.scpl.soft_gradient is False
