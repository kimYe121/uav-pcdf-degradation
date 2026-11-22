"""baseline 双流 concat 检测器、CMD 式 6 形式与阶段 1 冻结协议测试。"""

from __future__ import annotations

import pytest
import torch

from pcdf.conditions import ConditionTable
from pcdf.config import load_config
from pcdf.losses import cmd_loss
from pcdf.models.baseline import BaselineNet, load_baseline_into_pcdf
from pcdf.models.pcdf import PCDFNet
from pcdf.text_encoder import CLIPTextEncoder


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    table = ConditionTable.load(cfg.paths.condition_index, cfg.model.condition_spec)
    encoder = CLIPTextEncoder(cfg.model.text_encoder, device="cpu")
    pcdf = PCDFNet(cfg.model, nc=cfg.data.nc, table=table, text_encoder=encoder)
    baseline = BaselineNet(cfg.model, nc=cfg.data.nc)
    return cfg, pcdf, baseline


def test_cmd_loss_matches_equation6() -> None:
    """cmd_loss 必须逐项等于式 6：1/|b-a|·||E(x)-E(y)||_2 + Σ_k 1/|b-a|^k·||C_k||_2。"""
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    y = torch.randn(8, 16) + 0.5
    span = (
        torch.maximum(x.max(), y.max()) - torch.minimum(x.min(), y.min())
    ).clamp_min(1e-6)
    mx, my = x.mean(dim=0), y.mean(dim=0)
    expect = (mx - my).norm(p=2) / span
    for k in range(2, 6):
        expect = expect + ((x - mx).pow(k).mean(dim=0) - (y - my).pow(k).mean(dim=0)).norm(p=2) / span**k
    assert torch.allclose(cmd_loss(x, y, orders=5), expect, atol=1e-6)


def test_cmd_loss_is_nonnegative_and_zero_when_equal() -> None:
    torch.manual_seed(1)
    x = torch.randn(8, 16)
    assert cmd_loss(x, x.clone()).item() == pytest.approx(0.0, abs=1e-6)
    assert cmd_loss(x, torch.randn(8, 16)).item() > 0


def test_baseline_forward_shapes(setup) -> None:
    cfg, _, baseline = setup
    baseline.eval()
    with torch.no_grad():
        out = baseline(torch.rand(2, 6, 128, 128))
    decoded, raw = out["main"]
    assert decoded.shape == (2, 4 + cfg.data.nc + 1, 3 * (16 * 16 + 8 * 8 + 4 * 4))
    # anchor 头的 raw 是三个尺度的原始输出（每尺度 3*(5+1+nc) 通道）
    assert isinstance(raw, list) and len(raw) == 3
    assert raw[0].shape[1] == 3 * (5 + 1 + cfg.data.nc)


def test_baseline_keys_are_pcdf_subset(setup) -> None:
    """baseline 的参数是 PCDFNet 的子集，键名与形状完全一致。"""
    _, pcdf, baseline = setup
    pcdf_keys = dict(pcdf.state_dict())
    for name, tensor in baseline.state_dict().items():
        assert name in pcdf_keys, f"baseline 键 {name} 不在 PCDFNet 中"
        assert pcdf_keys[name].shape == tensor.shape


def test_load_baseline_into_pcdf(setup) -> None:
    """baseline 权重装入 PCDFNet 后，shared 键完全一致，missing 只含新增模块。"""
    _, pcdf, baseline = setup
    ckpt = {"model": baseline.state_dict(), "arch": "baseline"}
    info = load_baseline_into_pcdf(pcdf, ckpt)
    assert info["unexpected"] == []
    assert all(
        k.startswith(("scpl.", "text_adapter.", "cdf.", "v_rgb.", "v_ir.", "s."))
        for k in info["missing"]
    )
    merged = pcdf.state_dict()
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(merged[name], tensor)


def test_stage1_freeze_detector(setup) -> None:
    """阶段 1 只训练 SCPL 门控、CDF 与 text adapter，主干保持 eval 模式。"""
    _, pcdf, _ = setup
    pcdf.freeze_detector()
    pcdf.train()  # 模拟训练循环每 epoch 的调用；冻结模块必须仍处于 eval

    frozen = ("encoder_rgb.", "encoder_ir.", "neck.", "head.", "v_rgb.", "v_ir.", "s.")
    trainable_prefixes = set()
    for name, p in pcdf.named_parameters():
        if name.startswith(frozen):
            assert not p.requires_grad, f"主干参数 {name} 未被冻结"
        else:
            assert p.requires_grad, f"新增模块参数 {name} 被误冻结"
            trainable_prefixes.add(name.split(".")[0] + ".")
    assert trainable_prefixes == {"scpl.", "cdf.", "text_adapter."}
    assert pcdf.encoder_rgb.training is False
    assert pcdf.head.training is False


def test_stage1_text_mode_skips_pcd(setup) -> None:
    """论文 Fig. 10 Stage I：文本引导下主路径不经过 PCD。

    除了"没有梯度"，实现还按需跳过三支解耦的前向本身——阶段 1 因此省掉约
    三分之一计算量。
    """
    _, pcdf, _ = setup
    pcdf.freeze_detector()
    pcdf.train()
    pcdf.zero_grad(set_to_none=True)
    cond = torch.tensor([[0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1]])
    out = pcdf(torch.rand(2, 6, 128, 128), cond=cond, guidance="text")
    assert out["fs"] is None and out["fv_rgb"] is None and out["fs_gap"] is None
    out["guidance"].sum().backward()
    for branch in (pcdf.v_rgb, pcdf.v_ir, pcdf.s):
        assert all(p.grad is None for p in branch.parameters())
    gate_grad = pcdf.scpl.gate.mlp[0].weight.grad
    assert gate_grad is not None and gate_grad.abs().sum() > 0
