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
    # OBB 头 anchor-free：三尺度共 16·16+8·8+4·4 = 336 个预测
    assert decoded.shape == (2, 4 + cfg.data.nc + 1, 336)
    # 推理态不保留原始输出，raw 恒为 None
    assert raw is None


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
    # 检测头是例外：它冻的是参数，模式由 train(mode) 决定（train 态要返回
    # v8OBBLoss 需要的 dict），因此这里应处于 train 模式。
    assert pcdf.head.training is True
    assert all(not p.requires_grad for p in pcdf.head.parameters())

    pcdf.eval()  # 评估时必须切回 eval，否则 NMS 拿到的是 dict
    assert pcdf.head.training is False


def test_stage0_all_params_trainable(setup) -> None:
    """阶段 0 baseline 是全参数训练。

    回归用例：Ultralytics 的 ``.pt`` 把参数存成 ``requires_grad=False``，直接
    搬层会让两个 encoder（8.6M）静默冻结，只剩 neck+head 在学。

    唯一的例外是 DFL 的积分核 ``head.dfl.conv.weight``（16 个数）：它是固定的
    期望算子，不是可学参数，ultralytics 在 ``DFL.__init__`` 里就置了
    ``requires_grad=False``。这里白名单只放它一个，其它任何冻结都算回归。
    """
    _, _, baseline = setup
    frozen = [n for n, p in baseline.named_parameters() if not p.requires_grad]
    assert frozen == ["head.dfl.conv.weight"], f"意外冻结的参数：{frozen}"
    for branch in (baseline.encoder_rgb, baseline.encoder_ir, baseline.neck, baseline.head):
        assert any(p.requires_grad for p in branch.parameters())
    # 两个 encoder 必须完全可训练（8.6M），否则就是 .pt 权重把梯度关掉了
    enc = sum(p.numel() for p in baseline.encoder_rgb.parameters())
    assert enc > 4e6 and all(p.requires_grad for p in baseline.encoder_rgb.parameters())


def test_stage2_keeps_encoders_trainable(setup) -> None:
    """阶段 2 冻结 SCPL 与 text adapter，双流编码器必须仍然可训练。

    用同一个 CLIP 实例新建"刚构建"的网络，避免共用 fixture 已被前面的用例
    ``freeze_detector()`` 之后的冻结状态干扰判定。
    """
    cfg, pcdf, _ = setup
    table = ConditionTable.load(cfg.paths.condition_index, cfg.model.condition_spec)
    fresh = PCDFNet(cfg.model, nc=cfg.data.nc, table=table, text_encoder=pcdf.text_encoder)
    fresh.freeze_scpl()
    fresh.freeze_text_adapter()
    for branch in (fresh.encoder_rgb, fresh.encoder_ir, fresh.v_rgb, fresh.v_ir, fresh.s, fresh.cdf):
        assert any(p.requires_grad for p in branch.parameters())
    assert all(not p.requires_grad for p in fresh.scpl.parameters())
    assert all(not p.requires_grad for p in fresh.text_adapter.parameters())
    trainable = sum(p.numel() for p in fresh.parameters() if p.requires_grad)
    assert trainable > 15e6, f"阶段 2 可训练参数只有 {trainable/1e6:.2f}M"


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
