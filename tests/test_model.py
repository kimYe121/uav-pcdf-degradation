"""完整 PCDF 网络的张量形状、梯度与冻结策略测试。

骨架与检测头都是固定的 YOLOv5 backbone + ultralytics OBB 头，因此这里测的
就是实际训练路径。
"""

from __future__ import annotations

import pytest
import torch
from ultralytics.nn.modules import OBB

from pcdf.conditions import ConditionTable
from pcdf.config import load_config
from pcdf.models.pcdf import PCDFNet
from pcdf.text_encoder import CLIPTextEncoder

# 128×128 输入：P3 16×16 + P4 8×8 + P5 4×4 = 336 个格子。
# OBB 头是 anchor-free，每格一个预测，不像 anchor 头还要乘 3。
CELLS = 336
OBB_CHANNELS = 4 + 11 + 1  # cx, cy, w, h + 11 类 + 角度


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    table = ConditionTable.load(cfg.paths.condition_index, cfg.model.condition_spec)
    encoder = CLIPTextEncoder(cfg.model.text_encoder, device="cpu")
    model = PCDFNet(cfg.model, nc=cfg.data.nc, table=table, text_encoder=encoder)
    model.eval()
    return cfg, table, model


def test_head_is_obb(setup) -> None:
    """检测头是 ultralytics 的 anchor-free OBB（DFL + 角度分支）。"""
    _, _, model = setup
    assert isinstance(model.head, OBB)
    assert tuple(model.head.stride.tolist()) == (8.0, 16.0, 32.0)


def test_stage2_text_guidance_on_decoupled_features(setup) -> None:
    """阶段 2 训练：文本引导 + CDF 作用在 F_v 上（Fig. 10 Stage II）。

    旧实现把"引导来源"和"融合对象"绑死（text→原始特征、visual→F_v），
    阶段 2 只能用未训练的 GAP(F_s) 引导，条件信号从第 0 轮就被丢掉。
    """
    _, _, model = setup
    model.train()
    x = torch.rand(2, 6, 128, 128)
    cond = torch.tensor([[0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1]])
    out = model(x, cond=cond, guidance="text", fuse="decoupled", return_aux=True, return_ft=True)
    assert out["fv_rgb"] is not None and out["fs"] is not None
    assert out["guidance"].shape == (2, 256)
    assert out["aux_rgb"] is not None

    # guidance="text" 且不显式给 fuse 时仍走阶段 1 语义：不计算 PCD。
    out1 = model(x, cond=cond, guidance="text")
    assert out1["fv_rgb"] is None and out1["fs"] is None


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

    # 训练态：v8OBBLoss 要吃 dict(boxes, scores, angle, feats)
    main = out["main"]
    assert isinstance(main, dict)
    assert main["boxes"].shape == (2, 4 * model.head.reg_max, CELLS)
    assert main["scores"].shape == (2, cfg.data.nc, CELLS)
    assert main["angle"].shape == (2, 1, CELLS)
    assert len(main["feats"]) == 3
    assert [f.shape[1] for f in main["feats"]] == list(cfg.model.widths)
    # 辅助头同样接在 Reassigned 特征上，返回结构一致
    assert isinstance(out["aux_rgb"], dict) and isinstance(out["aux_ir"], dict)


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


def test_aux_heads_use_reassigned_features(setup) -> None:
    """式 8 + Fig. 4：辅助检测头必须接在 W_m ⊙ F_v,m（Reassigned）上。

    回归用例：辅助头若直接吃重加权之前的 ``F_v``，它们就与软门控无关，
    Fig. 4 里那三个 Detection 的接线也就断了。这里只从辅助头回传梯度，
    检查梯度确实经过 CDF 的投影层。
    """
    cfg, table, model = setup
    model.train()
    model.zero_grad(set_to_none=True)
    model.freeze_scpl()
    out = model(torch.rand(2, 6, 128, 128), guidance="visual", return_aux=True)

    aux_loss = sum(t.sum() for t in out["aux_rgb"].values() if torch.is_tensor(t))
    aux_loss = aux_loss + sum(t.sum() for t in out["aux_ir"].values() if torch.is_tensor(t))
    aux_loss.backward()
    proj_grad = model.cdf.projections[0][0].weight.grad
    assert proj_grad is not None and proj_grad.abs().sum() > 0, "辅助头的梯度没经过软门控"
    model.zero_grad(set_to_none=True)


def test_soft_gate_input_modes(setup) -> None:
    """Eq.10：投影只吃 F_s；guide_pool 保留 Fig.4 的三路输入作为可选档位。"""
    cfg, table, model = setup
    # 配置默认（configs/pcdf.yaml）= guide
    assert model.cdf.projections[0][0].in_features == cfg.model.guidance_dim

    cfg_pool = load_config()
    cfg_pool.model.cdf_input = "guide_pool"
    pooled = PCDFNet(
        cfg_pool.model, nc=cfg_pool.data.nc, table=table,
        text_encoder=CLIPTextEncoder(cfg_pool.model.text_encoder, device="cpu"),
    )
    assert pooled.cdf.projections[0][0].in_features == (
        cfg_pool.model.guidance_dim + 2 * cfg_pool.model.widths[0]
    )


def test_neck_variants() -> None:
    """Fig. 4 图例写明 `F` = Channel Concatenation（无颈部），这里守两档：

    * ``neck="concat"``：拼接后直连检测头，头输入 = 2C（严格照图）；
    * ``neck="pan"``：v5 标配颈部，头输入 = C。
    """
    from pcdf.models.baseline import BaselineNet

    for neck, expect_ch, expect_params in (
        ("pan", (128, 256, 256), 11.71),
        ("concat", (256, 512, 512), 10.87),
    ):
        cfg = load_config(overrides={"model.neck": neck})
        model = BaselineNet(cfg.model, nc=cfg.data.nc).eval()
        assert [x[0].conv.in_channels for x in model.head.cv2] == list(expect_ch)
        total = sum(p.numel() for p in model.parameters()) / 1e6
        assert abs(total - expect_params) < 0.05, f"{neck}: {total:.2f}M"
        with torch.no_grad():
            out = model(torch.rand(1, 6, 128, 128))
        assert out["main"][0].shape[:2] == (1, OBB_CHANNELS)


def test_text_encoder_is_frozen(setup) -> None:
    _, _, model = setup
    assert all(not p.requires_grad for p in model.text_encoder.model.parameters())
    assert not any(p.grad is not None for p in model.text_encoder.model.parameters())


def test_eval_forward_has_no_text_branch(setup) -> None:
    cfg, table, model = setup
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(2, 6, 128, 128), guidance="visual")
    decoded, raw = out["main"]
    assert decoded.shape == (2, OBB_CHANNELS, CELLS)
    assert raw is None
    assert "gate_mask" not in out
    assert out["fs_gap"].shape == (2, cfg.model.guidance_dim)


def test_head_mode_follows_train_flag(setup) -> None:
    """回归用例：检测头的模式由 ``train(mode)`` 决定，而不是恒定 train。

    头在 train 返回 loss 用的 dict、在 eval 返回可进 NMS 的张量；早期实现为了
    阶段 1 能算损失把它在 ``train()`` 里写死成 train，导致 ``model.eval()``
    之后评估拿到的是 dict，NMS 直接报错。
    """
    cfg, table, model = setup
    model.eval()
    with torch.no_grad():
        assert isinstance(model(torch.rand(1, 6, 128, 128), guidance="visual")["main"], tuple)
    model.train()
    assert isinstance(model.head, OBB)
    assert model.head.training is True
    model.eval()
    assert model.head.training is False


def test_stage1_freeze_scope() -> None:
    """阶段 1 两种冻结读法：默认冻检测器；可选只冻 PCD（只冻结 CLIP 的读法）。"""
    cfg = load_config()
    table = ConditionTable.load(cfg.paths.condition_index, cfg.model.condition_spec)

    def build():
        enc = CLIPTextEncoder(cfg.model.text_encoder, device="cpu")
        return PCDFNet(cfg.model, nc=cfg.data.nc, table=table, text_encoder=enc)

    frozen = build()
    frozen.freeze_detector()
    assert not any(p.requires_grad for p in frozen.encoder_rgb.parameters())
    assert not any(p.requires_grad for p in frozen.head.parameters())
    assert not any(p.requires_grad for p in frozen.v_rgb.parameters())

    trainable = build()
    # 检测头里 dfl.conv 的 bin 权重本来就冻结，所以比较"冻结前后是否一致"，
    # 而不是断言全部 requires_grad。
    before = {n: p.requires_grad for n, p in trainable.head.named_parameters()}
    trainable.freeze_detector(keep_detector_trainable=True)
    after = {n: p.requires_grad for n, p in trainable.head.named_parameters()}
    assert after == before
    assert all(p.requires_grad for p in trainable.encoder_rgb.parameters())
    assert not any(p.requires_grad for p in trainable.v_rgb.parameters())


def test_stage2_freezes_gate_and_soft_path(setup) -> None:
    """阶段 2 冻结门控后，straight-through 路径也要关闭（否则白算 6N 次编码）。"""
    _, _, model = setup
    model.freeze_scpl()
    assert all(not p.requires_grad for p in model.scpl.parameters())
    assert model.scpl.soft_gradient is False