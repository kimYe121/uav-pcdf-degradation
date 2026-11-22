"""PCD 解耦损失（论文式 6-9）与检测损失装配。

``L_dt``  —— 条件特定特征与文本条件嵌入之间的中心矩差异（式 6）；
``L_irr`` —— 条件不变分支与条件特定分支交叉协方差的 Frobenius 范数（式 7）；
``L_dc``  —— 同一 OBB head 分别作用于两个 ``F_v`` 得到的检测损失（式 8），
             由训练循环调用本模块的 criterion 计算。

检测损失的增益（box/cls/obj/dfl）来自 ``ModelConfig.loss``；两套头量纲不同，
因此不共用一组数字。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8OBBLoss

from .config import ModelConfig
from .models.anchor_head import AnchorOBBHead, AnchorOBBLoss


def cmd_loss(x: Tensor, y: Tensor, orders: int = 5) -> Tensor:
    """中心矩差异 CMD（Zellinger et al., ICLR 2017；式 6）。

    输入 ``x``、``y`` 为 ``(B, D)``；期望与中心矩沿 batch 维统计，并按随机
    变量的取值范围 ``|b-a|`` 归一化。``B < 2`` 时高阶中心矩恒为 0，训练循环
    不会在单样本 batch 上使用该损失。
    """
    x = x.flatten(1).float()
    y = y.flatten(1).float()
    if x.shape != y.shape:
        raise ValueError(f"CMD 需要形状一致，得到 {tuple(x.shape)} 与 {tuple(y.shape)}")
    lo = torch.minimum(x.min(), y.min())
    hi = torch.maximum(x.max(), y.max())
    span = (hi - lo).clamp_min(1e-6)

    mx, my = x.mean(dim=0), y.mean(dim=0)
    # 式 6：1/|b-a|·||E(x)-E(y)||_2 + Σ_k 1/|b-a|^k·||C_k(x)-C_k(y)||_2，
    # 每一项都是差向量的 L2 范数（对维度求和后开方），不是逐维均方。
    loss = (mx - my).norm(p=2) / span
    for k in range(2, orders + 1):
        cx = (x - mx).pow(k).mean(dim=0)
        cy = (y - my).pow(k).mean(dim=0)
        loss = loss + (cx - cy).norm(p=2) / span.pow(k)
    return loss


def irr_loss(fv_rgb: list[Tensor], fv_ir: list[Tensor], fs: list[Tensor]) -> Tensor:
    """无关损失 ``L_irr``，在 P3/P4/P5 三层上平均（式 7）。

    每层计算 ``||F_v^T F_s||_F^2 / (C·H·W)``，RGB/IR 两支相加。除以
    ``C·H·W`` 是本实现的数值归一化：原始 Frobenius 范数会随空间尺寸增长，
    直接使用会压过检测损失。该选择记录在 docs/ARCHITECTURE.md。
    """
    if not (len(fv_rgb) == len(fv_ir) == len(fs)):
        raise ValueError("L_irr 需要每个尺度各一份 F_v,rgb / F_v,ir / F_s")
    total = fs[0].new_zeros(())
    for xr, xi, xs in zip(fv_rgb, fv_ir, fs):
        if not (xr.shape == xi.shape == xs.shape):
            raise ValueError("L_irr 要求 F_v 与 F_s 形状完全一致")
        b, c, h, w = xr.shape
        # 半精度下 C×C 协方差可能溢出，统一升到 fp32。
        xr, xi, xs = xr.float(), xi.float(), xs.float()
        cov_r = xr.flatten(2).transpose(1, 2) @ xs.flatten(2)  # (B, C, C)
        cov_i = xi.flatten(2).transpose(1, 2) @ xs.flatten(2)
        denom = float(c * h * w)
        total = total + cov_r.pow(2).sum(dim=(1, 2)).mean() / denom
        total = total + cov_i.pow(2).sum(dim=(1, 2)).mean() / denom
    return total / len(fs)


class LossAdapter(nn.Module):
    """把共享检测头包装成 Ultralytics 损失函数可读取的"模型"。

    ``v8OBBLoss`` 会读取 ``model.model[-1]`` 的 ``reg_max`` / ``stride`` / ``nc``
    与 ``model.args`` 的损失增益；适配器只注册 head，避免构造一个假的完整模型。
    """

    def __init__(self, head: nn.Module, args) -> None:
        super().__init__()
        self.model = nn.ModuleList([head])
        self.args = args


def make_criterion(head: nn.Module, cfg: ModelConfig) -> nn.Module:
    """按头类型与配置创建检测损失。

    - :class:`AnchorOBBHead`：式 8 的 ``L_obj + L_cls + L_reg``（ProbIoU）；
    - Ultralytics ``OBB``：``v8OBBLoss``（分类 + 回归 + DFL + 角度）。

    v8 损失仍以 Ultralytics 的完整默认参数为基底，只覆盖三个增益：该损失
    还读取 label_smoothing、overlap 等字段，用精简对象替换容易在升级后缺字段。
    """
    if isinstance(head, AnchorOBBHead):
        return AnchorOBBLoss.from_config(head, cfg.loss)
    args = get_cfg()
    args.box = cfg.loss.v8_box
    args.cls = cfg.loss.v8_cls
    args.dfl = cfg.loss.v8_dfl
    return v8OBBLoss(LossAdapter(head, args))


def decoupling_terms(
    fs_gap: Tensor,
    ft: Tensor,
    text_adapter: nn.Module,
    fv_rgb: list[Tensor],
    fv_ir: list[Tensor],
    fs: list[Tensor],
    lambda_dt: float,
    lambda_irr: float,
    lambda_dc: float,
    l_dc: Tensor,
) -> dict[str, Tensor]:
    """组装 ``L_dec = λ1·L_dt + λ2·L_irr + λ3·L_dc``（式 9）。

    ``fs_gap`` 是 P5 ``F_s`` 的全局平均池化；``text_adapter(F_t)`` 是阶段 1
    文本引导所在的同一个引导空间。两者维度一致，CMD 才有定义。
    """
    l_dt = cmd_loss(fs_gap, text_adapter(ft))
    l_irr = irr_loss(fv_rgb, fv_ir, fs)
    total = lambda_dt * l_dt + lambda_irr * l_irr + lambda_dc * l_dc
    return {"l_dt": l_dt, "l_irr": l_irr, "l_dc": l_dc, "l_dec": total}
