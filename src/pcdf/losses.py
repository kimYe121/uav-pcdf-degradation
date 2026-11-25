"""PCD 解耦损失（论文式 6-9）。

``L_dt``  —— 条件特定特征与文本条件嵌入之间的中心矩差异（式 6）；
``L_irr`` —— 条件不变分支与条件特定分支交叉协方差的 Frobenius 范数（式 7）；
``L_dc``  —— 同一 OBB head 分别作用于两个 ``F_v`` 得到的检测损失（式 8），
             由训练循环调用 :func:`pcdf.models.heads.make_criterion` 计算。

本模块只包含论文新增的三个正则项。检测损失本身是 ultralytics ``v8OBBLoss``
（见 :mod:`pcdf.models.heads`），不在论文的贡献范围内，也不在这里改写。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


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
    """无关损失 ``L_irr``（式 7）：``||F_v^T F_s||_F²``，两个模态相加。

    论文原式没写归一化。照抄的后果是量纲随分辨率爆炸：P3（64×80）原始值
    是 6.7e9，λ₂=0.003 直接把梯度炸掉；而 P5 只有 5.2e7，三层根本不可加。
    因此这里把内积按**像素数**归一，即用协方差而不是原始和：

        ||F_v^T F_s / N||_F² ,  N = H·W

    这样三种分辨率下的量级一致（实测 P3/P4/P5 ≈ 255/512/510，与 C 同阶），
    且 λ₂=0.003 仍是论文值——贡献约 1.5，与检测损失同一量级，正好起到
    "把 F_v 与 F_s 拉开"的正则作用。

    逐像素 L2 归一后再算协方差会让该项塌到 ~0.01，正则强度消失，故不做。
    三层取平均：论文按层定义，未说明跨尺度合并方式，取均值最稳。
    """
    if not (len(fv_rgb) == len(fv_ir) == len(fs)):
        raise ValueError("L_irr 需要每个尺度各一份 F_v,rgb / F_v,ir / F_s")
    total = fs[0].new_zeros(())
    for xr, xi, xs in zip(fv_rgb, fv_ir, fs):
        if not (xr.shape == xi.shape == xs.shape):
            raise ValueError("L_irr 要求 F_v 与 F_s 形状完全一致")
        # 半精度下 C×C 协方差会溢出，统一升到 fp32。
        xr = xr.float().flatten(2)
        xi = xi.float().flatten(2)
        xs = xs.float().flatten(2)
        n = xr.shape[-1]
        cov_r = (xr.transpose(1, 2) @ xs) / n  # (B, C, C)
        cov_i = (xi.transpose(1, 2) @ xs) / n
        total = total + cov_r.pow(2).sum(dim=(1, 2)).mean()
        total = total + cov_i.pow(2).sum(dim=(1, 2)).mean()
    return total / len(fs)


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