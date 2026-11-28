"""SCPL / PCD / CDF 的基础模块（论文 Sec. 4.2-4.4）。

每个类都标注对应公式。论文未展开的结构（如 1×1 + 3×3 的具体组合）采用最
简单的标准选择，并统一记录在 ``docs/ARCHITECTURE.md``，不在代码里悄悄调参。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ultralytics.nn.modules import C3k2, Conv


class DecoupleBranch(nn.Module):
    """三支解耦网络中的单个编码器（论文式 5）。

    结构为 1×1 通道混合 + 3×3 空间卷积，输出通道保持不变，这样 ``F_s`` 与
    ``F_v`` 可以直接计算交叉协方差（式 7）。``S`` 分支的输入是两模态拼接，
    所以允许 ``in_channels=2C``。
    """

    def __init__(self, channels: int, in_channels: int | None = None) -> None:
        super().__init__()
        in_channels = channels if in_channels is None else in_channels
        self.block = nn.Sequential(
            Conv(in_channels, channels, k=1),
            Conv(channels, channels, k=3, p=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SCPLGate(nn.Module):
    """SCPL 的硬门控网络 ``Fc``（论文式 2）。

    ``Fc`` 输入为 RGB/IR P5 特征的全局 maxpool、以及初始提示 ``I_i`` 的 CLIP
    嵌入；输出六个属性的可用概率，再按 ``tau``（论文 0.15）二值化。

    硬指示函数不可微。本实现用 straight-through：前向值等于硬 0/1 掩码，
    反向梯度等于 softmax 概率，使阶段 1 的 CDF 检测损失能训练门控网络。
    """

    def __init__(
        self,
        feat_dim: int,
        text_dim: int,
        n_attrs: int,
        hidden: int = 256,
        tau: float = 0.15,
    ) -> None:
        super().__init__()
        self.tau = float(tau)
        # 输入维度 = 两模态池化特征 + 初始提示嵌入
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_attrs),
        )

    def forward(
        self,
        pool_rgb: Tensor,
        pool_ir: Tensor,
        init_embedding: Tensor,
        force_all: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """返回 ``(straight-through 掩码, 硬掩码, 属性概率)``。"""
        logits = self.mlp(torch.cat([pool_rgb, pool_ir, init_embedding], dim=1))
        probs = logits.softmax(dim=-1)
        if force_all:
            hard = torch.ones_like(probs)
        else:
            hard = (probs >= self.tau).float()
            empty = hard.sum(dim=1) == 0
            if empty.any():
                # 六个属性全被屏蔽会产生空提示，丢失全部条件引导；
                # 对这些样本保留概率最大的属性。
                keep = probs[empty].argmax(dim=1)
                hard[empty] = 0.0
                hard[empty, keep] = 1.0
        # straight-through：值等于 hard，梯度等于 probs。
        mask_ste = hard + probs - probs.detach()
        return mask_ste, hard, probs


class ChannelNormSoftGate(nn.Module):
    """CDF 的逐通道归一化软门控（论文式 10-11，Fig. 4）。

    输入按 Fig. 4 的箭头是**三路**：条件引导（绿）、条件不变 RGB 特征（蓝）、
    条件不变 IR 特征（橙）——软门控要"看着当前特征"再决定这个通道该信谁，
    只喂引导会让权重与特征内容脱钩。因此每层的投影输入是
    ``[guidance, GAP(F_v,rgb), GAP(F_v,ir)]``。

    每个尺度投影出 ``2*C_l`` 个 logits，reshape 成 ``(B, 2, C_l, 1, 1)`` 后在
    模态维做 softmax：``W_rgb + W_ir = 1``（逐通道），与式 10 一致。

    注意：softmax 沿"模态"维，不是沿通道维；这样才得到论文中 ``[0,1]`` 内、
    两模态互补的权重曲线（Fig. 4 中间那张以 0.5 为中心的图）。
    """

    def __init__(
        self, guidance_dim: int, channels: tuple[int, int, int], mode: str = "guide_pool"
    ) -> None:
        super().__init__()
        self.mode = mode
        self.projections = nn.ModuleList()
        for c in channels:
            hidden = max(c // 4, 32)
            in_dim = guidance_dim + 2 * c if mode == "guide_pool" else guidance_dim
            self.projections.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 2 * c),
                )
            )

    def forward(
        self, guidance: Tensor, fv_rgb: list[Tensor], fv_ir: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """返回 ``(融合特征, IR 权重, 重加权 RGB, 重加权 IR)``。

        后两项供式 8 的辅助检测头使用：Fig. 4 里三个 Detection 都接在
        *Reassigned* 特征上，而不是重加权之前的 ``F_v``。
        """
        fused: list[Tensor] = []
        weights: list[Tensor] = []
        weighted_rgb: list[Tensor] = []
        weighted_ir: list[Tensor] = []
        for proj, xr, xi in zip(self.projections, fv_rgb, fv_ir):
            b, c = xr.shape[:2]
            if self.mode == "guide_pool":
                pool_r = xr.mean(dim=(2, 3))
                pool_i = xi.mean(dim=(2, 3))
                inp = torch.cat([guidance, pool_r, pool_i], dim=1)
            else:
                inp = guidance
            logits = proj(inp).view(b, 2, c, 1, 1)
            w = logits.softmax(dim=1)  # 每个通道在两个模态间归一化
            wr, wi = w[:, 0] * xr, w[:, 1] * xi
            weighted_rgb.append(wr)
            weighted_ir.append(wi)
            fused.append(torch.cat([wr, wi], dim=1))
            weights.append(w[:, 1, :, 0, 0])
        return fused, weights, weighted_rgb, weighted_ir


class PANNeck(nn.Module):
    """融合特征上的轻量 PAN neck。

    这是 implementation choice：论文没有说明 neck 与融合尺度。本实现在
    P3/P4/P5 三层融合，做一次 top-down + 一次 bottom-up，使 OBB head 看到
    与编码器一致的三层 stride。
    """

    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        c3, c4, c5 = channels
        # 融合特征是两模态拼接的 2C 通道，先 1×1 压回 C。
        self.reduce = nn.ModuleList([Conv(2 * c, c, k=1) for c in channels])
        self.td4 = C3k2(c4 + c5, c4, n=1)
        self.td3 = C3k2(c3 + c4, c3, n=1)
        self.down3 = Conv(c3, c3, k=3, s=2)
        self.bu4 = C3k2(c4 + c3, c4, n=1)
        self.down4 = Conv(c4, c4, k=3, s=2)
        self.bu5 = C3k2(c5 + c4, c5, n=1)
        self.out_channels = list(channels)

    def forward(self, fused: list[Tensor]) -> list[Tensor]:
        p3, p4, p5 = [reduce_(x) for reduce_, x in zip(self.reduce, fused)]
        p4 = self.td4(
            torch.cat([p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")], dim=1)
        )
        p3 = self.td3(
            torch.cat([p3, F.interpolate(p4, size=p3.shape[-2:], mode="nearest")], dim=1)
        )
        p4 = self.bu4(torch.cat([p4, self.down3(p3)], dim=1))
        p5 = self.bu5(torch.cat([p5, self.down4(p4)], dim=1))
        return [p3, p4, p5]


class GuidanceAdapter(nn.Module):
    """把 SCPL 文本嵌入投影到 CDF 的引导空间。

    阶段 1 用 ``text_adapter(F_t)`` 驱动 CDF；阶段 2 用 ``L_dt`` 把
    ``GAP(F_s)`` 对齐到同一个 256 维空间。这样阶段 1 学到的 CDF 投影器在
    测试时可以直接接收视觉引导，不需要条件标注。
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
