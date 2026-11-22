"""样本级条件提示学习 SCPL（论文 Sec. 4.2，式 1-4）。

一个 batch 的处理流程：

1. 按式 1 构造初始提示 ``I_i``，用冻结 CLIP 编码；
2. 两模态 P5 特征分别全局池化，与 ``I_i`` 拼接后送入 ``Fc``，得到六个属性的
   可用概率（式 2）；
3. 以 ``tau`` 阈值得到硬掩码 ``G_i``，丢弃被屏蔽的属性块（式 3）；
4. 重新编码掩码后的完整提示，得到条件嵌入 ``F_t``（式 4）。

SCPL 只在训练阶段使用（Sec. 4.2）。测试时 CDF 由 PCD 产生的 ``F_s`` 驱动，
不再读取任何条件文本。

梯度路径
--------
硬指示函数不可微。为了让 ``Fc`` 真正收到阶段 1 的检测梯度，前向用硬掩码
构造提示，反向通过 ``Σ p_n · E(block_n)`` 把梯度传回属性概率（straight-
through）。阶段 2 冻结 SCPL 后该路径无意义，由
:meth:`pcdf.models.pcdf.PCDFNet.freeze_scpl` 显式关闭，省掉每批 6N 次
单属性提示编码。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..conditions import ConditionTable
from ..text_encoder import CLIPTextEncoder, PromptEmbeddingCache
from .blocks import SCPLGate


class SCPL(nn.Module):
    """可学习的硬门控提示构造器。"""

    def __init__(
        self,
        table: ConditionTable,
        text_encoder: CLIPTextEncoder,
        feat_dim: int,
        text_dim: int = 512,
        hidden: int = 256,
        tau: float = 0.15,
        pool: str = "max",
        cache_size: int = 8192,
    ) -> None:
        super().__init__()
        self.table = table
        self.pool = pool
        self.gate = SCPLGate(
            feat_dim=feat_dim,
            text_dim=text_dim,
            n_attrs=table.n_attributes,
            hidden=hidden,
            tau=tau,
        )
        self.cache = PromptEmbeddingCache(text_encoder, max_size=cache_size)
        self.text_dim = text_dim
        # 是否启用 straight-through 的软梯度路径；阶段 2 冻结门控后关闭。
        self.soft_gradient = True

    def _pool(self, x: Tensor) -> Tensor:
        """按论文式 2 的 ``Pool`` 做全局池化，默认 max。"""
        if self.pool == "max":
            return F.adaptive_max_pool2d(x, 1).flatten(1)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(
        self,
        feat_rgb: Tensor,
        feat_ir: Tensor,
        ids: Tensor,
        force_all: bool = False,
    ) -> dict[str, Tensor]:
        """``feat_*`` 为 P5 特征图，``ids`` 为 ``(B, N)`` 条件 id。"""
        ids_list = ids.detach().to("cpu").tolist()
        prompts = [self.table.prompt_from_ids(row) for row in ids_list]
        init_embedding = self.cache.encode(prompts)

        mask_ste, mask_hard, probs = self.gate(
            self._pool(feat_rgb), self._pool(feat_ir), init_embedding, force_all=force_all
        )

        # 硬掩码决定实际送入 CLIP 的句子。
        masked_prompts = [
            self.table.prompt_from_ids(row, [bool(v) for v in mask])
            for row, mask in zip(ids_list, mask_hard.detach().to("cpu").tolist())
        ]
        ft_hard = self.cache.encode(masked_prompts)

        # straight-through：前向值是硬提示的嵌入，梯度通过软属性混合传回 Fc。
        if self.soft_gradient and not force_all:
            blocks = self.cache.encode_blocks(self.table, ids)  # (B, N, text_dim)
            ft_soft = (probs.unsqueeze(-1) * blocks).sum(dim=1)
            ft_ste = ft_hard + (ft_soft - ft_soft.detach())
        else:
            ft_ste = ft_hard

        return {
            "ft": ft_hard,
            "ft_ste": ft_ste,
            "gate_probs": probs,
            "gate_mask": mask_hard,
            "kept": mask_hard.sum(dim=1),
        }
