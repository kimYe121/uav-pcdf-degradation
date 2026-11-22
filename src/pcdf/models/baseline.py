"""双流 concat 基线检测器（论文 Sec. 5.2 / Table 4 的 Baseline，58.4%）。

原文定义："Our baseline is a one-stage dual-stream detector that integrates
two modalities through concatenation fusion."

结构上与 :class:`pcdf.models.pcdf.PCDFNet` 共享全部基础设施：

- 双流 backbone（``encoder_rgb`` / ``encoder_ir``，键名一致）；
- :class:`PANNeck` 的 ``reduce``（1×1，2C→C）就是"拼接融合"的可学习部分；
- 同一个检测头（由 ``model.head_type`` 决定）。

因此 baseline 的 ``state_dict`` 是 PCDFNet 的子集，可直接作为论文三阶段训练
协议的起点：补充材料 B.3 规定阶段 1 只训练 SCPL 与 CDF，主干必须从预训练的
baseline 权重出发。

训练协议（对齐论文）：
    stage 0（本模块）→ stage 1（SCPL+CDF，主干冻结）→ stage 2（full pipeline）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..config import ModelConfig
from .blocks import PANNeck
from .encoder import Backbone
from .heads import build_head, initialise
from .anchor_head import AnchorOBBHead


class BaselineNet(nn.Module):
    """双流 concat 融合的单阶段 OBB 检测器（论文 Baseline）。"""

    arch = "baseline"

    def __init__(self, cfg: ModelConfig, nc: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.nc = int(nc)
        self.widths = tuple(int(c) for c in cfg.widths)

        weights = cfg.encoder_weights if cfg.pretrained_backbone else None
        # 键名与 PCDFNet 一致，state_dict 可无缝迁移。
        self.encoder_rgb = Backbone(weights=weights, nc=nc, yaml=cfg.backbone_yaml, scale=cfg.backbone_scale)
        self.encoder_ir = Backbone.from_backbone(self.encoder_rgb)
        if tuple(self.encoder_rgb.out_channels) != self.widths:
            raise RuntimeError(
                f"配置 widths={self.widths} 与 backbone 输出 "
                f"{self.encoder_rgb.out_channels} 不一致；请检查 profile"
            )

        # PANNeck 的 reduce（1×1，2C→C）承担 concat 融合。
        self.neck = PANNeck(self.widths)
        self.head = build_head(cfg, self.nc)
        initialise(self, self.head)

    @property
    def stride(self) -> Tensor:
        return self.head.stride

    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,  # noqa: ARG002 —— baseline 不使用条件
        guidance: str = "visual",  # noqa: ARG002
        return_aux: bool = False,  # noqa: ARG002
        return_ft: bool = False,  # noqa: ARG002
        force_all_gates: bool = False,  # noqa: ARG002
    ) -> dict[str, Tensor]:
        """``x`` 为 ``(B, 6, H, W)``：通道 0-2 RGB，3-5 IR。

        额外参数与 :meth:`PCDFNet.forward` 对齐并被忽略，使训练/评估循环
        无需按模型类型分支调用。
        """
        if x.shape[1] != 6:
            raise ValueError(f"期望 6 通道 RGB+IR 输入，实际为 {tuple(x.shape)}")
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        feats_rgb = self.encoder_rgb(x[:, :3])
        feats_ir = self.encoder_ir(x[:, 3:])
        fused = [torch.cat([r, i], dim=1) for r, i in zip(feats_rgb, feats_ir)]
        head_out = self.head(self.neck(fused))
        if isinstance(self.head, AnchorOBBHead):
            # 统一成 (decoded, raw)：评估取 [0]，损失取 [1]。
            return {"main": (self.head.decode(head_out), head_out)}
        return {"main": head_out}


def load_baseline_into_pcdf(
    pcdf: nn.Module,
    checkpoint: dict,
    *,
    strict: bool = True,
) -> dict[str, list[str]]:
    """把 baseline 检查点的权重装入 PCDFNet（论文阶段 1 的起点）。

    baseline 与 PCDFNet 共享 ``encoder_rgb/encoder_ir/neck/head`` 的键名；
    ``missing`` 只允许出现 PCDF 新增模块（SCPL/PCD/CDF/text_adapter），
    ``unexpected`` 必须为空——否则说明两个模型的骨架不一致，直接报错。

    Returns:
        ``{"missing": [...], "unexpected": [...]}``。
    """
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = pcdf.load_state_dict(state, strict=False)
    allowed_prefixes = ("scpl.", "text_adapter.", "cdf.", "v_rgb.", "v_ir.", "s.")
    bad_missing = [k for k in missing if not k.startswith(allowed_prefixes)]
    if strict and (bad_missing or unexpected):
        raise RuntimeError(
            "baseline 检查点与 PCDFNet 骨架不匹配：\n"
            f"  missing（非新增模块）: {bad_missing[:8]}\n"
            f"  unexpected: {unexpected[:8]}"
        )
    return {"missing": list(missing), "unexpected": list(unexpected)}
