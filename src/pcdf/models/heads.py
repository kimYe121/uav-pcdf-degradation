"""检测头工厂：把"按配置选头 + 初始化顺序"集中到一处。

两套头（Ultralytics ``OBB`` 与 :class:`AnchorOBBHead`）的构造细节不同，
但调用方（baseline / PCDFNet）需要完全一致的语义：

* 由 ``model.head_type`` 决定用哪个头；
* 先做模型级 ``initialize_weights``，再补偿被清零的先验（objectness bias）。

以前这两段在 ``baseline.py`` 与 ``pcdf.py`` 里各写一遍，改一处容易漏另一处。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.head import OBB
from ultralytics.nn.tasks import initialize_weights

from ..config import ModelConfig
from .anchor_head import AnchorOBBHead

__all__ = ["build_head", "initialise"]


def build_head(cfg: ModelConfig, nc: int) -> nn.Module:
    """按配置构造检测头（不负责初始化权重）。"""
    if cfg.head_type == "anchor":
        # 论文式 8 含 L_obj，对应 anchor-based 头。
        return AnchorOBBHead(nc=nc, ch=cfg.widths, anchors=cfg.anchors)
    head = OBB(nc=nc, ch=cfg.widths, reg_max=cfg.reg_max)
    head.stride = torch.tensor([float(s) for s in cfg.strides])
    head.bias_init()
    return head


def initialise(module: nn.Module, head: nn.Module) -> None:
    """统一初始化顺序：全局初始化 → 补偿先验。

    Ultralytics 的 ``initialize_weights`` 会把卷积 bias 清零，anchor 头的
    objectness 先验（0.02）必须在其后重设，否则训练初期正负样本失衡。
    """
    initialize_weights(module)
    if isinstance(head, AnchorOBBHead):
        head.init_bias()
