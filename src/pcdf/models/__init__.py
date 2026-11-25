"""PCDF 网络模块（按职责分层，避免循环导入）。

* :mod:`encoder`   —— 双流 backbone（YOLOv5 scale n/s/m/l/x + ``.pt`` 权重迁移）
* :mod:`heads`     —— OBB 检测头装配、统一初始化顺序与检测损失
* :mod:`blocks`    —— SCPLGate / 三支解耦 / CDF / PAN / 引导投影
* :mod:`scpl`      —— SCPL 前向与 straight-through
* :mod:`baseline`  —— 双流 concat baseline（阶段 0）
* :mod:`pcdf`      —— 完整网络（阶段 1/2 与测试）
"""

from .baseline import BaselineNet, load_baseline_into_pcdf
from .blocks import ChannelNormSoftGate, DecoupleBranch, GuidanceAdapter, PANNeck, SCPLGate
from .encoder import Backbone
from .heads import build_head
from .pcdf import PCDFNet
from .scpl import SCPL

__all__ = [
    "PCDFNet",
    "BaselineNet",
    "Backbone",
    "SCPL",
    "SCPLGate",
    "DecoupleBranch",
    "ChannelNormSoftGate",
    "PANNeck",
    "GuidanceAdapter",
    "build_head",
    "load_baseline_into_pcdf",
]