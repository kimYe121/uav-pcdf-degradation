"""检测头装配：ultralytics OBB（anchor-free + DFL）与 ``v8OBBLoss``。

为什么用 ultralytics 现成实现
-----------------------------
论文对检测器 family 只字未提（全文 0 次 backbone / anchor），式 8 的
``L_cls + L_reg + L_obj`` 是 YOLOv5 anchor 时代的口径，又没有公开代码可对照，
所以这里的检测头只能选"真的能收敛"的那一个。仓库早期版本写过一个
anchor-based OBB 头（5 邻域匹配 + ProbIoU + objectness），同一 16 图 overfit
（320×256）600 步只到 mAP50 0.6116；换成 ultralytics 的 anchor-free OBB 头
（TAL 分配 + DFL）同条件 300 步到 0.9611。复现目标 58.4 / 63.1 只能由后者
产生，因此自写头已整体删除，仓库只保留这一条路径。

``v8OBBLoss`` 对宿主的三个要求
------------------------------
它从 ``model.model[-1]`` 取 stride / nc / reg_max，从 ``model.args`` 取损失
增益，从 ``head.stride`` 取特征金字塔步长。我们的 ``BaselineNet`` /
``PCDFNet`` 不是 ultralytics 的 ``DetectionModel``，因此由
:func:`attach_loss_context` 显式补上 ``model`` 与 ``args`` 两个属性。

``OBB`` 的返回随模式变化：训练态 ``dict(boxes, scores, angle, feats)``，
推理态 ``(decoded, preds)``。调用方（:mod:`pcdf.models.baseline` /
:mod:`pcdf.models.pcdf`）必须两种都能吃下。
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from ultralytics.nn.modules import OBB
from ultralytics.nn.tasks import initialize_weights
from ultralytics.utils.loss import v8OBBLoss

from ..config import ModelConfig

__all__ = [
    "build_head",
    "initialise",
    "make_criterion",
    "attach_loss_context",
]

# ultralytics 的默认损失增益（hyp.scratch-low 一档）。OBB 有四项：框、分类、
# DFL 分布与角度；anchor-free 头没有 objectness。
OBB_HYP: dict[str, float] = dict(box=7.5, cls=0.5, dfl=1.5, angle=1.0)

# P3/P4/P5 的步长；必须与 ``constants.STRIDES`` 一致。
_STRIDES: tuple[float, float, float] = (8.0, 16.0, 32.0)


def build_head(cfg: ModelConfig, nc: int) -> OBB:
    """构造 OBB 头；``neck`` 决定输入通道（pan=C，concat=2C）。"""
    ch = tuple(2 * c for c in cfg.widths) if cfg.neck == "concat" else tuple(cfg.widths)
    head = OBB(nc=nc, ne=1, ch=ch)
    # OBB.__init__ 里 stride 是按 yaml 推断的占位值；这里按我们的固定金字塔写死。
    head.stride = torch.tensor(_STRIDES)
    return head


def initialise(model: nn.Module) -> None:
    """全局初始化 + 补一次 ``bias_init``（分类/框先验）。

    ultralytics 在 ``DetectionModel.__init__`` 里于 stride 就位后调用
    ``head.bias_init()``；``OBB.__init__`` 自己**不调**（docstring 明确要
    "requires stride availability"）。我们的容器不是 ``DetectionModel``，
    漏掉这一步的后果是分类 bias 停在 0：

    * 初始 sigmoid ≈ 0.5，cls_loss 起步 5500+（正常应在个位数）；
    * 全图 6720×11 个输出都在"自信地报背景不是背景"，前几百步全花在把背景
      压下去，相当于白烧 1–2 个 epoch。
    """
    initialize_weights(model)
    model.head.bias_init()


def attach_loss_context(model: nn.Module) -> None:
    """补上 ``v8OBBLoss`` 需要的 ``model.model[-1]`` 与 ``model.args``。"""
    model.model = nn.ModuleList([model.head])
    model.args = SimpleNamespace(**OBB_HYP)


def make_criterion(model: nn.Module) -> v8OBBLoss:
    """创建检测损失。必须传整个模型：``v8OBBLoss`` 从 ``model.model[-1]`` 取
    stride / nc / reg_max，从 ``model.args`` 取增益。"""
    return v8OBBLoss(model, tal_topk=10)