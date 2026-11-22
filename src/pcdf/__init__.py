"""PCDF 方法级复现包（ICCV 2025）。

对外入口：

* :class:`pcdf.models.PCDFNet` / :class:`pcdf.models.BaselineNet`：网络；
* :func:`pcdf.config.load_config`：唯一的配置入口；
* :func:`pcdf.train.train_stage`：三阶段训练与断点续训；
* :func:`pcdf.evaluate.evaluate_obb`：旋转框 mAP 评估。

论文公式在对应模块的 docstring 中标注；论文未公开的结构统一记录在
``docs/ARCHITECTURE.md``。
"""

from .config import Config, load_config
from .models.baseline import BaselineNet
from .models.pcdf import PCDFNet

__all__ = ["BaselineNet", "Config", "PCDFNet", "load_config"]
__version__ = "2.0.0"
