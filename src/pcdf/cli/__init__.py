"""命令行入口集合。

每个模块都可以直接运行（``python -m pcdf.cli.train``），``scripts/*.py`` 只是
一层薄启动器，保证"从仓库任意目录、未安装包"也能用。安装后可用的入口点：

``pcdf-train`` / ``pcdf-eval`` / ``pcdf-smoke`` / ``pcdf-overfit`` /
``pcdf-check`` / ``pcdf-prepare`` / ``pcdf-plot`` / ``pcdf-fetch-clip`` /
``pcdf-fetch-backbone``。
"""

from __future__ import annotations

__all__ = [
    "check_env",
    "evaluate",
    "fetch_backbone",
    "fetch_clip",
    "overfit",
    "plot_curves",
    "prepare_data",
    "smoke",
    "train",
]
