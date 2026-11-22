#!/usr/bin/env python
"""兼容入口：等价于 ``python -m pcdf.cli.train``。

保留这个文件是为了让文档里的 ``python scripts/train.py --stage 1`` 继续可用；
真正的实现与参数说明都在 :mod:`pcdf.cli.train`。
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401 —— 把 src/ 加入 sys.path

from pcdf.cli.train import main

if __name__ == "__main__":
    sys.exit(main())
