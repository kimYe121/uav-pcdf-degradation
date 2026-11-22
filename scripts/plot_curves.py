#!/usr/bin/env python
"""兼容入口：等价于 ``python -m pcdf.cli.plot_curves``。"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401 —— 把 src/ 加入 sys.path

from pcdf.cli.plot_curves import main

if __name__ == "__main__":
    sys.exit(main())
