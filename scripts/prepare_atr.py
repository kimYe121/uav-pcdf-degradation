#!/usr/bin/env python
"""兼容入口：等价于 ``python -m pcdf.cli.prepare_data``。"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401 —— 把 src/ 加入 sys.path

from pcdf.cli.prepare_data import main

if __name__ == "__main__":
    sys.exit(main())
