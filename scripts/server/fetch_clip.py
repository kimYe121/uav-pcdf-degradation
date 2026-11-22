#!/usr/bin/env python
"""兼容入口：等价于 ``python -m pcdf.cli.fetch_clip``。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import _bootstrap  # noqa: F401,E402 —— 把 src/ 加入 sys.path

from pcdf.cli.fetch_clip import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
