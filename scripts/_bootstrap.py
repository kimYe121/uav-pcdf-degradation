"""把 ``src/`` 加入 ``sys.path``，供仓库内的薄启动器使用。

只在"未安装包、直接跑脚本"时生效；``pip install -e .`` 之后这一步是空操作。
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
