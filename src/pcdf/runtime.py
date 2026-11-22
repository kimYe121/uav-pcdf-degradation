"""运行时设施：设备选择、随机种子、环境快照与原子写。

这些函数原先散落在 ``train.py`` / ``evaluate.py`` / ``smoke.py`` 里，各写一份
``cuda:0 if torch.cuda.is_available() else cpu``。这种写法在两种常见环境下是
错的：

* 容器里驱动存在但 ``CUDA_VISIBLE_DEVICES=""``（CPU 复验、卡被别人占满）：
  ``torch.cuda.is_available()`` 仍返回 ``True``，于是代码选 ``cuda:0``，
  第一次分配张量才崩；
* ``torch.backends.cudnn.version()`` 在无可用设备时会抛 ``ValueError``，
  而它被放在写 ``meta.json`` 的路径上，直接把训练打断。

因此设备判定统一走 :func:`resolve_device`，环境快照统一走
:func:`environment_meta`，两者都保证"取不到就降级并记录"，不静默失败。
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .paths import PROJECT_ROOT

# 设备字符串的合法写法：auto / cpu / cuda / cuda:0 / 0 / 0,1
_CPU_ALIASES = {"", "auto", "cpu"}
_CUDA_ALIASES = {"cuda", "gpu"}


def cuda_is_usable() -> bool:
    """判断当前进程是否真的能用 CUDA。

    比 ``torch.cuda.is_available()`` 严格：还要求至少有一个可见设备，并且
    能读到它的属性。后者是唯一能覆盖"驱动在、设备被隐藏"的探针。
    """
    if not torch.cuda.is_available():
        return False
    if torch.cuda.device_count() <= 0:
        return False
    try:
        torch.cuda.get_device_properties(0)
    except Exception:  # noqa: BLE001 —— 驱动/权限问题一律视为不可用
        return False
    return True


def resolve_device(spec: str | torch.device | None = "auto", *, verbose: bool = True) -> torch.device:
    """把设备说明解析成可用的 ``torch.device``。

    Args:
        spec: ``auto``（有可用卡就用 ``cuda:0``，否则 ``cpu``）、``cpu``、
            ``cuda``/``cuda:N``、显卡序号 ``"0"`` 或多卡 ``"0,1"``。
        verbose: 发生降级时是否打印原因。

    Raises:
        ValueError: 显式指定了 CUDA，但当前进程没有可用设备。

    Notes:
        显式请求 CUDA 却不可用时**不做静默降级**——那会让"以为在 GPU 上跑"
        的实验悄悄变成 CPU 训练。只有 ``auto`` 才允许自动选择。
    """
    if isinstance(spec, torch.device):
        return spec
    text = str(spec or "auto").strip().lower()

    if text in _CPU_ALIASES:
        if text == "auto":
            if cuda_is_usable():
                return torch.device("cuda:0")
            if verbose:
                print("[device] 未检测到可用 CUDA 设备，使用 CPU（仅用于功能验证）", flush=True)
            return torch.device("cpu")
        return torch.device("cpu")

    if not cuda_is_usable():
        raise ValueError(
            f"请求了 CUDA 设备 {spec!r}，但当前进程没有可用 GPU"
            f"（torch.cuda.device_count()={torch.cuda.device_count()}，"
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<未设置>')!r}）。"
            "CPU 复验请显式传 --device cpu，或设置 PCDF_DEVICE=cpu。"
        )

    if text in _CUDA_ALIASES:
        return torch.device("cuda:0")

    first = text.split(",")[0]
    if first.startswith("cuda:"):
        first = first.split(":", 1)[1]
    if not first.isdigit():
        raise ValueError(f"无法解析设备 {spec!r}；可用写法：auto / cpu / cuda:0 / 0")
    index = int(first)
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(f"显卡序号 {index} 超出可见设备数 {count}（可用 0..{count - 1}）")
    return torch.device(f"cuda:{index}")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """固定 Python / NumPy / PyTorch 随机种子。

    ``deterministic=True`` 时会关闭 cuDNN 的自动调优；这与论文的
    "可重复"诉求一致，代价是卷积可能略慢。
    """
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _safe(getter, default: Any = None) -> Any:
    """执行可能因缺少 CUDA 而失败的查询，失败时返回默认值。"""
    try:
        return getter()
    except Exception:  # noqa: BLE001 —— 环境探测不应中断训练
        return default


def environment_meta(device: torch.device) -> dict[str, Any]:
    """采集可复现实验所需的环境快照，供 ``meta.json`` 使用。

    所有 CUDA 相关查询都经过 :func:`_safe`：在纯 CPU 机器上
    ``torch.backends.cudnn.version()`` 会抛 ``ValueError``，不能让写元数据
    这个附带动作把训练打断。
    """
    cuda_device = device.type == "cuda"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": _safe(lambda: torch.version.cuda),
        "cudnn": _safe(lambda: torch.backends.cudnn.version()),
        "device": str(device),
        "gpu": _safe(lambda: torch.cuda.get_device_name(device)) if cuda_device else None,
        "gpu_count": _safe(lambda: torch.cuda.device_count(), 0),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_count": os.cpu_count(),
    }


def git_metadata() -> dict[str, Any]:
    """记录代码版本；不在 git 仓库里时返回空字典，不影响训练。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=3,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=3,
        )
    except Exception:  # noqa: BLE001 —— git 缺失/非仓库都只是少一条元数据
        return {}
    if commit.returncode != 0:
        return {}
    dirty_files = status.stdout.strip().splitlines()
    return {"commit": commit.stdout.strip(), "dirty": bool(dirty_files), "dirty_files": dirty_files[:20]}


def json_safe(value: Any) -> Any:
    """把 NaN/Inf 转成 ``None``，保证 ``metrics.json`` 是合法 JSON。

    Python 的 ``json`` 默认会写出 ``NaN``，那不是合法 JSON；下游用别的语言
    读时会直接失败。
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """原子写 JSON：先写 ``*.tmp`` 再 ``os.replace``。

    容器被抢占时不能留下半个文件；``metrics.json`` 是事后唯一的过程记录。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, target)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
