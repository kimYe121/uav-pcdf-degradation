"""运行时设施：设备解析、随机种子、环境快照与原子写。

重点回归两个曾在纯 CPU 环境直接打断训练的问题：
1. ``torch.cuda.is_available()`` 为 True 但可见设备数为 0 时不能选 cuda；
2. ``torch.backends.cudnn.version()`` 在无设备时会抛异常，不能让它中断 meta.json。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from pcdf.runtime import (
    environment_meta,
    json_safe,
    resolve_device,
    seed_everything,
    write_json,
)


def test_cpu_always_resolves() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device(torch.device("cpu")).type == "cpu"
    # None / 空串等价 auto：有卡走 cuda，无卡走 cpu，但绝不会抛异常
    assert resolve_device(None, verbose=False).type in ("cpu", "cuda")
    assert resolve_device("", verbose=False).type in ("cpu", "cuda")


def test_explicit_cuda_fails_fast_without_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式要 GPU 却没卡时必须报错，而不是静默降级成 CPU 训练。

    用 monkeypatch 固定 ``cuda_is_usable``，让这条分支在有卡/无卡机器上都跑到。
    """
    monkeypatch.setattr("pcdf.runtime.cuda_is_usable", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="没有可用 GPU"):
        resolve_device("cuda:0")


def test_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pcdf.runtime.cuda_is_usable", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert resolve_device("auto", verbose=False).type == "cpu"


def test_environment_meta_is_cpu_safe() -> None:
    """纯 CPU 环境下采集元数据不能抛异常（旧实现会崩在 cudnn.version）。"""
    meta = environment_meta(torch.device("cpu"))
    assert meta["device"] == "cpu"
    assert meta["gpu"] is None
    assert meta["torch"]
    assert "cudnn" in meta  # 允许为 None，但键必须存在


def test_seed_everything_is_reproducible() -> None:
    seed_everything(123, deterministic=True)
    first = torch.rand(3)
    seed_everything(123, deterministic=True)
    assert torch.equal(first, torch.rand(3))


def test_json_safe_replaces_non_finite() -> None:
    payload = {"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": float("-inf")}}
    cleaned = json_safe(payload)
    assert cleaned["a"] is None
    assert cleaned["b"][0] is None and cleaned["b"][1] == 1.0
    assert cleaned["c"]["d"] is None
    # 必须是合法 JSON（Python 默认会写出 NaN，那不是合法 JSON）
    json.loads(json.dumps(cleaned, allow_nan=False))


def test_write_json_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "metrics.json"
    write_json(target, {"loss": 1.0, "map50": float("nan")})
    assert target.is_file()
    assert not (tmp_path / "metrics.json.tmp").exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["map50"] is None
    assert math.isfinite(data["loss"])
