"""资源就绪检查：权重/数据清单与退出码（不依赖 torch）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pcdf.config import load_config
from pcdf.preflight import check_data, check_weights, main, ready


def test_repo_weights_are_complete() -> None:
    cfg = load_config()
    items = check_weights(cfg)
    assert ready(items), [item.detail for item in items if not item.ok]


def test_repo_data_is_complete() -> None:
    cfg = load_config()
    items = check_data(cfg, count=False)
    assert ready(items), [item.detail for item in items if not item.ok]


def test_missing_weight_is_reported(tmp_path: Path) -> None:
    cfg = load_config()
    cfg.model.encoder_weights = str(tmp_path / "nope.pt")
    items = check_weights(cfg)
    assert not ready(items)
    assert any("骨架权重" in item.name and not item.ok for item in items)


def test_missing_data_is_reported(tmp_path: Path) -> None:
    cfg = load_config()
    cfg.paths.data_root = str(tmp_path / "no-data")
    cfg.paths.condition_index = str(tmp_path / "no-data" / "condition_index.csv")
    assert not ready(check_data(cfg, count=False))


def test_cli_exit_codes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """数据在 → 0；数据根指向空目录 → 1。"""
    assert main(["data", "--quiet"]) == 0
    monkeypatch.setenv("PCDF_DATA_ROOT", str(tmp_path / "empty"))
    assert main(["data", "--quiet"]) == 1
