"""Config loading and shared paths."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    name: str = "pilot"
    data_root: str = "data"
    seed: int = 0
    imgsz: int = 320
    epochs: int = 8
    batch: int = 8
    device: str = "0"
    workers: int = 0
    model: str = "yolo11n-obb.pt"
    fuse_alpha: float = 0.5
    levels: list[float] | None = None
    protocols: list[str] | None = None
    p1_kind: str = "night"
    synth_train: int = 300
    synth_val: int = 80
    synth_size: list[int] | None = None
    runs_dir: str = "runs"
    results_dir: str = "results"

    def __post_init__(self) -> None:
        if self.levels is None:
            self.levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        if self.protocols is None:
            self.protocols = ["P1", "P2", "P3"]
        if self.synth_size is None:
            self.synth_size = [640, 512]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}; allowed: {sorted(allowed)}")
        return cls(**raw)

    def path(self, *parts: str | Path) -> Path:
        return PROJECT_ROOT.joinpath(*parts)
