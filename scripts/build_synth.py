"""Generate the synthetic RGB-IR dataset used for smoke testing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modality_decay.core import Config
from modality_decay.datasets import write_data_yaml
from modality_decay.synth import build_synthetic


def main() -> None:
    cfg = Config.load(ROOT / "configs" / "pilot.yaml")
    out = cfg.path(cfg.data_root) / "synth"
    build_synthetic(out, cfg.synth_train, cfg.synth_val, cfg.synth_size, cfg.seed)
    for view in ("rgb", "ir", "fused"):
        path = write_data_yaml(out / view)
        print(f"{view:6s} -> {path}")
    print(f"done: {cfg.synth_train} train / {cfg.synth_val} val, size={cfg.synth_size}")


if __name__ == "__main__":
    main()
