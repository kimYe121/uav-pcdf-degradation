"""Preliminary single-modality OBB controls on a specified data split.

These controls are not PCDF reproduction results. Their purpose is to establish
clean references under the same backbone and split. The pilot subset is not
comparable with the paper's full-data benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modality_decay.core import Config
from modality_decay.evaluate import eval_weights
from modality_decay.train import train_view


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/atr_umod_quick")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--name", default=None, help="unique run prefix; avoids overwriting earlier baselines")
    ap.add_argument("--views", nargs="+", choices=("rgb", "ir"), default=("rgb", "ir"))
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.name:
        cfg.name = args.name
    data_root = Path(args.data)
    output = cfg.path(cfg.results_dir) / f"{cfg.name}_baselines.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for view in args.views:
        print(f"[train] {view}: {cfg.epochs} epochs, imgsz={cfg.imgsz}", flush=True)
        weights = train_view(cfg, data_root / view / "data.yaml", f"base_{view}")
        res = eval_weights(weights, data_root / view / "data.yaml", cfg.imgsz, cfg.device)
        results[view] = {"weights": str(weights), **res}
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[clean val] {view}: mAP50={res['map50']:.4f}  mAP50-95={res['map50_95']:.4f}", flush=True)
        print(f"           per-class AP50={res['per_class_ap50']}", flush=True)
        print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
