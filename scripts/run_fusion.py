"""Train a six-channel early-fusion control; this is not a PCDF model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modality_decay.core import Config
from modality_decay.fusion import EarlyFusionTrainer, metrics_dict, write_early_fusion_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/atr_umod_head6")
    parser.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    parser.add_argument("--name", default="pilot_head6_early6")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    data_yaml = write_early_fusion_yaml(args.data)
    print(f"[fusion] true six-channel input: RGB(3) + IR(3), data={data_yaml}", flush=True)
    print(f"[train] early6: {cfg.epochs} epochs, imgsz={cfg.imgsz}, batch={cfg.batch}", flush=True)

    trainer = EarlyFusionTrainer(overrides={
        "model": cfg.model,
        "data": str(data_yaml),
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "device": cfg.device,
        "workers": cfg.workers,
        "amp": True,
        "seed": cfg.seed,
        "deterministic": True,
        "project": str(cfg.path(cfg.runs_dir)),
        "name": args.name,
        "exist_ok": True,
        "plots": False,
        "verbose": False,
        "val": True,
    })
    trainer.train()
    weights = Path(trainer.save_dir) / "weights" / "best.pt"
    result = {
        "model": "early6",
        "input": "RGB(3)+IR(3)",
        "weights": str(weights),
        **metrics_dict(trainer.validator),
    }
    output = cfg.path(cfg.results_dir) / f"{args.name}_clean.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[clean val] early6: mAP50={result['map50']:.4f}  mAP50-95={result['map50_95']:.4f}", flush=True)
    print(f"           per-class AP50={result['per_class_ap50']}", flush=True)
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
