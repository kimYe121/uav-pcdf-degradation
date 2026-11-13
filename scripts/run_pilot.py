"""Pilot run on converted ATR-UMOD: train three views, sweep degradation, plot."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from modality_decay.core import Config
from modality_decay.evaluate import prepare_and_eval
from modality_decay.metrics import summarize
from modality_decay.plots import plot_curves
from modality_decay.train import train_view

VIEWS = ("rgb", "ir", "fused")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/atr_umod_pilot")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data_root = Path(args.data)
    if not (data_root / "rgb" / "data.yaml").exists():
        raise SystemExit(f"missing {data_root/'rgb'/'data.yaml'}; run scripts/prepare_atr.py convert first")

    results_dir = cfg.path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    weights: dict[str, Path] = {}
    for view in VIEWS:
        yaml_path = data_root / view / "data.yaml"
        print(f"[1/3] training {view} ({cfg.epochs} ep, imgsz={cfg.imgsz})", flush=True)
        if args.skip_train:
            weights[view] = cfg.path(cfg.runs_dir) / f"{cfg.name}_{args.tag}_{view}" / "weights" / "best.pt"
        else:
            weights[view] = train_view(cfg, yaml_path, f"{args.tag}_{view}")
        print(f"      -> {weights[view]}", flush=True)

    print(f"[2/3] sweeping {len(cfg.protocols)} protocols x {len(cfg.levels)} levels", flush=True)
    rows: list[dict] = []
    for protocol in cfg.protocols:
        for level in cfg.levels:
            rows.extend(
                prepare_and_eval(
                    cfg=cfg,
                    src_root=data_root,
                    protocol=protocol,
                    level=level,
                    weights=weights,
                    work_root=results_dir / f"degraded_{args.tag}",
                    split="val",
                )
            )
            print(f"      {protocol} s={level:.2f}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = results_dir / f"{args.tag}_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = summarize(df)
    lines = []
    for protocol, s in summary.items():
        star = "n/a" if s["s_star"] is None else f"{s['s_star']:.3f}"
        lines.append(
            f"{protocol}: NF min={s['min_nf']:+.3f} | s*={star} | "
            f"slope fusion={s['slope_fusion']:+.3f} rgb={s['slope_rgb']:+.3f} ir={s['slope_ir']:+.3f}"
        )
        print("      " + lines[-1], flush=True)

    (results_dir / f"{args.tag}_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    png = plot_curves(summary, results_dir / f"{args.tag}_curves.png", title=f"{args.tag}: availability-performance")
    print(f"\ncsv    : {csv_path}\nsummary: {results_dir / f'{args.tag}_summary.txt'}\nfigure : {png}")
    print(f"elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
