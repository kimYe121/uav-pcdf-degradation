"""End-to-end smoke test: synth -> train(3 views) -> degrade -> eval -> curve.

This validates the pipeline only. Synthetic results are NOT evidence for the
research claim; rerun with --data-root pointing at real ATR-UMOD for that.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from modality_decay.core import Config
from modality_decay.datasets import write_data_yaml
from modality_decay.evaluate import prepare_and_eval
from modality_decay.metrics import summarize
from modality_decay.plots import plot_curves
from modality_decay.synth import build_synthetic
from modality_decay.train import train_view

VIEWS = ("rgb", "ir", "fused")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data_root = cfg.path(cfg.data_root) / "synth"
    results_dir = cfg.path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/4] building synthetic data -> {data_root}")
    build_synthetic(data_root, cfg.synth_train, cfg.synth_val, cfg.synth_size, cfg.seed)
    yamls = {v: write_data_yaml(data_root / v) for v in VIEWS}

    weights: dict[str, Path] = {}
    for view in VIEWS:
        print(f"[2/4] training {view} ({cfg.epochs} epochs, imgsz={cfg.imgsz})")
        if args.skip_train:
            weights[view] = cfg.path(cfg.runs_dir) / f"{cfg.name}_{view}" / "weights" / "best.pt"
        else:
            weights[view] = train_view(cfg, yamls[view], view)
        print(f"      -> {weights[view]}")

    print(f"[3/4] evaluating {len(cfg.protocols)} protocols x {len(cfg.levels)} levels")
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
                    work_root=results_dir / "degraded",
                    split="val",
                )
            )
            print(f"      {protocol} s={level:.2f} done")

    df = pd.DataFrame(rows)
    csv_path = results_dir / f"{cfg.name}_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("[4/4] summarizing")
    summary = summarize(df)
    lines = []
    for protocol, s in summary.items():
        star = "n/a" if s["s_star"] is None else f"{s['s_star']:.3f}"
        lines.append(
            f"{protocol}: NF min={s['min_nf']:+.2f} | s*={star} | "
            f"slope fusion={s['slope_fusion']:+.2f} rgb={s['slope_rgb']:+.2f} ir={s['slope_ir']:+.2f}"
        )
        print("      " + lines[-1])

    summary_path = results_dir / f"{cfg.name}_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    png = plot_curves(summary, results_dir / f"{cfg.name}_curves.png", title=f"{cfg.name}: availability-performance")
    print(f"\nresults csv : {csv_path}")
    print(f"summary     : {summary_path}")
    print(f"figure      : {png}")
    print(f"elapsed     : {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
