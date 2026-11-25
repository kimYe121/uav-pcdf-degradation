"""``pcdf-vis``：定性结果图（论文 Fig.5 风格，暗底报告版）。

单模型：RGB | IR+预测（漏检蓝虚线 / 误检红虚线，conf≥0.25）。
``--compare``：RGB | IR+模型A | IR+模型B，用于 baseline vs PCDF 对照图。

示例::

    python -m pcdf.cli.visualize --checkpoint runs/base50pub/best.pt --images 4
    python -m pcdf.cli.visualize --checkpoint runs/pcdf_s2/best.pt \
        --compare runs/base50pub/best.pt --stems 00012 00345 --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ..checkpoint import CheckpointError
from ..config import ConfigError
from ..data import build_loader
from ..runtime import resolve_device
from ..visualize import predict_batch, visualize_grid, visualize_samples
from .common import (
    add_config_args,
    add_device_arg,
    build_config,
    config_from_checkpoint,
    load_table,
)
from .evaluate import build_model, infer_arch, load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pcdf-vis", description="渲染 OBB 定性结果图（IR 面板 + 条件字幕）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    add_config_args(p)
    add_device_arg(p)
    p.set_defaults(device="cpu")  # 默认不挤训练 GPU；要快给 --device 0
    p.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    p.add_argument("--compare", default=None, help="第二个检查点（对照面板）")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--images", type=int, default=4, help="自动选图数量（gt 5–20 的图里均匀取）")
    p.add_argument("--stems", nargs="*", default=None, help="指定图像 stem（优先于 --images）")
    p.add_argument("--grid-by", default=None,
                   choices=("weather", "illumination", "altitude", "time", "angle", "scenario"),
                   help="报告网格图：每行一个条件取值（论文 Fig.15-20 组织）")
    p.add_argument("--rows", type=int, default=4, help="网格行数（取样本最多的几个取值）")
    p.add_argument("--per-row", type=int, default=1, help="每行图数")
    p.add_argument("--conf", type=float, default=0.25, help="显示阈值（论文 Fig.5 口径）")
    p.add_argument("--out", default="output/vis", help="输出目录")
    p.add_argument("--batch", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        env_cfg = build_config(args)
        specs = [(Path(args.checkpoint).parent.name, args.checkpoint)]
        if args.compare:
            specs.append((Path(args.compare).parent.name, args.compare))
        models, cfgs = [], []
        for tag, ck in specs:
            payload = load_checkpoint(ck)
            cfg = config_from_checkpoint(payload["cfg"], env_cfg) if payload.get("cfg") else env_cfg
            device = resolve_device(args.device)
            table = load_table(cfg)
            model = build_model(cfg, infer_arch(payload), table, device)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            models.append((tag, model))
            cfgs.append(cfg)
        cfg = cfgs[0]
        device = resolve_device(args.device)
        table = load_table(cfg)
        _, loader = build_loader(
            cfg.data, table, split=args.split, augment=False, root=cfg.paths.data_root,
            batch_size=args.batch, workers=0, shuffle=False, device=device,
        )
        # 网格模式：先按条件取值选 stem（不扫 loader），再单遍推理取回。
        selection: list[tuple[str, list[str]]] = []
        if args.grid_by:
            attr = table.keys.index(args.grid_by)
            spec = table.specs[attr]
            stems = table.stems(args.split)
            by_val: dict[int, list[str]] = {}
            for s in stems:
                v = int(table.ids_for(s, subset=args.split)[attr])
                by_val.setdefault(v, []).append(s)
            order = sorted(by_val, key=lambda v: -len(by_val[v]))[: args.rows]
            for v in order:
                pool = by_val[v]
                step = max(1, len(pool) // args.per_row)
                picks = pool[::step][: args.per_row]
                selection.append((spec.names[v], picks))
        wanted = set(args.stems or []) | {s for _, picks in selection for s in picks}
        cand: list[tuple[dict, ...]] = []
        with torch.inference_mode():
            for bi, batch in enumerate(loader):
                rows = [predict_batch(m, batch, device, cfg.data.nc, args.conf)
                        for _, m in models]
                for b in range(len(rows[0])):
                    stem = Path(rows[0][b]["im_file"]).stem
                    n_gt = int(rows[0][b]["gt_cls"].numel())
                    if wanted:
                        if stem in wanted:
                            cand.append(tuple(r[b] for r in rows))
                            wanted.discard(stem)
                    elif 5 <= n_gt <= 20:
                        cand.append(tuple(r[b] for r in rows))
                if wanted and not wanted:
                    break
                if not args.stems and not selection and len(cand) >= 4 * args.images:
                    break
                if bi >= 300:
                    break
        if not cand:
            print("[vis] 没选到图（检查 --stems / split / --grid-by）", file=sys.stderr)
            return 2
        by_stem = {Path(c[0]["im_file"]).stem: c for c in cand}
        if selection:
            rows_raw = [(label, [by_stem[s] for s in picks if s in by_stem])
                        for label, picks in selection]
            rows_raw = [(l, its) for l, its in rows_raw if its]
            rows_items = [(l, [t[0] for t in its]) for l, its in rows_raw]
            models_grid = [
                (tag, [[rows_raw[r][0][i]] for r in range(len(rows_raw))])
                for i, (tag, _) in enumerate(models)
            ] if len(models) > 1 else None
            out = visualize_grid(
                rows_items, cfg.data.names, Path(args.out), args.grid_by,
                models=models_grid, conf=args.conf,
            )
            print(f"[vis] {out}")
            return 0
        if args.stems:
            chosen = cand
        else:
            idx = [round(i * (len(cand) - 1) / (args.images - 1)) for i in range(args.images)] \
                if args.images > 1 else [0]
            chosen = [cand[i] for i in dict.fromkeys(idx)]
        items_a = [c[0] for c in chosen]
        multi = [(tag, [c[i] for c in chosen]) for i, (tag, _) in enumerate(models)] \
            if len(models) > 1 else None
        outs = visualize_samples(
            items_a, cfg.data.names, table, args.split, Path(args.out),
            models=multi, conf=args.conf,
        )
        for o in outs:
            print(f"[vis] {o}")
    except (ConfigError, CheckpointError, FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
