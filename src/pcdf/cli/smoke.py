"""``pcdf-smoke``：三阶段小数据端到端验收。

目的：在少量 batch 上验证数据管线、前向、全部损失、反向传播、检查点读写与
评估器能协同工作。它不是性能实验，也不用于判断方法有效性。

与旧版的区别
------------
* 路径、设备、采样比例、batch、尺寸全部可传参，不再写死在函数体里；
* 只清理自己创建的 ``runs/_smoke*`` 目录，且先校验路径在 ``runs_dir`` 之下；
* 三个阶段都走公开的 :func:`pcdf.train.train_stage`，不复制训练逻辑。

示例::

    python -m pcdf.cli.smoke                     # CPU 也可跑，约 1 分钟
    python -m pcdf.cli.smoke --device cpu --batch 2 --imgsz 320 256
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from ..config import Config, ConfigError
from ..constants import STAGE_NAMES
from ..data import build_loader
from ..evaluate import evaluate_obb
from ..models.pcdf import PCDFNet
from ..paths import display_path
from ..runtime import resolve_device
from ..text_encoder import CLIPTextEncoder
from ..train import train_stage
from .common import add_config_args, add_device_arg, apply_overrides, build_config, load_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcdf-smoke",
        description="三阶段端到端冒烟（baseline -> SCPL+CDF -> full pipeline -> 评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_config_args(parser)
    add_device_arg(parser)
    parser.add_argument("--fraction", type=float, default=0.004, help="采样比例（默认 0.4%%）")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, nargs=2, default=(320, 256), metavar=("H", "W"))
    parser.add_argument("--max-batches", type=int, default=2, help="每个 epoch 的 step 上限")
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument("--run-root", default=None, help="冒烟输出目录；默认 <runs_dir>/_smoke")
    parser.add_argument("--keep", action="store_true", help="保留冒烟产物（默认清理）")
    return parser


def _safe_clean(target: Path, runs_root: Path) -> None:
    """只删除位于 ``runs_root`` 之下的目录，避免误删用户数据。"""
    resolved = target.resolve()
    if runs_root.resolve() not in resolved.parents:
        raise RuntimeError(f"拒绝清理 {resolved}：不在 {runs_root} 之下")
    if resolved.exists():
        shutil.rmtree(resolved)


def _stage_dir(root: Path, stage: int) -> Path:
    return root / f"smoke_{STAGE_NAMES[stage]}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = build_config(args)
        device = resolve_device(args.device)
        # 冒烟只验证"能跑通"：小样本、单进程、每轮都验证、不写周期检查点。
        apply_overrides(cfg, {
            "data.fraction": args.fraction,
            "data.workers": 0,
            "train.eval_interval": 1,
            "train.save_period": 0,
            "train.save_every_minutes": 0,
            "train.save_every_steps": 0,
        })
        root = Path(args.run_root) if args.run_root else Path(cfg.paths.runs_dir) / "_smoke"
        if root.exists():
            _safe_clean(root, Path(cfg.paths.runs_dir))
        root.mkdir(parents=True, exist_ok=True)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    stages: list[Path] = []
    previous: Path | None = None
    for stage in (0, 1, 2):
        print("=" * 70)
        print(f"阶段 {stage}：{STAGE_NAMES[stage]}")
        print("=" * 70)
        run_dir = _stage_dir(root, stage)
        result = train_stage(
            cfg,
            stage=stage,
            init_checkpoint=previous,
            device=str(device),
            run_dir=run_dir,
            epochs=1,
            batch=args.batch,
            imgsz=tuple(args.imgsz),
            max_batches=args.max_batches,
            val_batches=args.val_batches,
        )
        stages.append(result.run_dir)
        # 下一阶段用本阶段的 last.pt（一轮训练时 last.pt 一定有）。
        previous = result.run_dir / "last.pt"

    # 重新加载检查点，验证 state_dict 严格匹配且评估路径不依赖条件输入。
    print("=" * 70)
    print("重新加载 stage2 并评估")
    print("=" * 70)
    table = load_table(cfg)
    encoder = CLIPTextEncoder(cfg.model.text_encoder, device=device)
    model = PCDFNet(cfg.model, nc=cfg.data.nc, table=table, text_encoder=encoder).to(device)
    payload = torch.load(stages[-1] / "last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    _, loader = build_loader(
        cfg.data, table, split=cfg.data.val_split, augment=False, root=cfg.paths.data_root,
        batch_size=args.batch, workers=0, shuffle=False, device=device,
    )
    result = evaluate_obb(
        model, loader, device, nc=cfg.data.nc, max_batches=args.val_batches, names=cfg.data.names
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[smoke] checkpoint={display_path(stages[-1] / 'last.pt')}")
    print("[smoke] OK")

    if not args.keep:
        _safe_clean(root, Path(cfg.paths.runs_dir))
        print(f"[smoke] 已清理 {display_path(root)}（--keep 可保留）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
