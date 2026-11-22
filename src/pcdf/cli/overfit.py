"""``pcdf-overfit``：小样本过拟合诊断。

做法：从训练集抽 N 对图（只取有目标的样本），复制成一个自包含的临时数据集，
train/val 用同一批图，跑少量 epoch。实现正确时 loss 应接近 0、同一批图上的
mAP 应明显偏高；如果连 30 张图都过拟合不了，先别跑全量。

为什么先跑阶段 0
----------------
阶段 1 会冻结 encoder/neck/head（补充材料 B.3），只能训练 SCPL 与 CDF。
在一个没有训练过检测头的 baseline 上做阶段 1，检测损失根本没有下降通道，
"过拟合不了"并不能说明实现有 bug。因此诊断以**阶段 0（全参数可训）**为准，
``--with-stage2`` 再把阶段 1/2 接上去，验证 ``L_dt``/``L_irr``/``L_dc`` 的
反向路径是否通畅。

示例::

    python -m pcdf.cli.overfit                          # 32 张，阶段 0
    python -m pcdf.cli.overfit --images 16 --epochs 40 --device cpu --imgsz 320 256
    python -m pcdf.cli.overfit --with-stage2             # 额外验证阶段 1/2
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from ..config import Config, ConfigError
from ..conditions import ConditionTable
from ..paths import display_path
from ..runtime import resolve_device, timestamp
from ..train import train_stage
from .common import add_config_args, add_device_arg, apply_overrides, build_config, load_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcdf-overfit",
        description="小样本过拟合诊断（验证标签、损失、检测头与评估器是否自洽）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_config_args(parser)
    add_device_arg(parser)
    parser.add_argument("--images", type=int, default=32, help="抽样图像对数")
    parser.add_argument("--epochs", type=int, default=60, help="过拟合 epoch 数")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02, help="诊断用学习率（比论文略高，便于快速过拟合）")
    parser.add_argument("--imgsz", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="默认沿用配置")
    parser.add_argument("--min-map", type=float, default=0.2, help="阶段 0 通过阈值（mAP50）")
    parser.add_argument("--with-stage2", action="store_true", help="额外跑阶段 1 与阶段 2")
    parser.add_argument("--out", default=None, help="输出目录；默认 runs/diagnostics/overfit_<时间>")
    parser.add_argument("--keep-data", action="store_true", help="保留复制出来的小数据集")
    return parser


def build_subset(
    cfg: Config, table: ConditionTable, out_dir: Path, n_images: int
) -> tuple[Path, int]:
    """从训练集复制 N 对"有目标"的样本，生成自包含的小数据集。"""
    source_root = Path(cfg.paths.data_root)
    data_root = out_dir / "data"
    for sub in (
        f"{cfg.data.modality_dir}/images/train",
        f"{cfg.data.modality_dir}/labels/train",
        f"{cfg.data.pair_dir}/images/train",
    ):
        (data_root / sub).mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, str]] = []
    for stem in table.stems(cfg.data.train_split):
        row = table.row_by_stem.get(stem, {})
        try:
            if int(row.get("n_objects", "0") or 0) <= 0:
                continue
        except ValueError:
            continue
        ir_img = source_root / cfg.data.modality_dir / "images" / "train" / f"{stem}.jpg"
        rgb_img = source_root / cfg.data.pair_dir / "images" / "train" / f"{stem}.jpg"
        label = source_root / cfg.data.modality_dir / "labels" / "train" / f"{stem}.txt"
        if not (ir_img.is_file() and rgb_img.is_file() and label.is_file()):
            continue
        selected.append(row)
        if len(selected) >= n_images:
            break

    if len(selected) < n_images:
        raise RuntimeError(
            f"只找到 {len(selected)} 对可用样本，少于要求的 {n_images}；"
            "请检查数据目录或减小 --images"
        )

    for row in selected:
        stem = str(row["stem"]).strip()
        shutil.copy2(
            source_root / cfg.data.modality_dir / "images" / "train" / f"{stem}.jpg",
            data_root / cfg.data.modality_dir / "images" / "train" / f"{stem}.jpg",
        )
        shutil.copy2(
            source_root / cfg.data.pair_dir / "images" / "train" / f"{stem}.jpg",
            data_root / cfg.data.pair_dir / "images" / "train" / f"{stem}.jpg",
        )
        shutil.copy2(
            source_root / cfg.data.modality_dir / "labels" / "train" / f"{stem}.txt",
            data_root / cfg.data.modality_dir / "labels" / "train" / f"{stem}.txt",
        )

    columns = table.columns or ["stem", "subset"]
    with open(data_root / "condition_index.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    print(f"[overfit] 小数据集：{len(selected)} 对图 -> {display_path(data_root)}")
    return data_root, len(selected)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = build_config(args)
        device = resolve_device(args.device)
        table = load_table(cfg)

        out_dir = (
            Path(args.out) if args.out
            else Path(cfg.paths.runs_dir) / "diagnostics" / f"overfit_{timestamp().replace(':', '').replace(' ', '_')}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        data_root, n_selected = build_subset(cfg, table, out_dir, args.images)

        # 临时数据集：train 与 val 用同一批图，目的就是"背下来"。
        cfg.paths.data_root = str(data_root)
        cfg.paths.condition_index = str(data_root / "condition_index.csv")
        cfg.data.train_split = "train"
        cfg.data.val_split = "train"
        cfg.train.run_name = "overfit"
        apply_overrides(cfg, {
            "data.fraction": 1.0,
            "data.workers": 0,
            "train.eval_interval": 1,
            "train.val_batches": 0,
            "train.final_lr_ratio": 1.0,      # 短跑用恒定 lr，避免被指数衰减压死
            "train.lr": args.lr,
            "train.save_period": 0,
            "train.save_every_minutes": 0,
            "train.save_every_steps": 0,
        })
        cfg.resolve_paths()
        cfg.validate()
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    imgsz = tuple(args.imgsz) if args.imgsz else tuple(cfg.data.imgsz)
    print(f"[overfit] 输出：{display_path(out_dir)}")
    print(f"[overfit] 阶段 0：{args.epochs} epoch，batch={args.batch}，imgsz={imgsz}，device={device}")

    result = train_stage(
        cfg, stage=0, init_policy="none", device=str(device),
        run_dir=out_dir / "stage0", epochs=args.epochs,
        batch=args.batch, imgsz=imgsz,
    )
    map50 = float(result.best_map50)

    if args.with_stage2:
        stage1 = train_stage(
            cfg, stage=1, init_checkpoint=result.run_dir / "last.pt", device=str(device),
            run_dir=out_dir / "stage1", epochs=max(5, args.epochs // 3),
            batch=args.batch, imgsz=imgsz,
        )
        print(f"[overfit] 阶段 1 best mAP50={stage1.best_map50:.4f}")
        stage2 = train_stage(
            cfg, stage=2, init_checkpoint=stage1.run_dir / "last.pt", device=str(device),
            run_dir=out_dir / "stage2", epochs=max(5, args.epochs // 3),
            batch=args.batch, imgsz=imgsz,
        )
        print(f"[overfit] 阶段 2 best mAP50={stage2.best_map50:.4f}")

    if not args.keep_data:
        shutil.rmtree(data_root, ignore_errors=True)

    if map50 < args.min_map:
        print(
            f"[overfit][FAIL] 阶段 0 best mAP50={map50:.4f} < {args.min_map:.2f}："
            "先查标签/角度约定/损失/评估，不要开全量训练",
            file=sys.stderr,
        )
        return 1

    print(f"[overfit][OK] 阶段 0 best mAP50={map50:.4f} >= {args.min_map:.2f}（{n_selected} 对图）")
    print(f"[overfit] 曲线与记录：{display_path(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
