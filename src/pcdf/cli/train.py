"""``pcdf-train``：按论文三阶段协议训练。

示例::

    python -m pcdf.cli.train --stage 0                    # baseline
    python -m pcdf.cli.train --stage 1                    # SCPL+CDF（自动接 baseline）
    python -m pcdf.cli.train --stage 2 --profile 48g      # full pipeline
    python -m pcdf.cli.train --stage 1 --resume auto      # 断点续训
    python -m pcdf.cli.train --stage 1 --device cpu --epochs 1 --max-batches 2
"""

from __future__ import annotations

import argparse
import sys

from ..checkpoint import CheckpointError
from ..config import ConfigError
from ..train import train_stage
from .common import add_config_args, add_device_arg, apply_overrides, build_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcdf-train",
        description="训练 PCDF 的某一阶段（0=baseline，1=SCPL+CDF，2=full pipeline）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_config_args(parser)
    add_device_arg(parser)
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--init", default=None, help="初始权重（只加载模型，不恢复优化器）")
    parser.add_argument(
        "--no-init", action="store_true",
        help="跳过阶段初始化，直接用主干预训练权重（诊断用，不是论文协议）",
    )
    parser.add_argument("--run-dir", default=None, help="运行目录；默认 <runs_dir>/<run_name>_stage<N>")
    parser.add_argument("--run-name", default=None, help="覆盖 train.run_name（影响默认运行目录）")
    parser.add_argument("--epochs", type=int, default=None, help="本次运行的总 epoch 数")
    parser.add_argument("--batch", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--imgsz", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--workers", type=int, default=None, help="DataLoader worker 数")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="每个 epoch 限制 step 数")
    parser.add_argument("--val-batches", type=int, default=None, help="验证时限制 batch 数")
    parser.add_argument(
        "--resume", default=None,
        help="断点续训：检查点路径、运行目录或 auto（读取 run-dir/last.pt）",
    )
    parser.add_argument("--lr", type=float, default=None, help="覆盖初始学习率")
    parser.add_argument("--warmup-epochs", type=float, default=None, help="覆盖 warmup 轮数")
    parser.add_argument("--warmup-bias-lr", type=float, default=None, help="warmup 起始学习率比例")
    parser.add_argument("--final-lr-ratio", type=float, default=None, help="指数衰减终点比例")
    parser.add_argument("--grad-accum", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--no-amp", action="store_true", help="关闭 AMP（排查数值问题时用）")
    parser.add_argument("--save-period", type=int, default=None, help="每 N 个 epoch 保存 epoch_XXXX.pt")
    parser.add_argument("--save-every-minutes", type=float, default=None, help="每 N 分钟保存 last.pt")
    parser.add_argument("--save-every-steps", type=int, default=None, help="每 N 个 step 保存 last.pt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = build_config(args)
        # 只把"训练配方"写回配置；batch/imgsz 由 train_stage 统一处理，
        # 避免同一参数在两条路径上各写一次。
        apply_overrides(cfg, {
            "train.seed": args.seed,
            "train.lr": args.lr,
            "train.warmup_epochs": args.warmup_epochs,
            "train.warmup_bias_lr": args.warmup_bias_lr,
            "train.final_lr_ratio": args.final_lr_ratio,
            "train.grad_accum": args.grad_accum,
            "train.save_period": args.save_period,
            "train.save_every_minutes": args.save_every_minutes,
            "train.save_every_steps": args.save_every_steps,
            "train.run_name": args.run_name,
            "data.workers": args.workers,
            "train.amp": False if args.no_amp else None,
        })
        result = train_stage(
            cfg,
            stage=args.stage,
            init_checkpoint=args.init,
            init_policy="none" if args.no_init else "auto",
            device=args.device,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=tuple(args.imgsz) if args.imgsz else None,
            max_batches=args.max_batches,
            val_batches=args.val_batches,
            resume=args.resume,
        )
    except (ConfigError, CheckpointError, FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    state = "interrupted" if result.interrupted else "finished"
    best = f"{result.best_map50:.4f}" if result.best_map50 >= 0 else "-"
    print(
        f"[done] run_dir={result.run_dir} best_mAP50={best} "
        f"epochs={len(result.history)}/{result.total_epochs} state={state}"
    )
    if result.interrupted:
        print(f"[done] 续训：python -m pcdf.cli.train --stage {args.stage} --resume auto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
