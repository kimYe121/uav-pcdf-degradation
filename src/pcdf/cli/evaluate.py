"""``pcdf-eval``：用旋转框 mAP 评估检查点。

评估只走视觉引导 ``F_s``，不读取条件标注。模型结构优先取自检查点里保存的
``cfg``——这样"用 48G 档训练、用 paper 档评估"这类错配不会静默发生；路径
（数据/权重）则以本机配置与环境变量为准。

口径约定（见 ``docs/STATE.md`` §4.6）：``val`` 是从官方 train 里整组留出的
跨场景子集，长尾类别样本极少，**同一权重比 test 低约 16 个点**。因此
``--split both`` 会同时打印 ``val`` 与 ``test``；对外报数只用 ``test``。

示例::

    python -m pcdf.cli.evaluate --checkpoint runs/base20/best.pt --split both
    python -m pcdf.cli.evaluate --checkpoint runs/pcdf_stage2/best.pt --split test --map50-95
    python -m pcdf.cli.evaluate --checkpoint runs/pcdf_stage0/best.pt --device cpu --max-batches 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ..checkpoint import CheckpointError
from ..config import Config, ConfigError
from ..data import build_loader
from ..evaluate import MAP5095_THRESHOLDS, evaluate_obb
from ..models.baseline import BaselineNet
from ..models.pcdf import PCDFNet
from ..runtime import resolve_device
from ..text_encoder import CLIPTextEncoder
from .common import (
    add_config_args,
    add_device_arg,
    build_config,
    config_from_checkpoint,
    load_table,
)

SPLITS = ("train", "val", "test")
#: ``--split both`` 展开成这两个：val 用于选权重，test 才是论文报告口径。
BOTH_SPLITS = ("val", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcdf-eval",
        description="评估检查点的旋转框 mAP（只走视觉引导，不需要条件标注）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_config_args(parser)
    add_device_arg(parser)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    parser.add_argument(
        "--split", default=None, choices=(*SPLITS, "both"),
        help="默认取 data.val_split；both = 同时跑 val 与 test（推荐，避免口径混用）",
    )
    parser.add_argument("--batch", type=int, default=None, help="评估 batch（默认沿用配置）")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker 数（默认 0）")
    parser.add_argument("--max-batches", type=int, default=0, help="限制评估 batch 数（0=全部）")
    parser.add_argument("--map50-95", action="store_true", help="额外计算 mAP50-95")
    parser.add_argument("--guidance", default="visual", choices=("visual", "text"),
                        help="stage1 检查点必须用 text（PCD 未训，视觉引导是随机初始化）")
    parser.add_argument("--fuse", default=None, choices=("raw", "decoupled"),
                        help="CDF 融合对象；默认随 guidance（text→raw，visual→decoupled）。"
                             "诊断 stage2 的文本引导时用 --guidance text --fuse decoupled")
    parser.add_argument("--output", default=None, help="结果 JSON；默认与检查点同目录")
    return parser


def load_checkpoint(path: str | Path) -> dict:
    """读取检查点；失败时给出可操作的提示，而不是 torch 的原始报错。"""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"未找到检查点：{checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 —— 损坏/格式不对都归为不可用
        raise CheckpointError(f"无法读取检查点 {checkpoint_path}：{exc}") from exc
    if not isinstance(payload, dict) or "model" not in payload:
        raise CheckpointError(f"{checkpoint_path} 不是训练检查点（缺少 model 字段）")
    return payload


def infer_arch(payload: dict) -> str:
    """确定架构：优先读 ``arch``，旧检查点按权重键名推断。"""
    arch = payload.get("arch")
    if arch in ("baseline", "pcdf"):
        return arch
    keys = payload.get("model", {})
    return "baseline" if not any(key.startswith("scpl.") for key in keys) else "pcdf"


def build_model(cfg: Config, arch: str, table, device: torch.device):
    """按架构构建模型；baseline 不加载 CLIP。"""
    if arch == "baseline":
        return BaselineNet(cfg.model, nc=cfg.data.nc).to(device)
    encoder = CLIPTextEncoder(cfg.model.text_encoder, device=device)
    model = PCDFNet(cfg.model, nc=cfg.data.nc, table=table, text_encoder=encoder)
    return model.to(device)


def evaluate_split(
    cfg: Config,
    model,
    table,
    device: torch.device,
    split: str,
    *,
    batch: int,
    workers: int,
    max_batches: int,
    thresholds: tuple[float, ...],
    guidance: str = "visual",
    fuse: str | None = None,
) -> dict:
    """在单个 split 上算指标。"""
    _, loader = build_loader(
        cfg.data, table, split=split, augment=False, root=cfg.paths.data_root,
        batch_size=batch, workers=workers, shuffle=False, device=device,
    )
    return evaluate_obb(
        model, loader, device, nc=cfg.data.nc,
        conf=cfg.data.eval_conf, iou=cfg.data.eval_iou, max_det=cfg.data.eval_max_det,
        max_batches=max_batches, thresholds=thresholds, names=cfg.data.names,
        guidance=guidance, fuse=fuse,
    )


def print_result(split: str, result: dict) -> None:
    print(f"[eval] split={split:<5} mAP50={result['map50']:.4f}  "
          f"mAP50-95={result['map50_95']:.4f}  "
          f"images={result['n_images']} gt={result['n_gt']} pred={result['n_pred']}")
    for name, ap in result["per_class_ap50"].items():
        print(f"    {name:<14}{ap:.4f}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        env_cfg = build_config(args)
        payload = load_checkpoint(args.checkpoint)
        arch = infer_arch(payload)

        ckpt_cfg = payload.get("cfg")
        if ckpt_cfg:
            cfg = config_from_checkpoint(ckpt_cfg, env_cfg)
        else:
            # 旧检查点没有配置快照：用当前配置，并在日志里明确提示。
            cfg = env_cfg
            print("[eval] 该检查点没有保存配置快照，按当前 --config/--profile 建模", flush=True)

        batch = int(args.batch or cfg.data.batch)
        if args.batch:
            cfg.data.batch = batch
        device = resolve_device(args.device)

        table = load_table(cfg)
        model = build_model(cfg, arch, table, device)
        # strict=True：架构判定或配置快照不对时立刻报错，不给出"看似能评估"的假结果。
        model.load_state_dict(payload["model"], strict=True)
        model.eval()

        if args.split == "both":
            splits: tuple[str, ...] = BOTH_SPLITS
        elif args.split:
            splits = (args.split,)
        else:
            splits = (cfg.data.val_split,)
        thresholds = MAP5095_THRESHOLDS if args.map50_95 else (0.5,)

        results: dict[str, dict] = {}
        for split in splits:
            results[split] = evaluate_split(
                cfg, model, table, device, split,
                batch=batch, workers=args.workers,
                max_batches=args.max_batches, thresholds=thresholds,
                guidance=args.guidance, fuse=args.fuse,
            )
    except (ConfigError, CheckpointError, FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    common = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "arch": arch,
        "profile": cfg.train.profile,
        "imgsz": list(cfg.data.imgsz),
    }
    print(f"[eval] arch={arch} profile={cfg.train.profile or 'base'} "
          f"imgsz={list(cfg.data.imgsz)} device={device}")
    for split, result in results.items():
        print_result(split, result)

    if "val" in results and "test" in results:
        gap = results["test"]["map50"] - results["val"]["map50"]
        print(f"[eval] 口径差 test - val = {gap:+.4f}"
              f"（对外报数用 test；val 只用于选 best.pt）")

    single = len(results) == 1
    if single:
        (split, result), = results.items()
        payload_out = {**result, **common, "split": split}
        default_name = f"eval_{split}.json"
    else:
        payload_out = {**common, "splits": results,
                       "map50": results[splits[0]]["map50"]}
        default_name = "eval_both.json"

    out = Path(args.output) if args.output else Path(args.checkpoint).with_name(default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())