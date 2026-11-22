"""``pcdf-fetch-backbone``：下载骨架预训练权重，并校验档位是否匹配。

为什么需要它
------------
服务器上最容易出的事故是"权重档位与配置不一致"：配置写 v5s、目录里放的却是
v5n，逐层加载时报一个某层 shape 不匹配的错误，看不出根因。这个命令把三件事
放在一起做：

1. 按配置里的文件名（``model.encoder_weights``）下载到 ``weights_dir``；
2. 下载后立刻反查真实规模，与 ``model.backbone_scale`` 比对；
3. 不一致就删掉坏文件并以非零码退出，避免"看起来装好了"。

示例::

    python -m pcdf.cli.fetch_backbone                  # 按配置下载并校验
    python -m pcdf.cli.fetch_backbone --url <镜像地址>  # 指定镜像
    PCDF_BACKBONE_URL=<地址1>,<地址2> python -m pcdf.cli.fetch_backbone
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..config import load_config
from ..download import download_first, fmt_size
from ..models.encoder import infer_v5_scale

# 默认源：官方 release + 国内常用 GitHub 代理。
# 允许用 --url / PCDF_BACKBONE_URL 完全覆盖（容器不通外网时应改用内网地址）。
DEFAULT_RELEASE = "v8.3.0"
ASSET_BASE = f"https://github.com/ultralytics/assets/releases/download/{DEFAULT_RELEASE}"
MIRROR_PREFIX = "https://ghproxy.net/"
MIN_SIZE = 1_000_000


def candidate_urls(asset: str, release: str, extra: list[str] | None = None) -> list[str]:
    """组装候选 URL：显式参数 > 环境变量 > 官方 + 代理。"""
    if extra:
        return list(extra)
    env = os.environ.get("PCDF_BACKBONE_URL", "").strip()
    if env:
        return [item.strip() for item in env.split(",") if item.strip()]
    official = f"https://github.com/ultralytics/assets/releases/download/{release}/{asset}"
    return [official, MIRROR_PREFIX + official]


def verify(path: Path, asset: str, expected_scale: str | None) -> tuple[bool, str]:
    """校验权重文件可读、是检测模型、且规模与配置一致。"""
    if not path.is_file():
        return False, "文件不存在"
    if path.stat().st_size < MIN_SIZE:
        return False, f"体积只有 {fmt_size(path.stat().st_size)}，疑似错误页"
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 —— 损坏/非 torch 文件
        return False, f"无法读取：{type(exc).__name__}: {exc}"
    module = payload.get("model") if isinstance(payload, dict) else payload
    state = module.state_dict() if hasattr(module, "state_dict") else None
    if state is None or "model.0.conv.weight" not in state:
        return False, "不是 Ultralytics 检测模型检查点（缺 model.0.conv.weight）"

    if "yolov5" in asset and expected_scale:
        actual = infer_v5_scale(path)
        if actual and actual != expected_scale:
            return False, f"规模不匹配：文件是 {actual!r}，配置要求 {expected_scale!r}"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pcdf-fetch-backbone",
        description="下载骨架预训练权重并校验档位",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dest", default=None, help="目标文件；默认取配置的 encoder_weights")
    parser.add_argument("--url", action="append", default=None, help="自定义下载地址，可重复")
    parser.add_argument("--release", default=DEFAULT_RELEASE, help=f"Ultralytics release（默认 {DEFAULT_RELEASE}）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    parser.add_argument("--check-only", action="store_true", help="只校验，不下载")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    cfg = load_config()
    dest = Path(args.dest).expanduser().resolve() if args.dest else Path(cfg.model.encoder_weights)
    asset = dest.name
    expected_scale = cfg.model.backbone_scale if "yolov5" in cfg.model.backbone_yaml else None

    if dest.is_file() and not args.force:
        ok, detail = verify(dest, asset, expected_scale)
        print(f"[backbone] 已存在：{dest}（{fmt_size(dest.stat().st_size)}）-> {detail}")
        if ok:
            return 0
        print("[backbone] 文件不可用，删除后重新下载", file=sys.stderr)
        dest.unlink()

    if args.check_only:
        print(f"[backbone] 缺失或不可用：{dest}", file=sys.stderr)
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    urls = candidate_urls(asset, args.release, args.url)
    print(f"[backbone] 目标：{dest}")
    try:
        used = download_first(urls, dest, min_size=MIN_SIZE, timeout=args.timeout)
    except OSError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print("手动方案：把本机 weights/ 下同名文件 scp 到该路径，然后跑 --check-only", file=sys.stderr)
        return 1

    ok, detail = verify(dest, asset, expected_scale)
    if not ok:
        dest.unlink(missing_ok=True)
        print(f"[error] 下载的文件未通过校验：{detail}", file=sys.stderr)
        return 1
    print(f"[backbone] ok {asset} -> {dest}（{fmt_size(dest.stat().st_size)}，来源 {used}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
