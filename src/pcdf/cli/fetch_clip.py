"""``pcdf-fetch-clip``：下载 CLIP ViT-B/16 文本编码器（ModelScope 优先，HF 镜像回退）。

只取 6 个必需文件，避免把 flax/tf 权重也拖下来；下完顺便从
``pytorch_model.bin`` 抽出 ``text_projection.pt``，训练时不必再读 571 MB。

示例::

    python -m pcdf.cli.fetch_clip                       # 下载到配置的 text_encoder 目录
    python -m pcdf.cli.fetch_clip --source hf-mirror
    python -m pcdf.cli.fetch_clip --prepare-only        # 不联网，只补投影矩阵
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from ..download import download_file, fmt_size
from ..text_encoder import CLIP_PROJECTION_FILE, CLIP_REQUIRED_FILES

# 文件 -> 最小合理体积，用来识别 HTML 错误页或被截断的下载。
# 清单与 text_encoder 的校验同源，避免"下载脚本说齐了、训练说缺了"。
MIN_SIZES: dict[str, int] = {
    "config.json": 1_000,
    "pytorch_model.bin": 500_000_000,
    "vocab.json": 500_000,
    "merges.txt": 300_000,
    "tokenizer_config.json": 100,
    "special_tokens_map.json": 100,
}
FILES: tuple[str, ...] = (*CLIP_REQUIRED_FILES, "pytorch_model.bin")

SOURCES: dict[str, str] = {
    "modelscope": "https://www.modelscope.cn/models/openai-mirror/clip-vit-base-patch16/resolve/master",
    "hf-mirror": "https://hf-mirror.com/openai/clip-vit-base-patch16/resolve/main",
    "huggingface": "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main",
}
DEFAULT_ORDER: tuple[str, ...] = ("modelscope", "hf-mirror", "huggingface")


def log(message: str) -> None:
    print(message, flush=True)


def file_ok(path: Path, min_size: int) -> bool:
    """体积 + 内容双重检查，识别截断的下载和伪装成文件的错误页。"""
    if not path.is_file() or path.stat().st_size < min_size:
        return False
    try:
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.name == "merges.txt":
            with path.open("rb") as handle:
                if sum(1 for _ in handle) < 40_000:
                    return False
    except (OSError, ValueError):
        return False
    return True


def make_projection(dest: Path) -> bool:
    """从完整权重里取出 ``text_projection.weight``，缓存成 1 MB 的小文件。"""
    cache = dest / CLIP_PROJECTION_FILE
    if cache.is_file() and cache.stat().st_size > 0:
        return True

    bin_path = dest / "pytorch_model.bin"
    safe_path = dest / "model.safetensors"
    weight = None
    try:
        if bin_path.is_file():
            import torch

            log("  读取 pytorch_model.bin 提取 text_projection.weight …")
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
            weight = state.get("text_projection.weight")
            del state
        elif safe_path.is_file():
            from safetensors import safe_open

            with safe_open(str(safe_path), framework="pt", device="cpu") as handle:
                if "text_projection.weight" in handle.keys():
                    weight = handle.get_tensor("text_projection.weight")
        if weight is not None:
            import torch

            torch.save(weight, cache)
    except Exception as exc:  # noqa: BLE001 —— 权重格式问题不应中断整个流程
        log(f"  [warn] 生成 {CLIP_PROJECTION_FILE} 失败：{type(exc).__name__}: {exc}")
        return False

    if weight is None:
        log("  [warn] 权重里没有 text_projection.weight，CLIP 将退化为未投影特征")
        return False
    return True


def _default_dest() -> Path:
    """默认目标目录 = 配置里的 ``model.text_encoder``。"""
    from ..config import load_config

    return Path(load_config().model.text_encoder)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pcdf-fetch-clip",
        description="下载 CLIP ViT-B/16 文本编码器（只取必需文件，支持断点续传）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dest", default=None, help="目标目录；默认取配置的 text_encoder")
    parser.add_argument(
        "--source", default="auto", choices=("auto", *SOURCES),
        help="下载源；auto 依次尝试 ModelScope → hf-mirror → huggingface",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="单次连接/读取超时秒数")
    parser.add_argument("--prepare-only", action="store_true", help="不下载，只补 text_projection.pt")
    parser.add_argument("--force", action="store_true", help="删掉已有权重重新下载")
    args = parser.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve() if args.dest else _default_dest()
    dest.mkdir(parents=True, exist_ok=True)

    if args.force and not args.prepare_only:
        for name in ("pytorch_model.bin", "model.safetensors", CLIP_PROJECTION_FILE):
            (dest / name).unlink(missing_ok=True)
        log("--force：已清除旧权重文件")

    if args.prepare_only:
        ok = make_projection(dest)
        log(f"目录：{dest}")
        return 0 if ok else 1

    order = DEFAULT_ORDER if args.source == "auto" else (args.source,)
    socket.setdefaulttimeout(args.timeout)

    missing = [name for name in FILES if not file_ok(dest / name, MIN_SIZES[name])]
    if not missing:
        log(f"文件已齐全：{dest}")
    else:
        log(f"目标目录：{dest}")
        log(f"待下载：{', '.join(missing)}")
        used: list[str] = []
        for name in missing:
            errors = []
            for source in order:
                url = f"{SOURCES[source]}/{name}"
                log(f"  [{source}] {name}")
                try:
                    download_file(url, dest / name, min_size=MIN_SIZES[name], timeout=args.timeout)
                    used.append(source)
                    break
                except Exception as exc:  # noqa: BLE001 —— 换下一个源
                    log(f"    [warn] {source} 失败：{type(exc).__name__}: {exc}")
                    errors.append(f"{source}: {exc}")
            else:
                log("")
                log("所有下载源都失败。可选做法：")
                log(f"  1) 本机已有 clip-vit-base-patch16，直接打包上传到 {dest}")
                log(f"  2) 换源重试：python -m pcdf.cli.fetch_clip --source hf-mirror --dest {dest}")
                log("  3) 检查容器出网：curl -I --max-time 8 https://www.modelscope.cn")
                for line in errors:
                    log(f"     - {line}")
                return 1
        log(f"下载完成，使用来源：{', '.join(sorted(set(used)))}")

    if not make_projection(dest):
        log(f"[warn] 缺 {CLIP_PROJECTION_FILE}，训练前的权重检查会失败")
        return 1

    log("校验：")
    all_ok = True
    for name in FILES:
        path = dest / name
        ok = file_ok(path, MIN_SIZES[name])
        all_ok = all_ok and ok
        size = path.stat().st_size if path.exists() else 0
        log(f"  {'ok  ' if ok else 'FAIL'} {name:28s} {fmt_size(size):>10s}")
    projection = dest / CLIP_PROJECTION_FILE
    if projection.is_file():
        log(f"  ok   {CLIP_PROJECTION_FILE:28s} {fmt_size(projection.stat().st_size):>10s}")
    else:
        all_ok = False
        log(f"  FAIL {CLIP_PROJECTION_FILE:28s}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
