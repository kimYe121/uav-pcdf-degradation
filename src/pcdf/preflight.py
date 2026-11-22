"""资源就绪检查（不导入 torch，秒级返回）。

用途
----
shell 启动器需要在训练前快速判断"权重齐不齐、数据在不在"。这一步以前由
``common.sh`` 里的 ``weights_ready`` / ``data_ready`` 承担，路径与文件名清单
与 Python 侧各写一份，改权重布局时要同时改两处。

现在清单只存在于这里：

* 权重清单派生自 ``ModelConfig``（``encoder_weights`` / ``text_encoder``），
  不再硬编码 ``yolov5su.pt``；
* 数据清单派生自 ``DataConfig``（split 名、modality 目录、condition_index）；
* 因为不 import torch，``python -m pcdf.preflight`` 在 CPU 跳板机上也能跑。

命令行::

    python -m pcdf.preflight weights          # 缺权重时退出码 1
    python -m pcdf.preflight data
    python -m pcdf.preflight all --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config, load_config
from .text_encoder import (
    CLIP_PROJECTION_FILE,
    CLIP_REQUIRED_FILES,
    CLIP_WEIGHT_FILES,
)


@dataclass
class Item:
    """单项检查结果。"""

    name: str
    ok: bool
    detail: str
    level: str = "error"  # error | warn

    def render(self) -> str:
        mark = "ok  " if self.ok else ("warn" if self.level == "warn" else "FAIL")
        return f"  [{mark}] {self.name:<34} {self.detail}"


def _format_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def check_weights(cfg: Config) -> list[Item]:
    """检查骨架权重与 CLIP 目录。

    CLIP 必需文件清单来自 :mod:`pcdf.text_encoder`，与 ``fetch_clip.py`` 的
    下载清单同源，避免"下载脚本说齐了、训练脚本说缺了"。
    """
    items: list[Item] = []
    backbone = Path(cfg.model.encoder_weights)
    if backbone.is_file():
        items.append(Item("骨架权重", True, f"{backbone}（{_format_size(backbone)}）"))
    else:
        items.append(Item(
            "骨架权重", False,
            f"缺少 {backbone}；把 Ultralytics 权重放到 {cfg.paths.weights_dir}/ 或用 profile 换档",
        ))

    clip_dir = Path(cfg.model.text_encoder)
    if not clip_dir.is_dir():
        items.append(Item(
            "CLIP 目录", False,
            f"缺少 {clip_dir}（跑 python -m pcdf.cli.fetch_clip，或 bash scripts/server/setup.sh）",
        ))
        return items

    missing = [name for name in CLIP_REQUIRED_FILES if not (clip_dir / name).is_file()]
    if not any((clip_dir / name).is_file() for name in CLIP_WEIGHT_FILES):
        missing.append(" / ".join(CLIP_WEIGHT_FILES))
    items.append(Item(
        "CLIP 文本编码器", not missing,
        f"{clip_dir}" if not missing else f"缺少 {missing}",
    ))
    projection = clip_dir / CLIP_PROJECTION_FILE
    items.append(Item(
        "CLIP text_projection", projection.is_file(),
        str(projection) if projection.is_file() else "缺少投影矩阵；跑 fetch_clip.py --prepare-only 生成",
        level="warn",
    ))
    return items


def _count_images(folder: Path) -> int:
    """数一个目录下的图片；只扫一层，避免在 5 万张图上做递归统计。"""
    if not folder.is_dir():
        return 0
    return sum(1 for entry in folder.iterdir() if entry.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))


def check_data(cfg: Config, *, count: bool = True) -> list[Item]:
    """检查配对数据目录与条件索引。"""
    root = Path(cfg.paths.data_root)
    items = [
        Item("数据根目录", root.is_dir(), str(root)),
        Item("条件索引", Path(cfg.paths.condition_index).is_file(), str(cfg.paths.condition_index)),
        Item("条件码本", Path(cfg.model.condition_spec).is_file(), str(cfg.model.condition_spec)),
    ]
    for split in (cfg.data.train_split, cfg.data.val_split):
        images = root / cfg.data.modality_dir / "images" / split
        labels = root / cfg.data.modality_dir / "labels" / split
        pairs = root / cfg.data.pair_dir / "images" / split
        n_images = _count_images(images) if count else int(images.is_dir())
        ok = images.is_dir() and labels.is_dir() and pairs.is_dir() and n_images > 0
        detail = f"{n_images} 张" if ok else f"缺少 {images} / {labels} / {pairs} 之一"
        items.append(Item(f"{split} 配对数据", ok, detail))
    return items


def check_all(cfg: Config) -> list[Item]:
    """权重 + 数据；任一 ``error`` 视为未就绪。"""
    return check_weights(cfg) + check_data(cfg)


def ready(items: list[Item]) -> bool:
    """是否所有必需项都通过（``warn`` 不影响就绪）。"""
    return all(item.ok or item.level == "warn" for item in items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="资源就绪检查（不导入 torch）")
    parser.add_argument("target", nargs="?", default="all", choices=("weights", "data", "all"))
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--profile", default=None, help="配置档位")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--quiet", action="store_true", help="只输出失败项")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, profile=args.profile)
    items = {
        "weights": check_weights,
        "data": check_data,
        "all": check_all,
    }[args.target](cfg)

    if args.json:
        print(json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2))
    else:
        shown = [item for item in items if not args.quiet or not item.ok]
        # --quiet 只用于 shell 的"通过就闭嘴"场景：成功时不打印任何东西。
        if not args.quiet:
            print(f"[preflight] {args.target}（data_root={cfg.paths.data_root}）")
        for item in shown:
            print(item.render())
    return 0 if ready(items) else 1


if __name__ == "__main__":
    sys.exit(main())
