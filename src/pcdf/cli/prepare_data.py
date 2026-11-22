"""``pcdf-prepare``：把 ATR-UMOD 原始标注转换成配对 YOLO-OBB 目录。

输入（官方发布结构）::

    <root>/train/{images,images_ir,labels,labels_ir}
    <root>/test/ {images,images_ir,labels,labels_ir}

输出::

    <out>/ir/images/{train,val,test}    红外图像（优先硬链接，节省磁盘）
    <out>/ir/labels/{train,val,test}    YOLO-OBB 标签（8 坐标）
    <out>/rgb/images/{train,val,test}   配对的 RGB 图像
    <out>/condition_index.csv           stem, subset, 六个原始条件字段

划分沿用官方 train/test；从 train 中按 ``location + 采集日期`` 的场景组留出
验证集，避免同场景泄漏。标签取自 ``labels_ir``，因为论文 Sec. 5.2 中多模态
检测器训练和评估都使用 IR 标签。

示例::

    python -m pcdf.cli.prepare_data check --root data/ATR-UMOD
    python -m pcdf.cli.prepare_data convert --root data/ATR-UMOD --val-frac 0.15
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import yaml

# 与 configs/conditions.yaml 的 source 字段一一对应；顺序即 CSV 列顺序。
SOURCE_ATTRIBUTES: tuple[str, ...] = (
    "UAVangle", "UAVheight", "weather", "illumination", "time", "location",
)
INDEX_COLUMNS: tuple[str, ...] = ("stem", "subset", *SOURCE_ATTRIBUTES, "n_objects")


def parse_xml(path: Path) -> tuple[int, int, dict[str, str], list[tuple[str, np.ndarray]]]:
    """解析一个 XML：图像尺寸、六个条件字段、以及每个目标的四点多边形。"""
    root = ET.parse(path).getroot()
    width = int(float(root.findtext("size/width") or 0))
    height = int(float(root.findtext("size/height") or 0))
    attrs = {key: (root.findtext(key) or "").strip() for key in SOURCE_ATTRIBUTES}
    objects: list[tuple[str, np.ndarray]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if not name:
            continue
        points: np.ndarray | None = None
        polygon = obj.find("polygon")
        if polygon is not None:
            xs = [polygon.findtext(f"x{i}") for i in range(1, 5)]
            ys = [polygon.findtext(f"y{i}") for i in range(1, 5)]
            if all(value is not None for value in xs + ys):
                points = np.array([[float(x), float(y)] for x, y in zip(xs, ys)], dtype=np.float32)
        if points is None:
            box = obj.find("robndbox")
            if box is not None:
                cx, cy = float(box.findtext("cx") or 0), float(box.findtext("cy") or 0)
                bw, bh = float(box.findtext("w") or 0), float(box.findtext("h") or 0)
                angle = float(box.findtext("angle") or 0)
                points = cv2.boxPoints(((cx, cy), (bw, bh), angle)).astype(np.float32)
        if points is None or points.shape != (4, 2) or not np.isfinite(points).all():
            continue
        objects.append((name, points))
    return width, height, attrs, objects


def order_points(points: np.ndarray) -> np.ndarray:
    """统一多边形绕向与起点，避免 YOLO 标签出现自交。"""
    ordered = points.copy()
    signed = 0.5 * float(
        np.dot(ordered[:, 0], np.roll(ordered[:, 1], -1))
        - np.dot(ordered[:, 1], np.roll(ordered[:, 0], -1))
    )
    if signed < 0:
        ordered = ordered[::-1]
    start = int(np.lexsort((ordered[:, 0], ordered[:, 1]))[0])
    return np.roll(ordered, -start, axis=0)


def to_yolo_line(points: np.ndarray, width: int, height: int, class_id: int) -> str:
    """把一个四点多边形转成归一化 YOLO-OBB 行。"""
    ordered = order_points(points)
    ordered[:, 0] = np.clip(ordered[:, 0] / width, 0.0, 1.0)
    ordered[:, 1] = np.clip(ordered[:, 1] / height, 0.0, 1.0)
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in ordered.reshape(-1))


def link_or_copy(src: Path, dst: Path) -> None:
    """优先硬链接；跨盘或不支持时退回复制。"""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #
def cmd_check(args: argparse.Namespace) -> int:
    """统计原始数据的文件数、类别分布与条件取值，用于转换前核对。"""
    root = Path(args.root)
    for split in ("train", "test"):
        parts = []
        for sub in ("images", "images_ir", "labels", "labels_ir"):
            folder = root / split / sub
            parts.append(f"{sub}={len(list(folder.glob('*'))) if folder.is_dir() else 0}")
        print(f"{split:6s} " + "  ".join(parts))

    files = sorted((root / "train" / "labels_ir").glob("*.xml"))[: args.sample]
    classes: Counter = Counter()
    attrs: dict[str, Counter] = {key: Counter() for key in SOURCE_ATTRIBUTES}
    for path in files:
        _, _, raw, objects = parse_xml(path)
        for key in SOURCE_ATTRIBUTES:
            attrs[key][raw.get(key, "")] += 1
        for name, _ in objects:
            classes[name] += 1
    print(f"\nsampled {len(files)} labels")
    print("classes:", dict(classes.most_common()))
    for key in SOURCE_ATTRIBUTES:
        print(f"{key:14s}", dict(attrs[key].most_common(8)))
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """执行转换、场景组划分、标签写出与 condition_index.csv 生成。"""
    root, out = Path(args.root), Path(args.out)
    train_xml = sorted((root / "train" / "labels_ir").glob("*.xml"))
    test_xml = sorted((root / "test" / "labels_ir").glob("*.xml"))
    if not train_xml:
        raise FileNotFoundError(f"{root / 'train' / 'labels_ir'} 下没有 XML 标注")
    if args.limit:
        train_xml = train_xml[: args.limit]
        test_xml = test_xml[: max(1, args.limit // 8)]

    meta: dict[str, tuple[int, int, dict[str, str]]] = {}
    class_set: set[str] = set()
    for path in train_xml:
        width, height, attrs, objects = parse_xml(path)
        if width <= 0 or height <= 0:
            continue
        meta[path.stem] = (width, height, attrs)
        class_set.update(name for name, _ in objects)
    if args.classes:
        missing = sorted(set(args.classes) - class_set)
        if missing:
            raise ValueError(f"--classes 中有训练集不存在的类别：{missing}")
        class_set = set(args.classes)
    class_names = sorted(class_set)
    class_id = {name: index for index, name in enumerate(class_names)}
    print(f"classes ({len(class_names)}): {class_names}")

    # 场景组 = 地点 + 采集日期，整组留出，避免同一场景同时出现在 train/val。
    groups = {
        stem: f"{attrs.get('location') or '?'}|{(attrs.get('time') or '').split('T')[0] or '?'}"
        for stem, (_, _, attrs) in meta.items()
    }
    group_sizes = Counter(groups.values())
    keys = sorted(group_sizes)
    random.Random(args.seed).shuffle(keys)
    target = max(1, int(round(len(meta) * args.val_frac)))
    val_groups: set[str] = set()
    accumulated = 0
    for key in keys:
        if accumulated >= target:
            break
        val_groups.add(key)
        accumulated += group_sizes[key]
    print(f"scene groups: {len(keys)}, hold out {len(val_groups)} groups / {accumulated} images")

    for view in ("ir", "rgb"):
        for subset in ("train", "val", "test"):
            (out / view / "images" / subset).mkdir(parents=True, exist_ok=True)
            (out / view / "labels" / subset).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for split, files in (("train", train_xml), ("test", test_xml)):
        for index, xml_path in enumerate(files):
            stem = xml_path.stem
            subset = "test" if split == "test" else ("val" if groups.get(stem) in val_groups else "train")
            width, height, attrs, objects = parse_xml(xml_path)
            if width <= 0 or height <= 0:
                continue
            rgb_path = root / split / "images" / f"{stem}.jpg"
            ir_path = root / split / "images_ir" / f"{stem}.jpg"
            if not rgb_path.is_file() or not ir_path.is_file():
                continue
            lines = [
                to_yolo_line(points, width, height, class_id[name])
                for name, points in objects
                if name in class_id
            ]
            link_or_copy(rgb_path, out / "rgb" / "images" / subset / f"{stem}.jpg")
            link_or_copy(ir_path, out / "ir" / "images" / subset / f"{stem}.jpg")
            (out / "ir" / "labels" / subset / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
            )
            rows.append({
                "stem": stem, "subset": subset,
                **{key: attrs.get(key, "") for key in SOURCE_ATTRIBUTES},
                "n_objects": len(lines),
            })
            if (index + 1) % 2000 == 0:
                print(f"  {split}: {index + 1}/{len(files)}")

    data_yaml = {
        "path": str((out / "ir").resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
        "channels": 6,
    }
    (out / "ir" / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8", newline="\n"
    )
    with open(out / "condition_index.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"done -> {out}  {dict(Counter(str(row['subset']) for row in rows))}")
    return 0


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pcdf-prepare",
        description="ATR-UMOD 原始标注 -> 配对 YOLO-OBB 目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", help="统计原始数据规模与类别/条件分布")
    check.add_argument("--root", required=True, help="原始数据根目录")
    check.add_argument("--sample", type=int, default=300, help="抽样统计的标注数")
    check.set_defaults(func=cmd_check)

    convert = sub.add_parser("convert", help="执行转换")
    convert.add_argument("--root", required=True)
    convert.add_argument("--out", default=None, help="输出目录；默认取配置的 data_root")
    convert.add_argument("--val-frac", type=float, default=0.15, help="验证集比例（按场景组）")
    convert.add_argument("--limit", type=int, default=0, help="只转换前 N 个训练样本（调试）")
    convert.add_argument("--seed", type=int, default=0, help="场景组打乱种子")
    convert.add_argument("--classes", nargs="+", default=None, help="显式指定类别列表与顺序")
    convert.set_defaults(func=cmd_convert)

    args = parser.parse_args(argv)
    if args.cmd == "convert" and not args.out:
        # 未指定 --out 时落到配置里的 data_root（环境变量可覆盖）。
        from ..config import load_config

        args.out = load_config().paths.data_root
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
