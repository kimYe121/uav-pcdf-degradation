"""ATR-UMOD -> YOLO-OBB conversion with scene-level splitting.

Layout expected (matches the released dataset):

    <root>/train/{images,images_ir,labels,labels_ir}
    <root>/test/ {images,images_ir,labels,labels_ir}

Each XML carries six condition attributes (UAVangle, UAVheight, weather,
illumination, time, location) and per-object <polygon> with four points.
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

ATTRS = ["UAVangle", "UAVheight", "weather", "illumination", "time", "location"]
SPLITS = ("train", "test")
VIEWS = ("rgb", "ir", "fused")


def parse_xml(path: Path) -> tuple[int, int, dict, list[tuple[str, np.ndarray]]]:
    root = ET.parse(path).getroot()
    width = int(float(root.findtext("size/width") or 0))
    height = int(float(root.findtext("size/height") or 0))
    attrs = {k: (root.findtext(k) or "").strip() for k in ATTRS}

    objects: list[tuple[str, np.ndarray]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if not name:
            continue
        pts = None
        poly = obj.find("polygon")
        if poly is not None:
            xs = [poly.findtext(f"x{i}") for i in range(1, 5)]
            ys = [poly.findtext(f"y{i}") for i in range(1, 5)]
            if all(v is not None for v in xs + ys):
                pts = np.array([[float(x), float(y)] for x, y in zip(xs, ys)], dtype=np.float32)
        if pts is None:
            rb = obj.find("robndbox")
            if rb is not None:
                cx = float(rb.findtext("cx") or 0)
                cy = float(rb.findtext("cy") or 0)
                bw = float(rb.findtext("w") or 0)
                bh = float(rb.findtext("h") or 0)
                ang = float(rb.findtext("angle") or 0)
                pts = cv2.boxPoints(((cx, cy), (bw, bh), ang)).astype(np.float32)
        if pts is None or pts.shape != (4, 2) or not np.isfinite(pts).all():
            continue
        objects.append((name, pts))
    return width, height, attrs, objects


def poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def order_points(p: np.ndarray) -> np.ndarray:
    """Force a consistent winding and start vertex so polygons never self-intersect."""
    q = p.copy()
    signed = 0.5 * float(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
    if signed < 0:
        q = q[::-1]
    start = int(np.lexsort((q[:, 0], q[:, 1]))[0])
    return np.roll(q, -start, axis=0)


def to_yolo_line(pts: np.ndarray, width: int, height: int, class_id: int) -> str:
    q = order_points(pts)
    q[:, 0] = np.clip(q[:, 0] / width, 0.0, 1.0)
    q[:, 1] = np.clip(q[:, 1] / height, 0.0, 1.0)
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in q.reshape(-1))


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def cmd_check(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for split in SPLITS:
        row = []
        for sub in ("images", "images_ir", "labels", "labels_ir"):
            n = len(list((root / split / sub).glob("*"))) if (root / split / sub).is_dir() else 0
            row.append(f"{sub}={n}")
        print(f"{split:6s} " + "  ".join(row))

    files = sorted((root / "train" / "labels_ir").glob("*.xml"))[: args.sample]
    classes: Counter = Counter()
    attr_vals: dict[str, Counter] = {k: Counter() for k in ATTRS}
    ratios, arity = [], Counter()
    for f in files:
        w, h, attrs, objs = parse_xml(f)
        if w and h:
            arity[f"{w}x{h}"] += 1
        for k in ATTRS:
            if attrs.get(k):
                attr_vals[k][attrs[k]] += 1
        for name, pts in objs:
            classes[name] += 1
            ratios.append(poly_area(pts))
        if objs:
            arity[f"objs/img={len(objs)}"] += 1

    print(f"\nsampled XMLs: {len(files)}")
    print("\nclass instances:")
    for name, n in classes.most_common():
        print(f"  {name:14s} {n}")
    print("\nattributes (value: count):")
    for k in ATTRS:
        vals = attr_vals[k].most_common(8)
        print(f"  {k:14s} {vals}")
    print("\nimage sizes / object counts:")
    for k, n in arity.most_common(6):
        print(f"  {k:20s} {n}")

    if ratios:
        arr = np.array(ratios)
        print(f"\npolygon area px: median={np.median(arr):.1f} p05={np.percentile(arr,5):.1f} p95={np.percentile(arr,95):.1f}")


def cmd_convert(args: argparse.Namespace) -> None:
    root, out = Path(args.root), Path(args.out)
    train_xml = sorted((root / "train" / "labels_ir").glob("*.xml"))
    if args.limit:
        train_xml = train_xml[: args.limit]
    test_xml = sorted((root / "test" / "labels_ir").glob("*.xml"))
    if args.limit:
        test_xml = test_xml[: max(1, args.limit // 8)]

    meta: dict[str, tuple[int, int, dict]] = {}
    class_set: set[str] = set()
    for p in train_xml:
        w, h, attrs, objs = parse_xml(p)
        meta[p.stem] = (w, h, attrs)
        class_set.update(name for name, _ in objs)
    class_names = sorted(class_set)
    class_id = {n: i for i, n in enumerate(class_names)}
    print(f"classes ({len(class_names)}): {class_names}")

    def group_of(stem: str) -> str:
        attrs = meta[stem][2]
        loc = attrs.get("location") or "?"
        day = (attrs.get("time") or "").split("T")[0] or "?"
        return f"{loc}|{day}"

    groups = {stem: group_of(stem) for stem in meta}
    keys = sorted(set(groups.values()))
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * args.val_frac)))
    val_keys = set(keys[:n_val])
    print(f"scene groups: {len(keys)}, holding out {n_val} -> val")

    subsets = {"train": "train", "test": "test"}
    for view in VIEWS:
        for subset in ("train", "val", "test"):
            (out / view / "images" / subset).mkdir(parents=True, exist_ok=True)
            (out / view / "labels" / subset).mkdir(parents=True, exist_ok=True)

    index_rows = []
    for split, files in (("train", train_xml), ("test", test_xml)):
        for i, xml_path in enumerate(files):
            stem = xml_path.stem
            if split == "test":
                subset = "test"
            else:
                subset = "val" if groups.get(stem) in val_keys else "train"

            w, h, attrs, objs = parse_xml(xml_path)
            rgb_path = root / split / "images" / f"{stem}.jpg"
            ir_path = root / split / "images_ir" / f"{stem}.jpg"
            if not rgb_path.exists() or not ir_path.exists():
                continue

            lines = [to_yolo_line(pts, w, h, class_id[name]) for name, pts in objs if name in class_id]
            text = "\n".join(lines) + ("\n" if lines else "")

            for view in ("rgb", "ir"):
                link_or_copy(rgb_path if view == "rgb" else ir_path, out / view / "images" / subset / f"{stem}.jpg")
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
            if rgb is not None and ir is not None:
                if ir.shape[:2] != rgb.shape[:2]:
                    ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                fused = rgb.copy()
                fused[..., 0] = ir
                cv2.imwrite(str(out / "fused" / "images" / subset / f"{stem}.jpg"), fused, [cv2.IMWRITE_JPEG_QUALITY, 95])

            for view in VIEWS:
                (out / view / "labels" / subset / f"{stem}.txt").write_text(text, encoding="utf-8")

            index_rows.append(
                {
                    "stem": stem,
                    "subset": subset,
                    **{k: attrs.get(k, "") for k in ATTRS},
                    "n_objects": len(lines),
                }
            )
            if (i + 1) % 2000 == 0:
                print(f"  {split}: {i + 1}/{len(files)}")

    for view in VIEWS:
        payload = {
            "path": str((out / view).resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: n for i, n in enumerate(class_names)},
        }
        (out / view / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with open(out / "condition_index.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stem", "subset", *ATTRS, "n_objects"])
        writer.writeheader()
        writer.writerows(index_rows)

    counts = Counter(r["subset"] for r in index_rows)
    print(f"\ndone -> {out}")
    print(f"  {dict(counts)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check")
    c.add_argument("--root", required=True)
    c.add_argument("--sample", type=int, default=300)
    c.set_defaults(func=cmd_check)

    v = sub.add_parser("convert")
    v.add_argument("--root", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--val-frac", type=float, default=0.15)
    v.add_argument("--limit", type=int, default=0, help="use only the first N training pairs")
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(func=cmd_convert)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
