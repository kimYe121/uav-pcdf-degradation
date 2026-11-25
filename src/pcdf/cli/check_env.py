"""``pcdf-check``：环境、依赖、权重、数据与提示自检。

除了打印版本号，还会真正 import 一遍训练路径依赖的符号——上游小版本升级
可能悄悄改掉某个类的位置，只看版本号发现不了，等训练启动才报错。

旧版本在这里有一个致命顺序错误：``load_config`` 在函数末尾才被 import，
但前面已经用到它，于是脚本必然抛 ``UnboundLocalError``——文档里"第一步跑
自检"直接不可用。现在所有 import 都在模块顶层，且权重清单来自
:mod:`pcdf.preflight`，与 shell 启动器同源。

示例::

    python -m pcdf.cli.check_env
    python -m pcdf.cli.check_env --skip-data      # 数据还没传完时
    python -m pcdf.cli.check_env --json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib import metadata

from ..config import ConfigError
from ..conditions import ConditionTable
from ..data import build_dataset
from ..preflight import check_data, check_weights, ready
from ..runtime import cuda_is_usable, environment_meta, resolve_device
from .common import add_config_args, add_device_arg, build_config

# 训练/评估实际依赖的符号。任何一项失败都意味着代码与已装版本不匹配。
PROBES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("ultralytics Conv / C3k2", "ultralytics.nn.modules", ("Conv", "C3k2")),
    ("ultralytics DetectionModel", "ultralytics.nn.tasks", ("DetectionModel", "initialize_weights")),
    ("ultralytics batch_probiou", "ultralytics.utils.metrics", ("batch_probiou",)),
    ("ultralytics NMS", "ultralytics.utils.nms", ("non_max_suppression",)),
    ("ultralytics dataset", "ultralytics.data.dataset", ("YOLODataset",)),
    ("ultralytics augment", "ultralytics.data.augment",
     ("Compose", "LetterBox", "RandomFlip", "RandomPerspective")),
    ("ultralytics get_cfg", "ultralytics.cfg", ("get_cfg",)),
    ("transformers CLIP", "transformers", ("CLIPTextModel", "CLIPTokenizer")),
)


def dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "缺失"


def probe(label: str, module: str, attrs: tuple[str, ...]) -> tuple[bool, str]:
    """import 模块并检查关键符号是否还在。"""
    try:
        mod = importlib.import_module(module)
        missing = [a for a in attrs if not hasattr(mod, a)]
        if missing:
            raise AttributeError(f"缺少 {', '.join(missing)}")
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 —— 任何 import 失败都算不兼容
        return False, f"{type(exc).__name__}: {exc}"


def cv_roundtrip() -> tuple[bool, str]:
    """numpy 2 / OpenCV 5 / torch 之间的 ABI 问题只有实际跑一遍数据往返才暴露。"""
    try:
        import cv2
        import numpy as np
        import torch

        buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1]
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        cv2.resize(img, (16, 16), interpolation=cv2.INTER_LINEAR)
        cv2.boxPoints(((4.0, 4.0), (2.0, 2.0), 30.0))
        torch.from_numpy(img)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcdf-check",
        description="环境、依赖、权重、数据与提示自检",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_config_args(parser)
    add_device_arg(parser)
    parser.add_argument("--skip-data", action="store_true", help="跳过数据目录检查")
    parser.add_argument("--skip-sample", action="store_true", help="跳过读取样本与提示拼接")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = build_config(args)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    env = environment_meta(device)
    failures: list[str] = []
    payload: dict = {"environment": env, "probes": {}, "items": []}

    print(f"python      : {env['python']}  ({sys.executable})")
    print(f"torch       : {env['torch']}  cuda_usable={cuda_is_usable()}")
    print(f"device      : {env['device']}  gpu={env['gpu']}  cuda_visible={env['cuda_visible']}")
    for name in ("ultralytics", "transformers", "opencv-python", "numpy", "tokenizers"):
        print(f"{name:<12}: {dist_version(name)}")

    print("兼容性：")
    for label, module, attrs in PROBES:
        ok, detail = probe(label, module, attrs)
        payload["probes"][label] = {"ok": ok, "detail": detail}
        if not ok:
            failures.append(label)
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label}  {'' if ok else detail}")

    ok, detail = cv_roundtrip()
    payload["probes"]["cv2/numpy/torch"] = {"ok": ok, "detail": detail}
    if not ok:
        failures.append("cv2/numpy/torch")
    print(f"  [{'ok  ' if ok else 'FAIL'}] cv2 / numpy / torch 数据往返  {'' if ok else detail}")

    items = check_weights(cfg) + ([] if args.skip_data else check_data(cfg))
    payload["items"] = [
        {"name": item.name, "ok": item.ok, "detail": item.detail, "level": item.level}
        for item in items
    ]
    print("资源：")
    for item in items:
        print(item.render())
    if not ready(items):
        failures.append("weights/data")

    if not args.skip_sample:
        try:
            table = ConditionTable.load(cfg.paths.condition_index, cfg.model.condition_spec)
            print(f"conditions  : {table.n_attributes} 个属性，类别数={table.class_counts}")
            dataset = build_dataset(
                cfg.data, table, split=cfg.data.train_split, augment=False,
                root=cfg.paths.data_root, batch_size=cfg.data.batch, fraction=0.001,
            )
            sample = dataset[0]
            prompt = table.prompt_from_ids(sample["cond"].tolist())
            print(f"train       : {len(dataset)} 张（采样），image={tuple(sample['img'].shape)}")
            print(f"prompt      : {prompt}")
            payload["sample"] = {"image_shape": list(sample["img"].shape), "prompt": prompt}
        except Exception as exc:  # noqa: BLE001 —— 数据问题要在自检里暴露
            failures.append("pcdf data")
            print(f"data        : 自检失败（{type(exc).__name__}: {exc}）")

    payload["failures"] = failures
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print("")
    if failures:
        print(f"结论：{len(failures)} 项失败（{', '.join(failures)}），先别开训练。")
        return 1
    print("结论：兼容性检查全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
