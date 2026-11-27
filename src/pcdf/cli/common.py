"""CLI 公共参数与配置装配。

所有命令行入口（train/evaluate/smoke/overfit/...）共用同一套规则：

1. ``--config`` / ``--profile`` / ``--set`` 由本模块解析，交给
   :func:`pcdf.config.load_config`；
2. 显式参数通过**一张映射表**写回配置（``{"train.lr": args.lr}``），不再一条
   一条 ``if args.x is not None`` 地补丁；
3. 设备默认值读 ``PCDF_DEVICE``，方便远端开发时用环境变量统一切换 CPU/GPU。

这样新增一个可覆盖参数只需要在表里加一行，且类型转换由配置层统一负责。
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Iterable, Mapping

import torch

from ..config import Config, apply_path_env, load_config
from ..conditions import ConditionTable
from ..paths import ENV_DEVICE, DEFAULT_CONFIG_PATH
from ..runtime import resolve_device


def add_config_args(parser: argparse.ArgumentParser, *, with_config: bool = True) -> None:
    """加入配置相关参数。"""
    if with_config:
        parser.add_argument(
            "--config", default=str(DEFAULT_CONFIG_PATH),
            help="基础配置文件（默认 configs/pcdf.yaml）",
        )
    parser.add_argument("--profile", default=None, help="配置档位名，如 paper / 48g / cpu")
    parser.add_argument(
        "--set", action="append", default=None, metavar="KEY=VALUE",
        help="点号键覆盖，可重复，例如 --set train.lr=0.02 --set model.tau=0.2",
    )


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    """加入设备参数，默认取 ``PCDF_DEVICE`` 或 auto。"""
    parser.add_argument(
        "--device", default=os.environ.get(ENV_DEVICE, "auto"),
        help="auto / cpu / cuda:0 / 0（也可用环境变量 PCDF_DEVICE）",
    )


def parse_set(entries: Iterable[str] | None) -> dict[str, str]:
    """把 ``--set k=v`` 列表转成字典；格式错误立刻报错。"""
    out: dict[str, str] = {}
    for item in entries or []:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式，收到 {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--set 的键不能为空：{item!r}")
        out[key] = value.strip()
    return out


def build_config(args: argparse.Namespace) -> Config:
    """按命令行参数加载配置。"""
    return load_config(
        getattr(args, "config", None),
        profile=getattr(args, "profile", None),
        overrides=parse_set(getattr(args, "set", None)),
    )


def apply_overrides(cfg: Config, mapping: Mapping[str, Any]) -> Config:
    """应用"参数名 -> 配置键"的映射，并在覆盖后重新校验。

    重新校验是必要的：``--batch 0`` 或 ``--imgsz 300 200`` 只有在覆盖之后
    才会被发现，越早报错越省 GPU 时间。
    """
    cfg.apply(mapping)
    cfg.validate()
    return cfg


def device_from_args(args: argparse.Namespace) -> torch.device:
    """解析 ``--device``。"""
    return resolve_device(getattr(args, "device", "auto"))


def load_table(cfg: Config) -> ConditionTable:
    """加载条件索引（所有入口一致的错误信息）。"""
    return ConditionTable.load(cfg.paths.condition_index, cfg.model.condition_spec)


def config_from_checkpoint(checkpoint_cfg: Mapping[str, Any], env: Config) -> Config:
    """用检查点的模型结构 + 当前环境的路径重建配置。

    检查点里存的是训练机器上的绝对路径（数据盘、权重目录）。直接复用会在
    另一台机器上指向不存在的目录，因此路径段一律以 ``env``（当前配置 +
    环境变量）为准，模型/数据形状仍取检查点——这样"用 checkpoint 的架构
    评估"与"用本机数据评估"两件事同时成立。
    """
    raw = dict(checkpoint_cfg)
    raw["paths"] = env.to_dict()["paths"]
    cfg = Config.from_dict(raw)
    # 结构取检查点，"去哪找文件"取本机：条件码本/骨干 yaml/预训练权重/CLIP。
    cfg.adopt_runtime_paths(env)
    cfg.train.profile = env.train.profile
    apply_path_env(cfg)
    cfg.resolve_paths()
    cfg.validate()
    return cfg
