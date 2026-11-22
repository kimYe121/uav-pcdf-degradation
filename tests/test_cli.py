"""CLI 装配：参数解析、点号覆盖与设备默认值（不真正训练）。"""

from __future__ import annotations

import pytest

from pcdf.cli import common
from pcdf.cli.evaluate import build_parser as build_eval_parser
from pcdf.cli.train import build_parser as build_train_parser


def test_train_parser_defaults() -> None:
    args = build_train_parser().parse_args(["--stage", "1"])
    assert args.stage == 1
    assert args.device in ("auto", "cpu")  # 受 PCDF_DEVICE 影响
    assert args.init is None and args.no_init is False


def test_eval_parser_accepts_two_checkpoint_spellings() -> None:
    parser = build_eval_parser()
    assert parser.parse_args(["--checkpoint", "a.pt"]).checkpoint == "a.pt"
    assert parser.parse_args(["--ckpt", "b.pt"]).checkpoint == "b.pt"


def test_parse_set_rejects_bad_format() -> None:
    with pytest.raises(SystemExit):
        common.parse_set(["train.lr"])
    assert common.parse_set(["train.lr=0.02", " train.amp = false "]) == {
        "train.lr": "0.02",
        "train.amp": "false",
    }


def test_build_config_applies_set() -> None:
    args = build_train_parser().parse_args(["--stage", "1", "--set", "train.lr=0.03"])
    cfg = common.build_config(args)
    assert cfg.train.lr == pytest.approx(0.03)


def test_apply_overrides_validates() -> None:
    args = build_train_parser().parse_args(["--stage", "1"])
    cfg = common.build_config(args)
    with pytest.raises(ValueError):
        common.apply_overrides(cfg, {"data.batch": 0})


def test_config_from_checkpoint_uses_current_paths() -> None:
    """检查点里的绝对路径必须被当前环境的路径替换。"""
    args = build_train_parser().parse_args(["--stage", "2"])
    env = common.build_config(args)
    saved = env.to_dict()
    saved["paths"]["data_root"] = "/nonexistent/from/training/machine"

    rebuilt = common.config_from_checkpoint(saved, env)
    assert rebuilt.paths.data_root == env.paths.data_root
    assert rebuilt.model.widths == env.model.widths
