"""断点续训：检查点字段、原子写入、结构校验与配方沿用。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pcdf.checkpoint import (
    CheckpointError,
    load_training_state,
    plan_resume,
    resolve_resume_path,
    save_checkpoint,
)
from pcdf.config import load_config


class _Toy(nn.Module):
    """最小可训练模块，用来验证检查点读写而不必建真实网络。"""

    arch = "pcdf"

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)


def _write(tmp_path, cfg, *, stage: int = 1, epochs_total: int = 50):
    model = _Toy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        cfg=cfg,
        stage=stage,
        epoch=2,
        next_epoch=3,
        step_in_epoch=0,
        global_step=17,
        best_map50=0.42,
        history=[{"epoch": 1, "loss": 1.0}],
        arch="pcdf",
        epochs_total=epochs_total,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        metrics={"loss": 0.5},
        interrupted=False,
    )
    return path


def test_checkpoint_contains_resume_state(tmp_path) -> None:
    cfg = load_config()
    path = _write(tmp_path, cfg)

    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()  # 原子写入后不残留临时文件
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "model", "optimizer", "scheduler", "next_epoch", "step_in_epoch",
        "global_step", "rng", "epochs_total", "arch", "cfg",
    ):
        assert key in ckpt
    assert ckpt["next_epoch"] == 3
    assert ckpt["stage"] == 1
    assert ckpt["best_map50"] == 0.42
    assert ckpt["history"][0]["epoch"] == 1
    assert ckpt["epochs_total"] == 50


def test_load_state_reads_epochs_total(tmp_path) -> None:
    """epochs_total 必须来自显式字段：--epochs 8 时 cfg.train.epochs 也已是 8，
    但用旧检查点恢复时不能再被当前配置覆盖。"""
    cfg = load_config()
    state = load_training_state(_write(tmp_path, cfg, epochs_total=8))
    assert state.epochs_total == 8
    assert state.start_epoch == 3


def test_resolve_resume_path(tmp_path) -> None:
    assert resolve_resume_path(None, tmp_path) is None
    assert resolve_resume_path("auto", tmp_path) is None  # 还没有 last.pt
    cfg = load_config()
    path = _write(tmp_path, cfg)
    assert resolve_resume_path("auto", tmp_path) == path
    assert resolve_resume_path(str(tmp_path), tmp_path) == path


def test_resume_plan_reuses_checkpoint_recipe(tmp_path) -> None:
    """48G 档重启后不能退回 YAML 的 batch16/lr0.01。"""
    saved = load_config(profile="48g")
    saved.train.epochs = 30
    state = load_training_state(_write(tmp_path, saved, epochs_total=30))

    current = load_config()  # batch16 / lr0.01
    plan = plan_resume(current, state, stage=1)
    assert plan.batch == 32
    assert plan.total_epochs == 30
    assert plan.effective["train.lr"] == pytest.approx(0.02)
    assert plan.effective["train.warmup_epochs"] == pytest.approx(1.0)


def test_resume_plan_cli_batch_wins(tmp_path) -> None:
    cfg = load_config()
    state = load_training_state(_write(tmp_path, cfg))
    plan = plan_resume(cfg, state, stage=1, batch_override=4)
    assert plan.batch == 4
    assert any("--batch" in w for w in plan.warnings)


def test_resume_plan_rejects_stage_mismatch(tmp_path) -> None:
    cfg = load_config()
    state = load_training_state(_write(tmp_path, cfg, stage=1))
    with pytest.raises(CheckpointError, match="stage"):
        plan_resume(cfg, state, stage=2)


def test_resume_plan_rejects_imgsz_change(tmp_path) -> None:
    """输入几何不同不能续训：优化器/BN 统计量都建立在旧尺度上。"""
    saved = load_config(overrides={"data.imgsz": "640,512"})
    state = load_training_state(_write(tmp_path, saved))

    current = load_config()  # 默认 (512,640)
    with pytest.raises(CheckpointError, match="imgsz"):
        plan_resume(current, state, stage=1)


def test_resume_plan_rejects_structural_mismatch(tmp_path) -> None:
    """结构字段（widths/backbone/guidance_dim）不一致时不能"续训"，只能 --init。"""
    saved = load_config(overrides={
        "model.widths": "256,256,512",
        "model.guidance_dim": "512",
    })
    state = load_training_state(_write(tmp_path, saved))

    current = load_config()  # widths 128/256/256、guidance_dim 256
    with pytest.raises(CheckpointError, match="结构字段"):
        plan_resume(current, state, stage=1)
