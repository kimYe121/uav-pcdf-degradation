"""检查点读写与续训调和。

为什么单独成模块
----------------
断点续训是补丁最容易堆积的地方：早期实现在恢复时逐字段 ``if`` 判断
（batch 缺了就补、lr 缺了就改、widths 不一致就报错），新增一个超参就得再补
一个分支，而且"哪些字段该沿用检查点、哪些该用当前配置"没有统一规则。

现在把这件事收敛为一条不变式：

    **结构字段必须一致；训练配方以检查点为准；本次命令行显式给出的值优先。**

- *结构字段*（``widths`` / ``backbone_*`` / ``guidance_dim`` / ``data.names``）
  决定权重形状，不一致无法加载，直接报错；
- *训练配方*（``batch`` / ``lr`` / ``warmup`` / ``grad_accum`` / ``amp``）
  决定优化轨迹，续训必须沿用检查点里的值，否则"续训"其实是换了个实验；
- *命令行李参数*（``--batch`` / ``--epochs``）显式覆盖时以用户为准，并在
  日志里说明。

检查点同时保存 ``epochs_total``，因为 ``cfg.train.epochs`` 未必等于本次实际
训练轮数（``--epochs`` 只影响本次运行）。早期版本漏了这一点，导致 shell 用
``cfg.train.epochs`` 判断"阶段是否跑满"时永远判不出来。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .paths import resolve
from .runtime import timestamp

# 2: 增加 epochs_total / arch 显式字段；1 仍可读取（缺字段时走回退逻辑）。
CHECKPOINT_FORMAT = 2

# 决定权重形状的字段：续训时必须与检查点完全一致。
STRUCTURAL_KEYS: tuple[tuple[str, str], ...] = (
    ("model", "widths"),
    ("model", "backbone_yaml"),
    ("model", "backbone_scale"),
    ("model", "guidance_dim"),
    ("data", "names"),
    # 输入几何决定特征的感受野与尺度统计；跨几何续训等价于换了数据分布，
    # 优化器/BN 统计量都不可沿用。老检查点必须用老 imgsz 续训。
    ("data", "imgsz"),
)

# 决定优化轨迹的字段：续训沿用检查点，除非命令行显式覆盖。
RESUME_TRAIN_KEYS: tuple[str, ...] = (
    "lr",
    "warmup_epochs",
    "warmup_bias_lr",
    "grad_accum",
    "amp",
)


class CheckpointError(RuntimeError):
    """检查点不可用（版本不支持、阶段不符、结构不匹配）。"""


# --------------------------------------------------------------------------- #
# 随机数状态
# --------------------------------------------------------------------------- #
def rng_state() -> dict[str, Any]:
    """采集 Python / NumPy / Torch / CUDA 的随机数状态。"""
    state: dict[str, Any] = {
        "python": __import__("random").getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any] | None) -> None:
    """恢复随机数状态；跨设备/跨版本不兼容时只告警，不中断训练。"""
    if not state:
        return
    import random

    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[checkpoint] 警告：随机数状态恢复失败（{exc}），将使用当前种子", flush=True)


# --------------------------------------------------------------------------- #
# 写入
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    cfg: Config,
    stage: int,
    epoch: int,
    next_epoch: int,
    step_in_epoch: int,
    global_step: int,
    best_map50: float,
    history: list[dict[str, Any]],
    arch: str,
    epochs_total: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    metrics: dict[str, Any] | None = None,
    interrupted: bool = False,
) -> None:
    """原子写入完整训练状态。

    先写 ``*.tmp`` 再 ``os.replace``，避免容器被抢占时留下半个文件。
    ``next_epoch`` / ``step_in_epoch`` 共同定义恢复位置：mid-epoch 保存时
    ``next_epoch == epoch``，恢复后从同一 epoch 的下一批继续。
    """
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT,
        "arch": arch,
        "model": model.state_dict(),
        "stage": int(stage),
        "epoch": int(epoch),
        "next_epoch": int(next_epoch),
        "step_in_epoch": int(step_in_epoch),
        "global_step": int(global_step),
        "epochs_total": int(epochs_total),
        "best_map50": float(best_map50),
        "history": history,
        "metrics": metrics or {},
        "names": list(cfg.data.names),
        "cfg": cfg.to_dict(),
        "rng": rng_state(),
        "interrupted": bool(interrupted),
        "saved_at": timestamp(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def prune_checkpoints(folder: str | Path, keep: int) -> None:
    """只保留最近 ``keep`` 个 ``epoch_*.pt``，控制云盘占用。"""
    if keep <= 0:
        return
    files = sorted(Path(folder).glob("epoch_*.pt"))
    for old in files[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
@dataclass
class ResumeState:
    """从检查点还原出的续训状态（不含模型对象）。"""

    path: Path
    stage: int
    start_epoch: int
    start_step: int
    global_step: int
    best_map50: float
    history: list[dict[str, Any]]
    model: dict[str, torch.Tensor]
    config: dict[str, Any]
    arch: str
    epochs_total: int | None
    rng: dict[str, Any] | None
    optimizer: dict[str, Any] | None = None
    scheduler: dict[str, Any] | None = None
    scaler: dict[str, Any] | None = None
    interrupted: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def load_training_state(path: str | Path) -> ResumeState:
    """读取检查点并校验版本；只做结构解析，不做配置调和。"""
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise CheckpointError(f"{checkpoint_path} 不是训练检查点（缺少 model 字段）")

    version = int(payload.get("format_version", 1))
    if version > CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"检查点格式 v{version} 高于当前代码支持的 v{CHECKPOINT_FORMAT}；"
            "请更新代码后再续训，或改用 --init 只加载权重"
        )

    saved_cfg = payload.get("cfg") or {}
    # 旧检查点没有 epochs_total，退回检查点内的 cfg.train.epochs。
    epochs_total = payload.get("epochs_total")
    if epochs_total is None:
        epochs_total = saved_cfg.get("train", {}).get("epochs")

    return ResumeState(
        path=checkpoint_path,
        stage=int(payload.get("stage", 0)),
        start_epoch=int(payload.get("next_epoch", payload.get("epoch", 0))),
        start_step=int(payload.get("step_in_epoch", 0)),
        global_step=int(payload.get("global_step", 0)),
        best_map50=float(payload.get("best_map50", -1.0)),
        history=list(payload.get("history", [])),
        model=payload["model"],
        config=saved_cfg,
        arch=str(payload.get("arch", "")),
        epochs_total=int(epochs_total) if epochs_total else None,
        rng=payload.get("rng"),
        optimizer=payload.get("optimizer"),
        scheduler=payload.get("scheduler"),
        scaler=payload.get("scaler"),
        interrupted=bool(payload.get("interrupted", False)),
        raw=payload,
    )


def resolve_resume_path(
    resume: str | Path | None,
    run_dir: str | Path,
) -> Path | None:
    """解析 ``--resume``：``auto`` / 运行目录 / 显式文件。

    ``auto`` 找不到 ``last.pt`` 时返回 ``None``（由调用方决定是提示还是报错）。
    """
    if resume is None or str(resume).strip() == "":
        return None
    text = str(resume).strip()
    if text.lower() == "auto":
        candidate = Path(run_dir) / "last.pt"
        return candidate if candidate.exists() else None
    candidate = resolve(text)
    if candidate.is_dir():
        candidate = candidate / "last.pt"
    return candidate


# --------------------------------------------------------------------------- #
# 调和
# --------------------------------------------------------------------------- #
@dataclass
class ResumePlan:
    """续训的生效参数与说明，供训练器一次性应用并记录。"""

    state: ResumeState
    batch: int
    total_epochs: int
    # 形如 ``{"train.lr": 0.02, "data.batch": 32}``，由 ``cfg.apply`` 应用。
    effective: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _structural_mismatch(cfg: Config, saved: dict[str, Any]) -> list[str]:
    """比较结构字段，返回人类可读的差异列表。"""
    current = cfg.to_dict()
    problems: list[str] = []
    for section, key in STRUCTURAL_KEYS:
        saved_value = (saved.get(section) or {}).get(key)
        if saved_value is None:
            continue  # 旧检查点缺字段：交给加载时的严格匹配去暴露
        now_value = (current.get(section) or {}).get(key)
        if _comparable(saved_value) != _comparable(now_value):
            problems.append(f"{section}.{key}: 检查点={saved_value!r} 当前={now_value!r}")
    return problems


def _comparable(value: Any) -> Any:
    """把 list/tuple 统一成 tuple，便于跨 YAML/torch 表示比较。"""
    if isinstance(value, (list, tuple)):
        return tuple(int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v for v in value)
    return value


def plan_resume(
    cfg: Config,
    state: ResumeState,
    *,
    stage: int,
    batch_override: int | None = None,
    epochs_override: int | None = None,
) -> ResumePlan:
    """把检查点状态与当前配置调和成一份可执行的续训计划。

    Raises:
        CheckpointError: 阶段不符或结构字段不一致。
    """
    if state.stage != stage:
        raise CheckpointError(
            f"检查点来自 stage {state.stage}，当前请求 stage {stage}。"
            "跨阶段请用 --init 只加载权重，而不是 --resume。"
        )

    problems = _structural_mismatch(cfg, state.config)
    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        raise CheckpointError(
            "检查点与当前配置的结构字段不一致，续训会得到无意义的权重：\n"
            f"{detail}\n"
            f"请用训练时的 --profile 续训（检查点记录的档位："
            f"{state.config.get('train', {}).get('profile') or 'base'}）。"
        )

    saved_train = state.config.get("train") or {}
    saved_data = state.config.get("data") or {}
    effective: dict[str, Any] = {}
    notes: list[str] = []
    warnings: list[str] = []

    # 训练配方以检查点为准，避免 48G 档重启后退回 YAML 的 batch16/lr0.01。
    for key in RESUME_TRAIN_KEYS:
        if key in saved_train:
            effective[f"train.{key}"] = saved_train[key]
    saved_batch = int(saved_data.get("batch") or 0)
    if batch_override is not None:
        batch = int(batch_override)
        effective["data.batch"] = batch
        if saved_batch and batch != saved_batch:
            warnings.append(f"命令行 --batch {batch} 覆盖了检查点记录的 {saved_batch}")
    else:
        batch = saved_batch or cfg.data.batch
        effective["data.batch"] = batch

    if epochs_override is not None:
        total_epochs = int(epochs_override)
        if state.epochs_total and total_epochs != state.epochs_total:
            notes.append(f"总轮数改为 {total_epochs}（检查点为 {state.epochs_total}）")
    else:
        total_epochs = int(state.epochs_total or cfg.train.epochs)

    if state.start_epoch >= total_epochs:
        notes.append(f"检查点已完成 {state.start_epoch}/{total_epochs} 轮，无需继续")

    profile = saved_train.get("profile")
    if profile:
        cfg.train.profile = str(profile)

    notes.insert(
        0,
        f"沿用检查点配方：profile={cfg.train.profile or 'base'} batch={batch} "
        f"lr={effective.get('train.lr', cfg.train.lr):g} "
        f"warmup={effective.get('train.warmup_epochs', cfg.train.warmup_epochs):g}",
    )

    return ResumePlan(
        state=state,
        batch=batch,
        total_epochs=total_epochs,
        effective=effective,
        notes=notes,
        warnings=warnings,
    )


def load_model_weights(model: nn.Module, path: str | Path, *, strict: bool = True) -> dict[str, list[str]]:
    """只加载模型参数（``--init`` 语义），不恢复优化器与轮次。"""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing": list(missing), "unexpected": list(unexpected)}
