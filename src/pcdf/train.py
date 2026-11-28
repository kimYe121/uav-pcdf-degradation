"""三阶段训练主循环（补充材料 B.3）。

阶段 0：baseline 双流 concat；阶段 1：SCPL + CDF（文本引导，主干冻结）；
阶段 2：full pipeline（视觉引导，SCPL/text adapter 冻结）。

除论文明确给出的优化器/学习率/epoch 外，warmup、梯度裁剪、AMP、增强等未公开
细节均记录在 ``docs/ARCHITECTURE.md``。

断点续训
--------
容器实例可能被抢占，因此检查点保存**完整训练状态**（模型、优化器、调度器、
AMP scaler、epoch/step、best mAP、日志历史、随机数状态）。恢复规则集中在
:mod:`pcdf.checkpoint`：结构字段必须一致，训练配方以检查点为准，命令行参数
优先。本模块只负责"按计划执行"，不再持有散落的续训分支。

每个 epoch 的落盘顺序
--------------------
1. 训练完一个 epoch → 若需要验证，**先**写 ``last.pt``（保护权重）；
2. 跑验证，把 ``map50`` 写进同一个 metrics 字典（它已经在 history 里）；
3. 再写 ``last.pt``，此时历史完整；
4. 若 mAP 提升写 ``best.pt``，按周期写 ``checkpoints/epoch_XXXX.pt``；
5. 追加 ``log.csv``、覆盖 ``metrics.json``。

早期实现只在验证前保存，导致 ``last.pt`` 的历史里永远缺当前 epoch 的 mAP，
排查曲线断裂时很难解释；现在两次写入把"权重不丢"和"记录完整"同时满足。
"""

from __future__ import annotations

import csv
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .checkpoint import (
    CheckpointError,
    ResumePlan,
    ResumeState,
    load_model_weights,
    load_training_state,
    plan_resume,
    prune_checkpoints,
    resolve_resume_path,
    restore_rng,
    save_checkpoint,
)
from .conditions import ConditionTable
from .config import Config, save_config
from .constants import EPOCH_FIELDS, STAGE_NAMES
from .data import batch_to_device, build_dataset, build_loader, epoch_loader
from .evaluate import evaluate_obb
from .losses import decoupling_terms
from .models.baseline import BaselineNet, load_baseline_into_pcdf
from .models.heads import make_criterion
from .models.pcdf import PCDFNet
from .paths import display_path, resolve, stage_dir
from .runtime import (
    environment_meta,
    git_metadata,
    json_safe,
    resolve_device,
    seed_everything,
    timestamp,
    write_json,
)
from .text_encoder import CLIPTextEncoder

# 初始化策略：auto 用上一阶段产物，required 缺了就报错，none 从主干预训练权重
# 直接开始（诊断/冒烟用，不是论文协议）。
INIT_POLICIES = ("auto", "required", "none")


@dataclass
class TrainResult:
    """一个阶段的训练结果。"""

    run_dir: Path
    stage: int
    history: list[dict[str, Any]] = field(default_factory=list)
    best_map50: float = -1.0
    best_epoch: int | None = None
    interrupted: bool = False
    finished: bool = True
    total_epochs: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "stage": self.stage,
            "best_map50": self.best_map50,
            "best_epoch": self.best_epoch,
            "interrupted": self.interrupted,
            "finished": self.finished,
            "epochs_done": len(self.history),
            "epochs_total": self.total_epochs,
        }


def best_epoch(history: list[dict[str, Any]]) -> int | None:
    """返回验证 mAP50 最高的 epoch（1-based）；没有有效结果时返回 None。"""
    best_ep, best_map = None, float("-inf")
    for row in history:
        try:
            value = float(row.get("map50"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > best_map:
            best_ep, best_map = int(row.get("epoch", 0)), value
    return best_ep


# --------------------------------------------------------------------------- #
# 训练器
# --------------------------------------------------------------------------- #
class Trainer:
    """单个阶段的训练执行器。

    只做"给定模型、优化器、数据，把 epoch 跑完并正确落盘"这一件事；模型构造、
    续训调和、初始化策略都在 :func:`train_stage` 中完成，便于单独测试。
    """

    def __init__(
        self,
        *,
        cfg: Config,
        stage: int,
        device: torch.device,
        run_path: Path,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        train_ds,
        val_loader: DataLoader | None,
        batch_size: int,
        total_epochs: int,
        max_batches: int,
        val_batches: int,
        history: list[dict[str, Any]],
        best_map: float,
        start_epoch: int,
        start_step: int,
        global_step: int,
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.device = device
        self.run_path = run_path
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.train_ds = train_ds
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.max_batches = max_batches
        self.val_batches = val_batches
        self.history = history
        self.best_map = best_map
        self.start_epoch = start_epoch
        self.start_step = start_step
        self.global_step = global_step

        self.arch = getattr(model, "arch", "pcdf")
        self.checkpoint_dir = run_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = run_path / "log.csv"
        self.batches_per_epoch = (len(train_ds) + batch_size - 1) // batch_size
        self.warmup_steps = int(cfg.train.warmup_epochs * self.batches_per_epoch)
        self.base_lr = cfg.train.lr
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.stop_flag = {"value": False}
        self.peak_mem_gb = 0.0
        # 最近处理完的 batch 下标，供中断/异常时定位恢复点。
        self.last_step = -1

    # ------------------------------------------------------------------ 入口
    def run(self) -> TrainResult:
        """按 epoch 执行训练，直到完成或被信号中断。"""
        previous = _install_signal_handlers(self.stop_flag)
        interrupted = False
        last_periodic = time.monotonic()
        current_epoch = self.start_epoch

        try:
            for epoch in range(self.start_epoch, self.total_epochs):
                metrics = self._train_epoch(epoch, last_periodic)
                last_periodic = metrics.pop("_last_periodic", time.monotonic())
                self.history.append(metrics)

                if self.stop_flag["value"]:
                    # 抢占/中断：保存到当前 step，恢复后从下一步继续。
                    self._save(
                        self.run_path / "last.pt",
                        epoch=epoch,
                        next_epoch=epoch,
                        step_in_epoch=self.last_step + 1,
                        metrics=metrics,
                        interrupted=True,
                    )
                    interrupted = True
                    print(
                        f"[interrupt] 已保存 last.pt（epoch {epoch + 1}, "
                        f"step {self.last_step + 1}），可 --resume 继续",
                        flush=True,
                    )
                    break

                if self._should_eval(epoch):
                    # 先落一次权重：验证阶段被抢占也不会丢掉整个 epoch。
                    self._save(
                        self.run_path / "last.pt",
                        epoch=epoch,
                        next_epoch=epoch + 1,
                        step_in_epoch=0,
                        metrics=metrics,
                        interrupted=False,
                    )
                    self._evaluate(epoch, metrics)
                    if np.isfinite(metrics.get("map50", float("nan"))) and metrics["map50"] > self.best_map:
                        self.best_map = float(metrics["map50"])
                        self._save(
                            self.run_path / "best.pt",
                            epoch=epoch,
                            next_epoch=epoch + 1,
                            step_in_epoch=0,
                            metrics=metrics,
                            interrupted=False,
                        )
                else:
                    metrics["map50"] = float("nan")

                self._save(
                    self.run_path / "last.pt",
                    epoch=epoch,
                    next_epoch=epoch + 1,
                    step_in_epoch=0,
                    metrics=metrics,
                    interrupted=False,
                )
                if self.cfg.train.save_period > 0 and (epoch + 1) % self.cfg.train.save_period == 0:
                    self._save(
                        self.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                        epoch=epoch,
                        next_epoch=epoch + 1,
                        step_in_epoch=0,
                        metrics=metrics,
                        interrupted=False,
                    )
                    prune_checkpoints(self.checkpoint_dir, self.cfg.train.keep_checkpoints)

                self._record(epoch, metrics)
                current_epoch = epoch
                last_periodic = time.monotonic()
        except Exception as exc:
            self._on_crash(exc, current_epoch, self.last_step)
            raise
        finally:
            _restore_signal_handlers(previous)

        if self.device.type == "cuda":
            print(f"[mem] 峰值显存 {self.peak_mem_gb:.2f} GB（batch={self.batch_size}）", flush=True)

        return TrainResult(
            run_dir=self.run_path,
            stage=self.stage,
            history=self.history,
            best_map50=self.best_map,
            best_epoch=best_epoch(self.history),
            interrupted=interrupted,
            finished=not interrupted,
            total_epochs=self.total_epochs,
        )

    # ------------------------------------------------------------------ epoch
    def _train_epoch(self, epoch: int, last_periodic: float) -> dict[str, Any]:
        """跑完一个 epoch 的训练部分，返回未含评估结果的 metrics。"""
        cfg = self.cfg
        self.model.train()
        if self.stage != 0:
            # CLIP 始终冻结；baseline 没有 text_encoder。
            self.model.text_encoder.model.eval()
        epoch_start = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        loader = epoch_loader(
            self.train_ds,
            batch_size=self.batch_size,
            workers=cfg.data.workers,
            shuffle=True,
            device=self.device,
            # 固定每个 epoch 的样本顺序，mid-epoch 恢复才能对上批次。
            seed=cfg.train.seed + epoch,
        )
        limit = min(self.max_batches, self.batches_per_epoch) if self.max_batches else self.batches_per_epoch
        skip = self.start_step if epoch == self.start_epoch else 0
        if skip >= limit:
            print(f"[resume] step {skip} 超出本 epoch 上限 {limit}，改为从 epoch 起点重跑", flush=True)
            skip = 0

        running: dict[str, float] = {}
        kept_sum, kept_n, steps = 0.0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        step = -1

        for step, raw_batch in enumerate(loader):
            if step < skip:
                continue
            if self.stop_flag["value"] or step >= limit:
                break
            batch = batch_to_device(raw_batch, self.device)
            loss, items, extra_kept = self._step(batch, step=step, limit=limit)
            self.last_step = step

            for key, value in items.items():
                running[key] = running.get(key, 0.0) + value
            if extra_kept is not None:
                kept_sum += extra_kept
                kept_n += 1
            steps += 1

            if (step + 1) % 20 == 0:
                elapsed = time.time() - epoch_start
                eta_min = elapsed / (step + 1) * (limit - step - 1) / 60.0
                print(
                    f"  ep{epoch + 1} {step + 1}/{limit} "
                    f"loss={running['loss'] / steps:.3f} "
                    f"box={running['box'] / steps:.3f} cls={running['cls'] / steps:.3f} "
                    f"dfl={running['dfl'] / steps:.3f} angle={running['angle'] / steps:.3f} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.5f} eta={eta_min:.1f}min",
                    flush=True,
                )

            last_periodic = self._maybe_periodic_save(
                epoch, step, limit, running, steps, last_periodic
            )

        if steps == 0:
            raise RuntimeError("没有任何训练 batch 被处理，请检查 max_batches / 数据路径")

        self.scheduler.step()
        lr_now = self.optimizer.param_groups[0]["lr"]
        metrics: dict[str, Any] = {k: v / steps for k, v in running.items()}
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr_now
        metrics["kept"] = kept_sum / kept_n if kept_n else float("nan")
        metrics["epoch_time_s"] = time.time() - epoch_start
        metrics["global_step"] = self.global_step
        metrics["timestamp"] = timestamp()
        if self.device.type == "cuda":
            epoch_mem = torch.cuda.max_memory_allocated(self.device) / 1e9
            self.peak_mem_gb = max(self.peak_mem_gb, epoch_mem)
            metrics["gpu_mem_gb"] = epoch_mem
        else:
            metrics["gpu_mem_gb"] = 0.0
        metrics["_last_periodic"] = last_periodic
        return metrics

    def _step(self, batch: dict, *, step: int, limit: int) -> tuple[Tensor, dict[str, float], float | None]:
        """单批前向 + 反向 + 优化，返回 (loss, 日志项, kept)。

        梯度累积窗口的收尾条件与训练循环一致：``(step+1) % grad_accum == 0``
        或到达本 epoch 的最后一批——否则末尾不足一个窗口的样本梯度永远不会
        被 ``optimizer.step()`` 消费，相当于白算。
        """
        cfg, model = self.cfg, self.model
        images = batch["img"]
        cond = batch.get("cond")
        batch_size_now = images.shape[0]

        # 手动 warmup；之后每个 epoch 由 ExponentialLR 指数衰减。
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            warm = (self.global_step + 1) / self.warmup_steps
            lr = self.base_lr * (
                cfg.train.warmup_bias_lr + (1.0 - cfg.train.warmup_bias_lr) * warm
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr

        with torch.autocast("cuda", enabled=self.scaler.is_enabled(), dtype=torch.float16):
            # Fig. 10：阶段 1 用文本引导 + 原始特征；阶段 2 训练仍由 SCPL 文本
            # 引导 CDF，但权重作用在 F_v 上（测试阶段才切换到 GAP(F_s)）。
            fwd = dict(
                cond=cond,
                guidance="text" if self.stage in (1, 2) else "visual",
                return_aux=(self.stage == 2),
                return_ft=(self.stage == 2),
            )
            if self.stage == 2:
                fwd["fuse"] = "decoupled"
            out = model(images, **fwd)
            loss_main, items = self.criterion(out["main"], batch)
            loss_main = loss_main.sum()
            loss = loss_main
            terms: dict[str, Tensor] = {}
            if self.stage == 2:
                # L_dc：共享 OBB head 分别作用于两个条件不变分支。
                l_dc = self.criterion(out["aux_rgb"], batch)[0].sum() + self.criterion(
                    out["aux_ir"], batch
                )[0].sum()
                terms = decoupling_terms(
                    out["fs_gap"],
                    out["ft"],
                    model.text_adapter,
                    out["fv_rgb"],
                    out["fv_ir"],
                    out["fs"],
                    cfg.model.lambda_dt,
                    cfg.model.lambda_irr,
                    cfg.model.lambda_dc,
                    l_dc,
                )
                loss = loss + terms["l_dec"]

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"epoch 训练出现非有限损失：{loss.item()}（检查 lr / AMP / 数据标签）"
            )
        self.scaler.scale(loss / cfg.train.grad_accum).backward()
        self.global_step += 1

        if (step + 1) % cfg.train.grad_accum == 0 or step + 1 == limit:
            # 梯度累积窗口结束（含末尾不足一个窗口的情况）。
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.params, cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        values = {
            "loss": float(loss.detach()) / batch_size_now,
            "box": float(items.get("box_loss", 0.0)),
            "cls": float(items.get("cls_loss", 0.0)),
            "dfl": float(items.get("dfl_loss", 0.0)),
            "angle": float(items.get("angle_loss", 0.0)),
            "l_dt": float(terms["l_dt"].detach()) if terms else 0.0,
            "l_irr": float(terms["l_irr"].detach()) if terms else 0.0,
            "l_dc": float(terms["l_dc"].detach()) / batch_size_now if terms else 0.0,
        }
        kept = float(out["kept"].float().mean()) if "kept" in out else None
        return loss, values, kept

    def _maybe_periodic_save(
        self, epoch: int, step: int, limit: int, running: dict[str, float],
        steps: int, last_periodic: float,
    ) -> float:
        """按 step / 时间周期保存 mid-epoch ``last.pt``；返回最近保存时刻。"""
        cfg = self.cfg.train
        due_steps = cfg.save_every_steps > 0 and (step + 1) % cfg.save_every_steps == 0
        due_time = (
            cfg.save_every_minutes > 0
            and time.monotonic() - last_periodic >= cfg.save_every_minutes * 60
        )
        if not (due_steps or due_time) or step + 1 >= limit:
            return last_periodic
        self._save(
            self.run_path / "last.pt",
            epoch=epoch,
            next_epoch=epoch,
            step_in_epoch=step + 1,
            metrics={"loss": running.get("loss", 0.0) / max(steps, 1)},
            interrupted=False,
        )
        print(f"  [checkpoint] 周期保存 last.pt（epoch {epoch + 1}, step {step + 1}）", flush=True)
        return time.monotonic()

    # ------------------------------------------------------------------ 评估
    def _should_eval(self, epoch: int) -> bool:
        if self.val_loader is None or self.cfg.train.eval_interval <= 0:
            return False
        return (epoch + 1) % self.cfg.train.eval_interval == 0 or epoch + 1 == self.total_epochs

    def _evaluate(self, epoch: int, metrics: dict[str, Any]) -> None:
        """在验证集上算 mAP，并把结果写进 metrics（同一个 dict 已在 history 里）。"""
        result = evaluate_obb(
            self.model,
            self.val_loader,
            self.device,
            nc=self.cfg.data.nc,
            conf=self.cfg.data.eval_conf,
            iou=self.cfg.data.eval_iou,
            max_det=self.cfg.data.eval_max_det,
            max_batches=self.val_batches,
            thresholds=(0.5,),
            names=self.cfg.data.names,
            # 监控走与训练一致的引导：阶段 1/2 训练都用 SCPL 文本引导，
            # 视觉引导在阶段 2 初期是未训练的 GAP(F_s)，用它选点会重演
            # 2026-11-24 的 best.pt 误选事故（见 STATE §9）。部署口径的
            # 视觉 test 在训练结束后单独评。
            guidance="text" if self.stage in (1, 2) else "visual",
            # 监控必须与训练走同一条融合路径：阶段 2 训练时 CDF 作用在 F_v 上，
            # 只传 guidance=text 会退回 fuse="raw"（阶段 1 的路径），val 量到的
            # 是另一条前向（实测 ep5 只有 3.3，而训练分支正常下降）。
            fuse="decoupled" if self.stage == 2 else None,
        )
        metrics["map50"] = result["map50"]
        metrics["eval_images"] = result["n_images"]
        print(
            f"[eval] epoch {epoch + 1}: mAP50={result['map50']:.4f} "
            f"images={result['n_images']} gt={result['n_gt']} pred={result['n_pred']}",
            flush=True,
        )

    # ------------------------------------------------------------------ 落盘
    def _save(
        self,
        path: Path,
        *,
        epoch: int,
        next_epoch: int,
        step_in_epoch: int,
        metrics: dict[str, Any],
        interrupted: bool,
    ) -> None:
        payload = {k: v for k, v in metrics.items() if not k.startswith("_")}
        save_checkpoint(
            path,
            model=self.model,
            cfg=self.cfg,
            stage=self.stage,
            epoch=epoch,
            next_epoch=next_epoch,
            step_in_epoch=step_in_epoch,
            global_step=self.global_step,
            best_map50=self.best_map,
            history=self.history,
            arch=self.arch,
            epochs_total=self.total_epochs,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            metrics=payload,
            interrupted=interrupted,
        )

    def _record(self, epoch: int, metrics: dict[str, Any]) -> None:
        """追加 CSV、覆盖 metrics.json、打印 epoch 摘要。"""
        write_header = not self.log_path.exists()
        with open(self.log_path, "a" if not write_header else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(EPOCH_FIELDS))
            if write_header:
                writer.writeheader()
            # 缺项写 NaN 而不是 0：0 会被误读成"真的算出 0 mAP"。
            writer.writerow({key: metrics.get(key, float("nan")) for key in EPOCH_FIELDS})

        write_json(
            self.run_path / "metrics.json",
            {
                "stage": self.stage,
                "profile": self.cfg.train.profile,
                "epochs_total": self.total_epochs,
                "best_map50": self.best_map,
                "best_epoch": best_epoch(self.history),
                "updated_at": metrics["timestamp"],
                "history": self.history,
            },
        )
        print(
            f"[epoch {epoch + 1}/{self.total_epochs}] loss={metrics['loss']:.3f} "
            f"box={metrics['box']:.3f} cls={metrics['cls']:.3f} dfl={metrics['dfl']:.3f} "
            f"angle={metrics['angle']:.3f} "
            f"l_dt={metrics['l_dt']:.4f} l_irr={metrics['l_irr']:.4f} "
            f"kept={metrics['kept']:.2f} lr={metrics['lr']:.5f} "
            f"time={metrics['epoch_time_s']:.0f}s",
            flush=True,
        )

    def _on_crash(self, exc: Exception, epoch: int, step: int) -> None:
        """异常退出时尽量保留现场，便于排查与续训。"""
        try:
            self._save(
                self.run_path / "crash.pt",
                epoch=max(epoch, 0),
                next_epoch=max(epoch, 0),
                step_in_epoch=max(step + 1, 0),
                metrics={},
                interrupted=True,
            )
            print(f"[crash] 已保存现场到 {self.run_path / 'crash.pt'}", flush=True)
        except Exception as save_exc:  # noqa: BLE001 —— 保存失败不应掩盖原始异常
            print(f"[crash] 现场保存失败：{save_exc}", flush=True)
        try:
            write_json(
                self.run_path / "summary.json",
                {
                    "stage": self.stage,
                    "profile": self.cfg.train.profile,
                    "status": "error",
                    "error": repr(exc),
                    "best_map50": self.best_map,
                    "best_epoch": best_epoch(self.history),
                    "epochs_done": len(self.history),
                    "epochs_total": self.total_epochs,
                    "failed_at": timestamp(),
                },
            )
        except Exception:  # noqa: BLE001 —— 摘要失败同样不应掩盖原始异常
            pass


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
def train_stage(
    cfg: Config,
    stage: int,
    *,
    init_checkpoint: str | Path | None = None,
    init_policy: str = "auto",
    device: str = "auto",
    run_dir: str | Path | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: tuple[int, int] | None = None,
    max_batches: int | None = None,
    val_batches: int | None = None,
    resume: str | Path | None = None,
) -> TrainResult:
    """训练一个阶段。

    Args:
        cfg: 已解析并校验的配置。
        stage: 0（baseline）、1（SCPL+CDF，主干冻结）、2（full pipeline）。
        init_checkpoint: 阶段 1/2 的初始权重；只加载模型，不恢复优化器。
        init_policy: ``auto`` 自动找上一阶段产物；``required`` 缺失即报错；
            ``none`` 不做阶段初始化（诊断/冒烟用，不是论文协议）。
        device: ``auto`` / ``cpu`` / ``cuda:0`` / ``0``。
        run_dir: 输出目录；默认 ``<runs_dir>/<run_name>_stage<N>``。
        epochs/batch/imgsz/max_batches/val_batches: 本次运行的覆盖项。
        resume: 断点续训：``auto`` / 运行目录 / 检查点文件。

    Returns:
        :class:`TrainResult`。``interrupted=True`` 表示被信号打断，可用
        ``--resume`` 继续。

    Raises:
        ValueError: 参数非法（未知 stage、未知 init_policy）。
        CheckpointError: 续训检查点与当前配置结构不符。
        FileNotFoundError: 阶段 1/2 缺少必需的初始权重。
    """
    if stage not in STAGE_NAMES:
        raise ValueError(f"stage 必须是 0/1/2，收到 {stage!r}")
    if init_policy not in INIT_POLICIES:
        raise ValueError(f"init_policy 只能是 {INIT_POLICIES}，收到 {init_policy!r}")

    if imgsz is not None:
        cfg.data.imgsz = tuple(int(v) for v in imgsz)
    if batch is not None:
        cfg.data.batch = int(batch)

    device_obj = resolve_device(device)
    seed_everything(cfg.train.seed, cfg.train.deterministic)

    run_path = Path(run_dir) if run_dir else stage_dir(cfg.paths.runs_dir, cfg.train.run_name, stage)
    resume_path = resolve_resume_path(resume or cfg.train.resume, run_path)

    # 允许直接指向 checkpoints/epoch_XXXX.pt：运行目录是其上两级。
    if run_dir is None and resume_path is not None:
        run_path = (
            resume_path.parent.parent
            if resume_path.parent.name == "checkpoints"
            else resume_path.parent
        )
    run_path.mkdir(parents=True, exist_ok=True)

    resume_requested = bool(resume or cfg.train.resume)
    if resume_requested and resume_path is None:
        print(
            f"[resume] 未找到 {run_path / 'last.pt'}，本次从零开始训练；"
            "如果是误操作，请检查 --run-dir / --resume",
            flush=True,
        )

    # ---- 续训调和：结构校验 + 训练配方以检查点为准 -----------------------
    plan: ResumePlan | None = None
    history: list[dict[str, Any]] = []
    best_map = -1.0
    start_epoch, start_step, global_step = 0, 0, 0
    if resume_path is not None:
        state = load_training_state(resume_path)
        plan = plan_resume(
            cfg, state, stage=stage, batch_override=batch, epochs_override=epochs
        )
        cfg.apply(plan.effective)
        for note in plan.notes:
            print(f"[resume] {note}", flush=True)
        for warning in plan.warnings:
            print(f"[resume] 警告：{warning}", flush=True)
        print(
            f"[resume] {display_path(resume_path)} -> epoch {state.start_epoch + 1}, "
            f"step {state.start_step}, global_step {state.global_step}, "
            f"best mAP50 {state.best_map50:.4f}",
            flush=True,
        )
        history = state.history
        best_map = state.best_map50
        start_epoch, start_step, global_step = state.start_epoch, state.start_step, state.global_step

    total_epochs = plan.total_epochs if plan else int(epochs or cfg.train.epochs)
    # 让配置快照与实际执行一致：--epochs 8 时 config.yaml 不应仍写 50。
    cfg.train.epochs = total_epochs
    if start_epoch >= total_epochs:
        print(f"[resume] 已到达总 epoch 数 {total_epochs}，无需继续训练", flush=True)
        write_json(
            run_path / "summary.json",
            {
                "stage": stage,
                "profile": cfg.train.profile,
                "status": "already_finished",
                "best_map50": best_map,
                "best_epoch": best_epoch(history),
                "epochs_done": len(history),
                "epochs_total": total_epochs,
                "finished_at": timestamp(),
            },
        )
        return TrainResult(
            run_dir=run_path, stage=stage, history=history, best_map50=best_map,
            best_epoch=best_epoch(history), interrupted=False, finished=True,
            total_epochs=total_epochs,
        )

    save_config(cfg, run_path / "config.yaml")

    # ---- 数据 ------------------------------------------------------------
    table = ConditionTable.load(cfg.paths.condition_index, cfg.model.condition_spec)
    train_ds = build_dataset(
        cfg.data, table, split=cfg.data.train_split, augment=True,
        root=cfg.paths.data_root, batch_size=cfg.data.batch,
    )
    val_loader = None
    val_batches = cfg.train.val_batches if val_batches is None else val_batches
    if cfg.train.eval_interval > 0:
        _, val_loader = build_loader(
            cfg.data, table, split=cfg.data.val_split, augment=False,
            root=cfg.paths.data_root, batch_size=cfg.data.batch,
            workers=max(0, cfg.data.workers // 2), shuffle=False, device=device_obj,
        )

    # ---- 模型 ------------------------------------------------------------
    model = _build_model(cfg, stage, table, device_obj)
    _apply_init(model, cfg, stage, init_checkpoint, init_policy, resume_path)
    if stage == 2:
        model.freeze_scpl()
        model.freeze_text_adapter()
    elif stage == 1:
        model.freeze_detector(
            keep_detector_trainable=cfg.train.stage1_train_detector
        )

    params = [p for p in model.parameters() if p.requires_grad]
    # 论文 Sec. 5.1 只声明 SGD + momentum + weight decay，未提 Nesterov，
    # 因此保持 PyTorch SGD 默认 nesterov=False，避免引入额外假设。
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
        nesterov=False,
    )
    gamma = cfg.train.final_lr_ratio ** (1.0 / max(total_epochs - 1, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = make_criterion(model)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device_obj.type == "cuda")

    # ---- 恢复优化器/调度器/AMP 与随机数 ---------------------------------
    if plan is not None:
        state: ResumeState = plan.state
        model.load_state_dict(state.model, strict=True)
        if state.optimizer:
            optimizer.load_state_dict(state.optimizer)
        if state.scheduler:
            scheduler.load_state_dict(state.scheduler)
            # 若用户延长了总 epoch 数，用当前配置重算衰减系数但保留 last_epoch。
            scheduler.gamma = gamma
        if state.scaler:
            scaler.load_state_dict(state.scaler)
        restore_rng(state.rng)

    _print_setup(cfg, stage, device_obj, model, params, len(train_ds))

    # ---- 归档与训练 ------------------------------------------------------
    meta = {
        "stage": stage,
        "stage_name": STAGE_NAMES[stage],
        "profile": cfg.train.profile,
        "run_dir": str(run_path),
        "started_at": timestamp(),
        "command": _command_line(),
        "cwd": str(Path.cwd()),
        "environment": environment_meta(device_obj),
        "batch": cfg.data.batch,
        "epochs": total_epochs,
        "imgsz": list(cfg.data.imgsz),
        "train_images": len(train_ds),
        "batches_per_epoch": (len(train_ds) + cfg.data.batch - 1) // cfg.data.batch,
        "trainable_params": sum(p.numel() for p in params),
        "total_params": sum(p.numel() for p in model.parameters()),
        "init": {
            "policy": init_policy,
            "checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "resume": str(resume_path) if resume_path else None,
        },
        "git": git_metadata(),
    }
    write_json(run_path / "meta.json", meta)
    (run_path / "command.txt").write_text(
        _command_line() + "\n", encoding="utf-8", newline="\n"
    )

    trainer = Trainer(
        cfg=cfg, stage=stage, device=device_obj, run_path=run_path, model=model,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        train_ds=train_ds, val_loader=val_loader, batch_size=cfg.data.batch,
        total_epochs=total_epochs, max_batches=max_batches or cfg.train.max_batches,
        val_batches=val_batches, history=history, best_map=best_map,
        start_epoch=start_epoch, start_step=start_step, global_step=global_step,
    )
    result = trainer.run()

    write_json(
        run_path / "summary.json",
        {
            "stage": stage,
            "profile": cfg.train.profile,
            "status": "interrupted" if result.interrupted else "finished",
            "best_map50": result.best_map50,
            "best_epoch": result.best_epoch,
            "epochs_done": len(result.history),
            "epochs_total": result.total_epochs,
            "batch": cfg.data.batch,
            "imgsz": list(cfg.data.imgsz),
            "peak_mem_gb": trainer.peak_mem_gb,
            "finished_at": timestamp(),
        },
    )
    return result


def _build_model(cfg: Config, stage: int, table: ConditionTable, device: torch.device) -> nn.Module:
    """按阶段构建模型；baseline 不需要 CLIP，跳过加载以省内存与启动时间。"""
    if stage == 0:
        return BaselineNet(cfg.model, nc=cfg.data.nc).to(device)
    text_encoder = CLIPTextEncoder(cfg.model.text_encoder, device=device)
    model = PCDFNet(cfg.model, nc=cfg.data.nc, table=table, text_encoder=text_encoder)
    return model.to(device)


def _apply_init(
    model: nn.Module,
    cfg: Config,
    stage: int,
    init_checkpoint: str | Path | None,
    init_policy: str,
    resume_path: Path | None,
) -> None:
    """处理阶段初始化（``--init``）。

    续训时不重复加载权重（检查点里已有完整模型），只有非续训场景才走这里。
    """
    if resume_path is not None or init_policy == "none":
        return

    if stage == 0:
        if init_checkpoint is not None:
            info = load_model_weights(model, resolve(init_checkpoint), strict=True)
            print(f"[stage0] 从 {init_checkpoint} 初始化（missing={len(info['missing'])}）")
        return

    checkpoint = Path(init_checkpoint) if init_checkpoint else _default_init_path(cfg, stage)
    if not checkpoint.exists():
        hint = (
            f"阶段 {stage} 需要{('基线（stage 0）' if stage == 1 else '阶段 1')}检查点："
            f"请传 --init，或先训练上一阶段"
        )
        if init_policy == "required":
            raise FileNotFoundError(f"{hint}（查找路径：{checkpoint}）")
        if init_policy == "none":
            return
        raise FileNotFoundError(f"{hint}（查找路径：{checkpoint}）")

    if stage == 1:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        info = load_baseline_into_pcdf(model, payload)
        print(
            f"[stage1] 从 baseline {display_path(checkpoint)} 初始化"
            f"（新增参数 {len(info['missing'])} 个张量）"
        )
    else:
        info = load_model_weights(model, checkpoint, strict=True)
        print(f"[stage2] 从 {display_path(checkpoint)} 初始化（missing={len(info['missing'])}）")


def _default_init_path(cfg: Config, stage: int) -> Path:
    """阶段 1/2 的默认初始权重：上一阶段的 ``best.pt``，退而求其次用 ``last.pt``。"""
    previous = stage_dir(cfg.paths.runs_dir, cfg.train.run_name, stage - 1)
    for name in ("best.pt", "last.pt"):
        candidate = previous / name
        if candidate.exists():
            return candidate
    return previous / "best.pt"


def _print_setup(
    cfg: Config, stage: int, device: torch.device, model: nn.Module,
    params: list[nn.Parameter], n_images: int,
) -> None:
    """打印一次性的启动摘要，便于在日志开头核对配置。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in params)
    clip = ""
    if stage != 0:
        clip = f" clip(frozen)={model.parameter_summary()['clip_frozen'] / 1e6:.1f}M"
    print(
        f"[train] stage={stage}({STAGE_NAMES[stage]}) profile={cfg.train.profile or 'base'} "
        f"batch={cfg.data.batch} lr={cfg.train.lr:g} warmup={cfg.train.warmup_epochs:g}epoch "
        f"workers={cfg.data.workers} amp={cfg.train.amp} "
        f"deterministic={cfg.train.deterministic} device={device}",
        flush=True,
    )
    print(f"[model] trainable={trainable / 1e6:.2f}M total={total / 1e6:.2f}M{clip}", flush=True)
    print(
        f"[data] train={n_images} images, "
        f"batches/epoch={(n_images + cfg.data.batch - 1) // cfg.data.batch}",
        flush=True,
    )


def _command_line() -> str:
    return " ".join([Path(sys.executable).name, *sys.argv])


# --------------------------------------------------------------------------- #
# 信号处理
# --------------------------------------------------------------------------- #
def _install_signal_handlers(stop_flag: dict[str, bool]) -> dict[int, Any]:
    """捕获 SIGTERM/SIGINT，在当前 step 后保存检查点。"""
    previous: dict[int, Any] = {}

    def handler(signum, _frame):
        stop_flag["value"] = True
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        print(f"\n[signal] 收到 {name}：将在当前 step 结束后保存检查点并退出……", flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # 非主线程或平台不支持时忽略。
            pass
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
