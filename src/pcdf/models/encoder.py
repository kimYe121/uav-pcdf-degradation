"""双流编码器的 backbone（论文 Fig. 4 的 RGB/IR Encoder）。

论文只写了“dual branch encoders”，没有公开检测器 family。本复现支持两种骨架，
都作为 implementation choice 记录在 ``docs/ARCHITECTURE.md``：

* YOLO11-OBB：``yolo11n-obb.yaml``（128/128/256）、``yolo11s-obb.yaml``
  （256/256/512），预训练用 ``yolo11*-obb.pt``；
* YOLOv5：``yolov5.yaml`` + ``scale``（s 档 128/256/256），预训练用
  ``yolov5su.pt`` 的 backbone 层。论文式 8 含 ``L_obj``、一作 fork 了 v5，
  因此这是默认家族。

关于规模推断
------------
Ultralytics 只从**文件名**推断 YOLOv5 的 ``n/s/m/l/x``，不同小版本行为不一致
（8.4.211 与 8.4.216 就不同）。因此这里显式写入 ``scale``，并在有检查点时用
首层通道数反查真实规模，两者不一致直接报错，而不是"加载成功但全是随机权重"。
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import yaml as _yaml
from torch import Tensor

from ..config import V5_SCALES

# YOLOv5 首层 ``Conv(3, 64*width)`` 的输出通道 -> 规模。
V5_SCALE_BY_CHANNELS: dict[int, str] = {16: "n", 32: "s", 48: "m", 64: "l", 80: "x"}


def _checkpoint_module(weights: str | Path | None) -> nn.Module | None:
    """从 Ultralytics 检查点里取出它自己的模型对象（含 backbone 层与权重）。

    返回 ``None`` 表示文件不存在、不是 Ultralytics 检查点、或只有 state_dict，
    此时由调用方回退到按 yaml 构造。
    """
    if weights is None:
        return None
    path = Path(weights)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 —— 损坏/非 torch 文件时回退到 yaml
        return None
    module = payload.get("model") if isinstance(payload, dict) else payload
    if isinstance(module, nn.Module) and hasattr(module, "model") and hasattr(module, "save"):
        return module
    return None


def infer_v5_scale(weights: str | Path) -> str | None:
    """从 YOLOv5 检查点的首层通道数反推规模；读不到返回 ``None``。

    用于把"配置写 s、权重其实是 n"挡在建模之前——这种情况逐层加载会失败，
    但错误信息通常指向某个具体层名，很难看出根因是权重档位错了。
    """
    module = _checkpoint_module(weights)
    if module is None:
        return None
    state = module.state_dict()
    weight = state.get("model.0.conv.weight")
    if weight is None:
        return None
    return V5_SCALE_BY_CHANNELS.get(int(weight.shape[0]))


@lru_cache(maxsize=8)
def _v5_yaml(scale: str) -> dict:
    """读取并缓存 Ultralytics 自带的 YOLOv5 yaml，写入目标 scale。"""
    import ultralytics

    source = Path(ultralytics.__file__).parent / "cfg" / "models" / "v5" / "yolov5.yaml"
    if not source.is_file():
        raise FileNotFoundError(
            f"当前 ultralytics 版本缺少 {source}；YOLOv5 骨架不可用。"
            "请改用 yolo11*-obb.yaml，或安装带 v5 配置的 ultralytics。"
        )
    with source.open(encoding="utf-8") as handle:
        config = _yaml.safe_load(handle)
    config["scale"] = scale
    return config


def _build_reference(yaml_name: str, nc: int, scale: str | None) -> nn.Module:
    """按 yaml 构造一个参考模型，稍后只取它的 backbone 层。"""
    from ultralytics.nn.tasks import DetectionModel, OBBModel

    if "yolov5" in yaml_name:
        chosen = scale or "s"
        if chosen not in V5_SCALES:
            raise ValueError(f"未知的 YOLOv5 规模 {chosen!r}，可选 {sorted(V5_SCALES)}")
        return DetectionModel(_v5_yaml(chosen), nc=nc, verbose=False)
    return OBBModel(yaml_name, nc=nc, verbose=False)


class Backbone(nn.Module):
    """YOLO backbone，输出 P3/8、P4/16、P5/32 三层特征。

    优先从 ``.pt`` 里直接取它自己的层与权重（层与权重天生一致）；没有可用
    检查点时按 yaml 构造并随机初始化，再可选地加载 backbone 权重。
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        nc: int = 11,
        yaml: str = "yolo11n-obb.yaml",
        scale: str | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        reference = _checkpoint_module(weights)
        source = "checkpoint"
        if reference is not None:
            # 官方 .pt 的权重常是 FP16，训练前统一转回 FP32。
            reference = reference.float()
            if "yolov5" in yaml:
                self._verify_v5_scale(Path(weights), scale)
        else:
            reference = _build_reference(yaml, nc, scale)
            source = "yaml"

        taps = tuple(int(i) for i in reference.save[:3])  # 11 系与 v5 均为 (4, 6, 10)
        self.layers = nn.ModuleList([reference.model[i] for i in range(max(taps) + 1)])
        self.taps = taps
        del reference

        self.out_channels = self._infer_channels()
        self._source = source
        if weights is not None and source == "yaml":
            self.load_pretrained(weights)
        if verbose:
            print(
                f"[backbone] src={source} yaml={yaml} scale={scale} out={self.out_channels} "
                f"weights={Path(weights).resolve() if weights else '随机初始化'}",
                flush=True,
            )

    @staticmethod
    def _verify_v5_scale(weights: Path, configured: str | None) -> None:
        """校验配置的 v5 规模与检查点首层通道一致。

        没配置 ``scale`` 时以权重为准（不报错）；配置了却对不上时必须报错，
        否则后续 ``load_state_dict`` 会在某一层给出难以定位的 shape 错误。
        """
        actual = infer_v5_scale(weights)
        if configured and actual and configured != actual:
            raise ValueError(
                f"model.backbone_scale={configured!r} 与权重 {weights.name} 的实际规模 "
                f"{actual!r} 不一致；请改配置或换权重"
            )

    def _infer_channels(self) -> list[int]:
        """用一次 dummy 前向确认三层通道数，防止 YAML 与 widths 写错。"""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            feats = self.forward(torch.zeros(1, 3, 128, 128))
        self.train(was_training)
        return [int(f.shape[1]) for f in feats]

    @classmethod
    def from_backbone(cls, other: "Backbone") -> "Backbone":
        """深拷贝另一个 backbone 的层与权重，作为双流的第二分支。

        两个分支必须是**同一架构 + 同一份初始化**；分别从检查点重建可能读到
        不同档位，因此统一由 RGB 分支拷贝。
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.layers = nn.ModuleList([copy.deepcopy(layer) for layer in other.layers])
        obj.taps = tuple(other.taps)
        obj.out_channels = list(other.out_channels)
        obj._source = f"clone:{other._source}"
        return obj

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index in self.taps:
                outputs.append(x)
        return outputs

    def load_pretrained(self, weights: str | Path) -> tuple[int, int]:
        """从 Ultralytics 检查点加载 ``model.{i}.*`` 的 backbone 权重。

        只加载 backbone 层（0-10），不加载预训练 head；返回值便于日志核对。
        若某层键不匹配直接报错，避免"看似加载成功、实际全是随机权重"。
        """
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(f"未找到 backbone 权重：{path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = payload.get("model", payload) if isinstance(payload, dict) else payload
        state = model.state_dict() if hasattr(model, "state_dict") else model

        loaded = 0
        for index, layer in enumerate(self.layers):
            prefix = f"model.{index}."
            sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if not sub:
                continue
            missing, unexpected = layer.load_state_dict(sub, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"backbone 第 {index} 层权重不匹配：missing={missing} unexpected={unexpected}"
                )
            loaded += 1
        if loaded == 0:
            raise RuntimeError(
                f"{path} 里没有可用的 backbone 层（期望 model.0.* … model.10.*）；"
                "请确认这是 Ultralytics 检测模型检查点"
            )
        return loaded, len(self.layers)
