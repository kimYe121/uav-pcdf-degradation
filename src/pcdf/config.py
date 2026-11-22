"""类型化配置：YAML 解析、档位合并、参数覆盖、路径归一与交叉校验。

分层
----
``Config`` 由四段组成，彼此职责不重叠：

* ``paths``  —— 数据/权重/运行/日志目录（唯一可被环境变量覆盖的一段）；
* ``data``   —— 配对数据集、增强、评估参数；
* ``model``  —— 骨架、SCPL/PCD/CDF、检测头与损失增益；
* ``train``  —— 优化器配方、周期保存与断点续训策略。

三条不可破坏的规则
------------------
1. **未知键直接报错**：YAML 拼写错误不会静默地改变复现实验。
2. **路径只解析一次**：加载结束时全部转成绝对路径，之后任何模块都不再拼相对
   路径，也不依赖 ``cwd``。
3. **交叉约束在一处校验**：例如 ``guidance_dim`` 必须等于 P5 通道数；这类
   约束以前散落在模型构造函数里，错误要等到建模时才暴露。

覆盖优先级：基础 YAML < 档位 YAML < ``--set``/CLI 参数 < 路径环境变量。
最后一步（环境变量）优先，是因为部署时容器挂载点必须压过仓库内默认值。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, get_args, get_origin, get_type_hints

import yaml

from .constants import DEFAULT_ANCHORS, DEFAULT_NAMES, STRIDES
from .paths import (
    DEFAULT_CONFIG_PATH,
    ENV_CLIP_DIR,
    PATH_ENV_VARS,
    PROFILES_DIR,
    PROJECT_ROOT,
    resolve,
)

__all__ = [
    "ConfigError",
    "Config",
    "DataConfig",
    "LossConfig",
    "ModelConfig",
    "PathsConfig",
    "TrainConfig",
    "V5_SCALES",
    "apply_path_env",
    "available_profiles",
    "load_config",
    "resolve_profile",
    "save_config",
]

# YOLOv5 各规模的 (depth_multiple, width_multiple)。
# Ultralytics 只从**文件名**推断 v5 规模，不同小版本行为不一致，因此这里显式
# 声明，并在加载时用它校验配置与权重是否匹配。
V5_SCALES: dict[str, tuple[float, float]] = {
    "n": (0.33, 0.25),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
    "x": (1.33, 1.25),
}


class ConfigError(ValueError):
    """配置非法（未知键、类型不符、交叉约束不满足）。"""


# --------------------------------------------------------------------------- #
# 配置段
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    """资源与产物目录。

    这些字段是唯一允许被环境变量覆盖的部分；其余字段都是"实验配方"，
    换机器不应改变。相对路径按仓库根解析，绝对路径原样保留。
    """

    data_root: str = "data/atr_umod"
    # 留空则取 ``<data_root>/condition_index.csv``，避免同一个路径写两遍。
    condition_index: str = ""
    weights_dir: str = "weights"
    runs_dir: str = "runs"
    logs_dir: str = "logs"


@dataclass
class DataConfig:
    """ATR-UMOD 配对 RGB/IR 数据集与评估参数。

    图像与标签都放在 IR 目录下；RGB 图像通过把路径中的 ``modality_dir``
    替换为 ``pair_dir`` 得到。论文 Sec. 5.2 明确多模态检测器使用 IR 标签，
    因此这里始终读取 ``ir/labels``。
    """

    train_split: str = "train"
    val_split: str = "val"
    modality_dir: str = "ir"
    pair_dir: str = "rgb"
    imgsz: tuple[int, int] = (640, 512)  # 论文 Sec. 5.1，顺序为 (H, W)
    batch: int = 16  # 论文 batch；小显存请用 --batch 或 profile 覆盖
    workers: int = 4
    names: list[str] = field(default_factory=lambda: list(DEFAULT_NAMES))

    # 评估/NMS
    eval_conf: float = 0.001
    eval_iou: float = 0.7
    eval_max_det: int = 300
    eval_thresholds: int = 0  # >0 时额外计算 mAP50-95（10 个阈值）

    # 数据增强：论文未说明，详见 docs/ARCHITECTURE.md。
    # 有意关闭 mosaic 与 HSV：mosaic 会强制方形画布（论文为 640×512），
    # HSV 会同时改变复制到 3 通道的红外图像。
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flip_p: float = 0.5
    flip_vertical: float = 0.0
    fraction: float = 1.0

    @property
    def nc(self) -> int:
        """类别数，由 ``names`` 推导，避免与模型侧各写一份。"""
        return len(self.names)

    @property
    def height(self) -> int:
        return int(self.imgsz[0])

    @property
    def width(self) -> int:
        return int(self.imgsz[1])


@dataclass
class LossConfig:
    """检测损失增益。

    两套检测头的量纲不同，因此增益分开配置：anchor 头的回归项是
    ``1 - ProbIoU``（0~1 量级），v8 头是 CIoU + DFL（数值大一个量级）。
    混用一组增益会让其中一族的损失被压死，所以不共用。
    """

    # anchor-based（论文式 8）：L_obj + L_cls + L_reg
    anchor_box: float = 0.05
    anchor_cls: float = 0.5
    anchor_obj: float = 1.0
    anchor_thr: float = 0.25  # 锚框匹配的边长比阈值（与 YOLOv5 的 1/4 一致）
    # Ultralytics v8/v11 anchor-free：box + cls + dfl
    v8_box: float = 7.5
    v8_cls: float = 0.5
    v8_dfl: float = 1.5


@dataclass
class ModelConfig:
    """PCDF 结构、检测头与损失权重。

    ``widths`` 是 P3/P4/P5 通道数。``guidance_dim`` 必须等于 P5 通道数：
    阶段 1 的文本引导与阶段 2/测试的 ``GAP(F_s)`` 共用同一空间，CDF 投影器
    才能直接迁移。
    """

    widths: tuple[int, int, int] = (128, 256, 256)
    # "v8"     —— Ultralytics anchor-free OBB 头（DFL + 角度）；
    # "anchor" —— 论文式 8 对应的 YOLOv5 风格 anchor-based 头（含 objectness）。
    head_type: str = "v8"
    # Ultralytics 的检测模型 yaml；换档位时与 widths、guidance_dim 一起改。
    backbone_yaml: str = "yolov5.yaml"
    # 仅 yolov5.yaml 使用：该 yaml 不带规模后缀，需显式指定 n/s/m/l/x。
    backbone_scale: str = "s"
    reg_max: int = 16
    # anchor 头的默认锚框；留空则用 constants 中的 k-means 结果。
    anchors: dict[int, list[tuple[int, int]]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_ANCHORS.items()}
    )

    tau: float = 0.15  # 补充材料 Sec. G
    lambda_dt: float = 0.01  # 式 9
    lambda_irr: float = 0.003
    lambda_dc: float = 0.01

    condition_spec: str = "configs/conditions.yaml"
    # 以下两项相对 ``paths.weights_dir`` 解析；加载后被替换为绝对路径。
    encoder_weights: str = "yolov5su.pt"
    text_encoder: str = "clip-vit-base-patch16"
    freeze_text: bool = True
    pretrained_backbone: bool = True

    gate_hidden: int = 256
    guidance_dim: int = 256
    gate_pool: str = "max"
    text_cache_size: int = 8192

    loss: LossConfig = field(default_factory=LossConfig)

    @property
    def strides(self) -> tuple[int, int, int]:
        return STRIDES


@dataclass
class TrainConfig:
    """优化器配方（论文 Sec. 5.1）与断点续训策略。"""

    epochs: int = 50
    lr: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    final_lr_ratio: float = 0.01  # 指数衰减终点相对 lr 的比例
    warmup_epochs: float = 0.0  # 论文未提及，默认关闭
    warmup_bias_lr: float = 0.1
    grad_clip: float = 10.0
    amp: bool = True
    grad_accum: int = 1
    seed: int = 0
    deterministic: bool = True

    eval_interval: int = 5  # <=0 表示不验证
    val_batches: int = 0  # 0 表示完整验证集
    max_batches: int = 0  # 0 表示完整 epoch

    run_name: str = "pcdf"
    profile: str = ""  # 由 load_config 写入，仅用于记录与巡检
    resume: str = ""  # 也可由 CLI --resume 覆盖；auto 读取 run_dir/last.pt

    # 周期保存：容器随时可能被抢占，mid-epoch 也要留完整状态。
    save_period: int = 1  # 每 N 个 epoch 写 checkpoints/epoch_XXXX.pt
    save_every_minutes: float = 15.0  # 0 表示关闭
    save_every_steps: int = 0  # 0 表示关闭
    keep_checkpoints: int = 3  # 只保留最近 N 个 epoch 检查点


@dataclass
class Config:
    """顶层配置容器。"""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Config":
        """从原始映射构造，并拒绝未知键。"""
        if not isinstance(raw, Mapping):
            raise ConfigError(f"配置必须是映射，收到 {type(raw).__name__}")
        return _build(cls, dict(raw))

    def to_dict(self) -> dict[str, Any]:
        """转成可安全写入 YAML/JSON 的纯 Python 结构（tuple -> list）。"""
        return _normalise(self)

    # ------------------------------------------------------------------ 覆盖
    def apply(self, overrides: Mapping[str, Any] | None) -> "Config":
        """按键值对覆盖配置；值为 ``None`` 的条目被忽略。

        键支持点号路径（``train.lr``、``model.loss.anchor_box``），值会按目标
        字段声明的类型做转换，因此 ``--set train.amp=false`` 也能正确得到布尔。
        """
        if not overrides:
            return self
        for key, value in overrides.items():
            if value is None:
                continue
            self.set_value(key, value)
        return self

    def set_value(self, key: str, value: Any) -> None:
        """设置单个点号路径，并按声明类型转换。"""
        parts = [p for p in str(key).split(".") if p]
        if not parts:
            raise ConfigError("配置键不能为空")
        holder: Any = self
        for index, part in enumerate(parts[:-1]):
            prefix = ".".join(parts[: index + 1])
            if not hasattr(holder, part):
                raise ConfigError(f"未知配置段 {prefix!r}（可用：{_section_names(holder)}）")
            holder = getattr(holder, part)
            if not is_dataclass(holder):
                raise ConfigError(f"{prefix!r} 不是配置段，无法继续下钻")
        name = parts[-1]
        declared = _field_types(type(holder)).get(name)
        if declared is None:
            raise ConfigError(
                f"未知配置键 {'.'.join(parts)!r}（{type(holder).__name__} 可用：{_section_names(holder)}）"
            )
        setattr(holder, name, _coerce(value, declared, key))

    # ------------------------------------------------------------------ 校验
    def validate(self) -> None:
        """交叉校验；不满足时一次性抛出所有问题，避免逐个试错。"""
        problems: list[str] = []
        paths, data, model, train = self.paths, self.data, self.model, self.train

        if not data.names:
            problems.append("data.names 不能为空")
        if len(data.imgsz) != 2:
            problems.append(f"data.imgsz 必须是 (H, W)，当前 {data.imgsz}")
        else:
            for value, axis in zip(data.imgsz, ("H", "W")):
                if int(value) <= 0 or int(value) % 32 != 0:
                    problems.append(f"data.imgsz 的 {axis}={value} 必须是 32 的正整数倍")
        if data.batch < 1:
            problems.append(f"data.batch 必须 >= 1，当前 {data.batch}")
        if data.workers < 0:
            problems.append(f"data.workers 必须 >= 0，当前 {data.workers}")
        if not 0.0 < data.fraction <= 1.0:
            problems.append(f"data.fraction 必须落在 (0, 1]，当前 {data.fraction}")
        for name in ("eval_conf", "eval_iou"):
            value = getattr(data, name)
            if not 0.0 < float(value) <= 1.0:
                problems.append(f"data.{name} 必须落在 (0, 1]，当前 {value}")
        if data.eval_max_det < 1:
            problems.append(f"data.eval_max_det 必须 >= 1，当前 {data.eval_max_det}")

        if len(model.widths) != 3:
            problems.append(f"model.widths 必须是三个尺度的通道数，当前 {model.widths}")
        elif model.guidance_dim != model.widths[2]:
            problems.append(
                "model.guidance_dim 必须等于 P5 通道数（阶段 2 的 GAP(F_s) 直接驱动 CDF），"
                f"当前 guidance_dim={model.guidance_dim}，P5={model.widths[2]}"
            )
        if model.head_type not in ("v8", "anchor"):
            problems.append(f"model.head_type 只能是 v8 / anchor，当前 {model.head_type!r}")
        if "yolov5" in model.backbone_yaml and model.backbone_scale not in V5_SCALES:
            problems.append(
                f"model.backbone_scale 只能是 {sorted(V5_SCALES)}，当前 {model.backbone_scale!r}"
            )
        if not 0.0 < model.tau < 1.0:
            problems.append(f"model.tau 必须落在 (0, 1)，当前 {model.tau}")
        for name in ("lambda_dt", "lambda_irr", "lambda_dc"):
            if getattr(model, name) < 0:
                problems.append(f"model.{name} 不能为负")
        if model.gate_pool not in ("max", "avg"):
            problems.append(f"model.gate_pool 只能是 max / avg，当前 {model.gate_pool!r}")
        if model.head_type == "anchor":
            missing = [s for s in STRIDES if s not in model.anchors]
            if missing:
                problems.append(f"model.anchors 缺少尺度 {missing} 的锚框")
            else:
                for stride, boxes in model.anchors.items():
                    if len(boxes) != 3:
                        problems.append(f"model.anchors[{stride}] 必须正好 3 个锚框")
                    if any(min(w, h) <= 0 for w, h in boxes):
                        problems.append(f"model.anchors[{stride}] 存在非正宽高")

        if train.epochs < 1:
            problems.append(f"train.epochs 必须 >= 1，当前 {train.epochs}")
        if train.lr <= 0:
            problems.append(f"train.lr 必须 > 0，当前 {train.lr}")
        if not 0.0 < train.final_lr_ratio <= 1.0:
            problems.append(f"train.final_lr_ratio 必须落在 (0, 1]，当前 {train.final_lr_ratio}")
        if train.grad_accum < 1:
            problems.append(f"train.grad_accum 必须 >= 1，当前 {train.grad_accum}")
        if train.warmup_epochs < 0:
            problems.append(f"train.warmup_epochs 不能为负，当前 {train.warmup_epochs}")
        if train.eval_interval < 0:
            problems.append(f"train.eval_interval 不能为负，当前 {train.eval_interval}")
        if train.save_period < 0 or train.keep_checkpoints < 0:
            problems.append("train.save_period / keep_checkpoints 不能为负")
        for name in ("runs_dir", "data_root", "weights_dir"):
            if not str(getattr(paths, name)).strip():
                problems.append(f"paths.{name} 不能为空")

        if problems:
            joined = "\n".join(f"  - {p}" for p in problems)
            raise ConfigError(f"配置校验失败：\n{joined}")

    # ------------------------------------------------------------------ 路径
    def resolve_paths(self) -> "Config":
        """把所有路径字段绝对化，并补全由其它字段推导出的路径。

        必须在校验与建模之前调用；调用之后 ``cfg.model.encoder_weights`` 等
        字段可以直接交给 ``torch.load`` / ``from_pretrained``。
        """
        paths = self.paths
        data_root = resolve(paths.data_root)
        paths.data_root = str(data_root)
        paths.condition_index = str(
            resolve(paths.condition_index) if str(paths.condition_index).strip()
            else data_root / "condition_index.csv"
        )
        weights_dir = resolve(paths.weights_dir)
        paths.weights_dir = str(weights_dir)
        paths.runs_dir = str(resolve(paths.runs_dir))
        paths.logs_dir = str(resolve(paths.logs_dir))

        self.model.encoder_weights = str(resolve(self.model.encoder_weights, base=weights_dir))
        self.model.text_encoder = str(resolve(self.model.text_encoder, base=weights_dir))
        self.model.condition_spec = str(resolve(self.model.condition_spec))
        return self

    # ------------------------------------------------------------------ 便捷
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def run_dir_name(self) -> str:
        return self.train.run_name

    def stage_dir(self, stage: int) -> Path:
        """某阶段的默认运行目录。"""
        from .paths import stage_dir

        return stage_dir(self.paths.runs_dir, self.train.run_name, stage)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _field_types(cls: type) -> dict[str, Any]:
    """返回 ``{字段名: 声明类型}``；``from __future__`` 下必须用 hints。"""
    return get_type_hints(cls)


def _section_names(obj: Any) -> str:
    if not is_dataclass(obj):
        return ""
    return ", ".join(f.name for f in fields(obj))


def _normalise(value: Any) -> Any:
    """递归转成纯 Python 结构：tuple -> list，dataclass -> dict。"""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _normalise(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _build(cls: type, raw: Mapping[str, Any], prefix: str = "") -> Any:
    """递归构造 dataclass，并拒绝未知键。"""
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"{prefix or cls.__name__} 下存在未知配置键：{unknown}")
    hints = _field_types(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        target = hints.get(f.name, f.type)
        if isinstance(value, Mapping) and is_dataclass(target):
            value = _build(target, value, f"{prefix}{f.name}.")
        else:
            value = _coerce(value, target, f"{prefix}{f.name}")
        kwargs[f.name] = value
    return cls(**kwargs)


def _coerce(value: Any, target: Any, key: str = "") -> Any:
    """把 YAML/CLI 的原始值转成字段声明的类型。

    只做"形状"转换，不做语义校验——语义由 :meth:`Config.validate` 统一负责。
    """
    origin = get_origin(target)

    if target is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off"):
            return False
        raise ConfigError(f"{key or '值'} 需要布尔值，收到 {value!r}")

    if target is int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{key or '值'} 需要整数，收到 {value!r}") from exc

    if target is float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{key or '值'} 需要数值，收到 {value!r}") from exc

    if target is str:
        return str(value)

    if origin is tuple:
        args = get_args(target)
        items = _as_sequence(value, key)
        if args and len(args) == 2 and args[0] is int and args[1] is Ellipsis:
            return tuple(_coerce(v, int, key) for v in items)
        if args and len(items) != len(args):
            raise ConfigError(f"{key or '值'} 需要 {len(args)} 个元素，收到 {items!r}")
        return tuple(_coerce(v, args[i], key) for i, v in enumerate(items)) if args else tuple(items)

    if origin is list:
        args = get_args(target)
        if args == (str,):
            if isinstance(value, str):
                return [part.strip() for part in value.split(",") if part.strip()]
            return [str(v) for v in value]
        items = _as_sequence(value, key)
        # 逐元素转换：``dict[int, list[tuple[int, int]]]`` 这类嵌套结构
        # 不能只保留 YAML 给出的 list，否则下游拿到的是 list 而非 tuple。
        return [_coerce(v, args[0], key) for v in items] if args else list(items)

    if origin is dict:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{key or '值'} 需要 JSON 对象，收到 {value!r}") from exc
        if not isinstance(value, Mapping):
            raise ConfigError(f"{key or '值'} 需要映射，收到 {type(value).__name__}")
        args = get_args(target)
        if len(args) != 2:
            return dict(value)
        key_t, val_t = args
        return {
            _coerce(k, key_t, f"{key}.key"): _coerce(v, val_t, f"{key}[{k}]")
            for k, v in value.items()
        }

    return value


def _as_sequence(value: Any, key: str = "") -> Sequence[Any]:
    if isinstance(value, str):
        text = value.replace(",", " ").split()
        if not text:
            raise ConfigError(f"{key or '值'} 需要一组数值，收到空字符串")
        return text
    if isinstance(value, (list, tuple)):
        return value
    raise ConfigError(f"{key or '值'} 需要列表/元组，收到 {type(value).__name__}")


# --------------------------------------------------------------------------- #
# 加载 / 保存
# --------------------------------------------------------------------------- #
def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到配置文件：{path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} 必须是 YAML 映射")
    return raw


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并：``override`` 优先，未出现的键沿用 ``base``。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_profile(profile: str | Path) -> Path:
    """把档位名解析成 YAML 路径。

    ``48g`` → ``configs/profiles/48g.yaml``；带目录或以 ``.yaml`` 结尾的名字
    按路径处理，便于在仓库外复用自己的档位。
    """
    candidate = Path(profile)
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return resolve(candidate)
    name = candidate.name if candidate.suffix else f"{candidate.name}.yaml"
    return PROFILES_DIR / name


def available_profiles() -> list[str]:
    """列出 ``configs/profiles`` 下的档位名，用于报错提示。"""
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def apply_path_env(cfg: Config) -> Config:
    """用环境变量覆盖路径段（部署时容器挂载点优先于仓库默认值）。

    公开出来是因为"从检查点重建配置"也需要同一套覆盖规则：训练机上写的绝对
    路径到了另一台机器必须让位给本机的 ``PCDF_DATA_ROOT``。
    """
    for env_name, key in PATH_ENV_VARS.items():
        raw = os.environ.get(env_name, "").strip()
        if raw:
            cfg.set_value(key, raw)
    clip_dir = os.environ.get(ENV_CLIP_DIR, "").strip()
    if clip_dir:
        cfg.model.text_encoder = clip_dir
    return cfg


def load_config(
    path: str | Path | None = None,
    *,
    profile: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    use_env: bool = True,
) -> Config:
    """读取 YAML、合并档位与覆盖项，并完成路径归一与校验。

    Args:
        path: 基础配置文件；``None`` 表示仓库内的 ``configs/pcdf.yaml``。
        profile: 档位名或 YAML 路径；档位只写差异项。
        overrides: 点号键覆盖（CLI 已解析的参数），优先级高于档位。
        use_env: 是否应用路径环境变量（测试里通常关掉）。

    Returns:
        已绝对化路径并通过校验的 :class:`Config`。
    """
    base_path = resolve(path or DEFAULT_CONFIG_PATH)
    raw = _read_yaml_mapping(base_path)

    profile_name = ""
    if profile:
        profile_path = resolve_profile(profile)
        if not profile_path.is_file():
            names = "、".join(available_profiles()) or "无"
            raise FileNotFoundError(
                f"未找到配置档位 {str(profile)!r}（{profile_path}）。可用档位：{names}"
            )
        raw = _deep_merge(raw, _read_yaml_mapping(profile_path))
        profile_name = profile_path.stem

    cfg = Config.from_dict(raw)
    cfg.train.profile = profile_name
    cfg.apply(overrides)
    if use_env:
        apply_path_env(cfg)
    cfg.resolve_paths()
    cfg.validate()
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """把生效配置写入运行目录，便于复现与续训对照。

    写入的是"已绝对化 + 已合并档位"的完整快照，因此事后只看 run 目录就能
    知道当时到底用了什么，不需要再猜 ``--profile`` 与命令行参数。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",  # 固定 LF：配置快照要能跨平台比对
    )
