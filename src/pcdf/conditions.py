"""条件属性、合并规则与初始提示构造（论文式 1，补充材料 B.1）。

提示模板逐字取自补充材料 Sec. B.1：

    An aerial image, over {scenario}, at a altitude of {altitude},
    at an angle of {angle}, at {Time}, in {Weather}, in {Illumination}.

六个属性统一编码为整数 id，便于整批搬到 GPU。自然语言名称只来自
``configs/conditions.yaml``；天气/光照/场景的官方码本未公开时保留匿名类，
不在代码里臆造语义。

本模块是条件语义的**唯一**来源：数据管线取 id、SCPL 拼提示、诊断脚本挑子集，
都走 :class:`ConditionTable`，不再各自解析 CSV。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


@dataclass
class AttributeSpec:
    """单个条件属性的定义：原始字段、提示前缀、类别名与合并规则。

    ``kind`` 支持三种：

    * ``identity``：原始整数码直接映射，可用 ``groups`` 把多个码合并成一类；
    * ``threshold``：按阈值二分（高度 120 m、角度 30°）；
    * ``time``：按小时落入 Morning / Afternoon / Night 三个区间。
    """

    key: str
    source: str
    prefix: str
    kind: str
    names: list[str] = field(default_factory=list)
    codes: list[int] | None = None
    groups: list[int] | None = None
    threshold: float | None = None
    low: str = ""
    high: str = ""
    bins: list[dict] = field(default_factory=list)
    _code_to_id: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind == "identity":
            if self.codes is None:
                raise ValueError(f"{self.key}: identity 属性必须提供 codes")
            # groups 为空时一个码对应一类；提供 groups 时用于 B.2 合并。
            if self.groups is None:
                self.groups = list(range(len(self.codes)))
            if len(self.groups) != len(self.codes):
                raise ValueError(f"{self.key}: groups 长度必须与 codes 一致")
            if len(self.names) != max(self.groups) + 1:
                raise ValueError(f"{self.key}: names 必须覆盖所有 group id")
            self._code_to_id = {int(c): int(g) for c, g in zip(self.codes, self.groups)}
        elif self.kind == "threshold":
            if self.threshold is None or not self.low or not self.high:
                raise ValueError(f"{self.key}: threshold 属性需要 threshold/low/high")
            self.names = [self.low, self.high]
        elif self.kind == "time":
            if not self.bins:
                raise ValueError(f"{self.key}: time 属性需要 bins")
            self.names = [str(b["name"]) for b in self.bins]
        else:
            raise ValueError(f"{self.key}: 未知 kind={self.kind!r}")

    @property
    def n_classes(self) -> int:
        return len(self.names)

    def resolve(self, raw_value: str) -> int:
        """把一个原始 CSV 字段映射为连续的类别 id。"""
        value = str(raw_value).strip()
        if self.kind == "identity":
            try:
                code = int(float(value))
            except ValueError as exc:
                raise ValueError(f"{self.key}: 非整数条件码 {value!r}") from exc
            if code not in self._code_to_id:
                raise ValueError(
                    f"{self.key}: 条件码 {code} 不在 configs/conditions.yaml 中；"
                    f"已知码为 {sorted(self._code_to_id)}"
                )
            return self._code_to_id[code]
        if self.kind == "threshold":
            number = float(value)
            return 0 if number <= float(self.threshold) else 1
        hour = _parse_hour(value)
        for idx, b in enumerate(self.bins):
            start, end = int(b["start"]), int(b["end"])
            inside = start <= hour < end if start < end else (hour >= start or hour < end)
            if inside:
                return idx
        raise ValueError(f"{self.key}: 小时 {hour} 未落入任何时间区间")


@dataclass
class ConditionTable:
    """样本到条件 id 的索引，以及提示句的拼装。

    ``row_by_stem`` 保存 CSV 原始行（包括 ``n_objects`` 之类的非条件字段），
    让诊断脚本不必为了取一个列而重新解析 CSV；``columns`` 是原始表头，
    生成子集索引时按同一顺序写出。
    """

    specs: list[AttributeSpec]
    subject: str
    ids_by_stem: dict[str, np.ndarray]
    row_by_stem: dict[str, dict[str, str]]
    subset_by_stem: dict[str, str]
    columns: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ 属性
    @property
    def n_attributes(self) -> int:
        return len(self.specs)

    @property
    def keys(self) -> list[str]:
        return [s.key for s in self.specs]

    @property
    def class_counts(self) -> list[int]:
        return [s.n_classes for s in self.specs]

    @property
    def raw_by_stem(self) -> dict[str, dict[str, str]]:
        """每个样本的六个原始条件字段（只保留码本声明的列）。"""
        sources = [s.source for s in self.specs]
        return {
            stem: {key: row.get(key, "") for key in sources}
            for stem, row in self.row_by_stem.items()
        }

    def stems(self, subset: str | None = None) -> list[str]:
        """按划分列出样本 stem（顺序与 CSV 一致）。

        ``subset=None`` 返回全部。诊断脚本用它挑子集，避免再解析一遍 CSV。
        """
        if subset is None:
            return list(self.ids_by_stem)
        return [stem for stem, split in self.subset_by_stem.items() if split == subset]

    def subset_of(self, stem: str) -> str:
        """返回样本所属划分；未知 stem 返回空串。"""
        return self.subset_by_stem.get(stem, "")

    # ------------------------------------------------------------------ 构建
    @classmethod
    def load(cls, index_csv: str | Path, spec_yaml: str | Path) -> "ConditionTable":
        """从 condition_index.csv 与 conditions.yaml 构建索引。

        Raises:
            FileNotFoundError: 索引或码本缺失。
            ValueError: CSV 缺少码本声明的列。
        """
        index_path, spec_path = Path(index_csv), Path(spec_yaml)
        if not spec_path.is_file():
            raise FileNotFoundError(f"未找到条件码本：{spec_path}")
        if not index_path.is_file():
            raise FileNotFoundError(f"未找到条件索引：{index_path}")

        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        specs = [AttributeSpec(**dict(item)) for item in spec.get("attributes", [])]
        if not specs:
            raise ValueError(f"{spec_path} 未声明任何属性")
        subject = str(spec.get("subject", "An aerial image"))

        raw_rows: dict[str, dict[str, str]] = {}
        subset_by_stem: dict[str, str] = {}
        ids_by_stem: dict[str, np.ndarray] = {}
        with open(index_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            header = set(columns)
            missing_columns = [s.source for s in specs if s.source not in header]
            if "stem" not in header:
                raise ValueError(f"{index_path} 缺少 stem 列")
            if missing_columns:
                raise ValueError(
                    f"{index_path} 缺少码本声明的列 {missing_columns}；"
                    f"实际列为 {sorted(header)}"
                )
            for row in reader:
                stem = str(row["stem"]).strip()
                clean = {key: str(value or "").strip() for key, value in row.items()}
                raw_rows[stem] = clean
                subset_by_stem[stem] = clean.get("subset", "")
                ids_by_stem[stem] = np.array(
                    [s.resolve(clean.get(s.source, "")) for s in specs], dtype=np.int64
                )
        if not ids_by_stem:
            raise RuntimeError(f"条件索引为空：{index_path}")
        return cls(
            specs=specs,
            subject=subject,
            ids_by_stem=ids_by_stem,
            row_by_stem=raw_rows,
            subset_by_stem=subset_by_stem,
            columns=columns,
        )

    # ------------------------------------------------------------------ 提示
    def _block(self, attr_idx: int, class_id: int) -> str:
        """单个条件块，例如 ``over scenario 3``。"""
        spec = self.specs[attr_idx]
        return f"{spec.prefix} {spec.names[int(class_id)]}"

    def compose(self, ids: Sequence[int], mask: Sequence[bool] | None = None) -> str:
        """拼装一条提示；``mask`` 用于丢弃属性（论文式 3-4）。"""
        blocks = []
        for i, class_id in enumerate(ids):
            if mask is None or bool(mask[i]):
                blocks.append(self._block(i, int(class_id)))
        if not blocks:
            raise ValueError("六个条件属性全部被屏蔽，无法构造提示")
        return f"{self.subject}, " + ", ".join(blocks) + "."

    def prompt(self, stem: str, mask: Sequence[bool] | None = None) -> str:
        return self.compose(self.ids_by_stem[stem], mask)

    def prompt_from_ids(self, ids: Sequence[int], mask: Sequence[bool] | None = None) -> str:
        return self.compose(ids, mask)

    def block_prompt(self, attr_idx: int, class_id: int) -> str:
        """单属性提示，供 SCPL 的 straight-through 梯度路径使用。"""
        return f"{self.subject}, {self._block(attr_idx, int(class_id))}."

    def ids(self, stem: str) -> np.ndarray:
        try:
            return self.ids_by_stem[stem]
        except KeyError as exc:
            raise KeyError(f"{stem} 不在 condition_index.csv 中") from exc

    def class_distribution(self) -> dict[str, dict[str, int]]:
        """统计每个属性的类别分布，便于检查条件码本是否正确。"""
        out: dict[str, dict[str, int]] = {}
        for i, spec in enumerate(self.specs):
            counts = np.bincount(
                np.stack([v[i] for v in self.ids_by_stem.values()]), minlength=spec.n_classes
            )
            out[spec.key] = {spec.names[j]: int(c) for j, c in enumerate(counts)}
        return out


def _parse_hour(value: str) -> int:
    """ATR-UMOD 时间形如 ``2022-6-5T16:10``，只取小时。"""
    text = value.strip()
    if "T" in text:
        clock = text.split("T", 1)[1]
    elif " " in text:
        clock = text.split(" ", 1)[1]
    else:
        clock = text
    try:
        return int(clock.split(":", 1)[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"无法解析采集时间 {value!r}") from exc
