"""冻结的 CLIP ViT-B/16 文本编码器与提示向量缓存。

论文 Fig. 4 与 Sec. 4.2 明确使用 CLIP 文本编码器并保持冻结，只训练 SCPL 的
门控网络。因此 CLIP 参数不会注册进检测器的 state_dict，也不会进入优化器。

数值对齐
--------
官方 CLIP 的文本特征是 ``text_projection(ln_final(EOT))``；而
``transformers.CLIPTextModel`` 只输出 ``ln_final(EOT)``，不含投影矩阵。
这里从权重文件中单独取出 ``text_projection.weight`` 并缓存成
``text_projection.pt``（约 1 MB），保证与 ``CLIPModel.get_text_features``
数值一致，同时避免每次启动都去读 571 MB 的全量权重。
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

from .conditions import ConditionTable

# CLIP 文本编码器的必需文件；preflight 与下载脚本共用同一份清单。
CLIP_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
CLIP_WEIGHT_FILES: tuple[str, ...] = ("pytorch_model.bin", "model.safetensors")
CLIP_PROJECTION_FILE = "text_projection.pt"


class CLIPTextEncoder:
    """``transformers.CLIPTextModel`` 的轻量封装（始终冻结）。"""

    def __init__(self, model_dir: str | Path, device: torch.device | str = "cpu") -> None:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"未找到 CLIP 文本编码器：{model_dir}。请把 openai/clip-vit-base-patch16 的 "
                "config.json、vocab.json、merges.txt、tokenizer_config.json、"
                "pytorch_model.bin 下载到该目录（可用 scripts/server/fetch_clip.py）。"
            )
        from transformers import CLIPTextModel, CLIPTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.device = torch.device(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = CLIPTextModel.from_pretrained(str(model_dir), local_files_only=True)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)

        self.proj_weight = self._load_text_projection(model_dir)
        if self.proj_weight is not None:
            self.proj_weight = self.proj_weight.to(self.device)

        self._tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": 77,
            "return_tensors": "pt",
        }

    @staticmethod
    def _load_text_projection(model_dir: Path) -> torch.Tensor | None:
        """读取 ``text_projection.weight``；优先用已缓存的小文件。

        没有该键时返回 ``None``，编码器退化为未投影的文本特征（并在日志里
        可见），不会静默改变维度约定——下游维度校验会先失败。
        """
        cache = model_dir / CLIP_PROJECTION_FILE
        if cache.is_file():
            return torch.load(cache, map_location="cpu", weights_only=True)

        weight = None
        bin_path = model_dir / "pytorch_model.bin"
        safe_path = model_dir / "model.safetensors"
        if bin_path.is_file():
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
            weight = state.get("text_projection.weight")
            del state
        elif safe_path.is_file():
            from safetensors import safe_open

            with safe_open(str(safe_path), framework="pt", device="cpu") as handle:
                if "text_projection.weight" in handle.keys():
                    weight = handle.get_tensor("text_projection.weight")
        if weight is not None:
            try:
                torch.save(weight, cache)
            except OSError:
                # 只读目录下不缓存，不影响功能。
                pass
        return weight

    def to(self, device: torch.device | str) -> "CLIPTextEncoder":
        """把冻结编码器和投影矩阵搬到目标设备。"""
        self.device = torch.device(device)
        self.model.to(self.device)
        if self.proj_weight is not None:
            self.proj_weight = self.proj_weight.to(self.device)
        return self

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """编码一组提示，返回 ``(B, 512)`` 的 float32 文本特征。"""
        if not texts:
            return torch.zeros(0, self.hidden_size, device=self.device)
        tokens = self.tokenizer(texts, **self._tokenizer_kwargs).to(self.device)
        # CLIP 对 fp16 数值较敏感；这段前向固定在 fp32 下执行。
        with torch.autocast(device_type=self.device.type, enabled=False):
            out = self.model(**tokens).pooler_output
            if self.proj_weight is not None:
                out = torch.nn.functional.linear(out, self.proj_weight)
        return out.float()

    def train(self, mode: bool = True) -> "CLIPTextEncoder":  # pragma: no cover
        """保持冻结编码器处于 eval，避免误切到训练模式。"""
        self.model.train(mode)
        return self


class PromptEmbeddingCache:
    """提示向量的 LRU 缓存。

    同一批内条件组合高度重复，跨 epoch 也会反复出现；缓存可消除绝大多数
    重复 CLIP 前向。向量存 CPU，给检测器留出显存。
    """

    def __init__(self, encoder: CLIPTextEncoder, max_size: int = 8192) -> None:
        self.encoder = encoder
        self.max_size = int(max_size)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        """清空缓存（设备迁移或显存紧张时使用）。"""
        self._cache.clear()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """批量编码；未命中的提示才调用 CLIP。"""
        unique = list(dict.fromkeys(texts))
        missing = [t for t in unique if t not in self._cache]
        if missing:
            emb = self.encoder.encode(missing).detach().to("cpu", dtype=torch.float32)
            for text, vec in zip(missing, emb):
                self._cache[text] = vec
                self._cache.move_to_end(text)
                while len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)
        stacked = torch.stack([self._cache[t] for t in texts], dim=0)
        return stacked.to(self.encoder.device, non_blocking=True)

    def encode_blocks(self, table: ConditionTable, ids: torch.Tensor) -> torch.Tensor:
        """编码每个样本的六个单属性块，返回 ``(B, N, 512)``。

        仅供 SCPL 的 straight-through 梯度路径使用：前向仍是硬掩码提示，
        反向通过 ``Σ p_n · E(block_n)`` 把梯度送到门控网络。
        """
        ids_cpu = ids.detach().to("cpu").tolist()
        n_attrs = len(table.specs)
        texts: list[str] = [
            table.block_prompt(attr, int(row[attr]))
            for row in ids_cpu
            for attr in range(n_attrs)
        ]
        return self.encode(texts).view(len(ids_cpu), n_attrs, -1)
