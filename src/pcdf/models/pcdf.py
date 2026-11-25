"""PCDF 完整网络：双流编码器 -> PCD -> CDF -> OBB head。

论文对应关系
------------
Sec. 4.1 / Fig. 4 : RGB/IR 双流编码器、三支解耦、软门控
式 1-4            : :class:`pcdf.models.scpl.SCPL`
式 5              : ``S`` / ``V_rgb`` / ``V_ir``（:class:`DecoupleBranch`）
式 6-9            : :mod:`pcdf.losses`
式 10-11          : :class:`pcdf.models.blocks.ChannelNormSoftGate`
检测头            : ultralytics OBB（anchor-free + DFL，式 8，见 :mod:`pcdf.models.heads`）

三阶段协议（正文 Sec. 5.2 + 补充材料 B.3）
--------------------------
阶段 0：baseline 双流 concat（:mod:`pcdf.models.baseline`），阶段 1 的权重起点。
阶段 1：从 baseline 初始化并冻结 encoder/neck/head，只训 SCPL + CDF；
        CDF 权重作用于 encoder 原始特征（Fig. 10 Stage I 尚无 PCD）。
阶段 2：冻结 SCPL 与 text adapter，训练 full pipeline（视觉引导 ``F_s``）；
        ``L_dt`` 把 ``F_s`` 对齐到阶段 1 的文本引导空间，CDF 作用于 ``F_v``。
测试：只走 ``F_s``，不读取条件标注，也不调用 CLIP。

关于计算量
----------
阶段 1 的前向不经过 PCD（Fig. 10 Stage I），因此三支解耦网络在该阶段既不
产生梯度、也不影响输出。本实现按需计算：只有视觉引导、或需要 ``F_t``/辅助
检测时才构建 ``F_v``/``F_s``。这省掉约三分之一的阶段 1 前向开销，且不改变
任何数值——被跳过的张量原本就是死代码。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..conditions import ConditionTable
from ..config import ModelConfig
from ..text_encoder import CLIPTextEncoder
from .blocks import ChannelNormSoftGate, DecoupleBranch, GuidanceAdapter, PANNeck
from .encoder import Backbone

from .heads import attach_loss_context, build_head, initialise
from .scpl import SCPL


class PCDFNet(nn.Module):
    """SCPL + PCD + CDF + OBB head 的完整检测器。"""

    arch = "pcdf"

    def __init__(
        self,
        cfg: ModelConfig,
        nc: int,
        table: ConditionTable,
        text_encoder: CLIPTextEncoder | None = None,
    ) -> None:
        super().__init__()
        # 阶段 2 的 GAP(F_s) 直接作为 CDF 引导，维度必须等于 P5 通道。
        if tuple(cfg.widths)[2] != cfg.guidance_dim:
            raise ValueError(
                "guidance_dim 必须等于 P5 通道数，才能让阶段 2 的 F_s 直接驱动 CDF；"
                f"当前 guidance_dim={cfg.guidance_dim}，P5={tuple(cfg.widths)[2]}"
            )
        self.cfg = cfg
        self.nc = int(nc)
        self.widths = tuple(int(c) for c in cfg.widths)
        # 阶段 1 冻结的主干模块（见 freeze_detector）；train() 覆写保证它们
        # 始终处于 eval 模式，避免 BatchNorm 统计量被前向悄悄更新。
        self._frozen_modules: list[nn.Module] = []

        # 双流编码器：两个分支独立，均从同一份预训练 backbone 初始化。
        weights = cfg.encoder_weights if cfg.pretrained_backbone else None
        self.encoder_rgb = Backbone(weights=weights, nc=nc, yaml=cfg.backbone_yaml, scale=cfg.backbone_scale)
        self.encoder_ir = Backbone.from_backbone(self.encoder_rgb)
        if tuple(self.encoder_rgb.out_channels) != self.widths:
            raise RuntimeError(
                f"配置 widths={self.widths} 与 backbone 输出 "
                f"{self.encoder_rgb.out_channels} 不一致；请检查 profile"
            )

        # 三支解耦：V_rgb / V_ir 保持通道；S 的输入是两模态拼接，故为 2C。
        self.v_rgb = nn.ModuleList([DecoupleBranch(c) for c in self.widths])
        self.v_ir = nn.ModuleList([DecoupleBranch(c) for c in self.widths])
        self.s = nn.ModuleList([DecoupleBranch(c, in_channels=2 * c) for c in self.widths])

        if text_encoder is None:
            text_encoder = CLIPTextEncoder(cfg.text_encoder, device="cpu")
        # CLIP 不是 nn.Module，不会被注册为子模块，也不会进 state_dict。
        self.text_encoder = text_encoder
        self.scpl = SCPL(
            table=table,
            text_encoder=text_encoder,
            feat_dim=self.widths[-1],
            text_dim=text_encoder.hidden_size,
            hidden=cfg.gate_hidden,
            tau=cfg.tau,
            pool=cfg.gate_pool,
            cache_size=cfg.text_cache_size,
        )
        # 文本嵌入 -> 引导空间；阶段 2 会冻结它作为 L_dt 的稳定目标。
        self.text_adapter = GuidanceAdapter(text_encoder.hidden_size, cfg.guidance_dim)

        self.cdf = ChannelNormSoftGate(cfg.guidance_dim, self.widths)
        # Fig. 4 的字面接线是 concat 直连检测头（图例：F = Channel Concatenation）；
        # v5 系检测器需要颈部，两档都保留，由 model.neck 决定。
        self.neck = PANNeck(self.widths) if cfg.neck == "pan" else None
        # 共享 OBB head：主融合特征与两个 F_v 辅助分支都使用同一组参数。
        self.head = build_head(cfg, self.nc)
        initialise(self)
        attach_loss_context(self)
        if cfg.freeze_text:
            self.freeze_text()

    # ------------------------------------------------------------------ 配置
    @property
    def stride(self) -> Tensor:
        return self.head.stride

    def move_text_encoder(self, device: torch.device | str) -> None:
        """CLIP 不是 nn.Module 子模块，需要显式搬设备。"""
        self.text_encoder.to(device)

    def freeze_text(self) -> None:
        """冻结 CLIP 文本编码器（论文 Sec. 5.1）。"""
        for param in self.text_encoder.model.parameters():
            param.requires_grad_(False)
        self.text_encoder.model.eval()

    def freeze_scpl(self) -> None:
        """阶段 2 冻结 SCPL 门控（论文 B.3），并关闭 straight-through 软梯度。

        软梯度只用于训练 ``Fc``；门控冻结后它只剩计算开销（每批 6N 次单属性
        提示编码），因此一并关闭。
        """
        for param in self.scpl.parameters():
            param.requires_grad_(False)
        self.scpl.soft_gradient = False

    def freeze_text_adapter(self) -> None:
        """固定阶段 1 的文本引导空间，作为 L_dt 的稳定目标。"""
        for param in self.text_adapter.parameters():
            param.requires_grad_(False)

    def freeze_detector(self, *, keep_detector_trainable: bool = False) -> None:
        """阶段 1 的冻结范围（补充材料 B.3）。

        默认冻结双流 encoder、neck、head 与三支解耦分支——对应论文
        "training only the SCPL and CDF modules"。``keep_detector_trainable=True``
        时只冻结三支解耦（阶段 1 不使用 PCD，避免 weight decay 污染随机初始化），
        encoder/neck/head 参与训练——对应"论文只显式冻结了 CLIP 文本编码器"
        这一读法。
        """
        modules = [self.v_rgb, self.v_ir, self.s]
        if not keep_detector_trainable:
            modules += [self.encoder_rgb, self.encoder_ir, self.neck, self.head]
        for module in modules:
            for param in module.parameters():
                param.requires_grad_(False)
            module.eval()
        self._frozen_modules = list(modules)
        # 头被冻的是参数，不是模式：eval 下的 OBB 只返回解码结果，没有 loss
        # 需要的 dict，阶段 1 就再也算不出检测损失了。
        self.head.train(True)

    def train(self, mode: bool = True) -> "PCDFNet":
        """保持被冻结模块始终处于 eval 模式。

        训练循环每次 epoch 都会调用 ``model.train()``，若不覆写，冻结的
        encoder 里的 BatchNorm 会继续更新 running stats，等效于泄训。

        检测头是例外：它的模式不是"学不学"，而是"返回什么"——train 返回
        ``v8OBBLoss`` 需要的 dict，eval 返回可直接进 NMS 的张量。因此头的
        模式始终跟随本次调用的 ``mode``，即使参数已被冻结（阶段 1）。
        """
        super().train(mode)
        for module in self._frozen_modules:
            module.eval()
        self.head.train(mode)
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_summary(self) -> dict[str, int]:
        """按模块统计参数量；CLIP 单独列出，不计入可训练参数。"""

        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "total": count(self),
            "encoder_rgb": count(self.encoder_rgb),
            "encoder_ir": count(self.encoder_ir),
            "decoupling": count(self.v_rgb) + count(self.v_ir) + count(self.s),
            "scpl_gate": count(self.scpl.gate),
            "text_adapter": count(self.text_adapter),
            "cdf": count(self.cdf),
            "neck": count(self.neck),
            "head": count(self.head),
            "clip_frozen": count(self.text_encoder.model),
        }

    # ---------------------------------------------------------------- 前向
    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,
        guidance: str = "visual",
        return_aux: bool = False,
        return_ft: bool = False,
        force_all_gates: bool = False,
        fuse: str | None = None,
    ) -> dict[str, object]:
        """单次前向。

        Args:
            x: ``(B, 6, H, W)``，通道 0-2 为 BGR，3-5 为 IR。
            cond: ``(B, N)`` 条件 id；使用 SCPL 时必填。
            guidance: 引导来源：``"text"``（SCPL 文本嵌入）或 ``"visual"``
                （``GAP(F_s)``，测试部署路径）。
            fuse: CDF 的融合对象：``"raw"``（encoder 原始特征，阶段 1）或
                ``"decoupled"``（``F_v``，阶段 2/测试）。默认沿用旧行为
                （text→raw、visual→decoupled），显式传入则解耦"引导来源"
                与"融合对象"——阶段 2 训练按 Fig. 10 用文本引导 + ``F_v``。
            return_aux: 是否额外用共享 head 处理两个 ``F_v``，供 ``L_dc``。
            return_ft: 是否计算并返回 ``F_t``，供阶段 2 的 ``L_dt``。
            force_all_gates: 门控全部打开（诊断用，绕过 ``tau``）。

        Returns:
            含 ``main`` / ``guidance`` / ``cdf_ir_weights`` 等键的字典。
            ``fs_gap``、``fs``、``fv_rgb``、``fv_ir``、``aux_*`` 在不需要 PCD
            的阶段（文本引导且不返回辅助输出）为 ``None``。
        """
        if x.shape[1] != 6:
            raise ValueError(f"期望 6 通道 RGB+IR 输入，实际为 {tuple(x.shape)}")
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)  # DataLoader 输出 uint8，模型内部归一化
        rgb, ir = x[:, :3], x[:, 3:]

        # 双流编码：每层输出 P3/P4/P5。
        feats_rgb = self.encoder_rgb(rgb)
        feats_ir = self.encoder_ir(ir)

        # 融合对象：默认 text→raw（阶段 1）、visual→decoupled（阶段 2/测试）。
        if fuse is None:
            fuse = "raw" if guidance == "text" else "decoupled"
        if fuse not in ("raw", "decoupled"):
            raise ValueError(f"未知融合对象 {fuse!r}")

        # 阶段 1 的前向不需要 PCD（Fig. 10 Stage I），按需计算避免无谓开销。
        need_pcd = fuse == "decoupled" or guidance == "visual" or return_aux or return_ft
        fv_rgb = fv_ir = fs = None
        if need_pcd:
            # 三支解耦（式 5）。
            fv_rgb = [branch(f) for branch, f in zip(self.v_rgb, feats_rgb)]
            fv_ir = [branch(f) for branch, f in zip(self.v_ir, feats_ir)]
            fs = [
                branch(torch.cat([a, b], dim=1))
                for branch, a, b in zip(self.s, feats_rgb, feats_ir)
            ]

        # SCPL 仅在文本引导或需要 F_t 时执行。
        scpl_out = None
        if guidance == "text" or return_ft:
            if cond is None:
                raise ValueError("使用 SCPL 时必须提供条件 id")
            scpl_out = self.scpl(
                feats_rgb[-1], feats_ir[-1], cond, force_all=force_all_gates
            )

        # 条件引导：阶段 1 用文本，阶段 2/测试用池化后的 F_s。
        fs_gap = F.adaptive_avg_pool2d(fs[-1], 1).flatten(1) if fs is not None else None
        if guidance == "text":
            assert scpl_out is not None
            guide = self.text_adapter(scpl_out["ft_ste"])
        elif guidance == "visual":
            assert fs_gap is not None
            guide = fs_gap
        else:
            raise ValueError(f"未知引导模式 {guidance!r}")

        # CDF（式 10-11）与检测。论文 Fig. 10：阶段 1（文本引导）还没有 PCD，
        # CDF 权重直接作用于 encoder 原始特征；阶段 2/测试作用于条件不变
        # 分支 F_v（式 11 明确"仅对条件不变特征进行权重重分配"）。
        if fuse == "raw":
            fuse_rgb, fuse_ir = feats_rgb, feats_ir
        else:
            assert fv_rgb is not None and fv_ir is not None
            fuse_rgb, fuse_ir = fv_rgb, fv_ir
        fused, weights, weighted_rgb, weighted_ir = self.cdf(guide, fuse_rgb, fuse_ir)
        features = self.neck(fused) if self.neck is not None else fused
        head_out = self.head(features)
        if isinstance(head_out, dict):          # 训练态：v8OBBLoss 要的 dict
            main: object = head_out
        else:                                   # 推理态：(decoded, preds)
            main = (head_out[0], None)

        aux_rgb = aux_ir = None
        if return_aux:
            # 式 8 + Fig. 4：三个 Detection 都接在 **Reassigned** 特征上
            # （W_m ⊙ F_v,m），不是重加权之前的 F_v。这样辅助头的判别能力
            # 与主融合头看到的是同一批特征。
            aux_rgb = self.head(weighted_rgb)
            aux_ir = self.head(weighted_ir)

        out: dict[str, object] = {
            "main": main,
            "aux_rgb": aux_rgb,
            "aux_ir": aux_ir,
            "guidance": guide,
            "fs_gap": fs_gap,
            "fs": fs,
            "fv_rgb": fv_rgb,
            "fv_ir": fv_ir,
            "cdf_ir_weights": weights,
        }
        if scpl_out is not None:
            out.update(
                {
                    "ft": scpl_out["ft"],
                    "gate_probs": scpl_out["gate_probs"],
                    "gate_mask": scpl_out["gate_mask"],
                    "kept": scpl_out["kept"],
                }
            )
        return out
