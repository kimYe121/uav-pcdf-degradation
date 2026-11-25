# ATR-UMOD RGB-IR 旋转目标检测：一个围绕 PCDF 的复现项目

本项目在 ATR-UMOD（UAV RGB-IR、旋转框、六类成像条件标注）上实现一套双流旋转框
检测管线：`baseline`（双流 concat）→ `stage 1`（SCPL + CDF）→ `stage 2`（PCD 全管线），
对应 ICCV 2025 论文 *Fusion Meets Diverse Conditions* 的 PCDF 方法。

这是一个**个人复现项目**，不是官方代码，也不是逐项对照论文的 benchmark 复现：
论文未公开检测器结构、neck、条件码本等细节，部分模块由本项目按论文描述补齐。
项目目标是把方法跑通、把每个模块的作用量清楚，并在文档中如实记录哪些结论成立、
哪些还没有复现出来。

> 当前状态：**baseline 的 test mAP50 为 59.43，高于论文报告的 58.4**；
> PCDF 三模块的净增益尚未复现出来，目前最大的嫌疑是条件引导没有真正进入融合模块。
> 证据与下一步见 [docs/AUDIT.md](docs/AUDIT.md)，开发过程中的经验见
> [docs/LEARNING.md](docs/LEARNING.md)。

## 结果（ATR-UMOD test，1503 张，mAP50）

| 模型 | 输入几何 | test mAP50 | 说明 |
|---|---|---:|---|
| 论文 baseline | 640×512 | 58.4 | ICCV 2025 论文报告值 |
| 本仓库 baseline（旧几何） | 内容 0.8× + 灰边 | 56.10 | 早期实现，被几何 bug 拖累 |
| **本仓库 baseline（修正后）** | **原生 640×512** | **59.43** | 在旧权重上原生几何热启动 15 轮 |
| stage 1（SCPL+CDF，检测器冻结） | 原生 640×512 | 56.75 | 中间态；结构性掉点，见笔记 §9 |
| stage 2（旧协议，0.8 几何） | 内容 0.8× + 灰边 | 56.49 | 引导协议错误，已被新实现替换 |
| PCDF（新协议） | 原生 640×512 | 待服务器复跑 | 文本引导 + `F_v`，本地只验证到代码路径 |

逐类结果与评估 JSON 在 [docs/results/](docs/results/)。

## 三个主要发现

1. **输入几何比调参重要。** 修好 `RandomPerspective` 的 `(W,H)` 传参后，
   同一个 baseline 从 56.10 涨到 59.43（+3.33），中位目标从 32.8 px 回到 41.0 px。
   根因是 `LetterBox` 把 640×512 的内容缩到 512×410 再补 115 px 灰边。
2. **条件引导可能"接上了但没流过去"。** 打乱 stage2 的引导向量只掉 0.21 个点；
   条件间权重差 0.04；文本/视觉引导逐样本余弦 ≈ 0。CDF 在学逐图权重，但没在学条件。
3. **冻结范围决定谁在学。** stage1 放开检测器时门控 10 轮不动（kept 6.0）；
   冻结检测器后门控开始分化（kept 1.74）。论文只写了"只训 SCPL 和 CDF"，
   我们用实验裁决。

完整的问题—原因—解决—教训见 **[docs/LEARNING.md](docs/LEARNING.md)**。

## 架构图

- baseline（双流 concat）：[PNG](docs/assets/figures/arch_baseline.png) · [PDF](docs/assets/figures/arch_baseline.pdf)
- PCDF（SCPL + PCD + CDF）：[PNG](docs/assets/figures/arch_pcdf.png) · [PDF](docs/assets/figures/arch_pcdf.pdf)
- 项目总览：[PNG](docs/assets/figures/arch_project.png)

## 仓库结构

```text
src/pcdf/
  data.py           配对 RGB/IR 数据集、几何增强、条件 id
  conditions.py     六属性码本、提示构造、B.2 合并规则
  models/           baseline / PCDF / SCPL / PCD / CDF / OBB head
  losses.py         CMD (L_dt)、L_irr、辅助检测 L_dc
  train.py          三阶段训练循环、验证、检查点
  evaluate.py       probIoU + 旋转 NMS + 101 点 AP
  cli/              python -m pcdf.cli.{train,evaluate,visualize,...}
configs/            pcdf.yaml + conditions.yaml + profiles
scripts/            train_native_chain.sh（推荐入口）、部署/维护脚本
tests/              单元测试（含输入几何回归）
docs/               ARCHITECTURE / WORKFLOW / AUDIT / LEARNING / results / assets
```

## 快速开始

```bash
pip install -e .
python -m pcdf.cli.check_env          # 依赖、权重、数据自检
```

数据按 `data/atr_umod/{rgb,ir}` 组织，标签在 `ir/labels` 下（多模态检测器用 IR 标签）。
准备脚本：`python scripts/prepare_atr.py --help`。

训练（原生几何三阶段，推荐入口）：

```bash
# 服务器 4090（b16）
bash scripts/train_native_chain.sh

# 本地 8 GB 卡
BATCH=8 ACCUM=2 bash scripts/train_native_chain.sh
```

评估（对外只报 test）：

```bash
python -m pcdf.cli.evaluate --checkpoint runs/pcdf_s2_native/best.pt --split test
python -m pcdf.cli.evaluate --checkpoint runs/pcdf_s2_native/best.pt --split test \
  --guidance text --fuse decoupled          # stage2 文本引导的诊断口径
```

## 训练协议

| 项 | 值 |
|---|---|
| 输入 | 640×512（W×H），原生、无灰边 |
| 优化器 | SGD，lr 0.01，mom 0.937，wd 5e-4，指数衰减 |
| batch | 16（8 GB 卡用 8 + 梯度累积 2） |
| epochs | baseline 50 + stage1 50 + stage2 50 |
| 文本编码器 | CLIP ViT-B/16，冻结 |
| `τ` / `λ` | 0.15；0.01 / 0.003 / 0.01（论文值） |

## 已知局限

- **PCDF 的增益尚未证实**：stage2 新协议只跑过代码冒烟，完整 50 轮需要在
  24 GB 以上显卡上复跑；本地 8 GB 跑不动。
- **CDF 的条件敏感性弱**：投影是三路输入（引导 + 两模态池化），条件被内容稀释；
  `L_dt` 只对齐 batch 边缘矩，不配对样本。下一步与备选方案写在 AUDIT。
- **条件码本部分匿名**：天气/光照/场景的官方码本未公开，按补充材料合并规则推断。
- 未做 mosaic/HSV（A/B 证明对小目标是负收益）；未用 TTA、模型集成。

## 文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：论文与代码逐条对应、张量形状、保真边界
- [docs/AUDIT.md](docs/AUDIT.md)：stage2 增益不足的审计与实验计划
- [docs/LEARNING.md](docs/LEARNING.md)：这一周踩过的坑与经验
- [docs/WORKFLOW.md](docs/WORKFLOW.md)：本地/远程训练与排障
- [REPRODUCTION.md](REPRODUCTION.md)：复现步骤与硬件实测

## 引用

```bibtex
@inproceedings{chen2025pcdf,
  title={Fusion Meets Diverse Conditions: A High-diversity Benchmark and Baseline
         for UAV-based Multimodal Object Detection with Condition Cues},
  author={Chen, Chen and Bin, Kangcheng and Hu, Ting and Qi, Jiahao and Liu, Xingyue
          and Liu, Tianpeng and Liu, Zhen and Liu, Yongxiang and Zhong, Ping},
  booktitle={ICCV},
  year={2025}
}
```

## License

MIT，见 [LICENSE](LICENSE)。