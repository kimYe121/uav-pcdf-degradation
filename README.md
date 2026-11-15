<p align="center">
  <a href="#chinese">简体中文</a> &nbsp;·&nbsp; <a href="#english">English</a>
</p>

<a name="chinese"></a>

# uav-pcdf-degradation

**PCDF 核心机制复现与无人机 RGB-IR 旋转目标检测的部署退化研究。**

基于 [ATR-UMOD / PCDF（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.html)，在统一、公开的检测骨架上研究：条件感知动态融合面对部署期单路退化、缺失和跨模态错位时，是否仍然可靠。

> **当前状态：研究策略与数据管线阶段。PCDF 尚未实现或复现成功；没有真实数据上的负融合阈值结论。**

## 研究主线

1. **方法级复现**：实现 PCDF 的样本特定条件提示学习（SCPL）、提示引导条件解耦（PCD）和条件感知动态融合（CDF）。
2. **公平对照**：在同一骨架、数据划分和训练预算下比较 RGB-only、IR-only、双流 concat、普通动态门控、条件引导融合与 PCDF 式完整模型。
3. **部署扩展**：冻结训练好的模型，只对推理输入施加 P1 连续污染、P2 单路缺失、P3 配准漂移，评价条件门控的失效与改进。
4. **改进验证**：检验当前模态状态观测与连续回退能否减少退化下的性能损失；不预设改进一定有效。

复现边界、公开公式、未公开实现细节和验收顺序见 **[REPRODUCTION.md](REPRODUCTION.md)**。

## 不做什么

- 不把六通道 early fusion、B 通道替换或合成数据结果称为 PCDF。
- 不把不同数据子集、骨架和划分下的结果与论文的 63.1 mAP50 直接比较。
- 不预设所有退化都存在统一、可部署的固定阈值。
- 不宣称首次发现负融合；ATR-UMOD 补充材料已讨论普通融合低于最佳单模态的现象。

## 当前进展

| 部分 | 状态 |
|---|---|
| ATR-UMOD 成对数据转换、场景级划分 | 已实现 |
| 单模态 OBB 基线入口 | 已实现；已有小子集对照，非论文基准 |
| 真实六通道 early fusion | 前期对照；不是 PCDF |
| 双流 concat / CDF / SCPL / PCD | 待实现 |
| PCDF 式完整模型及消融 | 未开始 |
| 部署退化协议与 `NF / RA` 工具 | 原型；待接入有效融合模型 |
| 模态状态观测、连续回退 | 未实现 |
| 推理期失效阈值结论 | **尚未获得** |

历史合成冒烟结果仅用于链路测试，不能作为研究证据。

## 安装与检查

```bash
git clone https://github.com/kimYe121/uav-pcdf-degradation.git
cd uav-pcdf-degradation
pip install -e .
python scripts/check_env.py
```

需要 Python 3.10+。训练需要 CUDA 显卡；合成冒烟测试可在 CPU 上运行。小批量实现验证可使用 8GB 显存，完整对照和消融建议采用 24GB 级显卡；正式租卡时长以本地显存和单 epoch 实测为准。

## 数据

使用 [ATR-UMOD](https://github.com/supercpy/ATR-UMOD)：配准 RGB-IR 图像、11 类目标、OBB 标注和六个成像条件属性。按官方仓库说明下载并解压：

```text
data/ATR-UMOD/
└── train|test/
    ├── images/
    ├── images_ir/
    ├── labels/
    └── labels_ir/
```

转换为 YOLO-OBB 并按“地点＋日期”划分验证集：

```bash
python scripts/prepare_atr.py check   --root data/ATR-UMOD
python scripts/prepare_atr.py convert --root data/ATR-UMOD --out data/atr_umod
```

`check` 会在写文件前报告类别、实例、坐标和条件属性。不可按帧随机切分；正式融合实验使用 IR 标签、官方 test，并从 train 中划分场景隔离的 val。论文复现实验采用 11 类；`--classes` 仅用于前期子集调试，不用于与论文数值比较。

## 实验协议

| 协议 | 推理输入操作 | 不变部分 |
|---|---|---|
| P1 连续污染 | 只退化一路：低照度、噪声、雾或模糊 | 另一路与模型参数 |
| P2 单路缺失 | 一路置零或缺失 | 另一路与模型参数 |
| P3 配准漂移 | 平移一路，破坏跨模态对应关系 | 两路各自图像质量 |

```text
NF(s,c) = mAP_fusion(s,c) - max(mAP_RGB(s,c), mAP_IR(s,c))
RA(s)   = mAP(s) / mAP(clean)
```

`NF < 0` 表示融合弱于最佳单模态。交点 `s*` 仅在确实存在且稳定时作为**条件性经验边界**报告；它不是部署常数。部署时没有真值，不能直接用 `NF` 做在线路由。

## 目录

```text
configs/                 冒烟与前期实验配置
docs/                    公开图示与复现说明
scripts/                 数据准备、检查与实验入口
src/modality_decay/      数据处理、前期融合、退化与评价工具
REPRODUCTION.md          复现边界、模块清单与分阶段验收标准
```

## 参考资料

- [PCDF 正文与补充材料（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.html)
- [ATR-UMOD 官方数据集仓库](https://github.com/supercpy/ATR-UMOD)：截至本说明更新时仍标记 PCDF 代码待发布。
- M²D-LIF、RSC-MD：训练期融合退化与模态梯度失衡。
- MultiCorrupt：推理期 corruption 与相对性能保持率。
- DRNet：模态缺失下的动态路由。

## 许可证

MIT，详见 [LICENSE](LICENSE)。

---

<a name="english"></a>

# uav-pcdf-degradation

**Reproducing core PCDF mechanisms and studying deployment-time degradation in UAV RGB-IR oriented object detection.**

Using [ATR-UMOD / PCDF (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.html), this project asks whether condition-aware dynamic fusion remains reliable when one modality degrades, disappears, or becomes misregistered during deployment.

> **Status: research redesign and data preparation. PCDF is not yet implemented or reproduced. No real-data negative-fusion threshold has been established.**

## Research plan

1. **Mechanism-level reproduction** of Sample-specific Condition Prompt Learning (SCPL), Prompt-guided Condition Decoupling (PCD), and Condition-aware Dynamic Fusion (CDF).
2. **Controlled comparisons** of RGB-only, IR-only, dual-stream concat, data-driven gating, condition-guided fusion, and the PCDF-style full model under one backbone, split, and training budget.
3. **Deployment extension**: freeze the trained model and corrupt only inference inputs using P1 continuous contamination, P2 modality absence, and P3 cross-modal misregistration.
4. **Targeted improvement**: test whether current modality-state observations and continuous fallback reduce degradation losses, without assuming that they must help.

See **[REPRODUCTION.md](REPRODUCTION.md)** for the reproduction boundary, published details, implementation choices, and staged acceptance criteria.

## Non-claims

- Six-channel early fusion, B-channel replacement, and synthetic results are not PCDF.
- Results from different backbones, subsets, and splits are not directly comparable with the paper's 63.1 mAP50.
- A universal deployment threshold is not assumed to exist.
- Negative fusion is not claimed as a new discovery; the official supplementary material discusses fusion below the best unimodal model.

## Status

| Part | State |
|---|---|
| Paired ATR-UMOD conversion and scene-level splits | Implemented |
| Unimodal OBB baseline entry point | Implemented; small-subset checks only |
| True six-channel early fusion | Preliminary baseline; not PCDF |
| Dual-stream concat / CDF / SCPL / PCD | Planned |
| PCDF-style full model and ablations | Not started |
| P1/P2/P3 protocols and `NF / RA` utilities | Prototypes; require a valid fusion model |
| Modality-state observation and continuous fallback | Not implemented |
| Deployment-failure threshold | **Not established** |

Synthetic smoke outputs are wiring checks, not scientific evidence.

## Setup and data

```bash
git clone https://github.com/kimYe121/uav-pcdf-degradation.git
cd uav-pcdf-degradation
pip install -e .
python scripts/check_env.py
python scripts/prepare_atr.py check   --root data/ATR-UMOD
python scripts/prepare_atr.py convert --root data/ATR-UMOD --out data/atr_umod
```

Python 3.10+ is required. Training requires CUDA. Small implementation checks are feasible with 8GB VRAM; full comparisons and ablations are better suited to a 24GB-class GPU. Rental duration should follow measured memory and epoch time.

Use the official train/test split, IR labels for multimodal training, and a scene-isolated validation subset from train. Eleven classes are required for the reproduction experiments; `--classes` is only for preliminary debugging.

## Deployment metrics

```text
NF(s,c) = mAP_fusion(s,c) - max(mAP_RGB(s,c), mAP_IR(s,c))
RA(s)   = mAP(s) / mAP(clean)
```

`NF < 0` denotes fusion below the best unimodal model. A crossing `s*` is reported only when it exists and is stable, as a conditional empirical boundary rather than a deployment constant. Ground-truth `NF` is unavailable for online routing at deployment.

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <a href="#chinese">简体中文</a> &nbsp;·&nbsp; <a href="#english">English</a>
</p>
