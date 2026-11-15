# PCDF 复现边界与实验路线 / PCDF Reproduction Plan

## 中文

### 1. 项目定位

本项目依据 Chen 等人在 ICCV 2025 发表的 PCDF 论文及其官方补充材料，在统一、公开的 OBB 检测骨架上实现核心机制，并研究部署期输入退化。

这是**论文指导的方法级复现与适配**，不是官方代码复现。项目不承诺复原论文的 63.1 mAP50，也不把其他 backbone、数据子集或划分的结果与论文数值直接横比。

### 2. 原文明确公开的内容

- 数据：ATR-UMOD，RGB-IR 成对图像，11 类，IR 标签用于多模态训练，官方 train/test 场景不重叠。
- 条件：高度、角度、时间、天气、光照、场景。
- 提示模板：

  ```text
  An aerial image, over {scenario}, at a altitude of {altitude}, at an angle of {angle}, at {Time}, in {Weather}, in {Illumination}.
  ```

- 文本编码器：冻结的 CLIP ViT-B/16。
- SCPL：按样本选择条件块，原文阈值 `tau=0.15`；只用于训练。
- PCD：条件相关编码器 `S`，两路条件无关编码器 `V_rgb / V_ir`，以及 `Ldt / Lirr / Ldc` 三种损失。
- 损失权重：`lambda1=0.01, lambda2=0.003, lambda3=0.01`。
- CDF：条件特征生成逐通道归一化 `W_rgb / W_ir`，加权两路条件无关特征后 concat。
- 训练：单张 RTX 4090，SGD，初始学习率 0.01，momentum 0.937，weight decay 0.0005，640×512，batch 16，两个阶段各 50 epochs。
- 阶段一训练 SCPL+CDF；阶段二在阶段一权重上训练 PCD，以移除实际测试对人工条件标注的依赖。
- 补充材料的条件合并规则：高度 120m、角度 30° 分界；时间分为 Morning/Afternoon/Night；After Rain 和 Rainy 合并进 Cloudy；Overexposure 和 Twilight 分别合并进 Normal 和 Dim。

### 3. 正文与补充材料合读后仍未明确的内容

- 原 PCDF 的 backbone、neck、OBB head 及预训练配置；
- 融合位置、特征尺度与各层通道数；
- `Fc / S / V_rgb / V_ir / Ft` 的具体层结构；
- 两阶段完整冻结列表和优化器分组；
- SCPL 硬指示门控的梯度传播方式；
- 总损失组合、部分增强和模型选择细节；
- 参数量、FLOPs、训练时长及显存占用。

因此，不能把 PCDF 简化为“YOLO 上加一个 soft-gating”，也不能默认原文使用 YOLOv5s、YOLOrs 或 YOLO11n-OBB。

### 4. 本项目的实现选择

任何自行补齐的结构都必须在实验配置和报告中标记为 `implementation choice`，包括：

1. 公开 OBB backbone 与三个检测头（RGB 辅助、IR 辅助、融合）；
2. 特征投影、池化、维度匹配与 CMD 的数值归一化；
3. 硬门控的可微处理，例如训练软门控、测试硬门控，或 STE；
4. 文本向量缓存，避免重复运行冻结 CLIP；
5. 小 batch、AMP 与梯度累积的等效训练策略。

先做维度、损失、梯度和数值单元测试，再做小数据过拟合。不得通过隐藏这些差异宣称精确复现。

### 5. 分阶段验收

| 阶段 | 实验 | 进入下一阶段的条件 |
|---|---|---|
| A | RGB-only、IR-only、双流 concat | 训练稳定；明确 clean 对照 |
| B | 普通数据驱动门控、条件引导 CDF | 门控随输入变化；公平比较其相对 concat 的贡献 |
| C | SCPL、PCD 及损失消融 | 条件预测/解耦可学习，结果可解释 |
| D | 冻结有效融合模型，执行 P1/P2/P3 | 得到 NF、RA、分类别和分条件曲线 |
| E | 状态观测与连续回退 | 与原始门控、直接单模态回退和 oracle 对照 |

若 concat 未优于最佳单模态，先修检测骨架和训练配置，不继续堆叠 PCDF 模块，也不把后续退化结果解释为有效融合的失效。

`NF` 与 `RA` 是离线诊断指标，需要真值；部署时只能使用 `q_self`、`q_cross`、预测一致性和稳定性等可观测量。`s*` 若存在，只表示特定模型、退化、场景与指标下的经验交点，允许“无交点”或“多交点”。

### 6. 算力策略

- 本地 8GB：模块实现、单元测试、小数据过拟合、batch/显存/单 epoch 实测。
- 24GB 级 GPU：全量 11 类、统一骨架的公平对照、关键消融和最终多 seed 实验。
- 4090 24GB 是与原文硬件接近的选择，但不保证复现其数值。
- 未实测前不承诺显存、单轮时间或租卡时长。

### 7. 资料

- [论文官方页面（正文、补充材料）](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.html)
- [arXiv](https://arxiv.org/abs/2510.13620)
- [数据集官方仓库](https://github.com/supercpy/ATR-UMOD)

---

## English

### Scope

This is a **paper-guided, mechanism-level reproduction and adaptation**, not an official-code reproduction. It implements SCPL, PCD, and CDF on a disclosed OBB backbone and evaluates their behavior under deployment-time input degradation. It does not promise to recover the paper's 63.1 mAP50 or compare incompatible experimental setups.

The paper and supplementary material disclose the condition prompt template, frozen CLIP ViT-B/16 text encoder, condition merging, CDF equations, PCD losses and weights, and a two-stage schedule of 50 epochs each at 640×512 with batch 16 on one RTX 4090. Stage one trains SCPL+CDF; stage two trains PCD from that initialization to remove reliance on manual condition annotations at deployment.

Even after reading both documents, the exact detector backbone/head, fusion scales, submodule layer definitions, complete freezing schedule, hard-gate gradient treatment, and compute costs remain unspecified. Every replacement is recorded as an `implementation choice`, not presented as an original detail.

### Acceptance order

1. Establish RGB-only, IR-only, and dual-stream concat baselines under one protocol.
2. Compare data-driven and condition-guided CDF against concat with dynamic gates checked for input dependence.
3. Add SCPL/PCD and perform loss ablations with explicit gradient and numerical tests.
4. Freeze a valid fusion model and evaluate P1 contamination, P2 modality absence, and P3 misregistration using `NF`, `RA`, per-class AP, and per-condition results.
5. Test state observations and continuous fallback against the original gate, direct unimodal fallback, and an oracle.

If concat does not outperform the best unimodal model, fix the detector first. Do not interpret later degradation as a failure of useful fusion. `NF` requires ground truth and is not an online routing signal; `s*` is an optional conditional crossing, not a universal threshold.

Use a local 8GB GPU for implementation and profiling, and a 24GB-class GPU for full comparisons and ablations. No memory, epoch-time, or reproduction guarantee is made before measurement.
