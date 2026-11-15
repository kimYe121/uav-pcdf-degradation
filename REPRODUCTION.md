# PCDF 复现、条件组合泛化与解耦审计 / PCDF Reproduction and Research Protocol

## 中文

### 1. 研究问题与边界

本项目依据 Chen 等人在 ICCV 2025 发表的 PCDF 论文及官方补充材料，在公开的 OBB 检测骨架上实现 SCPL、PCD、CDF，并研究：

> **各属性边际取值已见、但其组合未见过时，PCDF 能否保持检测性能与合理的条件行为？PCD 的条件不变特征是否仍存在可预测的条件信息？**

这是**论文指导的方法级复现与研究性适配**，不是官方代码复现。项目不承诺复原论文的 63.1 mAP50，也不把不同 backbone、子集或划分的结果与论文数值直接横比。仓库名中的 `degradation` 是历史名称，不再代表当前研究假设。

### 2. 论文证据如何导向研究问题

| 原文证据 | 可以得出的结论 | 不能得出的结论 |
|---|---|---|
| 正文 Table 3 按单一属性报告 PCDF 与其他方法结果 | 已评估各属性取值的边际表现 | 已验证未见属性组合泛化 |
| 补充 B.2 为缓解条件稀疏而合并条件，正文指向 Table 3 | 分组统计使用粗粒度条件，存在样本稀疏问题 | 所有训练提示必然采用相同合并标签 |
| 正文 Foggy 为 PCDF 32.6、YOLOrs 33.2，并提及雾浓度和能见度不一致 | 粗条件可能无法覆盖同一类别内部差异 | 细粒度标签或连续条件一定能提高性能 |
| 正文 Table 4 的 PCD/损失消融与补充 F 节特征图 | 模块对检测有贡献，存在定性机制证据 | 条件不变特征已被证明无条件泄漏 |
| 补充 D 节 DroneVehicle 只补标 day/night | 方法可迁移到较少条件属性的数据集 | 完整六类条件体系已实现跨域组合泛化 |

正文 Table 4 中完整 PCDF 为 63.1，baseline 为 58.4；`w/o PCD` 为 62.1，即表格差值为 1.0 个百分点，正文相应段落的 1.2 与表格不一致，应保留说明而不是任选其一。

### 3. 相近工作及排除的重复方向

- **CAFuser（RA-L 2025）**：比较粗提示与详细提示，并分析测试场景中的动态模态权重；没有将天气、昼夜或其组合从训练中留出。其“未见场景”不能等同于“未见条件组合”。
- **Distribution-Aligned Multimodal Fusion（CVPR 2026）**：训练 overexposure，测试 underexposure、night、blur 等未见退化，重点是融合特征向正常预训练分布对齐；不研究语义条件组合，也不审计 PCD。
- **D2E2-Det（TGRS 2026）**：跨模态共享/私有特征正交解耦和光照门控，不是多属性条件特定/条件不变解耦。
- **模态可靠性建模（中国图象图形学报 2026）**：训练期 VLM 属性蒸馏及全局—局部可靠性，未报告未见条件组合或泄漏探针。
- **RASM-Net、QA2FDet**：已研究局部可靠性、空间对应、小目标或错位，不再把“PCDF 全局门控改为局部门控”直接称为研究空白。
- 固定退化阈值、模态状态观测、连续回退和普通动态门控比较已归档，不属于本路线的预设贡献。

本次文献核查是针对性检索，不是系统综述；“尚未发现覆盖该交叉问题的论文”不等于数学意义上的绝无相关工作，实施前仍需复查最新文献。

### 4. 原文公开内容与复现缺口

公开内容包括：11 类 ATR-UMOD、六类条件属性、IR 标签、冻结 CLIP ViT-B/16、完整提示模板、SCPL `tau=0.15`、PCD 三种损失及权重 `0.01/0.003/0.01`、逐通道 CDF、两阶段各 50 epochs、640×512、batch 16、SGD 及单张 RTX 4090。

仍未明确：原检测 backbone/neck/OBB head、融合尺度、`Fc/S/V_rgb/V_ir/Ft` 层结构、完整冻结列表、硬门控梯度处理、部分总损失/增强/模型选择细节，以及显存、FLOPs、FPS 和训练时间。

自行补齐的检测骨架、投影层、池化、CMD 数值归一化、STE 或软门控近似、文本缓存、AMP 和梯度累积均标记为 `implementation choice`。先做维度、损失、梯度和数值单元测试，再做小数据过拟合；不能把自行补齐的结构描述为原论文细节。

### 5. 数据协议：先证明能测，再训练模型

#### 5.1 条件定义

`unseen composition` 指某个选定的属性值元组不在训练集中，但其中每个单属性取值仍出现于训练集。它不是新天气类别的零样本识别，也不是图像合成退化。

本地 13,353 份 IR XML 的初步只读审计：

- train：11,850 张，326 种“高度×角度×天气×光照×场景”组合；
- test：1,503 张，45 种同类组合；
- train 高度×角度有 54 对，其中 34 对样本数至少 20；test 有 21 对，其中 15 对至少 20；
- 这些计数只是可行性初查，不代表全部满足检测统计功效，也不能据此提前指定唯一留出对。

先选择 1–3 个支撑充分的低维属性对。高度和角度是优先候选，因为它们具有原始数值且不必依赖天气代码语义；天气×光照只在获得官方代码映射、样本量和场景覆盖支持后加入。完整六属性组合不作为起点。

#### 5.2 划分与混杂控制

1. 对完整数据只读统计组合、类别、目标尺寸、采集日期与地点编号；不得为追求阳性结果反复挑选留出组合。
2. 预留最终 test，划分生成与模型选择只使用 train/val；若组合只存在于 test，则它不能作为训练留出实验的有效测试。
3. 从训练池排除目标组合，同时保留每个边际属性值；构造已见组合的 matched-seen 参考集。
4. 按可核验的采集序列或航次分组，隔离相邻帧；不能只按帧随机切分。
5. 匹配或报告场景、类别、尺寸、昼夜与采集日期的分布差异；不能控制时降级为探索性结论。
6. 固定协议后再训练，公开划分清单、排除原因和随机种子。

按“地点编号＋日期”分组时，train 有 68 组、test 有 21 组，交集为 19。地点编号可能跨采集复用，故这不是已证实的图像泄漏，但必须结合原始序列、视觉近重复和时间连续性审查。不能将论文的非重叠场景声明直接替换为这个代理定义。

### 6. 模型对照、候选改进与机制审计

必要对照只服务于研究假设，不再以“重复证明 PCDF 优于 concat”为贡献：

| 模型 | 回答的问题 |
|---|---|
| 双流 concat | 没有显式动态融合时的参考下界 |
| 数据驱动动态门控 | 样本动态性本身能否解释组合表现 |
| PCDF 式完整模型 | 条件提示、PCD、CDF 的组合行为 |
| 属性分解提示候选 | 可组合教师是否比整句联合教师更好地外推 |

属性分解提示拟保持各属性独立语义表示，再通过样本相关的组合函数形成条件教师，而非为每个完整条件组合学习独立嵌入。它与 PCDF 使用同一检测骨架和训练预算，不能加入测试组合标签。其结构和正则化需在最小实验前固定，不预设必然改善。

PCD 机制审计包括：

1. 从冻结模型提取 `Fs`、`Fv_rgb`、`Fv_ir`，在训练场景组上拟合条件探针，在未见场景组上测试；报告 balanced accuracy、macro-F1 和 chance level。
2. 分别测试各单属性，避免六属性联合标签的稀疏探针造成虚假高准确率。
3. 检查条件分支对边际属性的恢复，以及条件不变分支中仍残留的条件可预测性。
4. 用跨场景组 bootstrap 关联泄漏强度与 `CGD`，但只称相关证据，不宣称因果。
5. 同时测量辅助检测性能，避免以丢失全部目标信息换取“无条件泄漏”。

### 7. 指标、验收与退出

主要指标：mAP50、mAP50-95、分类别 AP、最弱组合性能、已见参考与未见组合的绝对差：

```text
CGD = mAP_seen_reference - mAP_unseen
```

同时报告样本量、场景组 bootstrap 置信区间与多 seed 均值/标准差；不把帧数当作完全独立样本量。参数量、FLOPs、显存和时延在实现后实测。

| 阶段 | 验收条件 |
|---|---|
| A. 数据可行性 | 存在有效留出组合；边际属性已见；场景隔离与混杂报告通过 |
| B. 工程复现 | 单元测试、小数据过拟合、条件缓存与梯度检查通过 |
| C. 最小因果对照 | PCDF、无显式条件门控、concat 在固定协议下可重复训练 |
| D. 机制检验 | 条件探针与检测保留指标完整；不只展示特征图 |
| E. 候选改进 | 属性分解提示跨 seed 改善最弱组合/CGD，且已见性能无不可接受损失 |

出现以下任一情况就停止扩展，不继续堆模块：

- 没有统计上可用的留出组合，或混杂无法控制；
- PCDF 与无显式条件门控的差异不超过 seed 波动；
- 条件泄漏不随 PCD 改变，或泄漏与泛化无稳定关系；
- 分解提示只提升已见组合，未改善未见组合或最弱组合。

阴性结果仍是研究结论，但不能包装成方法改进成功。

### 8. 算力策略

本地 8GB 用于模块实现、单元测试、小数据过拟合、探针和显存/单 epoch profiling。完整 11 类、多 seed 与关键消融建议使用 24GB 级 GPU；4090 与原文硬件接近但不保证相同结果。当前没有 PCDF 实测，不承诺显存、训练时长或租卡时长。

### 9. 资料

- [PCDF 正文与补充材料](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.html)
- [ATR-UMOD 官方仓库](https://github.com/supercpy/ATR-UMOD)
- [CAFuser](https://arxiv.org/abs/2410.10791)
- [Distribution-Aligned Multimodal Fusion](https://openaccess.thecvf.com/content/CVPR2026/papers/Hao_Distribution-Aligned_Multimodal_Fusion_for_Robust_Object_Detection_CVPR_2026_paper.pdf)
- [模态可靠性建模的航空遥感可见光—红外目标检测](https://doi.org/10.11834/jig.260197)

---

## English

### Scope

This is a **paper-guided mechanism-level reproduction and research adaptation**, not an official-code reproduction. It asks whether PCDF generalizes to an attribute combination absent from training while every marginal attribute value remains observed, and whether condition-invariant features still contain predictable condition information. It does not promise the paper's 63.1 mAP50 or compare incompatible implementations.

### Evidence and non-duplication

PCDF Table 3 reports marginal per-attribute performance, not held-out compositions. Table 4 and supplementary feature maps support the utility of PCD but do not constitute an independent condition-leakage probe. The condition-merging rule is explicitly tied to Table 3 and is not assumed to govern every training prompt. Foggy remains a reported weakness, but this does not establish that finer or continuous labels will necessarily help.

CAFuser compares coarse and detailed prompts without holding out condition classes or combinations. Distribution-Aligned Multimodal Fusion evaluates unseen synthetic degradations rather than semantic condition compositions. D2E2-Det addresses shared/private modality disentanglement, while recent reliability-modeling work does not report composition holdouts or leakage probes. RASM-Net and QA2FDet already cover local reliability and alignment, so replacing global with local gating is not treated as an open contribution. This was a targeted literature audit, not an exhaustive systematic review.

### Reproduction boundary

Published details include the six condition attributes, prompt template, frozen CLIP ViT-B/16, SCPL threshold 0.15, PCD losses and weights, channel-wise CDF, and two 50-epoch stages at 640×512 with batch 16 on one RTX 4090. The detector backbone/head, fusion scales, submodule layers, complete freezing schedule, hard-gate gradient treatment, and compute costs remain unspecified. Every replacement is logged as an `implementation choice` and validated with tensor, loss, gradient, numerical, and small-overfit tests.

### Data and evaluation protocol

An unseen composition is a held-out attribute tuple whose individual values remain represented in training. It is neither zero-shot recognition of a new weather class nor a synthetic corruption. The initial XML audit finds 326 five-attribute combinations in train and 45 in test. Well-supported low-dimensional pairs, initially altitude-angle, will be screened before fixing the protocol; exhaustive six-dimensional holdouts are not viable. Weather codes require the official mapping before semantic interpretation.

Model selection uses train/validation only. The final test set is reserved. Each experiment must retain marginal attribute coverage, isolate acquisition groups and adjacent frames, construct a matched seen-combination reference, and report scene, class, object-size, date, and illumination confounds. Nineteen location-date groups occur in both official splits; this is an audit finding rather than proof of image leakage and requires sequence/near-duplicate inspection.

Compare dual-stream concat, data-driven dynamic gating, PCDF-style SCPL+PCD+CDF, and a candidate attribute-factorized teacher under one disclosed backbone and budget. The candidate is falsifiable and receives no test-combination labels. Condition probes are fit on training scene groups and tested on disjoint groups using balanced accuracy/macro-F1, alongside detection retention. Probe-performance associations are not causal proof.

Report mAP50, mAP50-95, per-class AP, worst-group performance, scene-group bootstrap intervals, multi-seed variation, and `CGD = mAP_seen_reference - mAP_unseen`. Stop with a negative result if supported holdouts are inadequate, confounding cannot be controlled, PCDF differs from dynamic gating only within seed noise, or factorization fails to improve unseen/worst-group performance. Do not replace a falsified hypothesis with another unverified fallback module.

Use a local 8GB GPU for implementation and profiling, and a 24GB-class GPU for full multi-seed experiments. No memory, training-time, or reproduction guarantee is made before measurement.
