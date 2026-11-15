# PCDF 复现与研究问题审查 / PCDF Reproduction and Research-Question Audit

## 中文

### 1. 当前问题与研究边界

本项目依据 Chen 等人的 ICCV 2025 PCDF 论文及官方补充材料开展方法级复现。目前有一个尚未确认的候选问题：

> **PCD 是否真正分离了条件信息与目标判别信息？这种分离或失败与检测性能有什么关系？**

这只是待论证的研究假设，尚不能视为最终主线或创新点。需要先确认其研究独立性、可实现性和最低成本证伪方案；若不能形成独立贡献，应继续依据论文证据筛选，而不是重新包装已否决的动态调权方案。

这是论文指导的方法级复现与机制审计候选，不是官方代码复现。项目不承诺复原 63.1 mAP50，也不把不同 backbone、子集或划分的结果与论文数值直接横比。仓库名中的 `degradation` 是历史名称，不再代表当前研究假设。

### 2. 撤回“未见条件组合”主线

补充材料 B.2 明确指出，条件变化过多导致每个条件样本不足、增加过拟合风险并妨碍训练，因此合并条件以提高数据效率：

- 高度：`<=120m` / `>120m`；
- 角度：`<=30°` / `>30°`；
- 时间：Morning `[07:00,14:00)`、Afternoon `[14:00,19:00)`、Night `[19:00,07:00)`；
- After Rain、Rainy 合并进 Cloudy；
- Overexposure、Twilight 分别合并进 Normal、Dim。

此前把这套规则限定为“只用于 Table 3 展示”是对原文动机的不当收窄，已经纠正。正文 Table 3 的分组语境不能推翻补充材料关于训练效率的明确说明。

按高度×角度×时间合并规则，理论上有 12 种组合。本地 XML 只读重算结果：

| 划分 | 图像数 | 实际组合数 | 组合覆盖 |
|---|---:|---:|---|
| train | 11,850 | 12 | 全部 12 种 |
| test | 1,503 | 12 | 全部 12 种 |

train 四种高度×角度组合的样本量分别为 1,142 / 2,907 / 3,589 / 4,212，test 为 153 / 397 / 449 / 504。没有任何合并后的高度×角度×时间组合在另一划分中缺失。因此不能用原始标签的 326/45 种细粒度组合，宣称论文的实验条件体系存在“未见组合”。

已撤回：未见条件组合划分、CGD、属性分解提示及其相应研究叙述。该错误说明研究缺口必须由**论文实际标签口径＋本地统计**共同验证，不能只从表格缺项推出。

### 3. 候选问题：解耦未被直接验证（尚未确认）

PCD 的三个目标分别是：

- `Ldt`：用 CMD 让条件特定特征 `Fs` 对齐文本条件教师；
- `Lirr`：降低条件不变特征 `Fv_rgb / Fv_ir` 与 `Fs` 的相关性；
- `Ldc`：通过辅助检测损失保留目标判别能力。

正文 Table 4 完整 PCDF 为 63.1，`w/o PCD` 为 62.1，`w/o Lirr` 为 62.7；补充材料 F 节展示中间特征图。这些证据支持“PCD 有助于检测”，但没有直接测量：

1. `Fv` 中残留的条件可预测性；
2. `Fs` 中混入的目标类别或目标存在性信息；
3. 条件泄漏下降是否伴随目标信息损失；
4. 泄漏变化与分条件检测性能的关系是否跨场景组稳定。

正文相应段落的 PCD 下降 1.2 与 Table 4 的 1.0 个百分点不一致，报告中应保留这一差异。

### 4. 与相近工作的区别

D2E2-Det 的共享/私有模态特征正交解耦、UOD 的模态不变内在特征学习，并不等价于 PCDF 的多属性条件特定/条件不变解耦。CAFuser、近期模态质量蒸馏与 RASM-Net/QA2FDet 研究条件动态融合、局部可靠性或错位，并未替代对 PCD 条件泄漏的独立审计。

本项目不再将“全局改局部”“动态调权”“普通门控比较”或“未见条件组合”称为研究空白。文献核查是针对性检索，不宣称系统综述意义上的绝对空白；实施前仍需复查最新相关论文。

### 5. 原文公开内容与复现缺口

公开内容包括：11 类 ATR-UMOD、六类条件属性、IR 标签、冻结 CLIP ViT-B/16、完整提示模板、SCPL `tau=0.15`、PCD 三种损失及权重 `0.01/0.003/0.01`、逐通道 CDF、两阶段各 50 epochs、640×512、batch 16、SGD 及单张 RTX 4090。

仍未明确：原检测 backbone/neck/OBB head、融合尺度、`Fc/S/V_rgb/V_ir/Ft` 层结构、完整冻结列表、硬门控梯度处理、部分总损失/增强/模型选择细节，以及显存、FLOPs、FPS 和训练时间。

自行补齐的检测骨架、投影层、池化、CMD 数值归一化、STE 或软门控近似、文本缓存、AMP 和梯度累积均标记为 `implementation choice`。先做维度、损失、梯度和数值单元测试，再做小数据过拟合；不能把自行补齐的结构描述为原论文细节。

### 6. 数据与探针协议

#### 6.1 防止场景泄漏

使用论文合并后的条件标签；天气和光照数值在取得官方映射前只作匿名类别，不擅自赋予语义。探针训练、模型选择和测试按采集序列/航次隔离，不能按帧随机切分。

按“地点编号＋日期”分组时，train 有 68 组、test 有 21 组，交集为 19。这不自动证明图像泄漏，因为地点编号可能跨采集复用；但需结合原始序列、时间连续性与视觉近重复审查。论文的非重叠场景声明与本地代理分组不能混为一谈。

#### 6.2 双向信息审计

| 被测表示 | 条件探针 | 目标探针/辅助检测 | 解释 |
|---|---|---|---|
| `Fs` | 应高于机会水平 | 测量目标信息是否混入 | 验证条件分支是否按预期工作 |
| `Fv_rgb / Fv_ir` | 与原始特征、无 PCD 对照比较 | 必须保留目标判别能力 | 检验去条件是否以损失目标信息为代价 |

条件探针分别预测高度、角度、时间等单属性，报告 balanced accuracy、macro-F1、类别先验和机会水平。联合六属性标签过于稀疏，不用于主探针。目标信息使用独立场景组上的目标存在性/类别探针与原辅助检测指标，不只用特征图判断。

线性探针反映易读取的信息；有限容量非线性探针用于检查非线性残留。探针容量、训练样本量和正则化固定并公开，避免探针自身过拟合造成“高泄漏”假象。原始未解耦特征、无 `Lirr`、无 PCD 是必要参照。

### 7. 最小实验与退出条件

1. **工程验收**：张量、损失、梯度、文本缓存与小数据过拟合通过。
2. **最小对照**：完整 PCD、无 `Lirr`、无 PCD 在相同数据和预算下训练，保留检测与中间特征。
3. **机制审计**：完成双向探针，比较条件泄漏、目标判别和检测变化。
4. **针对性调整（可选）**：只有当泄漏或目标信息损失可重复时，才测试一种与问题对应的约束，例如目标区域内的条件不变约束。不得预设该改进有效。

退出条件：

- 探针在不同场景组上的方向不一致，或结果主要由类别先验/探针过拟合解释；
- `Lirr` 不改变条件泄漏，或其变化与检测始终无稳定关系；
- 条件不变性提高仅由目标信息严重损失造成；
- 没有资源完成可信 PCDF 对照，则不宣称机制结论，只交付复现和审计工具。

探针关联不是因果证明。即使发现高泄漏，也不能直接宣称它是检测失败原因；需要对应的干预实验和跨 seed 复现。

### 8. 算力策略

本地 8GB 用于模块实现、单元测试、小数据过拟合、探针和显存/单 epoch profiling。完整 11 类、多 seed 与关键消融建议使用 24GB 级 GPU；4090 与原文硬件接近但不保证相同结果。当前没有 PCDF 实测，不承诺显存、训练时长或租卡时长。

### 9. 资料

- [PCDF 正文与补充材料](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.html)
- [ATR-UMOD 官方仓库](https://github.com/supercpy/ATR-UMOD)
- [D2E2-Det](https://doi.org/10.1109/TGRS.2026.3714592)
- [模态可靠性建模的航空遥感可见光—红外目标检测](https://doi.org/10.11834/jig.260197)

---

## English

### Scope

This is a paper-guided reproduction with an unconfirmed candidate question: whether PCD actually separates condition information from object-discriminative information, and how that separation or failure relates to detection. It is not yet a final research direction or claimed contribution. Research independence, feasibility, and a minimum-cost falsification plan must be established first. This is not an official-code reproduction and does not promise 63.1 mAP50.

### Correction of the previous plan

Supplementary B.2 explicitly merges altitude, angle, time, weather, and illumination conditions to reduce sparsity, overfitting, and training difficulty. The earlier claim that this concerned only Table 3 was an unjustified narrowing of the text.

Recomputing local XML labels using the paper's altitude×angle×time rules gives all 12 possible combinations in both train (11,850 images) and test (1,503 images), with none missing from either split. Raw fine-grained combinations therefore cannot support an unseen-composition claim under the paper's condition system. Composition holdouts, CGD, and factorized-prompt experiments have been withdrawn.

### Remaining falsifiable question

The paper supports PCD using detection ablations and intermediate feature visualizations. It does not directly measure residual condition predictability in `Fv_rgb/Fv_ir`, object information in `Fs`, or the trade-off between condition removal and object discriminability. Table 4 gives 63.1 for PCDF and 62.1 without PCD; the accompanying text's 1.2-point decrease differs from the table's 1.0-point difference.

Reproduce SCPL/PCD/CDF on a disclosed OBB backbone. Record every unspecified detector, projection, pooling, normalization, gradient, and scheduling choice as an `implementation choice` and validate tensor shapes, losses, gradients, text caching, and small-data overfitting.

Train and test condition and object-information probes on acquisition-disjoint groups. Use individual condition attributes rather than sparse six-way labels; report balanced accuracy, macro-F1, class priors, chance level, detection retention, and group intervals. Compare full PCD, PCD without `Lirr`, and no PCD. Linear and capacity-controlled nonlinear probes address different readings of feature information; probe overfitting must not be mistaken for feature leakage.

Only a reproducible leakage or discriminability problem justifies one targeted intervention. Stop with a negative result if group-level findings conflict, `Lirr` does not alter leakage, or improved invariance merely destroys object information. Probe associations are not causal evidence. Without a credible PCDF control, deliver only implementation and audit tools rather than a mechanism claim.

Use a local 8GB GPU for implementation and profiling and a 24GB-class GPU for full multi-seed experiments. No memory, training-time, or reproduction guarantee is made before measurement.
