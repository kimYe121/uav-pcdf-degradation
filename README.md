<p align="center">
  <a href="#chinese">简体中文</a> &nbsp;·&nbsp; <a href="#english">English</a>
</p>

<a name="chinese"></a>

# PCDF 条件组合泛化与解耦审计

**基于 ATR-UMOD / PCDF（ICCV 2025）的无人机 RGB-IR 旋转目标检测研究。**

研究问题：当每个属性在训练中都出现过、但其**组合没有出现过**时，PCDF 的条件提示与动态融合还能否泛化？所谓的“条件无关特征”是否仍泄露可预测的条件信息？

> **当前状态：研究设计与数据协议审计阶段。PCDF 尚未实现，尚无组合泛化或解耦结论。**
>
> GitHub 仓库名 `uav-pcdf-degradation` 是早期命名；当前主线不再是固定退化阈值、状态观测或连续回退。

## 为什么不是重复 PCDF

PCDF 已经实现逐样本、实时的条件动态融合，推理时不调用 CLIP 文本编码器。本项目不再提出“根据环境动态调权”，也不把 RGB-only、IR-only、concat 与 PCDF 的常规比较当作研究贡献。

原文及其补充材料留下了一个不同的问题：

- 正文 Table 3 按高度、角度、天气等**单个属性的边际取值**报告结果，不是未见属性组合测试。
- 补充材料 B.2 为缓解条件稀疏而合并部分条件；正文将 Foggy 的困难归因于雾浓度与能见度不一致。该合并明确用于 Table 3 的结果展示，不能未经证据扩展为训练提示也全部采用合并标签。
- PCD 的证据主要是检测消融和特征图可视化，没有条件不变特征上的独立泄漏探针。

因此，本项目研究的是**组合外推与机制验证**，不是重复证明条件提示有用。

## 研究假设与最小实验

### 数据可行性门

先读取 XML 元数据，筛选样本量、类别和场景组均有足够支撑的**低维属性对**。留出某个属性组合训练，但保证各边际属性值仍在训练中出现；从已见组合中构造规模、类别、场景相近的参考集。

当前全量审计得到 train 326 种、test 45 种“高度×角度×天气×光照×场景”组合，不能直接对全部六维组合做有效留出。按“地点编号＋日期”分组时，官方 train/test 有 19 个共享组；这不自动证明图像泄漏，但必须检查同采集序列与相邻帧，不能把官方“场景不重叠”与本地代理分组混为一谈。

### 对照与改进

在同一公开 OBB 骨架、划分和训练预算下，保留必要的机制对照：

1. 双流 concat：非条件融合下界；
2. 数据驱动动态门控：区分“动态”与“显式条件监督”；
3. PCDF 式 SCPL + PCD + CDF：待复现的核心模型；
4. 属性分解提示：以可组合的属性表示替代整句联合条件教师，检验它是否改善未见组合；这是待证伪的候选改进，不预设有效。

常规单模态和前期六通道结果仅用于工程检查，不再作为研究创新。

### 评价与证伪

- 报告已见参考集、未见组合、宏平均及最弱组合上的 mAP50 / mAP50-95、分类别 AP 和场景组 bootstrap 区间。
- 报告组合泛化差 `CGD = mAP_seen_reference - mAP_unseen`，同时保留绝对性能，避免用比例掩盖低基线。
- 冻结特征后训练独立条件探针，报告条件 balanced accuracy / macro-F1，并保留目标检测指标；低相关不等于语义独立，探针结果不直接证明因果。
- 多随机种子比较 PCDF、无显式条件动态门控和属性分解提示。

**否定主线的情况**：有效留出组合不足；场景或类别混杂无法排除；PCDF 与无显式条件门控的未见组合差异不超过随机种子波动；或泄漏探针与泛化始终无关。此时报告阴性结果，不继续堆叠动态回退模块。

详细证据、复现边界与退出条件见 **[REPRODUCTION.md](REPRODUCTION.md)**。

## 当前进展

| 部分 | 状态 |
|---|---|
| ATR-UMOD 成对转换与场景级划分工具 | 已实现；划分定义仍需针对本研究扩展 |
| 全量条件组合与 train/test 元数据审计 | 已完成初步只读统计 |
| RGB/IR 与六通道小子集基线 | 前期工程对照；不是 PCDF |
| 未见条件组合划分生成器 | 待实现 |
| SCPL / PCD / CDF | 待实现 |
| 条件泄漏探针与属性分解提示 | 待实现、待证伪 |
| 组合泛化结论 | **尚未获得** |

## 安装与数据

```bash
git clone https://github.com/kimYe121/uav-pcdf-degradation.git
cd uav-pcdf-degradation
pip install -e .
python scripts/check_env.py
python scripts/prepare_atr.py check --root data/ATR-UMOD
```

使用 [ATR-UMOD](https://github.com/supercpy/ATR-UMOD) 的配准 RGB-IR 图像、OBB 标注与六类条件属性。正式复现采用 11 类和 IR 标签；`--classes` 只用于前期调试。8GB 显存可用于模块和小批量验证，完整实验建议 24GB 级 GPU；租卡时长必须由实测显存与单 epoch 时间决定。

```text
data/ATR-UMOD/
└── train|test/
    ├── images/
    ├── images_ir/
    ├── labels/
    └── labels_ir/
```

## 主要依据

- [PCDF 正文与补充材料（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.html)：条件边际结果、PCD 消融与 Foggy 局限。
- [CAFuser（RA-L 2025）](https://arxiv.org/abs/2410.10791)：粗细提示比较，但没有条件类别/组合留出。
- [Distribution-Aligned Multimodal Fusion（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/papers/Hao_Distribution-Aligned_Multimodal_Fusion_for_Robust_Object_Detection_CVPR_2026_paper.pdf)：研究未见输入退化，不是语义条件组合。
- [RASM-Net（2026）](https://doi.org/10.3390/rs18193288) 与 [QA2FDet（2026）](https://doi.org/10.3390/s26113443)：已覆盖局部可靠性、错位及小目标融合，故本项目不将“全局改局部”作为空白。

## 许可证

MIT，详见 [LICENSE](LICENSE)。

---

<a name="english"></a>

# PCDF Condition-Composition Generalization and Decoupling Audit

**UAV RGB-IR oriented object detection based on ATR-UMOD / PCDF (ICCV 2025).**

Can PCDF generalize when every individual condition attribute was observed during training but their **combination was not**? Do the claimed condition-invariant features still contain predictable condition information?

> **Status: research design and data-protocol audit. PCDF is not yet implemented; no composition-generalization or decoupling result has been obtained.**
>
> `uav-pcdf-degradation` is a legacy repository name. Fixed degradation thresholds, state observation, and continuous fallback are no longer the research plan.

## Research gap

PCDF already performs sample-specific dynamic fusion at inference without running the CLIP text encoder. This project does not reinvent dynamic weighting or present routine unimodal/concat comparisons as contributions.

The paper reports marginal per-attribute results rather than held-out attribute combinations. Its PCD evidence consists of detection ablations and feature-map visualizations, without an independent condition-leakage probe. The supplementary merging rule is explicitly connected to Table 3 and must not be assumed to govern every training prompt without further evidence.

The resulting question is **compositional extrapolation and mechanism verification**, not another demonstration that conditions help.

## Hypothesis and falsification plan

1. Audit XML metadata and select well-supported low-dimensional attribute pairs. Hold out a combination while retaining every marginal attribute value in training. Construct a matched seen-combination reference set.
2. Under one disclosed OBB backbone and budget, compare dual-stream concat, data-driven dynamic gating, the PCDF-style model, and a candidate attribute-factorized condition teacher. The last item is a falsifiable hypothesis, not a presumed improvement.
3. Measure mAP50/mAP50-95, per-class AP, worst-group performance, scene-group bootstrap intervals, and `CGD = mAP_seen_reference - mAP_unseen`.
4. Train independent condition probes on frozen condition-invariant features and report balanced accuracy/macro-F1 alongside detection retention. Decorrelation alone is not semantic independence, and probe associations are not causal proof.
5. Repeat decisive comparisons across seeds.

The local audit finds 326 five-attribute combinations in train and 45 in test, so exhaustive six-attribute holdouts are not statistically viable. Nineteen location-date groups occur in both official splits; this does not itself establish image leakage, but acquisition sequences and adjacent frames must be checked before drawing conclusions.

The study stops with a negative result if supported holdouts are inadequate, confounding cannot be controlled, or the PCDF-versus-dynamic-gating difference is within seed variability. It will not fall back to another unverified gating module.

See **[REPRODUCTION.md](REPRODUCTION.md)** for source evidence, reproduction boundaries, and acceptance criteria.

## Status and setup

| Part | State |
|---|---|
| Paired ATR-UMOD conversion and scene-level split utilities | Implemented; research-specific splitting remains |
| Full condition-combination and split metadata audit | Initial read-only audit completed |
| Preliminary small-subset baselines | Engineering controls; not PCDF |
| Composition holdout generator | Planned |
| SCPL / PCD / CDF | Not implemented |
| Condition probes and factorized prompts | Planned and unfalsified |
| Composition-generalization conclusion | **Not established** |

```bash
git clone https://github.com/kimYe121/uav-pcdf-degradation.git
cd uav-pcdf-degradation
pip install -e .
python scripts/check_env.py
python scripts/prepare_atr.py check --root data/ATR-UMOD
```

Use all 11 classes and IR labels for the formal reproduction. An 8GB GPU is suitable for implementation checks and profiling; full experiments are better suited to a 24GB-class GPU. Rental duration must follow measured memory and epoch time.

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <a href="#chinese">简体中文</a> &nbsp;·&nbsp; <a href="#english">English</a>
</p>
