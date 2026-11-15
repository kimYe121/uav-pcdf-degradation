<p align="center">
  <a href="#chinese">简体中文</a> &nbsp;·&nbsp; <a href="#english">English</a>
</p>

<a name="chinese"></a>

# PCDF 方法级复现与研究问题审查

**基于 ATR-UMOD / PCDF（ICCV 2025）的无人机 RGB-IR 旋转目标检测研究。**

候选研究问题（尚未确认）：**PCD 所要求的条件特定/条件不变分离，是否真的发生？条件信息泄漏、目标判别能力与检测性能之间是什么关系？** 在确认其研究独立性、可实现性和最低成本证伪方案之前，不将其称为最终创新点。

> **当前状态：研究假设与复现设计阶段。PCDF 尚未实现，尚无解耦有效性或改进结论。**
>
> GitHub 仓库名 `uav-pcdf-degradation` 是早期命名；固定退化阈值、状态观测、连续回退均已退出主线。

## 纠正：不再研究“未见条件组合”

补充材料 B.2 已明确为减少条件稀疏和过拟合而合并条件：高度按 120m、角度按 30° 二分，时间合并为 Morning/Afternoon/Night，部分天气和光照也合并。此前将其限定为“只用于 Table 3 展示”是错误解读。

按论文的高度×角度×时间规则重新统计本地 XML，理论上的 12 种组合在 train 和 test 中**全部覆盖**，没有组合缺失。原始标签产生的更多细粒度组合不能用来证明论文实验口径下存在未见组合。因此，未见条件组合、组合泛化差和属性分解提示均不再作为当前研究计划。

## 候选问题：条件解耦是否真的成立（尚未确认）

PCDF 用 `Ldt` 让条件特定特征对齐文本条件教师，用 `Lirr` 降低两路条件不变特征与条件特征的相关性，再用 `Ldc` 保持目标判别能力。论文的 Table 4 消融和补充材料 F 节特征图说明这些模块有助于检测，但没有直接测量：

- 条件不变特征中还能否预测高度、角度、时间等条件；
- 条件特定特征中是否保留目标类别信息；
- 降低条件可预测性是否伴随着目标信息损失；
- 条件泄漏与分条件检测性能是否存在稳定关系。

这不是重复“动态调权”，而是检验 PCDF 的一个核心机制假设。

## 可证伪实验

### 1. 方法级复现

在同一公开 OBB 骨架上实现 SCPL、PCD、CDF。所有论文未公开的结构选择均记录为 `implementation choice`，不承诺复原 63.1 mAP50。基础 concat 与动态门控只作机制对照，不作为研究创新。

### 2. 冻结特征探针

在严格隔离的场景组上训练/测试轻量探针：

| 特征 | 探针目标 | 预期机制 |
|---|---|---|
| 条件特定 `Fs` | 单属性条件标签 | 应具有条件可预测性 |
| 条件不变 `Fv_rgb / Fv_ir` | 单属性条件标签 | 与原始特征、无 PCD 对照比较残留信息是否下降；不要求必然接近机会水平 |
| `Fs / Fv` | 目标类别或目标存在性 | 检查条件分支是否混入目标信息、目标分支是否损失判别信息 |

分别测试各单属性，报告 balanced accuracy、macro-F1、机会水平和场景组置信区间；不使用稀疏的六属性联合标签。探针比较线性模型与有限容量的非线性模型，避免把探针过拟合误判为特征泄漏。

### 3. 对照与最小改进

比较原始 PCD、移除 `Lirr`、移除 PCD 三组，先检查 `Lirr` 是否真的降低泄漏及其检测代价。只有发现“泄漏高且影响检测”或“约束过强损害目标信息”时，才测试一个针对性调整，例如按目标区域约束条件不变性，而不是无依据地再增加门控模块。

### 4. 退出条件

若条件探针不能稳定复现、不同场景组结论相反，或 `Lirr` 的泄漏变化与检测性能始终无关，则报告阴性结果，不把探针相关关系包装成因果改进。若没有足够训练资源完成可信 PCDF 对照，则只交付复现与审计工具，不声称机制结论。

完整证据和协议见 **[REPRODUCTION.md](REPRODUCTION.md)**。

## 当前进展

| 部分 | 状态 |
|---|---|
| ATR-UMOD 成对转换与场景级划分工具 | 已实现；正式协议需继续审计采集序列 |
| 合并后条件覆盖统计 | 高度×角度×时间 12 种组合 train/test 全覆盖 |
| RGB/IR 与六通道小子集基线 | 前期工程对照；不是 PCDF |
| SCPL / PCD / CDF | 待实现 |
| 条件与目标信息探针 | 待实现、待证伪 |
| 解耦机制结论 | **尚未获得** |

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

- [PCDF 正文与补充材料（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.html)：PCD 公式、Table 4 消融、补充 B.2 条件合并与 F 节特征图。
- [D2E2-Det（TGRS 2026）](https://doi.org/10.1109/TGRS.2026.3714592)：共享/私有模态特征解耦，与 PCD 的条件特定/条件不变解耦目标不同。
- [RASM-Net（2026）](https://doi.org/10.3390/rs18193288) 与 [QA2FDet（2026）](https://doi.org/10.3390/s26113443)：局部可靠性与错位融合已有人研究，不作为本项目空白。

## 许可证

MIT，详见 [LICENSE](LICENSE)。

---

<a name="english"></a>

# PCDF Method-Level Reproduction and Research-Question Audit

**UAV RGB-IR oriented object detection based on ATR-UMOD / PCDF (ICCV 2025).**

Candidate question, not yet confirmed: does the claimed condition-specific/condition-invariant separation actually occur, and how are condition leakage, object discriminability, and detection performance related? It must pass a research-independence, feasibility, and minimum-cost falsification review before being treated as the final contribution.

> **Status: hypothesis and reproduction design. PCDF is not yet implemented; no decoupling or improvement result has been obtained.**

## Correction: no unseen-composition claim

Supplementary Section B.2 explicitly merges conditions to alleviate sparsity and overfitting. The earlier interpretation that this was only for Table 3 was incorrect. Recomputing altitude×angle×time with the paper's rules gives 12 possible combinations, all present in both train and test. Finer raw-label combinations do not establish an unseen-composition gap under the paper's experimental condition system. Composition holdouts, CGD, and factorized prompts have therefore been removed from the plan.

## Falsifiable decoupling study

Reproduce SCPL, PCD, and CDF on a disclosed OBB backbone, recording every unspecified component as an implementation choice. Train and test lightweight probes on scene-disjoint groups:

- Predict individual condition attributes from `Fs`, `Fv_rgb`, and `Fv_ir`.
- Probe object category/presence to detect object information in the condition branch or discriminability loss in the invariant branches.
- Compare full PCD, PCD without `Lirr`, and no PCD using balanced accuracy, macro-F1, chance level, detection metrics, and scene-group intervals.
- Only if a reproducible leakage–detection trade-off exists, test one targeted adjustment rather than another generic gating module.

A probe association is not causal proof. Report negative or group-inconsistent results rather than claiming successful decoupling or improvement.

See **[REPRODUCTION.md](REPRODUCTION.md)** for source evidence, reproduction boundaries, and acceptance criteria.

## Status and setup

| Part | State |
|---|---|
| Paired ATR-UMOD conversion and scene-level split utilities | Implemented; acquisition-sequence audit remains |
| Merged condition coverage | All 12 altitude-angle-time combinations occur in both splits |
| Preliminary small-subset baselines | Engineering controls; not PCDF |
| SCPL / PCD / CDF | Not implemented |
| Condition and object-information probes | Planned and unfalsified |
| Decoupling conclusion | **Not established** |

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
