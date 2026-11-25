# 复现记录与验收标准

本文件记录 PCDF 方法级复现的范围、实现清单、训练协议和验收结果。
结构细节、公式映射和张量形状见 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**，
部署流程见 **[docs/WORKFLOW.md](docs/WORKFLOW.md)**。

## 1. 目标与范围

**目标**：在公开 OBB 检测器上实现论文的 SCPL、PCD、CDF 三个模块、三项解耦损失
和三阶段训练流程，使代码结构与论文 Eq. 1–11 一一对应，并能在 ATR-UMOD 上完成
训练、评估和小样本过拟合验证。

**不是**：官方代码复现。论文没有公开检测器 backbone/neck/OBB head、`S/V` 内部
结构、阶段 1 引导来源等细节（条件码本已由数据集仓库公开）。本仓库不承诺 63.1 mAP50，
也不把不同 backbone、batch 或数据子集的结果与论文表格直接横比。

## 2. 已实现清单

| 论文部分 | 代码 | 状态 |
|---|---|---|
| 双流 RGB/IR 编码器（Fig. 4） | `models/encoder.py` | 已实现，YOLOv5s backbone，预训练逐层严格加载 |
| 初始提示 `I_i`（Eq. 1，补充 B.1） | `conditions.py` | 已实现，模板逐字对应 |
| SCPL 硬门控 `Fc`、`τ=0.15`（Eq. 2–4） | `models/scpl.py`, `models/blocks.py` | 已实现，含 straight-through |
| 三支解耦 `S/V_rgb/V_ir`（Eq. 5） | `models/blocks.py` | 已实现，P3/P4/P5 |
| `L_dt` CMD 5 阶（Eq. 6） | `losses.py::cmd_loss` | 已实现 |
| `L_irr` 交叉协方差（Eq. 7） | `losses.py::irr_loss` | 已实现，三尺度平均 |
| `L_dc` 辅助检测（Eq. 8） | `train.py` + 共享检测头 | 已实现，含 objectness |
| `L_dec` 权重 0.01/0.003/0.01（Eq. 9） | `losses.py::decoupling_terms` | 已实现 |
| 逐通道软门控（Eq. 10–11） | `models/blocks.py::ChannelNormSoftGate` | 已实现，权重和为 1 |
| 三阶段训练（正文 5.2 + 补充 B.3） | `train.py` | 已实现，stage 0/1/2 |
| 无条件下测试 | `models/pcdf.py`, `evaluate.py` | 已实现，测试不调用 SCPL |
| 配对数据、IR 标签、条件 id | `data.py` | 已实现，6 通道输入 |
| OBB mAP50 / mAP50-95 | `evaluate.py` | 已实现，probiou + 旋转 NMS |

## 3. 三阶段训练

```text
Stage 0  baseline：双流 concat + 检测头，全参数训练
         loss = L_det
         论文 Table 4 的 Baseline（58.4）

Stage 1  init from stage 0; freeze encoder/neck/head/S,V
         guidance = text_adapter(F_t)
         loss     = L_det

Stage 2  init from stage 1; freeze SCPL + text adapter
         guidance = GAP(F_s)
         loss     = L_det + 0.01*L_dt + 0.003*L_irr + 0.01*L_dc

Test     guidance = GAP(F_s); no condition ids, no CLIP call
```

依据：补充 B.3 原文 "training **only** the SCPL module and the CDF module.
Building on the pre-trained parameters in this stage..."——阶段 1 只训 SCPL+CDF，
主干必须已从 baseline 预训练；Fig. 10 Stage I 中 CDF 权重直接作用于 encoder
特征，阶段 2 才出现 PCD。

阶段 1 的引导来源论文未说明。本实现用 `F_t` 驱动 CDF，阶段 2 用 `L_dt` 把 `F_s`
对齐到同一 256 维空间，使 CDF 投影器可从文本引导迁移到视觉引导。记录为
implementation choice。

## 4. 数据协议

- 训练/验证/测试：`data/atr_umod/{train 10054, val 1796, test 1503}`。
  **报告口径 = `--split test`**（论文同口径）；`val` 只用于选权重，其 mAP 比 test
  低约 16 点（跨场景整组留出 + 长尾类别几乎无样本），不可与论文数字并列比较。
- 输入：RGB 与 IR 配对；IR 灰度复制 3 通道后拼成 6 通道。
- 标签：`labels_ir` 的 YOLO-OBB 8 坐标，归一化。
- 条件：`condition_index.csv` 的 `UAVheight/UAVangle/time/weather/illumination/location`。
- 合并：高度 ≤120 m / >120 m；角度 ≤30° / >30°；时间
  Morning [07,14) / Afternoon [14,19) / Night [19,07)；天气/光照保持匿名码，
  待官方码本替换（补充 B.2 的合并规则已在 `configs/conditions.yaml` 注释中说明）。
- val 由 train 按 `location + 日期` 场景组留出，避免同场景泄漏。

## 5. 验收

### 5.1 已执行

```bash
python -m pytest -q
# 86 passed

python -m pcdf.cli.check_env
# 兼容性 + 权重 + 数据 + 提示 全部通过

python -m pcdf.cli.smoke --device cpu
python -m pcdf.cli.smoke --device 0
# baseline -> 阶段 1 -> 阶段 2 -> 评估，输出 [smoke] OK
```

GPU 实测（RTX 4060 Laptop 8 GB，AMP）：

| 配置 | 峰值显存 |
|---|---|
| 三阶段冒烟 batch 2 / 320×256 | 0.09 / 0.38 / 0.50 GB |
| 阶段 1，640×512，batch 4 | 0.62 GB |
| 阶段 2，640×512，batch 4 | 2.90 GB |

### 5.2 断点续训验收

- `--resume auto` 从 `last.pt` 继续，优化器/调度器/AMP/随机数状态正确恢复；
- 手工构造 mid-epoch 检查点（`next_epoch=0, step_in_epoch=1`），恢复后跳过
  第 1 个 batch，`global_step` 从 1 继续累加；
- 从 `checkpoints/epoch_0001.pt` 直接恢复时，运行目录自动识别为上一级；
- `--epochs` 延长总轮数后，调度器保留 `last_epoch` 并按新总轮数重算 gamma；
- 结构字段不一致（如用改过 `widths` 的配置续训）会直接报错，不会"形状
  对不上还硬跑"。

`SIGTERM` 拦截面向 Linux 容器；Windows 下 `os.kill(SIGTERM)` 会直接终止进程，
本地请以周期保存 + `--resume auto` 为准。

### 5.3 单元测试覆盖

| 文件 | 覆盖点 |
|---|---|
| `test_config.py` | 默认值、档位合并、点号覆盖与类型转换、交叉校验、路径归一、环境变量 |
| `test_conditions.py` | 阈值/时间段/分组、提示模板、掩码、按划分取样本 |
| `test_checkpoint.py` | 检查点字段、原子写、`epochs_total`、配方沿用、结构不匹配报错 |
| `test_runtime.py` | 无卡时 `auto→cpu`、显式 CUDA 快速失败、CPU 环境快照不崩、原子写 |
| `test_data.py` | 6 通道、条件 id、归一化 OBB、collate |
| `test_model.py` | 训练/评估前向形状、CDF 权重范围、门控梯度、文本冻结、阶段 2 冻结 |
| `test_baseline.py` | 键集对齐、stage 0 全参数可训练、stage 2 编码器不误冻结、阶段 1 冻结策略、文本模式跳过 PCD |
| `test_obb_head.py` | 训练态返回 loss 用的 dict、推理态返回可进 NMS 的张量、损失可反传、`model.model[-1]` / `args` 上下文 |
| `test_losses.py` | CMD 相同为 0/平移为正、`L_irr` 零特征为 0、`L_dec` 组合正确 |
| `test_evaluate.py` | 完美预测 AP=1、漏检为 0、无 GT 类别为 NaN |
| `test_report.py` | 状态读取、完成判定、`--require-stage` 退出码 |
| `test_preflight.py` | 权重/数据清单、缺失报送、CLI 退出码 |
| `test_cli.py` | 参数解析、`--set` 格式、覆盖后校验、从检查点重建配置 |

实测环境：Python 3.12，PyTorch 2.11 + CUDA 12.8，Ultralytics 8.4.211，
RTX 4060 Laptop 8 GB；同一套测试在 `CUDA_VISIBLE_DEVICES=""` 下也全部通过。

## 6. 与论文的差异

| 项目 | 论文 | 本仓库 |
|---|---|---|
| 检测器 | 未公开 | YOLOv5s backbone + PAN + ultralytics OBB head（anchor-free + DFL） |
| 融合尺度 | 未公开 | P3/P4/P5 |
| 天气/光照/场景 | B.2 合并后的语义类 | 官方码本 + B.2 合并，提示句为自然语言 |
| 训练 batch | 16 | 配置 16；8 GB 用 2–4 覆盖 |
| 文本编码器 | CLIP ViT-B/16 冻结 | 相同，额外应用官方 `text_projection` |
| 增强 | 未公开 | 仿射 + 水平翻转，无 mosaic/HSV |
| AMP/warmup/clip | 未公开 | 开启 AMP、`warmup=0`（paper 档）、clip 10 |
| 数值 | 63.1 mAP50 | 不承诺、不横比 |

完整列表见 `docs/ARCHITECTURE.md` 第 5 节。

## 7. 研究审计（与本复现分离）

本仓库只交付方法与工程实现，不对"PCD 是否真正分离条件信息与目标信息"下结论。
若后续要做条件/目标探针审计，需要：

1. 用本仓库冻结特征导出 `F_s`、`F_v,rgb`、`F_v,ir`；
2. 在场景隔离组上训练线性/受限非线性探针，预测单属性条件或目标类别；
3. 比较完整 PCD、去 `L_irr`、去 PCD 三组的泄漏与检测代价；
4. 只有发现可重复的泄漏—检测权衡，才测试针对性约束。

探针相关不等于因果；不得把本复现的检测数值当作解耦有效性的证据。

## 8. 下一步

- 条件码本已接入官方 attribute_details.txt，stage 1 可直接在语义提示上训练。
- 在 24 GB GPU 上按论文 batch 16、50+50 epochs 跑完整训练。
- 增加按条件分组的 mAP50 统计（论文 Table 3 口径）。
- 在同一 backbone 上重训 baseline，才能做论文级数值对照。
