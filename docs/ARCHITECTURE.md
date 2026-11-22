# PCDF 网络结构与代码对应说明

本文档把 ICCV 2025 论文 **Fusion Meets Diverse Conditions: A High-diversity
Benchmark and Baseline for UAV-based Multimodal Object Detection with Condition
Cues**（PCDF）及其补充材料逐条映射到 `src/pcdf` 的实现。凡论文没有写明的结构，
都在第 5 节标为 **implementation choice**，不冒充原文细节。

- 论文正文：<https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.html>
- 数据集：<https://github.com/supercpy/ATR-UMOD>

---

## 1. 整体数据流

```
RGB ──► RGB Encoder ──► F_rgb(P3,P4,P5) ──┬──► V_rgb ──► F_v,rgb ──┐
                                          │                        │
IR  ──► IR  Encoder ──► F_ir (P3,P4,P5) ──┼──► V_ir  ──► F_v,ir  ──┤
                                          │                        ├─► CDF ──► PAN ──► OBB Head
                                          └──► S(concat) ──► F_s ──┘        ▲
                                                                             │
Condition ids ──► IPC ──► CLIP(冻结) ──► Fc 硬门控 ──► 条件提示 ──► CLIP ──► F_t
                                                                  (仅训练，供 L_dt)
```

阶段 0（baseline）纯视觉 concat 训练；阶段 1 用 `F_t` 作为 CDF 的条件引导
（主干冻结）；阶段 2 用 `F_s`，并用 `L_dt` 把 `F_s` 拉到阶段 1 的文本引导空间。
测试时完全不读取条件标注，只走 `Encoder → S/V → CDF → Neck → Head`。

---

## 2. 模块与公式

### 2.1 双流编码器（Sec. 4.1 / Fig. 4）

- `RGB Encoder`、`IR Encoder` 各自输出 P3/8、P4/16、P5/32 三层特征。
- 代码：`src/pcdf/models/encoder.py::Backbone`
  - 默认 YOLOv5s：`yolov5.yaml` + `scale: s`，taps 为 `model.save[:3]`（4/6/10）。
  - 也支持 YOLO11-OBB（`yolo11*-obb.yaml`），由 profile 切换。
  - 输入为 6 通道，`x[:, :3]` 进 RGB 分支，`x[:, 3:]` 进 IR 分支；IR 是灰度复制
    成 3 通道，因此两个分支的第一层卷积都能直接加载 RGB 预训练权重，不需要
    通道膨胀。
  - 预训练权重：`weights/yolov5su.pt` 中 `model.{0..10}.*`，逐层严格匹配；
    加载前会反查首层通道数，校验权重档位与 `backbone_scale` 一致。

### 2.2 SCPL（Sec. 4.2，Eq. 1–4；补充 B.1）

代码：`src/pcdf/models/scpl.py::SCPL`、`src/pcdf/models/blocks.py::SCPLGate`、
`src/pcdf/conditions.py`。

| 论文 | 实现 |
|---|---|
| 初始提示 `I_i = s + Σ o_i^n`，`o_i^n = v_n + a_i^n`（Eq. 1） | `ConditionTable.prompt_from_ids()`，模板逐字取自补充 B.1 |
| `G_i = 1(σ(Fc(Pool(F_rgb,F_ir), I_i)) ≥ τ)`（Eq. 2） | P5 全局 maxpool → 两模态拼接 → 与 `I_i` 的 CLIP 嵌入拼接 → MLP → softmax(6) |
| `o' = o if g=1 else ∅`（Eq. 3） | `ConditionTable.compose(ids, mask)` 丢弃被屏蔽的属性块 |
| `P_i = I_i × G_i`，再经 CLIP 得 `F_t`（Eq. 4） | 重新编码屏蔽后的完整句子，得到 512 维 `F_t` |
| `τ = 0.15` | `model.tau` |
| SCPL 只在训练使用 | `PCDFNet.forward` 仅在 `guidance="text"` 或 `return_ft=True` 时调用 |

六个属性按提示模板顺序固定为
`scenario, altitude, angle, time, weather, illumination`。

**可学习性处理。** 硬门控本身不可微。为了让 `Fc` 真正收到阶段 1 的检测梯度，
实现采用 straight-through：前向用硬 0/1 掩码构造提示，反向通过
`Σ_n p_n · E(block_n)` 把梯度传回属性可用概率（`SCPL.forward` 中的 `ft_ste`）。
这是对论文未说明细节的补齐，不改变前向语义。

**空掩码保护。** 若某样本六个属性全低于阈值，保留概率最大的属性，避免出现空提示。

### 2.3 PCD（Sec. 4.3，Eq. 5–9）

代码：`src/pcdf/models/blocks.py::DecoupleBranch`、`src/pcdf/losses.py`。

- `F_s = S([F_rgb, F_ir])`：两模态特征通道拼接后过 1×1 + 3×3 卷积，输出通道
  与单模态一致。
- `F_v,rgb = V_rgb(F_rgb)`、`F_v,ir = V_ir(F_ir)`：各自 1×1 + 3×3 卷积。
- 三层都做同样的解耦，`L_irr` 在三层上取平均。

**`L_dt`（Eq. 6，CMD）**：`cmd_loss(fs_gap, text_adapter(F_t))`，5 阶中心矩，
按 `|b-a|` 归一化。`fs_gap` 是 P5 `F_s` 的全局平均池化（256 维），
`text_adapter` 是 512→256 的两层投影（见 2.5）。

**`L_irr`（Eq. 7）**：对每层计算
`||F_v^T F_s||_F^2 / (C·H·W)`，RGB/IR 两支相加；除以 `C·H·W` 是为了让不同尺度
可加，否则原始 Frobenius 范数会随空间尺寸增长并压过检测损失。

**`L_dc`（Eq. 8）**：把**同一个 OBB head** 分别作用在 `F_v,rgb` 和 `F_v,ir` 上，
用同一个 `v8OBBLoss` 计算分类/回归/目标性损失。共享 head 不增加参数，也保证
“判别能力”与主检测任务定义一致。OBB head 额外含角度损失，属于检测器本身。

**`L_dec`（Eq. 9）**：`0.01·L_dt + 0.003·L_irr + 0.01·L_dc`。

### 2.4 CDF（Sec. 4.4，Eq. 10–11）

代码：`src/pcdf/models/blocks.py::ChannelNormSoftGate`。

- 引导向量 `g`（256 维）经每层独立的 2 层 MLP 映射为 `2·C_l` 个 logits。
- reshape 成 `(B, 2, C_l, 1, 1)` 后在第 2 维（模态维）做 softmax：
  `W_rgb + W_ir = 1`（逐通道）。
- `F_f = Concat(W_rgb ⊙ F_v,rgb, W_ir ⊙ F_v,ir)`，只重加权条件不变特征。
- 引导来源：
  - 阶段 1：`text_adapter(F_t)`（文本条件引导）；
  - 阶段 2 / 测试：`GAP(F_s)`（视觉条件引导）。

### 2.5 三阶段训练协议（正文 Sec. 5.2 + 补充 B.3）

**阶段 0：baseline 双流 concat 检测器**（论文 Table 4 的 Baseline，58.4%）。
原文："Our baseline is a one-stage dual-stream detector that integrates two
modalities through concatenation fusion." 双流 encoder 特征逐层 concat 成 2C，
经 neck `reduce`（1×1，2C→C）融合后进 head，只训 `L_det`。它同时是阶段 1 的
权重起点和复现对照的锚点。

| | 阶段 0 baseline | 阶段 1 | 阶段 2 | 测试 |
|---|---|---|---|---|
| RGB/IR encoder | 训练 | **冻结** | 训练 | 使用 |
| Neck / Head | 训练 | **冻结** | 训练 | 使用 |
| SCPL `Fc` + 文本编码器 | — | 训练 / 冻结 CLIP | **冻结** | 不使用 |
| 文本投影 `text_adapter` | — | 训练 | **冻结**（稳定 `L_dt` 目标） | 不使用 |
| CDF | — | 训练 | 训练 | 使用 |
| `S` / `V_rgb` / `V_ir` | — | **冻结**（前向不经过） | 训练（`L_dt/L_irr/L_dc`） | 使用 |
| CDF 作用特征 | — | encoder 原始特征（Fig. 10 Stage I 无 PCD） | `F_v`（式 11） | `F_v` |
| 条件引导 | — | `F_t` | `F_s` | `F_s` |
| 损失 | `L_det` | `L_det` | `L_det + L_dec` | — |

依据：补充 B.3 原文 "the first stage focuses on learning the relationship
between conditions and multimodal reliability by **training only** the SCPL
module and the CDF module. Building on the pre-trained parameters in this
stage..."——阶段 1 只训 SCPL+CDF，意味着主干（encoder/neck/head）必须已从
baseline 预训练；Fig. 10 Stage I 中 CDF 权重直接作用于 encoder 特征，
阶段 2 才出现 "Prompt-guided Condition decoupling"。

阶段 1 的引导源（`F_t`）与阶段 2 的迁移方式（`L_dt` 把 `F_s` 对齐到同一空间）
论文未逐字说明，记录为 implementation choice；测试只走 `F_s`，无需条件标注。

---

## 3. 张量形状（B 张，640×512）

| 名称 | 形状 |
|---|---|
| 输入 `x` | `(B, 6, 512, 640)`，通道 0–2 为 BGR，3–5 为 IR |
| `F_rgb/F_ir` | P3 `(B,128,64,80)`，P4 `(B,256,32,40)`，P5 `(B,256,16,20)` |
| `F_s` / `F_v` | 与对应尺度同形状 |
| SCPL 池化输入 | `(B,256)` + `(B,256)` + `(B,512)` → MLP → `(B,6)` |
| `F_t` | `(B,512)` |
| CDF 引导 `g` | `(B,256)` |
| 融合特征 | P3 `(B,256,64,80)`，P4 `(B,512,32,40)`，P5 `(B,512,16,20)` |
| Neck 输出 | `(B,128,64,80)`、`(B,256,32,40)`、`(B,256,16,20)` |
| Head 输出（anchor） | `(B, 3×(5+1+11)=51, 20160)` |

20160 = 3 个 anchor × (64·80 + 32·40 + 16·20) 个格子；每格 51 维 =
`tx, ty, tw, th, tθ, obj, cls_0..cls_10`。解码后供 NMS 的张量是
`(B, 4+nc+1=16, 20160)`。

---

## 4. 参数量（默认：YOLOv5s + anchor 头）

| 部分 | 参数量 |
|---|---:|
| RGB encoder | 4.303 M |
| IR encoder | 4.303 M |
| S + V_rgb + V_ir | 4.579 M |
| SCPL gate | 0.264 M |
| text adapter | 0.197 M |
| CDF | 0.116 M |
| PAN neck | 2.247 M |
| anchor OBB head | 0.033 M |
| **合计** | **16.04 M** |
| 阶段 2 可训练（冻结 SCPL + adapter 后） | 6.98 M |
| CLIP ViT-B/16 文本编码器（冻结，不入 state_dict） | 63.17 M |

baseline（阶段 0）只含两个 encoder + neck + head = **10.89 M**。

实测（RTX 4060 Laptop 8 GB，AMP，640×512）：阶段 1 batch 4 峰值 0.62 GB，
阶段 2 batch 4（含两个辅助 head 前向）峰值 **2.90 GB**。batch 16 需要 24 GB
级显卡。

---

## 5. 论文已写明 vs 本实现的补齐

### 5.1 论文明确给出

- 双流编码器、三支解耦、CDF 软门控、SCPL 硬门控（Fig. 4）。
- 提示模板、`τ=0.15`、`λ=0.01/0.003/0.01`、CMD 5 阶、阶段 1/2 各 50 epochs。
- SGD，lr 0.01，momentum 0.937，weight decay 0.0005，640×512，batch 16。
  论文只声明 SGD，未提 Nesterov 与 warmup，因此 `nesterov=False`、
  `warmup_epochs=0`，保持框架默认值而非自行加料。
- CLIP ViT-B/16 文本编码器冻结；11 类、IR 标签；mAP50。
- 条件合并规则：高度 120 m、角度 30°、时间三段；天气/光照按 B.2 合并。
- 三阶段协议：baseline（Table 4，58.4%）→ 阶段 1 只训 SCPL+CDF（B.3，
  主干从 baseline 冻结）→ 阶段 2 full pipeline；baseline 训练轮数论文未给，
  按同配置 50 epochs。

### 5.2 论文未写明，本实现的明确选择

| 项目 | 选择 | 影响 |
|---|---|---|
| 检测器 family | YOLOv5s backbone + 自建 PAN + anchor OBB head | 无法与论文 63.1 直接横比 |
| 融合尺度 | P3/P4/P5 三层 | 论文未说明；单尺度会损失小目标 |
| `S/V` 内部结构 | 1×1 + 3×3 Conv，通道保持 | 论文只称 “encoder” |
| `F(·,·)` 融合函数 | 通道拼接 + 1×1 Conv | 论文未说明 |
| Neck | 一层 top-down + 一层 bottom-up PAN | 论文图中未画 neck |
| CDF 投影 | 每尺度 2 层 MLP，输入引导维 | Eq. 10 只写“非线性投影” |
| 阶段 1 引导 | `F_t` 经 text adapter | 补充 B.3 未说明阶段 1 引导源 |
| 门控梯度 | straight-through estimator | 硬掩码本身不可微 |
| `L_dt` 投影 | `F_s` 与 `F_t` 都投到引导空间 | 两者原始维度不同（256 vs 512） |
| `L_irr` 归一化 | 除以 `C·H·W` | 原式未归一化，直接相加会尺度爆炸 |
| `L_dc` | 共享主 head，含角度损失 | 论文 Eq. 8 写 cls/reg/obj；OBB 需角度 |
| 文本嵌入 | `CLIPTextModel` + 官方 `text_projection` | 与 CLIP `get_text_features` 对齐 |
| 增强 | 仿射 + 水平翻转；无 mosaic / HSV | mosaic 强制方形画布，HSV 会改 IR |
| AMP / clip | 开启 AMP、grad clip 10；warmup 关闭、`nesterov=False` | 论文未说明，取框架默认值 |
| 条件码本 | 天气/光照/场景保留匿名整数类 | 官方 codebook 未公开，不臆造语义 |

### 5.4 检测头对齐（式 8 的 L_obj）

论文式 8 写的是 `L_cls + L_reg + L_obj`，objectness 是 anchor-based 检测器的标志
（YOLOv8/v11 的 anchor-free 头没有这一项）。结合以下证据，本项目提供两套头：

- 一作 `supercpy` 公开 fork 了 **ultralytics/yolov5 v7.0**（跨仓库 compare：
  `ahead_by=0, files=[]`，即纯 fork、无改动），说明其代码基线是 v5；
- Sec. 5.1 的 momentum 0.937 / weight decay 0.0005 / lr 0.01 正是 Ultralytics
  默认配方（v5 的 `hyp.scratch-low.yaml`）；
- 对比表放 YOLOv5s 而非 v8/v11，实验框架属于 v5 时代。

| | `head_type: anchor`（默认，对齐式 8） | `head_type: v8`（YOLO11 系对照） |
|---|---|---|
| 结构 | 1×1 卷积 → 3 anchor × (5 + 1 + nc) | Ultralytics `OBB`：anchor-free、DFL(reg_max=16) + 角度 |
| objectness | **有**（bias 先验 0.02） | 无 |
| 分配 | 中心格 + 四邻域 + 宽高比阈值 0.25 | TAL |
| 损失 | `L_obj + L_cls + (1 − ProbIoU)` | `v8OBBLoss`（cls/box/dfl/angle） |
| 锚框 | 训练集 128,935 框 k-means，见 `model.anchors` | — |

`configs/pcdf.yaml` 默认把骨架与这个头绑定（v5s 4.303 M/分支 + anchor 头，
总参数 16.04 M，baseline 10.89 M）。式 9 的三项权重仍为 0.01 / 0.003 / 0.01，
`L_dc` 包含 objectness——三个分支共用同一个头，因此 `L_obj` 也会回传到
`F_v,rgb` 与 `F_v,ir`。

切换骨架只需改配置（`profiles/yolo11s.yaml`）：`backbone_yaml / backbone_scale /
widths / guidance_dim / head_type / encoder_weights`，代码无需改动。

### 5.3 已知不可复现项

1. **官方检测器结构、neck、OBB head 实现未公开**，因此不承诺 63.1 mAP50。
2. **天气/光照/场景的官方 code→name 映射未公开**，默认使用
   `weather type k` / `illumination level k` / `scenario k`。替换
   `configs/conditions.yaml` 中的 `codes/names` 即可接入官方码本；高度/角度/时间
   已按 B.2 合并。
3. 论文的 batch 16、三阶段共 150 epochs 在 8 GB 本地无法完整执行；
   配置保留论文值，小显存请用 `--batch` / `--imgsz` 覆盖。
4. 论文正文说 `w/o PCD` 下降 1.2，Table 4 是 63.1 vs 62.1（1.0），本仓库不裁决
   该差异，也不把它当作复现目标。

---

## 6. 数据管线

`src/pcdf/data.py::PairedRGBIRDataset` 继承 Ultralytics `YOLODataset`：

1. `img_path` 指向 `data/atr_umod/ir/images/{split}`，标签自动取
   `ir/labels/{split}`（论文用 IR 标签训练多模态检测器）。
2. `_load_source()` 读取 IR 与同 stem 的 RGB（把路径中的 `ir` 替换为 `rgb`），
   检查尺寸一致后拼成 6 通道。
3. `build_transforms()` 使用矩形 `imgsz=(H,W)` 的仿射、水平翻转、LetterBox；
   最后追加 `Format(return_obb=True)`，输出归一化 `xywhr`。
4. `__getitem__` 附加 `cond`（6 个整数 id）；`collate_fn` 将其堆叠为 `(B,6)`。

`configs/conditions.yaml` 是唯一的条件语义来源；`conditions.py` 负责
阈值/时间段/identity 映射、提示拼接与掩码。

数据划分：`train 10054 / val 1796 / test 1503`，val 由 train 内按
`location + 日期` 的场景组留出，避免同场景泄漏。

---

## 7. 评估

`src/pcdf/evaluate.py`：

- 模型 `eval()`，只走视觉引导，不需要条件。
- OBB head 解码 → `non_max_suppression(rotated=True, conf=0.001, iou=0.7)`。
- GT 为归一化 `xywhr`，按 letterbox 后的 `(W,H)` 还原。
- 逐类按置信度贪心匹配，`batch_probiou` 计算旋转 IoU；101 点插值 AP。
- 默认报告 mAP50；`--map50-95` 报告 0.5:0.05:0.95 平均。

---

## 8. 文件职责

```text
src/pcdf/
  constants.py     类别顺序、stride、默认锚框、log.csv 列（跨模块契约）
  paths.py         项目根、环境变量名、运行目录命名（唯一来源）
  config.py        类型化配置：YAML 合并、点号覆盖、交叉校验、路径归一
  runtime.py       设备解析、随机种子、环境快照、原子写 JSON
  checkpoint.py    检查点读写、续训调和（结构校验 + 配方沿用）
  train.py         Trainer：epoch 循环、验证、周期保存、信号处理
  report.py        运行目录巡检（只读 JSON，不依赖 torch）
  preflight.py     权重/数据就绪检查（不依赖 torch）
  conditions.py    六属性、合并规则、提示模板、掩码、按划分取样本
  text_encoder.py  冻结 CLIP、text_projection、提示缓存
  data.py          配对 RGB/IR 数据集、6 通道、条件 id、两个 loader 工厂
  losses.py        CMD / L_irr / L_dec / 检测损失装配
  evaluate.py      probIoU + 旋转 NMS + AP
  download.py      断点续传下载（CLIP 与骨架权重共用）
  models/
    encoder.py     双流 backbone（YOLOv5 / YOLO11 两族）+ 档位校验
    heads.py       检测头工厂 + 统一初始化顺序
    anchor_head.py 式 8 的 anchor 头与损失（增益来自配置）
    blocks.py      SCPLGate / DecoupleBranch / CDF / PAN / 引导投影
    scpl.py        SCPL 前向与 straight-through
    baseline.py    双流 concat baseline（阶段 0）
    pcdf.py        整体网络（阶段 1/2 与测试）
  cli/             train / evaluate / smoke / overfit / check_env /
                   prepare_data / plot_curves / fetch_clip / fetch_backbone
scripts/           同名薄启动器（保证 `python scripts/train.py` 仍可用）
scripts/server/    setup / prepare_data / smoke / train / train_all / eval / status
configs/           pcdf.yaml（基础）+ conditions.yaml（码本）+ profiles/（差异档）
tests/             83 项单元测试
```

**分层原则**：`paths` 只回答"东西在哪"，`config` 只回答"用什么参数"，
`runtime` 只回答"跑在什么设备上"，`checkpoint` 只回答"从哪继续"，
`train` 只回答"怎么跑一轮"。任何一层都不重复实现另一层的知识。

---

## 9. 运行

```bash
pip install -e .
python -m pcdf.cli.check_env
python -m pcdf.cli.smoke --device 0          # 三阶段端到端
python -m pytest -q                          # 单元测试

python -m pcdf.cli.train --stage 0 --device 0   # baseline（对照论文 58.4）
python -m pcdf.cli.train --stage 1 --device 0   # 从 baseline 初始化，主干冻结
python -m pcdf.cli.train --stage 2 --device 0   # 从阶段 1 初始化
python -m pcdf.cli.evaluate --checkpoint runs/pcdf_stage2/best.pt --split val
```

小显存 / 无卡：

```bash
python -m pcdf.cli.train --stage 1 --batch 2 --imgsz 320 256 \
    --max-batches 2 --val-batches 1 --device cpu --profile cpu
```

### 9.1 断点续训设计

检查点用"临时文件 + `os.replace`"原子写入，内容包含：

| 字段 | 作用 |
|---|---|
| `model` | 网络权重（不含冻结 CLIP） |
| `optimizer` / `scheduler` / `scaler` | 恢复 SGD 动量、指数衰减和 AMP 状态 |
| `next_epoch` / `step_in_epoch` | 恢复位置；mid-epoch 时 `next_epoch == epoch` |
| `global_step` | 继续 warmup 计数 |
| `epochs_total` | 本次运行的总轮数（`--epochs` 覆盖后也准确） |
| `best_map50` / `history` | 恢复 best 选择与日志历史 |
| `rng` | Python / NumPy / Torch / CUDA 随机数状态 |
| `cfg` / `stage` / `arch` | 配置快照、阶段与架构，用于续训校验与评估重建 |

保存时机：

1. 每个 epoch 结束、**验证之前**先写一次 `last.pt`（保护权重，防验证阶段被抢占）；
2. 验证后把 `map50` 写入同一个 metrics 字典，再写一次 `last.pt`（记录完整）；
3. 提升时写 `best.pt`，按周期写 `checkpoints/epoch_XXXX.pt` 并按 `keep_checkpoints` 修剪；
4. 训练中每 `save_every_minutes` / `save_every_steps` 写 mid-epoch `last.pt`；
5. `SIGTERM`/`SIGINT` 置停止标志，当前 step 结束后写 `last.pt` 退出；
6. 未捕获异常写 `crash.pt` 并写 `summary.json(status=error)`。

恢复规则（`checkpoint.plan_resume`，单一实现）：

| 类别 | 字段 | 规则 |
|---|---|---|
| 结构 | `widths` / `head_type` / `backbone_*` / `guidance_dim` / `data.names` | 必须一致，否则报错并提示训练时档位 |
| 配方 | `batch` / `lr` / `warmup_*` / `grad_accum` / `amp` | 以检查点为准 |
| 轮数 | `epochs_total` | 默认沿用；`--epochs` 显式覆盖 |
| 环境 | `imgsz` | 不一致只警告（通常不该改） |

恢复时用 `seed + epoch` 重建 DataLoader 并跳过 `step_in_epoch` 个 batch，
保证同一 epoch 的样本顺序一致。`--resume auto` 读取运行目录的 `last.pt`；
也支持直接指向 `checkpoints/epoch_XXXX.pt`（运行目录取其上两级）。

---

## 10. 与论文数值的关系

本仓库是**方法级复现**，不是官方代码复现：

- 结构、公式、损失权重、三阶段流程、数据条件协议均按论文实现；
- detector family、neck、`S/V` 内部结构、条件码本等未公开项按第 5 节补齐；
- 因此不把本仓库的 mAP 与论文 63.1 直接比较，也不承诺复现该数值。

若需要论文级数值对照，必须先获得官方 detector/codebook，或在同一 backbone 上
重训所有 baseline；这超出当前代码范围。
