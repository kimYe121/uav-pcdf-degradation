# PCDF 方法级复现（UAV RGB-IR 旋转目标检测）

**PCDF: Prompt-guided Condition-aware Dynamic Fusion**，ICCV 2025，ATR-UMOD 数据集。
本仓库按论文正文与补充材料实现 SCPL、PCD、CDF 三个模块、三阶段训练、配对
RGB/IR 数据管线和 OBB 评估。

> **状态**：单元测试 83 项通过；三阶段端到端在 CPU 与 GPU（RTX 4060 8 GB，AMP）
> 均已跑通。这不是官方代码复现：论文未公开检测器结构、neck、条件码本等细节，
> 因此**不承诺复现 63.1 mAP50**，也不把本仓库数值与论文表格直接横比。
> 结构逐条对应见 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**；
> 远端开发/部署照 **[docs/WORKFLOW.md](docs/WORKFLOW.md)** 操作。

## 论文方法一句话

RGB/IR 双流编码器提取单模态特征；SCPL 用 CLIP 编码六类成像条件并做样本级硬门控；
PCD 把特征解耦为条件特定 `F_s` 与条件不变 `F_v`，用 CMD、交叉协方差和辅助检测
损失约束；CDF 由条件引导逐通道生成两模态权重，重加权 `F_v` 后拼接送入 OBB head。
测试阶段不需要条件标注。

## 目录结构

```text
.
├── configs/
│   ├── pcdf.yaml               # 唯一基础配置：paths / data / model / train
│   ├── conditions.yaml         # 六属性码本、B.2 合并规则、提示模板
│   └── profiles/               # 只写差异项的档位
│       ├── paper.yaml          # 论文 Sec. 5.1（batch16 / lr0.01 / 无 warmup）
│       ├── 48g.yaml            # 工程加速（batch32 / lr0.02 / warmup1）
│       ├── debug.yaml          # 短跑排查
│       ├── cpu.yaml            # 无卡验证（320×256 / batch2 / 关 AMP）
│       └── yolo11s.yaml        # 骨架容量对照（需自备 yolo11s 权重）
├── docs/
│   ├── ARCHITECTURE.md         # 论文-代码逐条对应、张量形状、保真边界
│   └── WORKFLOW.md             # 本地 → 服务器：同步/权重重装/训练/续训/排障
├── scripts/                    # 薄启动器，实现都在 src/pcdf/cli
│   ├── train.py  evaluate.py  smoke.py  overfit.py
│   ├── check_env.py  prepare_atr.py  plot_curves.py
│   └── server/                 # 服务器运维脚本（见下表）
├── src/pcdf/
│   ├── constants.py            # 类别顺序、stride、默认锚框、日志列
│   ├── paths.py                # 项目根、环境变量、运行目录命名（唯一来源）
│   ├── config.py               # 类型化配置：合并/校验/覆盖/路径归一
│   ├── runtime.py              # 设备解析、随机种子、环境快照、原子写
│   ├── checkpoint.py           # 检查点读写与续训调和
│   ├── train.py                # Trainer：epoch 循环、验证、落盘
│   ├── report.py               # 运行目录巡检（只读 JSON，不依赖 torch）
│   ├── preflight.py            # 权重/数据就绪检查（不依赖 torch）
│   ├── conditions.py           # 条件 id、提示构造、掩码
│   ├── text_encoder.py         # 冻结 CLIP ViT-B/16 + 提示缓存
│   ├── data.py                 # 6 通道配对数据集与 DataLoader
│   ├── losses.py               # CMD / L_irr / L_dec / 检测损失装配
│   ├── evaluate.py             # probIoU + 旋转 NMS + AP
│   ├── models/
│   │   ├── encoder.py          # 双流 backbone（YOLOv5 / YOLO11 两族）
│   │   ├── heads.py            # 检测头工厂 + 统一初始化顺序
│   │   ├── anchor_head.py      # 式 8 的 anchor 头与损失
│   │   ├── blocks.py           # SCPLGate / S,V / CDF / PAN / 引导投影
│   │   ├── scpl.py             # SCPL 前向 + straight-through
│   │   ├── baseline.py         # 双流 concat baseline（阶段 0）
│   │   └── pcdf.py             # 完整网络
│   └── cli/                    # 所有命令行入口（python -m pcdf.cli.X）
├── tests/                      # 83 项单元测试
├── weights/                    # 预训练权重（gitignore）
├── data/                       # ATR-UMOD（gitignore）
└── runs/                       # 训练输出（gitignore）
```

`archive/` 是早期“模态衰减”实验的历史代码，不参与当前复现；`output/` 是课程
选题说明，不属于代码。

## 安装与权重

Python ≥ 3.10，PyTorch 2.x。

```bash
pip install -e .
python -m pcdf.cli.check_env          # 环境/依赖/权重/数据/提示自检
```

权重只有两项，路径由配置推导，不靠手工约定：

```text
weights/yolov5su.pt                 # 骨架（默认配置；Ultralytics 资产）
weights/clip-vit-base-patch16/      # CLIP 文本编码器
  ├── config.json  vocab.json  merges.txt
  ├── tokenizer_config.json  special_tokens_map.json
  ├── pytorch_model.bin
  └── text_projection.pt            # 下载脚本自动生成
```

一条命令装齐（ModelScope 优先，失败自动回退 hf-mirror / HF）：

```bash
python -m pcdf.cli.fetch_backbone    # 下载并按 backbone_scale 校验档位
python -m pcdf.cli.fetch_clip        # 下载 CLIP 并抽取 text_projection
```

`fetch_backbone` 会反查权重的真实规模（n/s/m/l/x）与 `model.backbone_scale`
比对，不一致直接删除并非零退出——避免"配置写 s、权重是 n"这类隐蔽事故。

## 数据

```text
data/atr_umod/
├── ir/images/{train,val,test}/*.jpg
├── ir/labels/{train,val,test}/*.txt      # YOLO-OBB，8 坐标
├── rgb/images/{train,val,test}/*.jpg
└── condition_index.csv
```

从原始数据重建（train 10054 / val 1796 / test 1503，val 按地点+日期场景组留出）：

```bash
python -m pcdf.cli.prepare_data check   --root data/ATR-UMOD
python -m pcdf.cli.prepare_data convert --root data/ATR-UMOD --out data/atr_umod --val-frac 0.15
```

## 快速开始

```bash
python -m pytest -q                                     # 83 项单元测试
python -m pcdf.cli.check_env                            # 自检
python -m pcdf.cli.smoke --device cpu                    # 无卡也能跑通三阶段
python -m pcdf.cli.smoke --device 0                      # 有卡（默认 auto）
```

冒烟依次执行：baseline → 阶段 1（文本引导）→ 阶段 2（视觉引导）→ 旋转 mAP
评估，末尾打印 `[smoke] OK`，默认清理 `runs/_smoke`。

## 服务器运行（智川云 / Linux 容器）

完整步骤见 **[docs/WORKFLOW.md](docs/WORKFLOW.md)**，这里只列脚本职责：

| 脚本 | 用途 |
|---|---|
| `setup.sh` | 安装 editable、补依赖、下载/校验权重、跑自检；`--force-weights` 重装权重 |
| `prepare_data.sh` | 原始 ATR-UMOD → YOLO-OBB + `condition_index.csv` |
| `smoke.sh` | `pytest` + 三阶段端到端；`--bench [--batch N]` 追加 50 step 基准 |
| `train.sh` | 单阶段训练，自动续训：`baseline` / `stage1` / `stage2` / `resume0..2`；`--dry-run` |
| `train_all.sh` | 三阶段一键，未跑满不会误开下一阶段 |
| `eval.sh` | val/test mAP，默认取阶段 2 的 `best.pt` |
| `status.sh` | GPU、磁盘、三阶段进度、最新日志；`--json` 机器可读 |

```bash
bash scripts/server/setup.sh
bash scripts/server/prepare_data.sh --raw /root/rivermind-data/ATR-UMOD
bash scripts/server/smoke.sh --bench --batch 8
tmux new -s pcdf 'bash scripts/server/train_all.sh --profile paper'
bash scripts/server/status.sh
bash scripts/server/eval.sh --split test --map50-95
```

**所有路径与设备都可环境变量覆盖**，脚本里没有写死的绝对路径或超参：

```bash
PYTHON=/opt/conda/bin/python DEVICE=1 \
DATA_ROOT=/data/atr_umod WEIGHTS_DIR=/data/weights RUNS_DIR=/data/runs \
bash scripts/server/train.sh stage1 --profile 48g
```

变量清单：`PYTHON`、`DEVICE`、`PROFILE`、`RUN_NAME`、`DATA_ROOT`、`WEIGHTS_DIR`、
`RUNS_DIR`、`LOG_DIR`、`TEE`、`CLIP_SOURCE`、`PCDF_BACKBONE_URL`。

若存在 `/root/rivermind-data`（数据盘），`RUNS_DIR`/`LOG_DIR` 默认落到
`/root/rivermind-data/pcdf/{runs,logs}`，容器重建不丢检查点。

## 配置

三层合并，优先级从低到高：

```
configs/pcdf.yaml  <  configs/profiles/<档位>.yaml  <  --set / CLI 参数  <  路径环境变量
```

- **未知键直接报错**，YAML 拼写错误不会静默改变实验；
- **路径只解析一次**：加载后全部转绝对路径，从任意 cwd 启动都指向同一份资源；
- **交叉约束集中校验**：`guidance_dim == widths[2]`、`imgsz % 32 == 0`、
  `head_type ∈ {v8, anchor}`、锚框覆盖三个 stride 等，建模前就失败。

| 档位 | batch | lr | warmup | workers | 用途 |
|---|---:|---:|---:|---:|---|
| `paper` | 16 | 0.01 | 0 | 8 | 对齐论文 Sec. 5.1，用于最终报告 |
| `48g` | 32 | 0.02 | 1 | 12 | 工程加速；lr 线性缩放，不等同论文设置 |
| `debug` | 16 | 0.01 | 0 | 8 | 8 epoch、每轮验证，排查用 |
| `cpu` | 2 | 0.01 | 0 | 0 | 320×256、关 AMP，无卡验证 |

任意字段可临时覆盖，不必新建文件：

```bash
python -m pcdf.cli.train --stage 1 --set train.lr=0.005 --set data.workers=8
```

## 训练

论文协议：SGD，lr 0.01，momentum 0.937，weight decay 0.0005，640×512，batch 16，
阶段 1/2 各 50 epochs，指数衰减；baseline（阶段 0）论文未给轮数，按同配置 50。

```bash
python -m pcdf.cli.train --stage 0 --device 0    # baseline（Table 4，58.4）
python -m pcdf.cli.train --stage 1 --device 0    # SCPL+CDF，主干从 baseline 冻结
python -m pcdf.cli.train --stage 2 --device 0    # full pipeline，从阶段 1 初始化
```

输出：`runs/pcdf_stage{0,1,2}/{config.yaml,meta.json,log.csv,metrics.json,
last.pt,best.pt,checkpoints/}`。

### 断点续训

检查点保存**完整训练状态**：模型、优化器、调度器、AMP scaler、`next_epoch`、
`step_in_epoch`、`global_step`、`epochs_total`、best mAP、日志历史、随机数状态。

```bash
python -m pcdf.cli.train --stage 1 --resume auto
python -m pcdf.cli.train --stage 2 --resume runs/pcdf_stage2/checkpoints/epoch_0010.pt
```

续训规则是固定不变式（详见 `src/pcdf/checkpoint.py`）：

- **结构字段必须一致**（widths / head_type / backbone / guidance_dim / 类别名），
  否则报错并提示用训练时的 `--profile`；
- **训练配方以检查点为准**（batch / lr / warmup / grad_accum / amp），48G 档重启
  不会退回 batch16/lr0.01；
- **命令行显式参数优先**，并在日志里说明被覆盖的项。

恢复时用 `seed + epoch` 重建 DataLoader 并跳过已训练 step；`SIGTERM` 会在当前
step 结束后写 `last.pt` 退出；异常退出另写 `crash.pt`。周期保存按配置
（`save_every_minutes` / `save_every_steps`），不写在 shell 里。

## 评估

```bash
python -m pcdf.cli.evaluate --checkpoint runs/pcdf_stage2/best.pt --split val
python -m pcdf.cli.evaluate --checkpoint runs/pcdf_stage2/best.pt --split test --map50-95
```

模型结构取自检查点内的配置快照，数据/权重路径取当前环境，因此**换机器或换档位
评估不会报 shape 错**。输出 mAP50、逐类 AP50、预测/GT 数量，并写
`eval_{split}.json`。

## 关键参数

| 参数 | 值 | 来源 |
|---|---|---|
| 输入尺寸 | 640×512 | 论文 Sec. 5.1 |
| batch | 16 | 论文 Sec. 5.1（小显存覆盖） |
| epochs | 50 + 50 | 论文 Sec. 5.1 |
| 优化器 | SGD, lr 0.01, mom 0.937, wd 5e-4 | 论文 Sec. 5.1 |
| 骨架 | YOLOv5s backbone（4.303 M/分支） | 论文未公开，见 ARCHITECTURE 5.4 |
| 检测头 | anchor-based，含 objectness | 式 8 的 `L_obj` |
| 文本编码器 | CLIP ViT-B/16，冻结 | 论文 Sec. 5.1 |
| `τ` | 0.15 | 补充 Sec. G |
| `λ1/λ2/λ3` | 0.01 / 0.003 / 0.01 | 论文 Eq. 9 |
| 条件合并 | 高度 120 m、角度 30°、时间三段 | 补充 B.2 |

## 与论文的保真边界

已按论文实现：双流编码器、SCPL 提示与硬门控、三支解耦、CMD/`L_irr`/`L_dc`、
逐通道软门控、三阶段流程、条件合并规则、IR 标签、旋转框评估。

论文未公开、由本仓库补齐（详见 `docs/ARCHITECTURE.md` 第 5 节）：

- 检测器 family / neck / OBB head：YOLOv5s backbone + 自建 PAN + anchor OBB head；
- `S/V` 内部结构、`F(·,·)` 融合方式、CDF 投影层；
- 阶段 1 的引导来源（本实现用 `F_t`，阶段 2 用 `L_dt` 对齐到 `F_s`）；
- 硬门控的 straight-through 梯度；
- 天气/光照/场景的官方码本与 B.2 合并映射（默认匿名类）；
- 增强、AMP、warmup、梯度裁剪。

## 硬件实测

RTX 4060 Laptop 8 GB、AMP、640×512：

| 配置 | 峰值显存 |
|---|---|
| 阶段 1，batch 4 | 0.62 GB |
| 阶段 2，batch 4（含两个辅助 head 前向） | 2.90 GB |

阶段 1 的前向不经过 PCD（Fig. 10 Stage I），实现按需跳过三支解耦，省掉约
三分之一计算量。batch 16 建议 24 GB 级显卡；先用 `smoke.sh --bench --batch N`
确认峰值再决定。

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
