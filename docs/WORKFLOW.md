# 远端开发与运行流程（本地 → 服务器）

本文是**唯一需要照着敲**的操作手册。所有路径都通过环境变量注入，脚本里没有
写死的绝对路径或超参；换机器只需要改环境变量。

```
约定
  本地仓库      <LOCAL>            例如 D:/CodeResources/DeepLearning/courseTest
  服务器代码    <REMOTE>           例如 /root/pcdf
  服务器数据盘  /root/rivermind-data（智川云等，可选）
```

---

## 0. 本地自查（30 秒，可选）

不需要 GPU，只验证"代码本身没坏"：

```bash
cd <LOCAL>
python -m pytest -q                        # 单元测试
python -m pcdf.preflight weights --quiet   # 权重清单（本地有权重时）
python -m pcdf.cli.smoke --device cpu --batch 2 --imgsz 320 256 --fraction 0.002
```

CPU 冒烟只证明"数据管线/前向/损失/检查点/评估能串起来"，**不产出精度结论**。
真正的验收在服务器 GPU 上（第 4 步）。

---

## 1. 推送仓库（本地）

```bash
cd <LOCAL>
git status
git add -A
git commit -m "refactor: 配置集中化 + 检查点/续训重构 + CLI/shell 去硬编码"
git push origin HEAD
```

`data/`、`weights/`、`runs/`、`logs/` 已在 `.gitignore` 里，不会被推上去。
如果还没配远端：

```bash
git remote add origin <你的仓库地址>
git branch -M main
git push -u origin main
```

---

## 2. 覆盖服务器代码（tar 打包 + scp + 远端解压）

服务器上代码和数据在**同一个目录**（`<REMOTE>/` 下同时有 `src/`、`data/`、
`weights/`），所以“删了再传”风险不对称：一次 exclude 写漏就删掉 3 GB 数据。
因此改成“打包 → 上传 → 远端先备份再覆盖”：

```powershell
# 本地 Windows PowerShell，默认取当前仓库根
pwsh scripts/deploy.ps1 -DryRun      # 只打包，看会带哪些文件
pwsh scripts/deploy.ps1              # 打包 + scp + 远端覆盖
pwsh scripts/deploy.ps1 -NoPrune     # 只覆盖，不清理陈旧文件
```

参数：`-SshHost`（形如 `user@your-host`）、`-Port`（SSH 端口）、`-RemoteDir`（远端仓库目录）。
默认值请在本地脚本里填写，不要写进仓库。

打包排除 `data/ weights/ runs/ logs/ archive/ output/ .git/ .workbuddy/
__pycache__/`，包体 0.1 MB 量级，秒级传完。

远端由 `scripts/server/sync_code.sh` 完成三件事：

1. 先把现有代码（`src configs scripts tests docs` + 顶层三份文件）打一份
   `$REMOTE/.backup/code_<时间戳>.tar.gz`，回滚只需 `tar xzf` 覆盖回去；
2. 解压到临时目录，**只对代码目录**做“新包没有的文件就删”的清理，
   `data/ weights/ runs/ logs/` 完全不参与；
3. 覆盖后清掉 `__pycache__`，避免“源码换了、字节码还是旧的”。

没有 PowerShell 的机器等价写法：

```bash
tar czf /tmp/pcdf_code.tar.gz --exclude=./data --exclude=./weights \
  --exclude=./runs --exclude=./logs --exclude=./archive --exclude=./output \
  --exclude=./.git --exclude=__pycache__ -C <LOCAL> .
scp -P <PORT> /tmp/pcdf_code.tar.gz user@<HOST>:/tmp/
scp -P <PORT> <LOCAL>/scripts/server/sync_code.sh user@<HOST>:/tmp/
ssh -p <PORT> user@<HOST> 'bash /tmp/sync_code.sh /tmp/pcdf_code.tar.gz <REMOTE>'
```

---
## 3. 权重：彻底重装（推荐，最省心）

服务器上出现"权重混乱"（档位不对、CLIP 缺文件、旧权重残留）时，直接删掉
重装，不要手工比对：

```bash
# 服务器上
cd <REMOTE>
export WEIGHTS_DIR=${WEIGHTS_DIR:-$PWD/weights}

# 1) 备份或直接删（确认没有别的项目在用这个目录）
mv "$WEIGHTS_DIR" "${WEIGHTS_DIR}_bak_$(date +%Y%m%d_%H%M%S)"

# 2) 一条命令重装：骨架权重 + CLIP + 完整性校验 + 环境自检
bash scripts/server/setup.sh
```

`setup.sh` 会按**配置**推导文件名，不依赖任何手工约定：

| 文件 | 来源 |
|---|---|
| `$WEIGHTS_DIR/yolov5su.pt` | `pcdf-fetch-backbone`（Ultralytics release，失败自动走 ghproxy 镜像） |
| `$WEIGHTS_DIR/clip-vit-base-patch16/` | `pcdf-fetch-clip`（ModelScope → hf-mirror → HF） |
| `…/text_projection.pt` | 下载后从 `pytorch_model.bin` 自动抽取 |

容器不通外网时：

```bash
# 本地打包权重，scp 上去，再只做校验
tar czf weights.tgz -C <LOCAL> weights
scp weights.tgz user@<HOST>:<REMOTE>/
ssh user@<HOST> "cd <REMOTE> && tar xzf weights.tgz && \
  bash scripts/server/setup.sh --check-only"
```

只想重新下载、不删目录：

```bash
bash scripts/server/setup.sh --force-weights
```

`--force-weights` 会重新拉骨架权重和 CLIP，并在下载后**反查真实规模**与
`model.backbone_scale` 比对；不一致会删掉坏文件并非零退出——这正是过去
"配置写 s、权重是 n"事故的防线。

---

## 4. 服务器验收（必须先过这一步再开长训练）

```bash
cd <REMOTE>
export PYTHON=/opt/conda/bin/python     # 换成你的解释器
export DEVICE=0                          # 有卡就写卡号；auto 也行

bash scripts/server/setup.sh             # 安装 + 权重 + 自检
bash scripts/server/smoke.sh             # pytest + 三阶段端到端（GPU）
bash scripts/server/smoke.sh --bench --batch 8   # 追加 50 step 速度/显存
```

`smoke.sh` 通过的标准输出：末尾出现 `[smoke] OK`，且进程退出码为 0。

---

## 5. 训练

```bash
# 三阶段一键：放进 screen，断线不影响
screen -dmS pcdf bash -lc 'cd <REMOTE> && bash scripts/server/train_all.sh --profile paper'
screen -ls                 # 看到 pcdf 在跑
screen -r pcdf             # 进去看实时输出；Ctrl-a d 退出但不停止

# 也可以先进 screen 再敲命令
screen -S pcdf
bash scripts/server/train_all.sh --profile paper
# Ctrl-a d 断开；screen -r pcdf 回来；screen -X -S pcdf quit 结束

# 或单阶段
bash scripts/server/train.sh baseline
bash scripts/server/train.sh stage1
bash scripts/server/train.sh stage2

# 只看命令不执行
bash scripts/server/train.sh stage1 --profile 48g --dry-run
```

阶段衔接是自动的：阶段 1 从 `runs/pcdf_stage0/best.pt` 初始化，阶段 2 从
`runs/pcdf_stage1/best.pt` 初始化（Python 侧按运行目录解析，脚本不拼路径）。

路径全部可覆盖：

```bash
WEIGHTS_DIR=/data/weights \
DATA_ROOT=/data/atr_umod \
RUNS_DIR=/data/pcdf/runs \
DEVICE=1 PYTHON=/opt/conda/bin/python \
bash scripts/server/train.sh stage1 --profile 48g
```

---

## 6. 监控与续训

```bash
bash scripts/server/status.sh          # GPU/磁盘/三阶段进度/最新日志
bash scripts/server/status.sh --json   # 机器可读
```

容器被抢占或手动停止后：

```bash
bash scripts/server/train.sh resume1   # 或直接重跑 stage1，会自动 --resume auto
```

续训规则（固定不变式）：

- **结构字段**（widths / backbone / guidance_dim / 类别名）必须与
  检查点一致，否则直接报错——不允许"形状对不上还硬跑"；
- **训练配方**（batch / lr / warmup / grad_accum / amp）以检查点为准，48G 档
  重启不会退回 batch16/lr0.01；
- **命令行显式参数**（`--batch` / `--epochs`）优先，并在日志里说明。

---

## 7. 评估

```bash
bash scripts/server/eval.sh                          # val / mAP50
bash scripts/server/eval.sh --split test --map50-95   # test / mAP50-95
bash scripts/server/eval.sh --checkpoint runs/pcdf_stage2/best.pt --max-batches 20
```

模型结构取自检查点内的配置快照，路径取当前环境——所以**换档位评估不会再报
shape 错**。结果写在检查点同目录的 `eval_<split>.json`。

画曲线：

```bash
python -m pcdf.cli.plot_curves --run-dir runs/pcdf_stage2
```

---

## 8. 排障

| 现象 | 原因 | 处理 |
|---|---|---|
| `请求了 CUDA 设备 'cuda:0'，但当前进程没有可用 GPU` | 卡被占满/平台隐藏了设备 | 换卡号，或 `--device cpu` 先跑通逻辑 |
| `检查点与当前配置的结构字段不一致` | 用了别的 profile 续训 | 用训练时的 `--profile`，或 `--init` 只加载权重 |
| `骨架权重 … 规模不匹配` | 权重档位与 `backbone_scale` 不符 | `setup.sh --force-weights` |
| 训练中途 `FloatingPointError: 非有限损失` | lr 过大 / 标签异常 | 先 `overfit.py` 小样本诊断 |
| `阶段 N 未跑满` | 被抢占或手动停止 | `train.sh resume<N>` |
| 显存不足 | batch/尺寸偏大 | `--batch 8` 或 `--imgsz 512 512`，先 `smoke.sh --bench` 测 |

诊断小样本能否过拟合（判断标签/损失/评估是否自洽）：

```bash
python -m pcdf.cli.overfit --images 32 --epochs 60 --batch 8 --device 0
python -m pcdf.cli.overfit --with-stage2 --device 0     # 额外验证 L_dt/L_irr/L_dc
```
