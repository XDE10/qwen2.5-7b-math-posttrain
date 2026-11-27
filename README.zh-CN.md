[English](README.md) | **简体中文**

# Qwen2.5-7B-Instruct 数学推理后训练（QFFT / SFT + LoRA-GRPO）

单卡 GPU 上完成的 `Qwen2.5-7B-Instruct` 数学推理后训练极简工程与实证研究，覆盖 **公开去污数据构建、单卡全参 QFFT/SFT（DeepSpeed ZeRO-2 CPU Offload + Liger）、单卡 LoRA-GRPO / MMR-DR-GRPO（TRL + 同卡 vLLM）、统一 32K 评测与按题配对统计检验**。

> **面向 AI Agent 自动部署与开发**：请优先阅读 [`AGENTS.md`](AGENTS.md)，内含环境矩阵、自检命令、标准部署剧本与仓库护栏规范。

---

## 公开模型权重（Hugging Face）

训练成品已发布至 Hugging Face Hub，可直接通过 `transformers` / `peft` / `vLLM` 或本仓库一键拉取运行：

| 模型 | Hugging Face 仓库 | 形式与体积 | 竞赛 100 题 | 说明 |
|---|---|---|---:|---|
| **`R1 QFFT-750`（主线最终模型）** | [`XDE10/Qwen2.5-7B-Math-QFFT-R1`](https://huggingface.co/XDE10/Qwen2.5-7B-Math-QFFT-R1) | 全量 bf16（14.18 GiB） | **33.63%** | 比底座 `+7.56 pp` 统计显著（rev `2afcc63`） |
| **`MMR-50` LoRA Adapter** | [`XDE10/Qwen2.5-7B-Math-QFFT-R1-MMR50-LoRA`](https://huggingface.co/XDE10/Qwen2.5-7B-Math-QFFT-R1-MMR50-LoRA) | LoRA Adapter（616 MiB） | **34.50%** | 挂载于主模型之上，5/5 评测最均衡 RL 节点（rev `ec6c8a5`） |

---

## 核心实测结果（统一 32K 原生口径，`mean@n`）

所有模型采用完全一致的 prompt token、原生 32K RoPE（无 YaRN）、`temperature=0.6`、`top_p=0.95`、`Math-Verify 0.9.0` 最终段末框判分（无 LLM 复核；每模型 4,599 条：AIME24/25 各 ×16、AMC23 ×8、MATH-500 ×4、GSM8K ×1）。

| 模型 | AIME24 (×16) | AIME25 (×16) | AMC23 (×8) | **竞赛 100 题** | MATH-500 (×4) | GSM8K (×1) | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen2.5-7B-Instruct`（原始底座） | 11.25% | 7.71% | 50.94% | **26.06%** | 76.80% | 91.58% | 固定 revision `a09a354` |
| `lwl-uestc/QFFT-S1-7B`（公开 QFFT 锚点） | 20.83% | 16.04% | 61.25% | **35.56%** | 81.10% | 92.27% | 同口径重测 |
| `PJiNH/QFFT-repro-S1.1-SFT-7B`（公开 SFT） | 18.75% | 16.88% | 61.25% | **35.19%** | 81.70% | 92.34% | 输出长约 2×（GSM8K 2,191 tok） |
| `Qwen/Qwen2.5-Math-7B-Instruct`（4K 原生） | 12.29% | 11.46% | 55.63% | **29.38%** | 83.40% | 95.30% | 数学专用参考 |
| **自训 `R1 QFFT-750`（主线模型）** | **18.54%** | **14.79%** | **59.06%** | **33.63%** | **80.50%** | **91.43%** | **`+7.56 pp` `[+3.88, +11.44]`** |
| **自训 `R1 + MMR-DR-GRPO 50 步`** | **18.75%** | **16.25%** | **60.00%** | **34.50%** | **80.35%** | **92.42%** | 比 QFFT `+0.88 pp` `[−1.38, +3.31]` |

### 核心结论速览
- **全参 QFFT 复现取得统计显著提升且保持高效推理**：仅用 1,000 条去污 DeepSeek-R1 长链轨迹做单卡全参 QFFT，竞赛 100 题由 `26.06% → 33.63%`（**`+7.56 pp`，95% CI `[+3.88, +11.44]`**），MATH-500 由 `76.80% → 80.50%`，同口径逼近官方 `QFFT-S1-7B`（差值 `−1.94 pp`，CI 跨零），同时保持简单题短推理（GSM8K 均值 476 token vs SFT 2,191 token）。
- **RL 阶段诚实结论与根因诊断**：
  - `MMR-50` 在全部 5 项上最均衡（竞赛 `34.50%`、GSM8K `92.42%`、AIME24/25 `maj@16` 分别提升 `+6.7 / +10.0 pp`），但相对 QFFT 的配对 95% CI 跨零，不宣称额外显著涨分；
  - 将同卡 vLLM 重要性采样由 `sequence_mask` 改为 `token_mask` 后，2026 独立竞赛 59 题留出集精度持平（`+0.21 pp`），**平均输出显著缩短 16.8%（−1,743.5 token）**；
  - 完整定位了长步 MMR-DR-GRPO（311 步）提分受阻根因：16K `dr_grpo` 固定分母稀释梯度 + 极小 LoRA LR `1e-6` + `beta=0.04` 使 KL 钉死在 `0.0022–0.0031`，叠加复合奖励导致全对组 145/181 条正确作答拿到负优势。

详细评测对照与技术复盘见 [`results/eval_summary.md`](results/eval_summary.md) 与 [`docs/experiment_report.md`](docs/experiment_report.md)。

---

## 快速开始（Clone 即一键运行）

> 所有路径均通过环境变量可配（默认读取 `./data/`，输出写入 `./outputs/`），零硬编码路径。国内网络可按需设置 `HF_ENDPOINT` 镜像。

### 1. 安装并自检（无需 GPU）

```bash
git clone https://github.com/XDE10/qwen2.5-7b-math-posttrain.git
cd qwen2.5-7b-math-posttrain

pip install -e ".[dev,data]"
math-posttrain doctor
pytest -q
```

### 2. 一键推理（直接调用已发布的 Hugging Face 模型）

```bash
pip install -e ".[infer]"

# 使用主线 R1 QFFT-750 模型推理
math-posttrain infer --prompt "Find all real values of x such that x^2 - 5x + 6 = 0."

# 叠加 MMR-50 LoRA Adapter 推理
math-posttrain infer --adapter mmr50_lora --prompt "Find the remainder when 2^100 is divided by 7."
```

### 3. 一键评测（自动下载 5 套固定 revision 评测集 + vLLM 生成 + Math-Verify 评分）

```bash
pip install -r requirements/eval.txt && pip install -e . --no-deps

# 评测已发布的主模型（支持注册表别名 / 任意 HF repo ID / 本地路径）
bash scripts/run_eval.sh qfft_r1 qfft_r1 full

# 无 GPU 时可用 fake 引擎 5 秒跑通完整评测与打分闭环：
bash scripts/run_eval.sh base smoke_demo smoke --engine fake

# 对两个评测标签做按题配对 bootstrap 统计检验：
math-posttrain score --tags qfft_r1,base --compare qfft_r1:base --per-problem
```

### 4. 一键训练（自动从 Hugging Face 拉取公开数据并执行 13-gram 去污）

```bash
# 4a. 单卡全参 QFFT（保留系统匹配 CUDA 的 PyTorch，推荐 ≥80 GB 显存 + ≥128 GB 内存）
pip install -r requirements/train.txt && pip install -e . --no-deps
bash scripts/run_train_qfft.sh                       # 默认 configs/train/qfft_r1.json

# 4b. 单卡 LoRA-GRPO / MMR-DR-GRPO（默认直接从 HF 公开的 qfft_r1 启动）
pip install -r requirements/rl.txt && pip install -e . --no-deps
bash scripts/run_train_grpo.sh                       # 默认 configs/grpo/mmr_dr50.json
bash scripts/run_train_grpo.sh grpo/dapo_16k.json    # 或 DAPO-GRPO 配置
```

---

## 目录结构

```text
qwen2.5-7b-math-posttrain/
├── AGENTS.md                      # 面向 AI Agent 的自动部署、运行与开发规范
├── README.md                      # 英文文档（主 README）
├── README.zh-CN.md                # 简体中文文档
├── pyproject.toml                 # 打包定义、可选依赖与 `math-posttrain` CLI
├── configs/
│   ├── eval.json                  # 固定 revision 模型注册表与统一 32K 评测套件
│   ├── deepspeed_zero2.json       # 单卡 7B 全参 ZeRO-2 CPU Offload 配置
│   ├── decontam_review.json       # s1K-1.1 与 5 套评测集去污人工复核清单
│   ├── train/                     # 全参 QFFT / SFT 配置（qfft_r1.json, sft_r1.json）
│   └── grpo/                      # LoRA-GRPO 配置（mmr_dr50.json, dapo_16k.json）
├── requirements/                  # 分阶段锁定依赖（train.txt, eval.txt, rl.txt）
├── scripts/                       # 一键评测与训练脚本（自动准备缺失数据）
├── src/math_posttrain/            # 核心源码
├── results/eval_summary.md        # 完整评测汇总、统计检验与长度对照
├── docs/experiment_report.md      # 技术复盘与 RL 深度根因诊断报告
└── tests/                         # 覆盖核心链路与 MMR 算法的高速单测
```

### 可选环境变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `MATH_POSTTRAIN_DATA_DIR` | `./data` | 评测集与训练集存放目录 |
| `MATH_POSTTRAIN_OUTPUT_DIR` | `./outputs` | 训练权重与评测结果输出目录 |
| `MATH_POSTTRAIN_CONFIG_DIR` | `./configs` | 自定义配置目录 |
| `HF_ENDPOINT` | `https://huggingface.co` | 可选 Hugging Face 镜像源地址 |
