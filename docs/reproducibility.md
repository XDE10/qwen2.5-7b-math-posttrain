# 环境安装与复现指南

本项目支持在**任意单卡 Linux GPU 服务器（推荐 ≥80 GB 显存，如 RTX PRO 6000 96 GB / H100 80 GB）**上直接运行，同时也提供可选的 Google Colab CLI 远程调度脚本。

---

## 1. 快速安装（CPU / 开发验证环境）

```bash
git clone https://github.com/XDE10/qwen2.5-7b-math-posttrain.git
cd qwen2.5-7b-math-posttrain

# 安装主包与测试依赖（Python >= 3.10）
pip install -e ".[dev,data]"

# 自检安装状态、配置文件与依赖版本（无需 GPU）
math-posttrain doctor

# 运行全部核心单元测试
pytest -q
```

### 按阶段安装 GPU 环境（推荐为训练 / 推理 / RL 分离虚拟环境）

由于 `vllm`、`trl` 与系统 `deepspeed` 对 `torch`/`transformers` 版本有不同约束，建议分环境安装（锁定版本见 [`requirements/`](../requirements/)）：

| 阶段 | 推荐安装方式 | 说明 |
|---|---|---|
| 数据准备 / 离线去污 | `pip install -e ".[data]"` | CPU 即可运行 |
| 离线评分 / 制图 | `pip install -e ".[data]" && pip install "math-verify[antlr4_13_2]==0.9.0" matplotlib` | CPU 即可运行 |
| 全参 SFT / QFFT 训练 | 保留系统 CUDA 匹配的 `torch`，再运行 `pip install -r requirements/train.txt && pip install -e . --no-deps` | 需要单卡 GPU + ≥128 GiB 主机内存（ZeRO-2 CPU Offload） |
| vLLM 推理评测 | `uv venv .venv-eval && uv pip install --python .venv-eval/bin/python -r requirements/eval.txt -e . --no-deps` | `vllm==0.33.0`, `transformers==5.20.2` |
| LoRA-GRPO 训练 | `uv venv .venv-rl && uv pip install --python .venv-rl/bin/python -r requirements/rl.txt -e . --no-deps` | `vllm==0.32.0`, `trl==1.17.1`, `peft==0.21.2` |
| MMR-DR-GRPO 扩展 | 在 RL 环境中追加 `pip install -r requirements/rl_mmr.txt` | CPU ONNX 编码 Jina 向量 |

---

## 2. 工作区路径配置（零 Colab 依赖）

默认情况下（未设置环境变量且不在 Colab 中）：
- **模型、检查点与评测输出**自动保存在仓库内 `./outputs/`（已加入 `.gitignore`）；
- **本地缓存与临时目录**自动保存在 `./.cache/math_posttrain/`；
- 可通过环境变量覆盖：
  - `MATH_POSTTRAIN_WORK_DIR`（或 `P1_DRIVE_ROOT`）：持久化输出根目录；
  - `MATH_POSTTRAIN_CACHE`（或 `P1_LOCAL_ROOT`）：高速本地 SSD 缓存根目录；
  - `MATH_POSTTRAIN_CONFIG_DIR`：自定义配置目录。

---

## 3. 端到端运行命令

### 3.1 下载评测集（按固定 Hugging Face revision 与 SHA-256）

```bash
math-posttrain fetch-benchmarks --sources
# 或使用一键脚本：
bash scripts/prepare_data.sh --benchmarks-only
```

下载后的 5 套评测集保存在 `data/benchmarks_v93/`（`aime24.jsonl`、`aime25.jsonl`、`amc23.jsonl`、`math500.jsonl`、`gsm8k.jsonl`）。

### 3.2 构建 1,000 题 SFT / QFFT 训练数据

```bash
math-posttrain build-data plan
math-posttrain build-data status
# 如需调用 DeepSeek API 补齐教师轨迹（需配置 DEEPSEEK_API_KEY）：
# math-posttrain fill-teacher fill --fallback-official
math-posttrain build-data build
```

### 3.3 全参 QFFT / SFT 训练（单卡）

```bash
# 主线正向复现配置（R1 1,000 题，16K，effective batch 8，6 epochs = 750 steps）
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m math_posttrain.train_sft \
  --config train/qfft_r1_16k_b8.json \
  --data data/qfft1000_v93/r1/train.jsonl

# 32K 不截断 QFFT / SFT 对照配置
# --config train/qfft_r1_32k_b16.json
# --config train/sft_r1_32k_b16.json
# --config train/qfft_ds_32k_b16.json
```

### 3.4 LoRA-GRPO / MMR-DR-GRPO 训练

```bash
# DAPO 16K + token_mask 正式配置
math-posttrain train-grpo --config grpo/grpo_dapo_16k.json

# MMR-DR-GRPO 50 步配置（先 CPU 校验配置与题池，再上卡）
math-posttrain train-grpo --config grpo/grpo_mmr_dr50.json --check
math-posttrain train-grpo --config grpo/grpo_mmr_dr50.json
```

### 3.5 vLLM 生成与离线评分（支持断点续跑与按题配对检验）

```bash
# 生成五项评测（支持注册表别名 base / qfft_official / sft_repro / qwen_math，或 run:<run_name>）
math-posttrain generate --model base --tag base --suite full
math-posttrain generate --model run:qfft_r1_16k_b8_v93 --tag qfft_r1 --suite full

# CPU 离线多进程判分与按题配对 bootstrap CI
math-posttrain score \
  --tags qfft_r1,base \
  --compare qfft_r1:base \
  --per-problem

# 生成 Pareto / 长度-难度分层 / pass@k 图表
math-posttrain analyze --tags base,qfft_r1 --out outputs/figures
```

---

## 4. 可选：Google Colab CLI 远程调度

如需在 Linux/WSL2 上通过 `google-colab-cli` + `rclone` 调度远程 Colab GPU：
1. 参考 [`docs/credential_templates/README.md`](credential_templates/README.md) 配置本地凭据（**严禁提交进 Git**）；
2. 运行：
   ```bash
   bash scripts/colab_launch.sh --gpu cpu ping:30
   bash scripts/colab_pipeline.sh 0   # 下载固定 revision 模型到 Drive
   bash scripts/colab_pipeline.sh 2   # 跑四锚点评测
   ```
