# Qwen2.5-7B-Instruct 数学推理后训练（QFFT / SFT + LoRA-GRPO）

单卡 96 GB GPU（RTX PRO 6000 / H100）上完成的 `Qwen2.5-7B-Instruct` 数学推理后训练工程与系统性实证研究，覆盖 **1,000 题去污长链数据构建、全参 QFFT / SFT（DeepSpeed ZeRO-2 CPU Offload + Liger）、单卡 LoRA-GRPO / MMR-DR-GRPO（TRL + 同卡 vLLM）、统一 32K 评测协议与按题配对统计检验**。

---

## 核心实测结果（统一 32K 原生口径，`mean@n`）

所有模型采用完全一致的 prompt token、原生 RoPE（无 YaRN）、`temperature=0.6`、`top_p=0.95`、`Math-Verify 0.9.0` 最终段末框判分（无 LLM 复核）；每模型共生成 4,599 条（AIME24/25 各 ×16、AMC23 ×8、MATH-500 ×4、GSM8K ×1）。

| 模型 | AIME24 (×16) | AIME25 (×16) | AMC23 (×8) | **竞赛 100 题** | MATH-500 (×4) | GSM8K (×1) | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen2.5-7B-Instruct`（原始底座） | 11.25% | 7.71% | 50.94% | **26.06%** | 76.80% | 91.58% | 固定 revision `a09a354` |
| `lwl-uestc/QFFT-S1-7B`（公开 QFFT 锚点） | 20.83% | 16.04% | 61.25% | **35.56%** | 81.10% | 92.27% | 同口径重测 |
| `PJiNH/QFFT-repro-S1.1-SFT-7B`（公开 SFT） | 18.75% | 16.88% | 61.25% | **35.19%** | 81.70% | 92.34% | 输出长约 2× |
| `Qwen2.5-Math-7B-Instruct`（4K 原生） | 12.29% | 11.46% | 55.63% | **29.38%** | 83.40% | 95.30% | 数学专用参考 |
| 自训 `Flash QFFT`（DS-V4.1-Flash） | 5.00% | 3.96% | 41.56% | **19.31%** | 67.00% | 88.70% | 截断率 44.6%（负结果） |
| **自训 `R1 QFFT-750`（主线最终模型）** | **18.54%** | **14.79%** | **59.06%** | **33.63%** | **80.50%** | **91.43%** | **比底座 +7.56 pp `[+3.88, +11.44]`** |
| **自训 `R1 + MMR-DR-GRPO 50 步`** | **18.75%** | **16.25%** | **60.00%** | **34.50%** | **80.35%** | **92.42%** | 比 QFFT +0.88 pp `[−1.38, +3.31]` |
| 自训 `R1 + MMR-DR-GRPO 250 步`（4/5 项） | 未完成 | 16.25% | 56.25% | — | 80.85% | 92.12% | AMC23 回落 −2.81 pp |

### 核心结论速览
1. **全参 QFFT 复现取得统计显著提升**：仅用 1,000 条去污 R1 长链轨迹做单卡全参 QFFT，竞赛 100 题由 `26.06% → 33.63%`（**`+7.56 pp`，95% CI `[+3.88, +11.44]`**），MATH-500 由 `76.80% → 80.50%`，同口径逼近官方 `QFFT-S1-7B`（差值 `−1.94 pp`，CI 跨零），同时保持简单题短推理（GSM8K 均长 476 token vs SFT 2,191 token）。
2. **RL 阶段诚实结论与核心诊断**：
   - 旧 50 步 `MMR-DR-GRPO` 在全部 5 项上最均衡（竞赛 `34.50%`、GSM8K `92.42%`、AIME24/25 的 `maj@16` 分别提升 `+6.7 / +10.0 pp`），但相对 `R1 QFFT-750` 的配对 95% CI 跨零，不宣称额外显著涨分；
   - `token_mask` 相比 `sequence_mask` 在 2026 独立竞赛 59 题上精度持平（`+0.21 pp`），**平均输出显著缩短 16.8%（−1,743.5 token）**；
   - 完整定位了新版 MMR-DR-GRPO 跑至 311 步仍难提分的根因：**`dr_grpo` 固定 16K 分母稀释梯度 + LoRA 极小 LR `1e-6` + `beta=0.04` 使 KL 全程钉死在 `0.0022–0.0031`**，叠加离线奖励重放发现复合奖励导致**全对组 145/181 条正确作答拿到负优势、全错组 20/60 条错误作答拿到正优势**。

详细分析见 [results/eval_summary.md](results/eval_summary.md) 与 [docs/experiment_report.md](docs/experiment_report.md)。

---

## 目录结构

```text
qwen2.5-7b-math-posttrain/
├── README.md
├── pyproject.toml                   # 标准打包配置、可选依赖与 `math-posttrain` CLI
├── configs/                         # 固定模型 revision、DeepSpeed 与正式 SFT/GRPO 配置
│   ├── pipeline.json
│   ├── deepspeed_zero2_offload.json
│   ├── decontam_review.json
│   ├── scoring_recovery.json
│   ├── train/                       # QFFT / SFT 全参配置
│   └── grpo/                        # DAPO / MMR-DR-GRPO 配置与审批清单
├── requirements/                    # 分环境锁定清单（train / eval / rl / rl_mmr）
├── src/math_posttrain/              # 主包（数据、模板、判分、指标、训练、RL、评测与可选 Colab 调度）
├── scripts/                         # 一键数据准备、评测与可选 Colab 启动脚本
├── results/                         # 完整评测汇总表与可审计的配对结果 JSON
├── docs/                            # 实验复盘报告、复现指南与脱敏凭据模板
└── tests/                           # 覆盖模板、判分、去污、断点和 GRPO/MMR 的核心测试
```

---

## 快速开始（Clone 即跑）

### 1. 安装并自检（无需 GPU）

```bash
git clone https://github.com/XDE10/qwen2.5-7b-math-posttrain.git
cd qwen2.5-7b-math-posttrain

pip install -e ".[dev,data]"
math-posttrain doctor
pytest -q
```

### 2. 下载 5 套固定版本数学评测集

```bash
math-posttrain fetch-benchmarks --sources
```

### 3. 单卡全参 QFFT 训练

```bash
# 安装训练依赖（保留系统匹配 CUDA 的 PyTorch）
pip install -r requirements/train.txt

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m math_posttrain.train_sft \
  --config train/qfft_r1_16k_b8.json \
  --data data/qfft1000_v93/r1/train.jsonl
```

### 4. 单卡 LoRA-GRPO 训练

```bash
pip install -r requirements/rl.txt -r requirements/rl_mmr.txt

# 离线检查配置与题池
math-posttrain train-grpo --config grpo/grpo_mmr_dr50.json --check
# 启动训练
math-posttrain train-grpo --config grpo/grpo_mmr_dr50.json
```

### 5. 统一 32K vLLM 评测与配对打分

```bash
pip install -r requirements/eval.txt

bash scripts/run_eval.sh base base full
# 两个模型配对对比：
math-posttrain score --tags qfft_r1,base --compare qfft_r1:base --per-problem
```

完整环境拆分、断点续跑与可选 Colab 调度说明见 [docs/reproducibility.md](docs/reproducibility.md)。
