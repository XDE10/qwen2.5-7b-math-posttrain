**English** | [简体中文](README.zh-CN.md)

# Qwen2.5-7B-Instruct Math Reasoning Post-Training (QFFT / SFT + LoRA-GRPO)

Minimalist, zero-hardcoding single-GPU mathematical reasoning post-training pipeline and empirical study for `Qwen2.5-7B-Instruct`, covering **decontaminated public dataset preparation, single-GPU full-parameter QFFT/SFT (DeepSpeed ZeRO-2 CPU Offload + Liger), single-GPU LoRA-GRPO / MMR-DR-GRPO (TRL + colocated vLLM), unified 32K evaluation, and per-problem paired bootstrap testing**.

> **For AI coding & deployment agents**: read [`AGENTS.md`](AGENTS.md) first for environment setup matrices, verification commands, deployment playbooks, and repository guardrails.

---

## Published Models (Hugging Face)

Trained weights are hosted on Hugging Face Hub and load directly via `transformers`, `peft`, `vLLM`, or the `math-posttrain` CLI:

| Model | Hugging Face Repository | Format / Size | Comp100 (`mean@n`) | Highlights |
|---|---|---|---:|---|
| **`R1 QFFT-750` (Primary)** | [`XDE10/Qwen2.5-7B-Math-QFFT-R1`](https://huggingface.co/XDE10/Qwen2.5-7B-Math-QFFT-R1) | Full bf16 (14.18 GiB) | **33.63%** | `+7.56 pp` `[+3.88, +11.44]` vs base (`rev: 2afcc63`) |
| **`MMR-50` LoRA Adapter** | [`XDE10/Qwen2.5-7B-Math-QFFT-R1-MMR50-LoRA`](https://huggingface.co/XDE10/Qwen2.5-7B-Math-QFFT-R1-MMR50-LoRA) | LoRA Adapter (616 MiB) | **34.50%** | Best balanced 5/5 RL checkpoint on `qfft_r1` (`rev: ec6c8a5`) |

---

## Key Benchmark Results (Unified 32K Native RoPE, `mean@n`)

All models share identical prompt tokens, native 32K RoPE (no YaRN), `temperature=0.6`, `top_p=0.95`, deterministic per-(problem, draw) seeds, and `Math-Verify 0.9.0` final-segment last-`\boxed{}` grading (no LLM judge; 4,599 generations per model: AIME24 ×16, AIME25 ×16, AMC23 ×8, MATH-500 ×4, GSM8K ×1):

| Model | AIME24 (×16) | AIME25 (×16) | AMC23 (×8) | **Comp100** | MATH-500 (×4) | GSM8K (×1) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen2.5-7B-Instruct` (`base`) | 11.25% | 7.71% | 50.94% | **26.06%** | 76.80% | 91.58% | Pinned rev `a09a354` |
| `lwl-uestc/QFFT-S1-7B` (`qfft_official`) | 20.83% | 16.04% | 61.25% | **35.56%** | 81.10% | 92.27% | Re-evaluated under same protocol |
| `PJiNH/QFFT-repro-S1.1-SFT-7B` (`sft_repro`) | 18.75% | 16.88% | 61.25% | **35.19%** | 81.70% | 92.34% | ~2× longer output (GSM8K 2,191 tok) |
| `Qwen/Qwen2.5-Math-7B-Instruct` (`qwen_math`, 4K) | 12.29% | 11.46% | 55.63% | **29.38%** | 83.40% | 95.30% | Math-specialized 4K reference |
| **`R1 QFFT-750` (Primary)** | **18.54%** | **14.79%** | **59.06%** | **33.63%** | **80.50%** | **91.43%** | **`+7.56 pp` `[+3.88, +11.44]` vs base** |
| **`R1 + MMR-DR-GRPO 50 steps`** | **18.75%** | **16.25%** | **60.00%** | **34.50%** | **80.35%** | **92.42%** | `+0.88 pp` `[−1.38, +3.31]` vs QFFT |

### Key Takeaways
- **Statistically significant QFFT gain with concise easy-problem reasoning**: Full-parameter QFFT on 1,000 decontaminated DeepSeek-R1 trajectories lifts Comp100 by **`+7.56 pp` (`[+3.88, +11.44]`)** and MATH-500 from `76.80% → 80.50%`, matching the official QFFT anchor within `−1.94 pp` (CI crosses 0) while keeping GSM8K responses concise (`476` tokens vs `2,191` tokens for SFT).
- **Honest RL findings & root-cause diagnostics**:
  - `MMR-50` is the most balanced complete 5-benchmark RL checkpoint (`34.50%` Comp100, `92.42%` GSM8K, `maj@16` +6.7 / +10.0 pp on AIME24/25), though its paired 95% CI vs QFFT crosses zero (`[−1.38, +3.31]`);
  - Switching colocated vLLM importance sampling from `sequence_mask` to `token_mask` keeps accuracy unchanged (`+0.21 pp`) on a 59-problem 2026 competition holdout while **shortening outputs by 16.8% (−1,743.5 tokens)**;
  - Detailed telemetry and offline rollout replay explain why extending MMR-DR-GRPO to 311 steps stalled (16K `dr_grpo` denominator dilution + tiny `1e-6` LoRA LR + `beta=0.04` pinning KL at `0.0022–0.0031`, compounded by MMR/cosine reward inversion giving negative advantages to 145/181 correct responses in all-correct groups).

See [`results/eval_summary.md`](results/eval_summary.md) and [`docs/experiment_report.md`](docs/experiment_report.md) for full tables and technical analysis.

---

## Quick Start (Clone & Run)

> All paths are configurable via environment variables (defaulting to `./data/` and `./outputs/`), with zero hardcoded local paths or cloud-specific coupling. Set `HF_ENDPOINT` if using a Hugging Face mirror.

### 1. Install & Self-Check (No GPU Required)

```bash
git clone https://github.com/XDE10/qwen2.5-7b-math-posttrain.git
cd qwen2.5-7b-math-posttrain

pip install -e ".[dev,data]"
math-posttrain doctor
pytest -q
```

### 2. One-Click Inference (Published Hugging Face Weights)

```bash
pip install -e ".[infer]"

# Primary R1 QFFT-750 checkpoint
math-posttrain infer --prompt "Find all real values of x such that x^2 - 5x + 6 = 0."

# With companion MMR-50 LoRA adapter
math-posttrain infer --adapter mmr50_lora --prompt "Find the remainder when 2^100 is divided by 7."
```

### 3. One-Click Evaluation (Auto-Fetch Benchmarks + vLLM 32K + Math-Verify)

```bash
pip install -r requirements/eval.txt && pip install -e . --no-deps

# Evaluate published checkpoint (accepts registry alias, any HF repo ID, or local path)
bash scripts/run_eval.sh qfft_r1 qfft_r1 full

# Zero-GPU end-to-end smoke check (5 seconds using built-in fake engine):
bash scripts/run_eval.sh base smoke_demo smoke --engine fake

# Paired per-problem bootstrap comparison between two evaluated tags:
math-posttrain score --tags qfft_r1,base --compare qfft_r1:base --per-problem
```

### 4. One-Click Single-GPU Training (Auto-Fetch & 13-Gram Decontaminate Data)

```bash
# 4a. Full-parameter QFFT (keep CUDA-matched PyTorch installed; >=80 GB VRAM + >=128 GB RAM)
pip install -r requirements/train.txt && pip install -e . --no-deps
bash scripts/run_train_qfft.sh                       # default: configs/train/qfft_r1.json

# 4b. LoRA-GRPO / MMR-DR-GRPO (initializes from published Hugging Face qfft_r1 by default)
pip install -r requirements/rl.txt && pip install -e . --no-deps
bash scripts/run_train_grpo.sh                       # default: configs/grpo/mmr_dr50.json
bash scripts/run_train_grpo.sh grpo/dapo_16k.json    # or DAPO-GRPO config
```

---

## Repository Structure

```text
qwen2.5-7b-math-posttrain/
├── AGENTS.md                      # Machine-readable guide for AI coding & deployment agents
├── README.md                      # English documentation
├── README.zh-CN.md                # Simplified Chinese documentation
├── pyproject.toml                 # Packaging metadata, optional extras, `math-posttrain` CLI
├── configs/
│   ├── eval.json                  # Pinned HF model registry & unified 32K evaluation suites
│   ├── deepspeed_zero2.json       # Single-GPU 7B ZeRO-2 CPU Offload config
│   ├── decontam_review.json       # Reviewed 13-gram decontamination decisions for s1K-1.1
│   ├── train/                     # Full-parameter QFFT / SFT configs (`qfft_r1.json`, `sft_r1.json`)
│   └── grpo/                      # LoRA-GRPO configs (`mmr_dr50.json`, `dapo_16k.json`)
├── requirements/                  # Stage-isolated dependency files (`train.txt`, `eval.txt`, `rl.txt`)
├── scripts/                       # One-click evaluation & training entrypoints
├── src/math_posttrain/            # Core Python package
├── results/eval_summary.md        # Complete benchmark results, paired CIs, and length stats
├── docs/experiment_report.md      # Technical report & RL failure-mode post-mortem
└── tests/                         # Fast unit & zero-GPU end-to-end test suite
```

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MATH_POSTTRAIN_DATA_DIR` | `./data` | Directory for downloaded benchmarks and prepared datasets |
| `MATH_POSTTRAIN_OUTPUT_DIR` | `./outputs` | Directory for training checkpoints, exports, and eval reports |
| `MATH_POSTTRAIN_CONFIG_DIR` | `./configs` | Custom configuration directory |
| `HF_ENDPOINT` | `https://huggingface.co` | Optional Hugging Face Hub mirror endpoint |
