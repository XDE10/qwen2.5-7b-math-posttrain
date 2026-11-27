# AGENTS.md

Operational guide for autonomous AI coding and deployment agents working in `qwen2.5-7b-math-posttrain`.

## 1. Project Summary

- **Goal**: Minimalist, reproducible single-GPU post-training and 32K evaluation pipeline for `Qwen/Qwen2.5-7B-Instruct` mathematical reasoning (full-parameter QFFT/SFT + LoRA-GRPO / MMR-DR-GRPO).
- **Package & CLI**: `qwen25-math-posttrain` (`src/math_posttrain/`), entrypoint `math-posttrain`.
- **Published Hugging Face Checkpoints** (pinned in `configs/eval.json`):
  - Primary full-parameter QFFT model (`qfft_r1`): `XDE10/Qwen2.5-7B-Math-QFFT-R1` (`rev: 2afcc6360893fd8955aa15c61145f7f391c64a85`)
  - Companion 50-step RL LoRA adapter (`mmr50_lora`): `XDE10/Qwen2.5-7B-Math-QFFT-R1-MMR50-LoRA` (`rev: ec6c8a52e07e1d8f479a415301707b65ed60ff17`)

## 2. Repository Map

| Path | Role |
|---|---|
| `src/math_posttrain/cli.py` | Unified CLI dispatcher (`doctor`, `prepare-data`, `infer`, `train-sft`, `train-grpo`, `merge-lora`, `generate`, `score`) |
| `src/math_posttrain/common.py` | Portable workspace resolution (`workspace()`), atomic JSON/JSONL IO, hashing, config loader |
| `src/math_posttrain/models.py` | Model/tokenizer resolver accepting local dir, `run:<name>`, alias in `configs/eval.json`, or HF `owner/repo` |
| `src/math_posttrain/formats.py` | Frozen Qwen2.5 prompt template & SFT/QFFT encoding invariants |
| `src/math_posttrain/decontam.py` | 13-gram + sequence-similarity decontamination against `configs/decontam_review.json` |
| `src/math_posttrain/prepare_data.py` | Downloads 5 pinned benchmarks, builds decontaminated `r1_qfft.jsonl` (`s1K-1.1`) and `rl_pool.jsonl` (`Openr1-Math-46k`) |
| `src/math_posttrain/train_sft.py` | Single-GPU full-parameter QFFT/SFT (`DeepSpeed ZeRO-2 CPU Offload` + `Liger`), restores `use_cache=True` on export |
| `src/math_posttrain/grpo.py` | Single-GPU LoRA-GRPO / MMR-DR-GRPO (`TRL` + colocated `vLLM`) |
| `src/math_posttrain/mmr_grpo.py` | MMR-DR-GRPO trainer, CPU ONNX Jina embeddings, Algorithm 1 reweighting, cosine reward |
| `src/math_posttrain/merge_lora.py` | Merges LoRA adapter into base weights and saves bf16 safetensors |
| `src/math_posttrain/generate.py` | Resumable 32K vLLM evaluation generation + zero-GPU `FakeEngine` |
| `src/math_posttrain/grading.py` | Process-isolated `Math-Verify 0.9.0` final-segment last-`\boxed{}` grader |
| `src/math_posttrain/score.py` | CPU scoring, majority clustering, and per-problem paired bootstrap CIs |
| `src/math_posttrain/metrics.py` | `mean@n`, unbiased `pass@k`, `maj@n`, bootstrap CI, budget curves |
| `src/math_posttrain/infer.py` | One-shot / interactive inference CLI against published HF weights |
| `configs/` | `eval.json`, `deepspeed_zero2.json`, `decontam_review.json`, `train/*.json`, `grpo/*.json` |
| `scripts/` | One-click bash wrappers: `run_eval.sh`, `run_train_qfft.sh`, `run_train_grpo.sh` |
| `tests/` | Fast unit & zero-GPU E2E tests (`test_pipeline.py`, `test_mmr_grpo.py`) |

## 3. Environment Setup Matrix

Use **separate virtual environments** for `eval`, `train`, and `rl` because `vllm`, `trl`, and `deepspeed` pin different `torch`/`transformers` versions. Python `>= 3.10` is required.

| Target Workflow | Hardware Required | Install Commands |
|---|---|---|
| **CPU Dev / Tests / Data Prep** | CPU only | `pip install -e ".[dev,data]"` |
| **Inference (Transformers / PEFT)** | 1× GPU (≥16 GB bf16) or CPU | `pip install -e ".[infer]"` |
| **Evaluation (vLLM 32K + Math-Verify)** | 1× GPU (≥24 GB; ≥48 GB for full 32K batching) | `pip install -r requirements/eval.txt && pip install -e . --no-deps` |
| **Full-Param QFFT / SFT Training** | 1× GPU (≥80 GB VRAM) + ≥128 GB Host RAM | Keep system CUDA-matched `torch`, then `pip install -r requirements/train.txt && pip install -e . --no-deps` |
| **LoRA-GRPO / MMR-DR-GRPO Training** | 1× GPU (≥80 GB VRAM recommended for 16K rollouts) | `pip install -r requirements/rl.txt && pip install -e . --no-deps` |

> If operating behind a restricted network or in mainland China, set `export HF_ENDPOINT=https://hf-mirror.com` before running data preparation or model downloads.

## 4. Preflight Verification (Always Run First)

Run these zero-GPU checks immediately after installation and before committing any edits:

```bash
math-posttrain doctor
pytest -v
```

To verify the full evaluation shell entrypoint (`generate` → `score` → `results.md`/`results.json`) without a GPU or network access:

```bash
TMP_WS=$(mktemp -d)
mkdir -p "$TMP_WS/data/benchmarks"
printf '{"sample_id":"math500/00","problem":"2+2","answer":"4","level":1}\n' > "$TMP_WS/data/benchmarks/math500.jsonl"
printf '{"sample_id":"aime24/00","problem":"3*7","answer":"21"}\n' > "$TMP_WS/data/benchmarks/aime24.jsonl"
MATH_POSTTRAIN_DATA_DIR="$TMP_WS/data" MATH_POSTTRAIN_OUTPUT_DIR="$TMP_WS/outputs" \
  bash scripts/run_eval.sh base smoke_demo smoke --engine fake
rm -rf "$TMP_WS"
```

## 5. Deployment & Execution Playbooks

### 5.1 Serve / Run Published Models

```bash
# CLI inference (default: XDE10/Qwen2.5-7B-Math-QFFT-R1)
math-posttrain infer --prompt "Solve x^2 - 5x + 6 = 0."

# CLI inference with companion MMR-50 LoRA adapter
math-posttrain infer --adapter mmr50_lora --prompt "Find the remainder when 2^100 is divided by 7."

# OpenAI-compatible vLLM API server (requires eval env)
vllm serve XDE10/Qwen2.5-7B-Math-QFFT-R1 --dtype bfloat16 --max-model-len 32768 --port 8000
```

### 5.2 Prepare Datasets (Pinned & Decontaminated)

```bash
math-posttrain prepare-data --benchmarks         # 5 pinned math benchmarks -> data/benchmarks/
math-posttrain prepare-data --sft                # decontaminated s1K-1.1 -> data/train/r1_qfft.jsonl
math-posttrain prepare-data --rl --rl-limit 400  # decontaminated OpenR1 pool -> data/train/rl_pool.jsonl
math-posttrain prepare-data --all                # run all three
```

### 5.3 Evaluate Any Model (Alias, HF Repo ID, or Local Path)

```bash
# Auto-fetches benchmarks if missing, runs resumable vLLM generation, then grades with Math-Verify
bash scripts/run_eval.sh qfft_r1 qfft_r1 full
bash scripts/run_eval.sh base base full

# Paired bootstrap significance test between two tags
math-posttrain score --tags qfft_r1,base --compare qfft_r1:base --per-problem
```

### 5.4 Train Full-Parameter QFFT / SFT

```bash
# Auto-prepares data/train/r1_qfft.jsonl if missing
bash scripts/run_train_qfft.sh                        # configs/train/qfft_r1.json
bash scripts/run_train_qfft.sh train/sft_r1.json      # SFT baseline
# Append `--resume` to continue from the latest checkpoint-* in outputs/runs/<run_name>
```

### 5.5 Train LoRA-GRPO / MMR-DR-GRPO & Merge Weights

```bash
# Auto-prepares data/train/rl_pool.jsonl if missing; starts from HF qfft_r1 by default
bash scripts/run_train_grpo.sh                        # configs/grpo/mmr_dr50.json
bash scripts/run_train_grpo.sh grpo/dapo_16k.json     # DAPO-GRPO config

# Standalone LoRA merge (if trained with --no-merge or merging published adapter)
math-posttrain merge-lora \
  --base qfft_r1 \
  --adapter mmr50_lora \
  --out outputs/merged_mmr50
```

## 6. Path Configuration (Zero-Hardcoding Policy)

Never hardcode machine-specific paths. All runtime locations resolve through `math_posttrain.common.workspace()`:

- `MATH_POSTTRAIN_DATA_DIR` (default: `./data`) — benchmarks (`benchmarks/`) and training sets (`train/`)
- `MATH_POSTTRAIN_OUTPUT_DIR` (default: `./outputs`) — training runs (`runs/`) and evaluation outputs (`evals/`)
- `MATH_POSTTRAIN_CONFIG_DIR` (default: `./configs`) — JSON configuration directory
- `HF_ENDPOINT` (default: `https://huggingface.co`) — Hugging Face download base URL

## 7. Guardrails for Agents Modifying This Repository

1. **No symlinks or hard links**: Never commit symlinks (`120000`) or filesystem hard links, and never add `file://` or machine-local absolute paths in code, configs, or docs.
2. **Keep the repository clean**: Never commit `data/`, `outputs/`, `.cache/`, model binaries (`*.safetensors`, `*.bin`, `*.pt`), logs, or credentials.
3. **Preserve encoding & grading invariants**:
   - `formats.encode_qfft` must supervise all response tokens with no prompt markup and no `<|im_end|>`.
   - `formats.encode_sft` must mask prompt tokens with `-100` and append `<|im_end|>`.
   - `grading.py` must grade only the final segment after the last `<|im_start|>answer` / `</think>` and must never silently convert grader timeouts/errors into `wrong`.
4. **Export inference-ready checkpoints**: `train_sft.py` and `merge_lora.py` must set `model.config.use_cache = True` before saving; published LoRA `adapter_config.json` must never reference a local temporary directory in `base_model_name_or_path`.
5. **Bilingual docs sync**: When changing CLI commands, configs, or results, update both `README.md` (English) and `README.zh-CN.md` (Chinese) together, then run `math-posttrain doctor` and `pytest -v`.
