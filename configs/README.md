# configs/ —— v26 工程配置（历史）

v93 的配置随实验包放在 [experiments/qfft_v93/configs/](../experiments/qfft_v93/configs/)：`pipeline.json`（模型 revision、评测口径与套件）、`train/*.json`（全参 SFT/QFFT）、`grpo/*.json`（LoRA-GRPO）、`ds_zero2_offload.json`、`decontam_review_v93.json`。

本目录只保留 `project1_aimo_sft_v26.json`（AIMO LoRA SFT 配方 v26），用于复现 v26 历史结果；其余 20 个历史配置已移入 `archive/configs/`。

## project1_aimo_sft_v26.json

| 分组 | 键 | 值 |
|---|---|---|
| 身份 | `experiment` / `stage` | `aimo_original_sft_v26` / `sft` |
| 模型 | `model` / `model_revision` | `Qwen/Qwen2.5-7B-Instruct` / `a09a35458c702b33eeacc393d103063234e8bc28` |
| 数据 | `train_file` / `manifest` | `data/aimo_sft_v26/train.jsonl` / `data/aimo_sft_v26/manifest.json` |
| 序列 | `max_length` / `seed` | `8192` / `42` |
| 优化 | `num_train_epochs` / `learning_rate` / `batch_size` / `gradient_accumulation_steps` | `3` / `2e-05` / `2` / `8`（有效 batch 16） |
| LoRA | `lora_r` / `lora_alpha` / `lora_dropout` / `target_modules` | `32` / `64` / `0.05` / 7 个线性层 |
| 追踪 | `swanlab` / `swanlab_required` / `swanlab_mode` | `true` / `true` / `online` |
| tokenizer | `tokenizer_sha256` | `tokenizer.json`、`tokenizer_config.json` 两个 SHA256 固定 |
| 评测 | `evaluation.max_new_tokens` / `max_model_len` | `32768` / `40960` |
| 采样 | `evaluation.temperature` / `top_p` | `0.6` / `0.95` |
| 次数 | `aime_samples_per_question` / `math500_samples_per_question` | `64` / `1`（MATH-500 为单次辅助，非 64 次估计） |
| RoPE | `evaluation.rope_scaling` | `yarn`，`factor 4.0`，`original_max_position_embeddings 32768` |

要点：训练窗口 8192 与评测生成 32768 是两套预算；YaRN factor 4 对 base 与 SFT 一致，用于容纳 32768 新 token + 题目；`max_model_len 40960` 必须放得下 prompt + 生成。

## 使用与边界

- 训练入口 `experiments/aimo_sft_v26/train.py`、Colab 启动入口 `tools/aimo/start_aimo_sft_v28.py` 都以本配置为准；改配置前先核对 `data/aimo_sft_v26/manifest.json` 的获批文件与条数。
- `scripts/training/*.py` 是早期 project1 入口，与现行 AIMO 配置**不直接配对**（其配套 smoke 配置已归档）；不要把两者混用。
- 历史配置（含 `approved_release_20261024` 的六个 smoke 配置、各代 project1/v1…v92 配置）在 `archive/configs/` 与 `docs/blueprints/configs/approved_release_20261024/`，只作追溯，不保证可直接运行。
- 配置文件属于版本范围，改动要有对应报告与实测记录；不要为了让旧脚本跑通而改回旧键名。
