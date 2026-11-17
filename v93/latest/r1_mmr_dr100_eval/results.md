# v93 results (mean@n %, ±95% CI half-width / mean output tokens)

Grader: v93-final-segment-last-box/math-verify-0.9.0-timeout-status; generated 2026-11-18T01:55:14+08:00.

| model | aime24 | aime25 | amc23 | gsm8k | math500 | AIME24+25+AMC23 |
|---|---|---|---|---|---|---|
| grpo_qfft_r1_mmr_dr100_eval_v93_merged | 18.8 ±10.2 / 14.7K | 15.8 ±11.2 / 11.5K | 57.5 ±11.7 / 7.6K | 92.5 ±1.4 / 0.4K | 80.9 ±2.8 / 3.1K | 33.4 ±7.7 |
| qfft_r1_16k_b8_restart_v93 | 18.5 ±10.2 / 14.7K | 14.8 ±10.5 / 11.1K | 59.1 ±12.2 / 7.5K | 91.4 ±1.5 / 0.5K | 80.5 ±2.8 / 3.1K | 33.6 ±7.9 |

## Paired comparisons (per-problem bootstrap)

| A − B | benchmark | Δ (pp) | 95% CI | p |
|---|---|---|---|---|
| grpo_qfft_r1_mmr_dr100_eval_v93_merged − qfft_r1_16k_b8_restart_v93 | aime24 | +0.2 | [-1.9, +2.3] | 0.921 |
| grpo_qfft_r1_mmr_dr100_eval_v93_merged − qfft_r1_16k_b8_restart_v93 | aime25 | +1.0 | [-1.0, +4.0] | 0.481 |
| grpo_qfft_r1_mmr_dr100_eval_v93_merged − qfft_r1_16k_b8_restart_v93 | amc23 | -1.6 | [-5.9, +2.8] | 0.531 |
| grpo_qfft_r1_mmr_dr100_eval_v93_merged − qfft_r1_16k_b8_restart_v93 | gsm8k | +1.1 | [+0.2, +2.0] | 0.033 |
| grpo_qfft_r1_mmr_dr100_eval_v93_merged − qfft_r1_16k_b8_restart_v93 | math500 | +0.4 | [-1.1, +1.8] | 0.601 |
| grpo_qfft_r1_mmr_dr100_eval_v93_merged − qfft_r1_16k_b8_restart_v93 | competition | -0.2 | [-2.3, +1.8] | 0.822 |
