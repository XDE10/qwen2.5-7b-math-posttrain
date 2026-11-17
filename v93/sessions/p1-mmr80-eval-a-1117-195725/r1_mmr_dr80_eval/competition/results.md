# v93 results (mean@n %, ±95% CI half-width / mean output tokens)

Grader: v93-final-segment-last-box/math-verify-0.9.0-timeout-status; generated 2026-11-17T20:36:02+08:00.

| model | aime24 | aime25 | amc23 | AIME24+25+AMC23 |
|---|---|---|---|---|
| grpo_qfft_r1_mmr_dr80_eval_v93_merged | 16.7 ±11.0 / 14.9K | 17.5 ±11.7 / 10.5K | 55.3 ±11.7 / 7.1K | 32.4 ±7.6 |
| qfft_r1_16k_b8_restart_v93 | 18.5 ±10.2 / 14.7K | 14.8 ±10.5 / 11.1K | 59.1 ±12.2 / 7.5K | 33.6 ±7.9 |

## Paired comparisons (per-problem bootstrap)

| A − B | benchmark | Δ (pp) | 95% CI | p |
|---|---|---|---|---|
| grpo_qfft_r1_mmr_dr80_eval_v93_merged − qfft_r1_16k_b8_restart_v93 | aime24 | -1.9 | [-4.4, +1.0] | 0.200 |
| grpo_qfft_r1_mmr_dr80_eval_v93_merged − qfft_r1_16k_b8_restart_v93 | aime25 | +2.7 | [-0.4, +6.5] | 0.123 |
| grpo_qfft_r1_mmr_dr80_eval_v93_merged − qfft_r1_16k_b8_restart_v93 | amc23 | -3.8 | [-10.6, +3.1] | 0.295 |
| grpo_qfft_r1_mmr_dr80_eval_v93_merged − qfft_r1_16k_b8_restart_v93 | competition | -1.2 | [-4.2, +1.9] | 0.444 |
