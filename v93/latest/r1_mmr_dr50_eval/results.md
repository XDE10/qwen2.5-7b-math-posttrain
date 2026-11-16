# v93 results (mean@n %, ±95% CI half-width / mean output tokens)

Grader: v93-final-segment-last-box/math-verify-0.9.0-timeout-status; generated 2026-11-16T10:16:26+08:00.

| model | aime24 | aime25 | amc23 | gsm8k | math500 | AIME24+25+AMC23 |
|---|---|---|---|---|---|---|
| grpo_qfft_r1_mmr_dr50_v93_merged | 18.8 ±9.6 / 15.0K | 16.2 ±9.8 / 11.7K | 60.0 ±12.2 / 7.6K | 92.4 ±1.4 / 0.5K | 80.3 ±2.9 / 3.0K | 34.5 ±7.7 |
| qfft_r1_16k_b8_restart_v93 | 18.5 ±10.2 / 14.7K | 14.8 ±10.5 / 11.1K | 59.1 ±12.2 / 7.5K | 91.4 ±1.5 / 0.5K | 80.5 ±2.8 / 3.1K | 33.6 ±7.9 |

## Paired comparisons (per-problem bootstrap)

| A − B | benchmark | Δ (pp) | 95% CI | p |
|---|---|---|---|---|
| grpo_qfft_r1_mmr_dr50_v93_merged − qfft_r1_16k_b8_restart_v93 | aime24 | +0.2 | [-2.9, +3.3] | 0.953 |
| grpo_qfft_r1_mmr_dr50_v93_merged − qfft_r1_16k_b8_restart_v93 | aime25 | +1.5 | [-1.2, +4.6] | 0.368 |
| grpo_qfft_r1_mmr_dr50_v93_merged − qfft_r1_16k_b8_restart_v93 | amc23 | +0.9 | [-3.8, +5.9] | 0.765 |
| grpo_qfft_r1_mmr_dr50_v93_merged − qfft_r1_16k_b8_restart_v93 | gsm8k | +1.0 | [+0.0, +2.0] | 0.062 |
| grpo_qfft_r1_mmr_dr50_v93_merged − qfft_r1_16k_b8_restart_v93 | math500 | -0.1 | [-1.7, +1.4] | 0.880 |
| grpo_qfft_r1_mmr_dr50_v93_merged − qfft_r1_16k_b8_restart_v93 | competition | +0.9 | [-1.4, +3.3] | 0.472 |

## Scoring recovery

The Colab session saved all 4,599 generations for the step-50 model. Scoring stopped on the baseline AIME25 sample `aime25/25`, draw index `10`, when Math-Verify timed out comparing the final boxed expression. Local scoring evaluated that box exactly as `216112050`, then verified it against the reference `113` using the same scorer. The verdict was incorrect. The saved generations and benchmark files were used unchanged. Both models have 4,599 scored generations and zero remaining verifier failures. See `scoring_recovery.json` for the exact expression and original timeout record.
