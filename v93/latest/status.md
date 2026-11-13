# v93 status (2026-11-13T10:27:21, session p1-r1-len12k)

```
stage benchmarks                                    complete  0.003 
stage eval:base:full                                complete  0.394 
stage eval:qfft_official:full                       complete  3.439 
stage eval:qwen_math:full                           complete  0.171 
stage eval:run:qfft_ds_v93:full                     complete  0.058 
stage eval:run:qfft_r1_16k_b8_restart_v93:full      complete  3.463 
stage eval:sft_repro:full                           complete  6.882 
stage grpo:compare_grpo_qfft_r1_12k                 complete  0.513 
stage grpo:compare_grpo_qfft_r1_16k                 complete  0.702 
stage grpo:compare_grpo_qfft_r1_8k                  complete  0.339 
stage grpo:diagnose_grpo_qfft_r1_8k                 complete  0.424 
stage grpo:smoke_grpo_qfft_r1_16k_b8                complete  0.325 
stage grpo:smoke_grpo_qfft_r1_fallback_8k           complete  0.331 
stage models:base,qwen_math,qfft_official,sft_repro complete  0.173 
stage pilot                                         failed     ValueError: Unknown stage 'pilot'
stage ping:30                                       complete  0.008 
stage smoke-eval                                    complete  0.1 
stage smoke-grpo                                    complete  0.342 
stage smoke-train                                   complete  0.234 
stage sync-data                                     complete  0.002 
stage train:qfft_ds                                 complete  0.016 
stage train:qfft_r1_16k_b8                          running   on p1-1112-183656@ff2652a8744b, heartbeat 1332 min ago (stale) 
stage train:qfft_r1_16k_b8_restart                  complete  6.252 
run compare_grpo_qfft_r1_12k_v93 complete step 2/ updated 2026-11-13T18:23:41+08:00
run compare_grpo_qfft_r1_16k_v93 complete step 2/ updated 2026-11-13T16:18:56+08:00
run compare_grpo_qfft_r1_8k_v93  complete step 2/ updated 2026-11-13T15:32:04+08:00
run diagnose_grpo_qfft_r1_8k_v93 complete step 2/ updated 2026-11-13T10:55:24+08:00
run qfft_ds_v93                  complete step 378/ updated 2026-11-12T03:37:21+08:00
run qfft_r1_16k_b8_restart_v93   complete step 750/ updated 2026-11-13T05:49:15+08:00
run qfft_r1_16k_b8_v93           training step 209/750 updated 2026-11-12T20:20:25+08:00
run smoke_grpo_qfft_r1_16k_b8_v93 complete step 2/ updated 2026-11-13T10:09:46+08:00
run smoke_grpo_qfft_r1_fallback_8k_v93 complete step 2/ updated 2026-11-13T11:22:34+08:00
run smoke_grpo_v93               complete step 2/ updated 2026-11-09T19:11:43+08:00
run smoke_peak_v93               smoke_done step 2/ updated 2026-11-09T18:45:06+08:00
eval base/aime24     480/480 complete=True
eval base/aime25     480/480 complete=True
eval base/amc23      320/320 complete=True
eval base/gsm8k      1319/1319 complete=True
eval base/math500    2000/2000 complete=True
eval qfft_ds_v93/aime24     480/480 complete=True
eval qfft_ds_v93/aime25     480/480 complete=True
eval qfft_ds_v93/amc23      320/320 complete=True
eval qfft_ds_v93/gsm8k      1319/1319 complete=True
eval qfft_ds_v93/math500    2000/2000 complete=True
eval qfft_official/aime24     480/480 complete=True
eval qfft_official/aime25     480/480 complete=True
eval qfft_official/amc23      320/320 complete=True
eval qfft_official/gsm8k      1319/1319 complete=True
eval qfft_official/math500    2000/2000 complete=True
eval qfft_r1_16k_b8_restart_v93/aime24     480/480 complete=True
eval qfft_r1_16k_b8_restart_v93/aime25     480/480 complete=True
eval qfft_r1_16k_b8_restart_v93/amc23      320/320 complete=True
eval qfft_r1_16k_b8_restart_v93/gsm8k      1319/1319 complete=True
eval qfft_r1_16k_b8_restart_v93/math500    2000/2000 complete=True
eval qwen_math/aime24     480/480 complete=True
eval qwen_math/aime25     480/480 complete=True
eval qwen_math/amc23      320/320 complete=True
eval qwen_math/gsm8k      1319/1319 complete=True
eval qwen_math/math500    2000/2000 complete=True
eval sft_repro/aime24     480/480 complete=True
eval sft_repro/aime25     480/480 complete=True
eval sft_repro/amc23      320/320 complete=True
eval sft_repro/gsm8k      1319/1319 complete=True
eval sft_repro/math500    2000/2000 complete=True
eval smoke_base/aime24     8/8 complete=True
eval smoke_base/math500    16/16 complete=True
# v93 results (mean@n %, ±95% CI half-width / mean output tokens)

Grader: v93-final-segment-last-box/math-verify-0.9.0; generated 2026-11-13T09:17:29+08:00.

| model | math500 | aime24 | aime25 | amc23 | gsm8k | AIME24+25+AMC23 |
|---|---|---|---|---|---|---|
| qfft_r1_16k_b8_restart_v93 | 80.5 ±2.8 / 3.1K | 18.5 ±10.2 / 14.7K | 14.8 ±10.5 / 11.1K | 59.1 ±12.2 / 7.5K | 91.4 ±1.5 / 0.5K | 33.6 ±7.9 |

```
