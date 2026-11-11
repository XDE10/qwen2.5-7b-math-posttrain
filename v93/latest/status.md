# v93 status (2026-11-11T23:10:54, session p1-1111-231532)

```
stage benchmarks                                    complete  0.003 
stage eval:base:full                                complete  0.394 
stage eval:qfft_official:full                       complete  3.439 
stage eval:qwen_math:full                           complete  0.171 
stage eval:run:qfft_ds_v93:full                     running   on p1-1111-231532@321507151d58, heartbeat 3 min ago 
stage eval:sft_repro:full                           complete  6.882 
stage models:base,qwen_math,qfft_official,sft_repro complete  0.173 
stage pilot                                         failed     ValueError: Unknown stage 'pilot'
stage ping:30                                       complete  0.008 
stage smoke-eval                                    complete  0.1 
stage smoke-grpo                                    complete  0.342 
stage smoke-train                                   complete  0.234 
stage sync-data                                     complete  0.002 
stage train:qfft_ds                                 complete  4.299 
run qfft_ds_v93                  complete step 378/ updated 2026-11-12T03:37:21+08:00
run smoke_grpo_v93               complete step 2/ updated 2026-11-09T19:11:43+08:00
run smoke_peak_v93               smoke_done step 2/ updated 2026-11-09T18:45:06+08:00
eval base/aime24     480/480 complete=True
eval base/aime25     480/480 complete=True
eval base/amc23      320/320 complete=True
eval base/gsm8k      1319/1319 complete=True
eval base/math500    2000/2000 complete=True
eval qfft_official/aime24     480/480 complete=True
eval qfft_official/aime25     480/480 complete=True
eval qfft_official/amc23      320/320 complete=True
eval qfft_official/gsm8k      1319/1319 complete=True
eval qfft_official/math500    2000/2000 complete=True
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

Grader: v93-final-segment-last-box/math-verify-0.9.0; generated 2026-11-11T10:37:48+08:00.

| model | math500 | aime24 | aime25 | amc23 | gsm8k | AIME24+25+AMC23 |
|---|---|---|---|---|---|---|
| qwen_math | 83.4 ±2.9 / 0.7K | 12.3 ±9.1 / 1.6K | 11.5 ±7.9 / 1.4K | 55.6 ±11.7 / 1.0K | 95.3 ±1.1 / 0.3K | 29.4 ±7.3 |

```
