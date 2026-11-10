# v93 status (2026-11-10T18:16:59, session p1-1110-234216)

```
stage benchmarks                                    complete  0.003 
stage eval:base:full                                complete  0.394 
stage eval:qfft_official:full                       running   on p1-1110-234216@4b5c96d35876, heartbeat 4 min ago 
stage models:base,qwen_math,qfft_official,sft_repro complete  0.173 
stage ping:30                                       complete  0.008 
stage smoke-eval                                    complete  0.1 
stage smoke-grpo                                    complete  0.342 
stage smoke-train                                   complete  0.234 
stage sync-data                                     complete  0.002 
run smoke_grpo_v93               complete step 2/ updated 2026-11-09T19:11:43+08:00
run smoke_peak_v93               smoke_done step 2/ updated 2026-11-09T18:45:06+08:00
eval base/aime24     480/480 complete=True
eval base/aime25     480/480 complete=True
eval base/amc23      320/320 complete=True
eval base/gsm8k      1319/1319 complete=True
eval base/math500    2000/2000 complete=True
eval smoke_base/aime24     8/8 complete=True
eval smoke_base/math500    16/16 complete=True
# v93 results (mean@n %, ±95% CI half-width / mean output tokens)

Grader: v93-final-segment-last-box/math-verify-0.9.0; generated 2026-11-11T00:08:18+08:00.

| model | math500 | aime24 | aime25 | amc23 | gsm8k | AIME24+25+AMC23 |
|---|---|---|---|---|---|---|
| base | 76.8 ±3.1 / 0.6K | 11.2 ±9.2 / 1.5K | 7.7 ±5.1 / 1.5K | 50.9 ±11.7 / 1.0K | 91.6 ±1.5 / 0.3K | 26.1 ±7.0 |

```
