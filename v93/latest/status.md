# v93 status (2026-11-09T11:11:54, session p1-1109-180121)

```
stage benchmarks                                    complete  0.003 
stage models:base,qwen_math,qfft_official,sft_repro complete  0.173 
stage ping:30                                       complete  0.009 
stage smoke-eval                                    complete  0.1 
stage smoke-grpo                                    complete  0.342 
stage smoke-train                                   complete  0.234 
run smoke_grpo_v93               complete step 2/ updated 2026-11-09T19:11:43+08:00
run smoke_peak_v93               smoke_done step 2/ updated 2026-11-09T18:45:06+08:00
eval smoke_base/aime24     8/8 complete=True
eval smoke_base/math500    16/16 complete=True
# v93 results (mean@n %, ±95% CI half-width / mean output tokens)

Grader: v93-final-segment-last-box/math-verify-0.9.0; generated 2026-11-09T18:51:16+08:00.

| model | math500 | aime24 | aime25 | amc23 | gsm8k | AIME24+25+AMC23 |
|---|---|---|---|---|---|---|
| smoke_base | 62.5 ±25.0 / 0.6K | 12.5 ±18.8 / 1.0K | – | – | – | – |

```
