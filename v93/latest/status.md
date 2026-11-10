# v93 status (2026-11-10T15:26:40, session p1-1110-232041)

```
stage benchmarks                                    complete  0.003 
stage eval:base:full                                failed     RuntimeError: gen_base failed with exit code 1; see /content/mydrive/runs/project1/v93/logs/20261110-152436_gen_base.log
stage models:base,qwen_math,qfft_official,sft_repro complete  0.173 
stage ping:30                                       complete  0.008 
stage smoke-eval                                    complete  0.1 
stage smoke-grpo                                    complete  0.342 
stage smoke-train                                   complete  0.234 
stage sync-data                                     complete  0.002 
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
