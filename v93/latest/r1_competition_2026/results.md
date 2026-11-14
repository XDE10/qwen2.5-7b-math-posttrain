# v93 results (mean@n %, ±95% CI half-width / mean output tokens)

Grader: v93-final-segment-last-box/math-verify-0.9.0-timeout-status; generated 2026-11-14T20:08:42+08:00.

| model | r1_competition_2026 | AIME24+25+AMC23 |
|---|---|---|
| r1 | 29.4 ±9.4 / 10.8K | – |
| sequence_mask | 30.1 ±9.4 / 10.4K | – |
| token_mask | 30.3 ±9.3 / 8.7K | – |

## Paired comparisons (per-problem bootstrap)

| A − B | benchmark | Δ (pp) | 95% CI | p |
|---|---|---|---|---|
| token_mask − sequence_mask | r1_competition_2026 | +0.2 | [-2.5, +3.0] | 0.935 |
| token_mask − r1 | r1_competition_2026 | +0.8 | [-2.3, +4.0] | 0.635 |
| sequence_mask − r1 | r1_competition_2026 | +0.6 | [-2.8, +3.8] | 0.738 |
