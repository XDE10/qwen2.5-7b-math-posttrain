"""Benchmark metrics: mean@n, unbiased pass@k, maj@n, bootstrap CIs, paired tests, length/behaviour stats.

All statistics are computed per problem first (the unit of sampling), then averaged, so a benchmark
with 30 problems and 16 samples each is treated as 30 observations, not 480.
"""
from __future__ import annotations

import math
import random
import re
import statistics
from collections import defaultdict

REFLECTION = re.compile(
    r"\b(?:wait|hmm+|alternatively|on second thought|let me (?:double[- ]check|verify|check|re-?check|"
    r"re-?examine|reconsider|think again))\b", re.IGNORECASE)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator 1 - C(n-c, k) / C(n, k) (Chen et al., 2021)."""
    if k > n:
        raise ValueError('k cannot exceed n')
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def bootstrap_ci(values, iters: int = 10000, alpha: float = 0.05, seed: int = 0):
    values = list(values)
    if not values:
        return [None, None]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return [means[int(alpha / 2 * iters)], means[min(iters - 1, int((1 - alpha / 2) * iters))]]


def paired_bootstrap(a, b, iters: int = 10000, seed: int = 0) -> dict:
    """Difference mean(a) - mean(b) over the same problems, CI and two-sided bootstrap p-value."""
    if len(a) != len(b) or not a:
        raise ValueError('Paired comparison needs two equal-length, non-empty per-problem lists')
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    mean = sum(diffs) / n
    below = sum(d <= 0 for d in boot) / iters
    above = sum(d >= 0 for d in boot) / iters
    return {'diff': mean, 'ci95': [boot[int(0.025 * iters)], boot[int(0.975 * iters)]],
            'p_two_sided': min(1.0, 2 * min(below, above)), 'problems': n}


def tail_repetition(text: str, window: int = 32, min_repeats: int = 4, tail_words: int = 6000) -> bool:
    """True if any `window`-word span repeats >= `min_repeats` times in the tail (degenerate loop)."""
    words = text.split()[-tail_words:]
    if len(words) < window * min_repeats:
        return False
    counts = defaultdict(int)
    for i in range(0, len(words) - window + 1, 4):
        key = ' '.join(words[i:i + window])
        counts[key] += 1
        if counts[key] >= min_repeats:
            return True
    return False


def reflection_count(text: str) -> int:
    return len(REFLECTION.findall(text))


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(records: list, problems: dict, majority: dict | None = None, ks=(1, 2, 4, 8, 16)) -> dict:
    """records: scored generations {sample_id, draw, correct, tokens, finish_reason, status, ...};
    problems: sample_id -> benchmark row (for level); majority: sample_id -> bool (maj@n correct)."""
    by_problem = defaultdict(list)
    for r in records:
        by_problem[r['sample_id']].append(r)
    missing = sorted(set(problems) - set(by_problem))
    per_problem = {}
    for sid, rows in by_problem.items():
        n, c = len(rows), sum(bool(r['correct']) for r in rows)
        per_problem[sid] = {'n': n, 'c': c, 'acc': c / n,
                            'pass': {k: pass_at_k(n, c, k) for k in ks if k <= n}}
    accs = [p['acc'] for p in per_problem.values()]
    n_values = sorted({p['n'] for p in per_problem.values()})
    summary = {
        'problems': len(per_problem), 'missing_problems': missing, 'generations': len(records),
        'samples_per_problem': n_values,
        'mean_at_n': _mean(accs), 'mean_at_n_ci95': bootstrap_ci(accs),
        'pass_at_k': {str(k): _mean([p['pass'].get(k) for p in per_problem.values()])
                      for k in ks if all(k <= p['n'] for p in per_problem.values())},
        'mean_tokens': _mean([r['tokens'] for r in records]),
        'mean_tokens_correct': _mean([r['tokens'] for r in records if r['correct']]),
        'mean_tokens_wrong': _mean([r['tokens'] for r in records if not r['correct']]),
        'median_tokens': statistics.median([r['tokens'] for r in records]) if records else None,
        'truncation_rate': _mean([r['finish_reason'] == 'length' for r in records]),
        'no_answer_rate': _mean([r['status'] == 'no_answer' for r in records]),
        'grader_failure_rate': _mean([r['status'] in ('timeout', 'error') for r in records]),
        'tail_repetition_rate': _mean([bool(r.get('tail_repetition')) for r in records]),
        'reflection_per_response': _mean([r.get('reflections', 0) for r in records]),
        'any_reflection_rate': _mean([r.get('reflections', 0) > 0 for r in records]),
        'metric_note': 'mean@n = average over problems of per-problem accuracy over n samples (pass@1 '
                       'estimate); pass@k unbiased; maj@n = plurality answer correct.',
    }
    if majority:
        summary['maj_at_n'] = _mean([float(majority[s]) for s in per_problem if s in majority])
    levels = defaultdict(list)
    for sid, rows in by_problem.items():
        level = problems.get(sid, {}).get('level')
        if level is not None:
            levels[str(level)].extend(rows)
    if levels:
        summary['by_level'] = {
            lv: {'acc': _mean([bool(r['correct']) for r in rows]), 'mean_tokens': _mean([r['tokens'] for r in rows]),
                 'any_reflection_rate': _mean([r.get('reflections', 0) > 0 for r in rows]),
                 'generations': len(rows)} for lv, rows in sorted(levels.items())}
    summary['per_problem_acc'] = {sid: p['acc'] for sid, p in sorted(per_problem.items())}
    return summary


def budget_curve(records: list, budgets=(1024, 2048, 4096, 8192, 16384, 32768)) -> dict:
    """Accuracy if generation had been capped at B tokens with no forcing: a response counts only if
    it finished within B tokens and was correct. Free to compute from one long-budget run."""
    by_problem = defaultdict(list)
    for r in records:
        by_problem[r['sample_id']].append(r)
    out = {}
    for b in budgets:
        accs = [sum(bool(r['correct']) and r['tokens'] <= b and r['finish_reason'] != 'length'
                    for r in rows) / len(rows) for rows in by_problem.values()]
        out[str(b)] = _mean(accs)
    return out
