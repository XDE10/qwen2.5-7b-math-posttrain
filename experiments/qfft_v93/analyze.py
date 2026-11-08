"""Figures for the report (CPU): Pareto, adaptive length by difficulty, pass@k, budget curve, GRPO dynamics.

    python -m experiments.qfft_v93.analyze --tags base,sft_ds_v93,qfft_ds_v93,grpo_qfft_ds_v93 \
        --grpo-runs grpo_qfft_ds_v93 --out reports/v93_figures

Reads evals/<tag>/<benchmark>/summary.json (written by score.py) and runs/<run>/metrics.jsonl.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.qfft_v93.common import drive_layout, log, read_json, read_jsonl

COMPETITION = ('aime24', 'aime25', 'amc23')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def load(evals: Path, tags):
    out = {}
    for tag in tags:
        out[tag] = {}
        for path in sorted((evals / tag).glob('*/summary.json')):
            out[tag][path.parent.name] = read_json(path)
    return out


def pareto(data, out: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for tag, s in data.items():
        comp = [s[n] for n in COMPETITION if n in s]
        if len(comp) != len(COMPETITION):
            continue
        accs = [v for x in comp for v in x['per_problem_acc'].values()]
        tokens = sum(x['mean_tokens'] * x['generations'] for x in comp) / sum(x['generations'] for x in comp)
        ax.scatter(tokens, 100 * sum(accs) / len(accs), s=60)
        ax.annotate(tag, (tokens, 100 * sum(accs) / len(accs)), textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel('mean output tokens (AIME24+25+AMC23)')
    ax.set_ylabel('mean@n accuracy (%)')
    ax.set_title('Accuracy vs. tokens')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'pareto_competition.png', dpi=160)


def adaptive_length(data, out: Path):
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, s in data.items():
        xs, tokens, refl = [], [], []
        if 'gsm8k' in s:
            xs.append('GSM8K')
            tokens.append(s['gsm8k']['mean_tokens'])
            refl.append(100 * s['gsm8k']['any_reflection_rate'])
        for level, v in sorted((s.get('math500', {}).get('by_level') or {}).items()):
            xs.append(f'L{level}')
            tokens.append(v['mean_tokens'])
            refl.append(100 * v['any_reflection_rate'])
        if xs:
            ax1.plot(xs, tokens, marker='o', label=tag)
            ax2.plot(xs, refl, marker='o', label=tag)
    ax1.set_ylabel('mean output tokens')
    ax1.set_title('Length by difficulty (GSM8K, MATH-500 L1-L5)')
    ax2.set_ylabel('% responses with reflection words')
    ax2.set_title('Long-CoT triggering by difficulty')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'adaptive_length.png', dpi=160)


def pass_at_k(data, out: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for tag, s in data.items():
        comp = [s[n] for n in COMPETITION if n in s]
        if len(comp) != len(COMPETITION):
            continue
        ks = sorted({int(k) for x in comp for k in x['pass_at_k']})
        ks = [k for k in ks if all(str(k) in x['pass_at_k'] for x in comp)]
        weights = [x['problems'] for x in comp]
        ys = [100 * sum(x['pass_at_k'][str(k)] * w for x, w in zip(comp, weights)) / sum(weights) for k in ks]
        ax.plot(ks, ys, marker='o', label=tag)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('k')
    ax.set_ylabel('pass@k (%)')
    ax.set_title('pass@k on AIME24+25+AMC23 (unbiased estimator)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'pass_at_k.png', dpi=160)


def budget(data, out: Path, benchmark='aime24'):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for tag, s in data.items():
        if benchmark not in s:
            continue
        curve = s[benchmark]['budget_curve_no_forcing']
        xs = sorted(int(k) for k in curve)
        ax.plot(xs, [100 * curve[str(x)] for x in xs], marker='o', label=tag)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('token budget (finished within budget, no forcing)')
    ax.set_ylabel('accuracy (%)')
    ax.set_title(f'Anytime accuracy on {benchmark}')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f'budget_{benchmark}.png', dpi=160)


def grpo_curves(runs_root: Path, runs, out: Path):
    plt = _plt()
    keys = [('reward', 'reward'), ('completions/mean_length', 'mean completion length'),
            ('entropy', 'token entropy'), ('frac_reward_zero_std', 'zero-variance groups'),
            ('clip_ratio/region_mean', 'clip ratio')]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.4))
    for run in runs:
        rows = read_jsonl(runs_root / run / 'metrics.jsonl')
        for ax, (key, title) in zip(axes, keys):
            pts = [(r['step'], r[key]) for r in rows if key in r]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], label=run, lw=1)
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'grpo_dynamics.png', dpi=160)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tags', required=True)
    p.add_argument('--grpo-runs', default='')
    p.add_argument('--evals-root')
    p.add_argument('--runs-root')
    p.add_argument('--out', default='reports/v93_figures')
    a = p.parse_args()
    layout = drive_layout()
    evals = Path(a.evals_root) if a.evals_root else layout.evals
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(evals, a.tags.split(','))
    pareto(data, out)
    adaptive_length(data, out)
    pass_at_k(data, out)
    for b in ('aime24', 'aime25', 'math500'):
        budget(data, out, b)
    if a.grpo_runs:
        grpo_curves(Path(a.runs_root) if a.runs_root else layout.runs, a.grpo_runs.split(','), out)
    log('Figures written to', out)


if __name__ == '__main__':
    main()
