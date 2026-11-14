"""Grade saved generations on CPU and build per-model summaries plus the cross-model results table.

    python -m experiments.qfft_v93.score                       # every tag under the Drive evals folder
    python -m experiments.qfft_v93.score --tags base,qfft_ds_v93 --compare qfft_ds_v93:sft_ds_v93

Outputs per <tag>/<benchmark>/: scored.jsonl, summary.json. Cross-model: evals/results.md + results.json.
Combined "competition" = AIME24 + AIME25 + AMC23 pooled at the problem level (100 problems), which has
more statistical power than any single 30-problem AIME year.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.qfft_v93.common import drive_layout, log, now_iso, read_json, read_jsonl, write_json, write_jsonl
from experiments.qfft_v93.grading import GRADER_VERSION, Grader
from experiments.qfft_v93.metrics import (bootstrap_ci, budget_curve, paired_bootstrap, reflection_count, summarize,
                                          tail_repetition)

COMPETITION = ('aime24', 'aime25', 'amc23')


def score_one(grader: Grader, out_dir: Path, rows: dict, rescore: bool, require_gradable: bool = False) -> dict | None:
    raw_path = out_dir / 'raw.jsonl'
    if not raw_path.exists():
        return None
    raw_rows = read_jsonl(raw_path)
    raw = {(r['sample_id'], r['draw']): r for r in raw_rows}
    id_path = out_dir / 'identity.json'
    if id_path.is_file():
        protocol = read_json(id_path)['protocol']
        if protocol.get('limit'):
            rows = dict(list(rows.items())[:protocol['limit']])
        expected = {(sid, draw) for sid in rows for draw in range(protocol['n'])}
        if raw.keys() != expected or len(raw) != len(raw_rows):
            raise RuntimeError(f'{out_dir}: incomplete draws: {len(expected - raw.keys())} missing, '
                               f'{len(raw.keys() - expected)} unexpected, {len(raw_rows) - len(raw)} duplicates')
    scored_path = out_dir / 'scored.jsonl'
    previous = {}
    if scored_path.exists() and not rescore:
        previous = {(r['sample_id'], r['draw']): r for r in read_jsonl(scored_path)
                    if r.get('grader') == GRADER_VERSION and r['status'] in ('correct', 'wrong', 'no_answer')}
    todo = [k for k in raw if k not in previous and k[0] in rows]
    results = grader.grade_many([(raw[k]['text'], rows[k[0]]['answer']) for k in todo]) if todo else []
    for key, grade in zip(todo, results):
        r = raw[key]
        previous[key] = {'sample_id': key[0], 'draw': key[1], 'status': grade['status'],
                         'correct': grade['status'] == 'correct', 'pred': grade.get('pred'), 'how': grade.get('how'),
                         'tokens': r['tokens'], 'finish_reason': r['finish_reason'],
                         'tail_repetition': tail_repetition(r['text']), 'reflections': reflection_count(r['text']),
                         'grader': GRADER_VERSION}
    records = [previous[key] for key in sorted(raw) if key in previous and key[0] in rows]
    write_jsonl(scored_path, records)
    if require_gradable:
        failures = sum(row['status'] not in ('correct', 'wrong', 'no_answer') for row in records)
        if failures:
            raise RuntimeError(f'{out_dir}: {failures} unscorable generations; re-run scoring to retry')
    by_problem = {}
    for r in records:
        by_problem.setdefault(r['sample_id'], []).append(r)
    majority = {}
    for sid, group in by_problem.items():
        if len(group) > 1:
            cluster = grader.majority_cluster([g['pred'] if g['status'] in ('correct', 'wrong') else None for g in group])
            majority[sid] = bool(cluster) and group[cluster[0]]['correct']
    summary = summarize(records, rows, majority or None)
    summary['budget_curve_no_forcing'] = budget_curve(records)
    summary.update(grader=GRADER_VERSION, scored=now_iso(), benchmark_problems=len(rows))
    write_json(out_dir / 'summary.json', summary)
    return summary


def pooled(summaries: dict, names=COMPETITION) -> dict | None:
    accs = []
    for name in names:
        s = summaries.get(name)
        if not s:
            return None
        accs += list(s['per_problem_acc'].values())
    return {'problems': len(accs), 'mean_at_n': sum(accs) / len(accs), 'mean_at_n_ci95': bootstrap_ci(accs),
            'per_problem_acc': None}


def fmt(summary) -> str:
    if not summary:
        return '–'
    lo, hi = summary['mean_at_n_ci95']
    half = (hi - lo) / 2 * 100 if lo is not None else 0
    tokens = summary.get('mean_tokens')
    return f"{summary['mean_at_n'] * 100:.1f} ±{half:.1f}" + (f" / {tokens / 1000:.1f}K" if tokens else '')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evals-root', help='Default: Drive runs/project1/v93/evals')
    p.add_argument('--bench-dir', help='Default: Drive data/benchmarks_v93')
    p.add_argument('--tags', help='Comma list; default: every folder under evals-root')
    p.add_argument('--compare', action='append', default=[], help='A:B paired comparison (repeatable)')
    p.add_argument('--rescore', action='store_true')
    p.add_argument('--require-gradable', action='store_true', help='Refuse a score containing verifier failures')
    p.add_argument('--workers', type=int)
    a = p.parse_args()
    layout = drive_layout()
    evals = Path(a.evals_root) if a.evals_root else layout.evals
    bench_dir = Path(a.bench_dir) if a.bench_dir else layout.data / 'benchmarks_v93'
    benchmarks = {path.stem: {r['sample_id']: r for r in read_jsonl(path)} for path in sorted(bench_dir.glob('*.jsonl'))}
    order = list(benchmarks)
    tags = a.tags.split(',') if a.tags else sorted(p.name for p in evals.iterdir() if p.is_dir())
    table = {}
    with Grader(workers=a.workers) as grader:
        for tag in tags:
            table[tag] = {}
            for name, rows in benchmarks.items():
                summary = score_one(grader, evals / tag / name, rows, a.rescore, a.require_gradable)
                if summary:
                    table[tag][name] = summary
                    log(f"{tag}/{name}: mean@n={summary['mean_at_n']:.3f} CI={summary['mean_at_n_ci95']} "
                        f"tokens={summary['mean_tokens']:.0f} trunc={summary['truncation_rate']:.3f}")
    comparisons = []
    for pair in a.compare:
        left, right = pair.split(':')
        for name in order + (['competition'] if set(COMPETITION) <= benchmarks.keys() else []):
            if name == 'competition':
                lp = {(n, sid): v for n in COMPETITION for sid, v in
                      table.get(left, {}).get(n, {}).get('per_problem_acc', {}).items()}
                rp = {(n, sid): v for n in COMPETITION for sid, v in
                      table.get(right, {}).get(n, {}).get('per_problem_acc', {}).items()}
                keys = sorted(lp) if lp.keys() == rp.keys() else []
                la, ra = [lp[k] for k in keys], [rp[k] for k in keys]
            else:
                lp = table.get(left, {}).get(name, {}).get('per_problem_acc', {})
                rp = table.get(right, {}).get(name, {}).get('per_problem_acc', {})
                common = sorted(lp) if lp.keys() == rp.keys() else []
                la, ra = [lp[s] for s in common], [rp[s] for s in common]
            if la and len(la) == len(ra):
                comparisons.append({'left': left, 'right': right, 'benchmark': name, **paired_bootstrap(la, ra)})
    lines = ['# v93 results (mean@n %, ±95% CI half-width / mean output tokens)', '',
             f'Grader: {GRADER_VERSION}; generated {now_iso()}.', '',
             '| model | ' + ' | '.join(order) + ' | AIME24+25+AMC23 |', '|---|' + '---|' * (len(order) + 1)]
    for tag, summaries in table.items():
        lines.append(f'| {tag} | ' + ' | '.join(fmt(summaries.get(n)) for n in order) +
                     f' | {fmt(pooled(summaries))} |')
    if comparisons:
        lines += ['', '## Paired comparisons (per-problem bootstrap)', '', '| A − B | benchmark | Δ (pp) | 95% CI | p |',
                  '|---|---|---|---|---|']
        for c in comparisons:
            lines.append(f"| {c['left']} − {c['right']} | {c['benchmark']} | {c['diff'] * 100:+.1f} | "
                         f"[{c['ci95'][0] * 100:+.1f}, {c['ci95'][1] * 100:+.1f}] | {c['p_two_sided']:.3f} |")
    (evals / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    slim = {tag: {n: {k: v for k, v in s.items() if k != 'per_problem_acc'} for n, s in sm.items()}
            for tag, sm in table.items()}
    write_json(evals / 'results.json', {'grader': GRADER_VERSION, 'updated': now_iso(), 'results': slim,
                                        'comparisons': comparisons})
    log('Wrote', evals / 'results.md')


if __name__ == '__main__':
    main()
