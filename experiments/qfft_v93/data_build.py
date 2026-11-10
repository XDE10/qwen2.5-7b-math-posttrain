"""Build the frozen 1000-question, two-teacher training sets (runs on the local workstation, CPU only).

Steps (each re-runnable; nothing here calls an API or a GPU):

    python -m experiments.qfft_v93.data_build plan     # 1000 questions, v93 replacement, decontamination
    python -m experiments.qfft_v93.data_build status   # which questions still lack an accepted DS trace
    python -m experiments.qfft_v93.teacher fill        # (API) fill the gaps until all 1000 are accepted
    python -m experiments.qfft_v93.data_build build    # write r1/ and ds/ train.jsonl + manifest + teacher report

Acceptance rule (identical for every question and both teachers, decided before looking at results):
a trace is usable when the teacher stopped normally (finish_reason == "stop"), both the reasoning and
the final answer are non-empty, and the SFT-encoded example (chat prompt + response + <|im_end|>) fits
the 32768-token training window. Answer correctness is NOT a filter (s1K-1.1 and QFFT keep incorrect
traces too); it is measured and reported per teacher. For DeepSeek the first usable attempt in
chronological order wins (v66 -> v86 -> v91 -> v93), so later attempts can never cherry-pick.
Traces are never truncated; the optional `ds_verified` variant swaps in a later usable attempt only
when the first one is checkably wrong.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import statistics
from collections import Counter
from pathlib import Path

from experiments.qfft_v93 import decontam
from experiments.qfft_v93.common import (ROOT, log, now_iso, read_json, read_jsonl, sha256_file, sha256_json,
                                         write_json, write_jsonl)
from experiments.qfft_v93.formats import ANSWER_TAG, canonical_response, render_prompt
from experiments.qfft_v93.grading import Grader, reference_for_s1
from experiments.qfft_v93.metrics import reflection_count

WINDOW = 32768
OUT = ROOT / 'data/qfft1000_v93'
PLAN = OUT / 'question_plan.json'
INPUTS = {
    'prepared': ROOT / 'reports/runs/romescape_reproduction_20261028_v57/train1000_prepared.jsonl',
    'questions': ROOT / 'reports/runs/qfft_ds_dataset_20261029_v66/questions1000.json',
    'replacements_v57': ROOT / 'reports/runs/romescape_reproduction_20261028_v57/replacement6_checks.json',
    's1k_parquet': ROOT / 'artifacts/research_s1_7b/s1K-1.1.parquet',
    'raw_pool': ROOT / 'data/portable_release/project1_raw_paired.jsonl',
    'tokenizer': ROOT / 'data/tokenizers/Qwen2.5-7B-Instruct/tokenizer.json',
    'benchmarks': ROOT / 'data/benchmarks_v93',
}
PREPARED_SHA = '0e8c6c83c5916e64d57247caac68bef70cb0ae41c20e07ed96cf4015f0e26781'
S1K_SHA = '10352d1f67d21b95385a364e8cc5a111a82603071acd4e36b0cbfbd6fc628c81'
TOKENIZER_SHA = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
# Pinned files that `fetch_benchmarks --sources` downloads and verifies; used when the older local copy is
# missing or differs, so a re-saved local file cannot block the build. Same revision, same bytes.
PINNED = {
    's1k_parquet': (S1K_SHA, ROOT / 'data/benchmarks_v93/_sources/s1K-1.1.parquet'),
    'tokenizer': (TOKENIZER_SHA, ROOT / 'data/benchmarks_v93/_sources/Qwen2.5-7B-Instruct-tokenizer.json'),
}
DS_DIRS = {
    'v66': ROOT / 'artifacts/qfft_ds_dataset_v66',
    'v86': ROOT / 'artifacts/qfft_length_retry_v86',
    'v91': ROOT / 'artifacts/qfft_incomplete_high_v91',
    'v93': ROOT / 'artifacts/qfft_teacher_v93',
}
# Replaced in v93: MATH-500 near-duplicate (0918) and 19 ultra-long insoluble proof outliers replaced with approved raw pool items.
V93_REPLACE = {
    's1k11/row/0918': 'complex',
    's1k11/row/0021': '65a9104d83967dc53c3766c2c86db39687b7b0e0b932161afb284af476ea4a3d',
    's1k11/row/0024': '63ae1cf0b64bc83850e3dc68cff2bedee8dbe5fa623a7404155d502195e84c01',
    's1k11/row/0057': '56b66761b29af0a70b421fc2916e00bf7c6afa70f5d77aa5ebc4268b4fa1026a',
    's1k11/row/0268': '5c9180c4cbecfe6704774b9f60ec02a96e2b18ba2cbc2d4a9f86422426e50b7b',
    's1k11/row/0339': '35327b646ee1006b6428829ac544051c8aaca0ab5e1ed4130669479e06b174aa',
    's1k11/row/0459': '0fe9a8e8752b1fcd1826c8a2439f9e7a9aa11f292375f81595e2c961709fdad9',
    's1k11/row/0576': '0b123b77ff6b6fc4a9add15f91f0c3c5056f20bc1a293ea1b4196d7551d9155e',
    's1k11/row/0693': '4371131d3016c8a687f92bd7de3387095758f8be99a97b8d950428dbe42b3e33',
    's1k11/row/0704': '0d285b61a26fc23fb1c14066e5ee43f8df8de1754e359dae7e7e92829d56e0f2',
    's1k11/row/0731': '6e73565f676a6ed1d10f6b889fc0a38a545d09cc57a3ac26f0bd130d3eea4d3a',
    's1k11/row/0740': '33a8b81e02eb9ad9dd64930e58ae3be2ee0ce7b0d64e3fe943e84701c9681f0c',
    's1k11/row/0745': '6acd9534d48f1e2ab36af2c21134b26ccdcff84806bf86d16fd8cf82397dbdb6',
    's1k11/row/0769': '0ca1f5f2de4299a43e5629c0de3b369b0886ed19567cecb39d5ffbf38d7cf4a3',
    's1k11/row/0808': 'eb4f6468d2a4a17bf5f93fa2631939c3fe87fbba69bb641a8027812c58dc283b',
    's1k11/row/0840': '68812eb5896f69763a82e3202b5c3fe857f6296a3ef29adee1b45fb856178733',
    's1k11/row/0870': '413d5205047d6a031b7b8a401b8b82aba63f578ca21aade305caabda311ed350',
    's1k11/row/0920': '0ba6fd578dbc1dbef8f9e902de2a230cc225e3b8c9b3cb6cddfa42b4e978e335',
    's1k11/row/0943': '3a3655a023cec69b14ff2fd852fa40dcedad01ff5c2ea6c3ea31a12dc3a6a674',
    's1k11/row/0974': '8810aaf66ec8b609679f0f8943758301d88b06e1cc6f2910c7c6be79badc38cf',
}
# Robust regex for DeepSeek meta-prompt / oververbosity remnants
META = re.compile(
    r'(?i)'
    r'(?:'
    r'(?:(?<=[\n.!?])|^)\s*'
    r'[^.!?\n]*?\b(?:oververbos\w*|desired\s*5\b)[^.!?\n]*?'
    r'(?:[.!?]|\n|$)'
    r'|'
    r';\s*[^;.\n]*?\b(?:oververbos\w*|desired\s*5\b)[^;.\n]*?(?=[;.\n]|$)'
    r'|'
    r'\([^)\n]*?\b(?:oververbos\w*|desired\s*5\b)[^)\n]*?\)'
    r'|'
    r',\s*(?:but\s+)?(?:final\s+)?(?:answer\s+)?(?:desired\s+)?oververbos\w*\s*5?\s*(?=[.!?\n]|$)'
    r')'
)


# ----------------------------------------------------------------------------- helpers

class TokenCounter:
    """Qwen2.5 token counts with the `tokenizers` library only (render_prompt == chat template, verified)."""

    def __init__(self, path=None):
        from tokenizers import Tokenizer
        if path is None:
            path = pinned_input('tokenizer')
        elif sha256_file(path) != TOKENIZER_SHA:
            raise RuntimeError(f'Unexpected tokenizer.json at {path}; expected Qwen2.5-7B-Instruct@a09a354')
        self.tok = Tokenizer.from_file(str(path))

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def lengths(self, question: str, response: str) -> dict:
        r = self.count(response)
        return {'response': r, 'qfft_total': r, 'sft_total': self.count(render_prompt(question)) + r + 1}


def clean_meta(text: str):
    edits = [m.group() for m in META.finditer(text or '')]
    return META.sub('', text or ''), edits


def split_think(solution: str):
    """'<think>X</think>Y' -> (X, Y) for OpenR1 raw-pool rows."""
    if not solution.startswith('<think>') or solution.count('</think>') != 1:
        raise ValueError('Expected exactly one leading <think>...</think> block')
    cut = solution.index('</think>')
    return solution[len('<think>'):cut], solution[cut + len('</think>'):]


def rel(path) -> str:
    """Repository-relative path when possible (portable manifests), absolute otherwise."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _order_key(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f'{seed}:{sample_id}'.encode()).hexdigest()


def _check_input(name: str) -> Path:
    path = INPUTS[name]
    if not path.exists():
        raise FileNotFoundError(f'Missing local input {name}: {path}')
    return path


_PINNED_CACHE = {}


def pinned_input(name: str) -> Path:
    """The local copy if its SHA-256 matches the pinned revision, otherwise the verified download."""
    if name in _PINNED_CACHE:
        return _PINNED_CACHE[name]
    expected, fallback = PINNED[name]
    tried = []
    for path in (INPUTS[name], fallback):
        if not path.is_file():
            tried.append(f'{path} (missing)')
            continue
        if sha256_file(path) == expected:
            if path != INPUTS[name]:
                log(f'{name}: using the pinned download {rel(path)} ({INPUTS[name].name} locally is '
                    f'{"different" if INPUTS[name].is_file() else "missing"})')
            _PINNED_CACHE[name] = path
            return path
        tried.append(f'{path} (sha256 differs)')
    raise RuntimeError(f'No verified {name}: ' + '; '.join(tried) +
                       '. Run `python -m experiments.qfft_v93.fetch_benchmarks --sources` first.')


# ----------------------------------------------------------------------------- plan

def _s1k_frame():
    import pandas as pd
    return pd.read_parquet(pinned_input('s1k_parquet'))  # s1K-1.1 at the pinned revision 96c411f


def _parse_meta(value):
    import ast
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value) if isinstance(value, str) else {}
    except (ValueError, SyntaxError):
        return {}


def choose_replacement(raw_rows, used_raw_ids, plan_questions, benchmarks, topic, override=None, grader=None):
    """Deterministic pick from the approved OpenR1 raw pool for a removed s1K row."""
    eligible = []
    for i, row in enumerate(raw_rows):
        if override and row['sample_id'] != override:
            continue
        if row['sample_id'] in used_raw_ids or row.get('training_approved') is not True:
            continue
        if row.get('split') != 'train' or row.get('arm') != 'raw':
            continue
        if not (row.get('source_position') or {}).get('complete'):
            continue
        try:
            reasoning, final = split_think(row['solution'])
        except (KeyError, ValueError):
            continue
        if not reasoning.strip() or not final.strip() or not str(row.get('reference_answer', '')).strip():
            continue
        eligible.append((i, row, reasoning, final))
    if not eligible:
        raise RuntimeError('No eligible raw-pool replacement' + (f' for override {override}' if override else ''))
    eligible.sort(key=lambda x: (topic.lower() not in x[1]['problem'].lower(), _order_key('v93', x[1]['sample_id'])))
    train = [(sid, q) for sid, q in plan_questions]
    for i, row, reasoning, final in eligible[:200]:
        flags = []
        for name, items in benchmarks.items():
            flags += decontam.find_flags([(row['sample_id'], row['problem'])], items)
        flags += decontam.find_flags(train, [(row['sample_id'], row['problem'])])
        if flags:
            continue
        if grader is not None:
            grade = grader.grade_many([(final, row['reference_answer'])])[0]
            if grade['status'] != 'correct':
                continue
        return i, row, reasoning, final
    raise RuntimeError('No raw-pool replacement passed decontamination and the R1 answer check')


def cmd_plan(args) -> None:
    prepared_path, questions_path = _check_input('prepared'), _check_input('questions')
    if sha256_file(prepared_path) != PREPARED_SHA:
        raise RuntimeError('train1000_prepared.jsonl changed; the 1000-question order is frozen')
    prepared = read_jsonl(prepared_path)
    questions = read_json(questions_path)
    if len(prepared) != 1000 or len(questions) != 1000:
        raise RuntimeError('Expected 1000 prepared rows and 1000 v66 questions')
    frame = _s1k_frame()
    raw_rows = read_jsonl(_check_input('raw_pool'))
    v57 = read_json(_check_input('replacements_v57'))
    used_raw = {r['new_sample_id'].split('/', 1)[1] for r in v57}
    benchmarks = decontam.load_benchmarks(_check_input('benchmarks'))
    plan = []
    with Grader(workers=args.workers) as grader:
        for i, (p, q) in enumerate(zip(prepared, questions)):
            if p['sample_id'] != q['sample_id'] or q['index'] != i or p['question'] != q['problem']:
                raise RuntimeError(f'v57/v66 question mismatch at index {i}')
            row = {'index': i, 'sample_id': p['sample_id'], 'question': p['question'], 'origin': 's1k11',
                   'source': q.get('source'), 'task_type': q.get('task_type'), 'category': q.get('category'),
                   'ds_effort': q['difficulty']['reasoning_effort']}
            if p['sample_id'].startswith('s1k11/'):
                s1 = frame.iloc[int(p['original_s1_row'])]
                ref, kind = reference_for_s1(_parse_meta(s1['metadata']), s1['solution'])
                row.update(reference_answer=ref, reference_kind=kind,
                           r1={'reasoning': p['deepseek_thinking_trajectory'], 'final': p['deepseek_attempt'],
                               'from': 's1K-1.1@96c411f deepseek_thinking_trajectory/deepseek_attempt'})
            else:
                row.update(origin='v57_replacement', reference_answer=str(q['reference_answer']),
                           reference_kind='raw_pool_reference',
                           r1={'reasoning': p['deepseek_thinking_trajectory'], 'final': p['deepseek_attempt'],
                               'from': 'OpenR1 raw pool via v57 replacement'})
            plan.append(row)
        for old_id, topic in V93_REPLACE.items():
            pos = next(k for k, r in enumerate(plan) if r['sample_id'] == old_id)
            others = [(r['sample_id'], r['question']) for r in plan if r['sample_id'] != old_id]
            override = (args.replacement or {}).get(old_id) or (topic if len(topic) == 64 else None)
            src_i, raw, reasoning, final = choose_replacement(raw_rows, used_raw, others, benchmarks, topic,
                                                              override=override, grader=grader)
            used_raw.add(raw['sample_id'])
            plan[pos] = {'index': pos, 'sample_id': 'v93-topup/' + raw['sample_id'], 'question': raw['problem'],
                         'origin': 'v93_replacement', 'replaces': old_id, 'raw_pool_row': src_i,
                         'source': 'previously-approved-openr1-raw', 'task_type': 'math',
                         'category': raw.get('category') or raw.get('problem_type'), 'ds_effort': 'high',
                         'reference_answer': str(raw['reference_answer']), 'reference_kind': 'raw_pool_reference',
                         'r1': {'reasoning': reasoning, 'final': final, 'from': 'OpenR1 raw pool (v93 replacement)'}}
            log(f'{old_id} -> {plan[pos]["sample_id"]} (raw pool row {src_i}): {raw["problem"][:120]!r}')
    report = decontam.check([(r['sample_id'], r['question']) for r in plan], benchmarks)
    bad = [d for d in report['duplicates'] if d['train_id'] in {r['sample_id'] for r in plan}]
    if report['unreviewed'] or bad:
        for flag in report['unreviewed'] + bad:
            log('DECONTAM', flag)
        raise RuntimeError('Decontamination failed: review new pairs in configs/decontam_review_v93.json')
    if len({r['sample_id'] for r in plan}) != 1000 or len({r['question'] for r in plan}) != 1000:
        raise RuntimeError('Plan must contain 1000 distinct questions')
    used = {**INPUTS, **{name: pinned_input(name) for name in PINNED}}  # the files actually read
    write_json(PLAN, {'version': 'v93', 'created': now_iso(), 'window': WINDOW,
                      'inputs': {k: {'path': rel(v), 'sha256': sha256_file(v) if v.is_file() else None}
                                 for k, v in used.items() if v.exists()},
                      'decontamination': report, 'questions': plan})
    kinds = Counter(r['reference_kind'] for r in plan)
    log(f'Plan written: {PLAN} | origins {dict(Counter(r["origin"] for r in plan))} | references {dict(kinds)}')


# ----------------------------------------------------------------------------- DS attempts

def ds_attempts(index: int):
    """Every saved DeepSeek candidate for this index, in chronological order."""
    out = []
    p = DS_DIRS['v66'] / 'candidates' / f'{index:04d}.json'
    if p.exists():
        out.append(('v66', p))
    for effort in ('max', 'high'):
        p = DS_DIRS['v86'] / 'attempts' / f'{index:04d}' / effort / 'candidate.json'
        if p.exists():
            out.append((f'v86-{effort}', p))
    p = DS_DIRS['v91'] / 'candidates' / f'{index:04d}.json'
    if p.exists():
        out.append(('v91', p))
    base = DS_DIRS['v93'] / 'attempts' / f'{index:04d}'
    if base.exists():
        for d in sorted(base.glob('a[0-9][0-9]')):
            if (d / 'candidate.json').exists():
                out.append((f'v93-{d.name}', d / 'candidate.json'))
    return out


def judge_candidate(candidate: dict, plan_row: dict, counter: TokenCounter) -> dict:
    if candidate.get('sample_id') != plan_row['sample_id']:
        return {'usable': False, 'reason': 'different_question'}
    if candidate.get('finish_reason') != 'stop':
        return {'usable': False, 'reason': 'not_stop:' + str(candidate.get('finish_reason'))}
    reasoning, edits = clean_meta(candidate.get('reasoning_content_raw') or '')
    final = candidate.get('final_content_raw') or ''
    if not reasoning.strip():
        return {'usable': False, 'reason': 'missing_reasoning'}
    if not final.strip():
        return {'usable': False, 'reason': 'missing_final'}
    if ANSWER_TAG in reasoning or ANSWER_TAG in final:
        return {'usable': False, 'reason': 'separator_in_text'}
    response = canonical_response(reasoning, final)
    tokens = counter.lengths(plan_row['question'], response)
    if tokens['sft_total'] > WINDOW:
        return {'usable': False, 'reason': 'over_window', 'tokens': tokens}
    return {'usable': True, 'reason': 'ok', 'response': response, 'final': final.strip(),
            'reasoning': reasoning.strip(), 'tokens': tokens, 'meta_edits': edits}


def ds_selection(plan_row: dict, counter: TokenCounter):
    """(first usable attempt, all judged attempts)."""
    judged = []
    for tag, path in ds_attempts(plan_row['index']):
        candidate = read_json(path)
        verdict = judge_candidate(candidate, plan_row, counter)
        judged.append((tag, path, candidate, verdict))
    first = next((j for j in judged if j[3]['usable']), None)
    return first, judged


def load_plan() -> dict:
    if not PLAN.exists():
        raise FileNotFoundError('Run `python -m experiments.qfft_v93.data_build plan` first')
    plan = read_json(PLAN)
    if len(plan['questions']) != 1000:
        raise RuntimeError('Plan must contain 1000 questions')
    return plan


def gap_report(plan: dict, counter: TokenCounter) -> dict:
    accepted, gaps = Counter(), []
    for row in plan['questions']:
        first, judged = ds_selection(row, counter)
        if first:
            accepted[first[0].split('-')[0]] += 1
        else:
            reasons = [j[3]['reason'] for j in judged] or ['no_attempt']
            gaps.append({'index': row['index'], 'sample_id': row['sample_id'], 'attempts': len(judged),
                         'reasons': reasons, 'ds_effort': row['ds_effort']})
    return {'accepted': sum(accepted.values()), 'accepted_by_round': dict(accepted), 'gaps': gaps}


def cmd_status(args) -> None:
    plan, counter = load_plan(), TokenCounter()
    report = gap_report(plan, counter)
    write_json(OUT / 'ds_status.json', {'created': now_iso(), **report})
    log(f"DS usable: {report['accepted']}/1000 by round {report['accepted_by_round']}; gaps: {len(report['gaps'])}")
    for gap in report['gaps'][:80]:
        log('  gap', gap['index'], gap['sample_id'], gap['reasons'])


# ----------------------------------------------------------------------------- build

def _stats(values):
    values = sorted(values)
    if not values:
        return {}
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {'n': len(values), 'mean': round(statistics.mean(values), 1), 'median': pick(0.5),
            'p90': pick(0.9), 'p99': pick(0.99), 'max': values[-1], 'over_16k': sum(v > 16384 for v in values)}


def _teacher_summary(rows: list) -> dict:
    grades = Counter(r['teacher_answer_grade'] for r in rows)
    checkable = [r for r in rows if r['teacher_answer_grade'] in ('correct', 'wrong', 'no_answer')]
    words = sum(len(r['response'].split()) for r in rows) or 1
    return {'rows': len(rows), 'grades': dict(grades),
            'accuracy_on_checkable': (sum(r['teacher_answer_grade'] == 'correct' for r in checkable) / len(checkable)
                                      if checkable else None),
            'checkable': len(checkable),
            'response_tokens': _stats([r['tokens']['response'] for r in rows]),
            'sft_tokens': _stats([r['tokens']['sft_total'] for r in rows]),
            'reflections_per_1k_words': round(1000 * sum(r['reflections'] for r in rows) / words, 2)}


def _agreement(a_rows, b_rows) -> dict:
    a = {r['sample_id']: r['teacher_answer_grade'] for r in a_rows}
    b = {r['sample_id']: r['teacher_answer_grade'] for r in b_rows}
    both = [s for s in a if s in b and a[s] in ('correct', 'wrong', 'no_answer') and b[s] in ('correct', 'wrong', 'no_answer')]
    cell = Counter((a[s] == 'correct', b[s] == 'correct') for s in both)
    return {'common_checkable': len(both), 'both_correct': cell[(True, True)], 'only_first': cell[(True, False)],
            'only_second': cell[(False, True)], 'both_wrong': cell[(False, False)]}


def _effort(tag: str, candidate: dict):
    return candidate.get('reasoning_effort_requested') or (tag.split('-', 1)[1] if tag.startswith('v86-') else None)


def _ds_provenance(rows: list) -> dict:
    """Where the DeepSeek traces of a variant come from: round, effort, prompt and API, counted per row."""
    details = [r['teacher_detail'] for r in rows]

    def count(key):
        return dict(sorted(Counter(str(d.get(key)) for d in details).items()))
    return {'rounds': dict(sorted(Counter((d.get('verified_swap_to') or d['attempt']).split('-')[0]
                                          for d in details).items())),
            'efforts': count('effort'), 'prompt_variants': count('prompt_variant'), 'providers': count('provider')}


def cmd_build(args) -> None:
    if args.allow_gaps and args.paired_exclusion:
        raise SystemExit('--allow-gaps (preview) and --paired-exclusion (final, paired) cannot be combined')
    plan, counter = load_plan(), TokenCounter()
    variants = {'r1': [], 'ds': []}
    if args.ds_verified:
        variants['ds_verified'] = []
    gaps, excluded = [], []
    pending_grades = []
    for row in plan['questions']:
        base = {'sample_id': row['sample_id'], 'index': row['index'], 'question': row['question'],
                'reference_answer': row.get('reference_answer'), 'reference_kind': row['reference_kind'],
                'origin': row['origin'], 'source': row.get('source'), 'task_type': row.get('task_type')}
        r1_response = canonical_response(row['r1']['reasoning'], row['r1']['final'])
        r1_tokens = counter.lengths(row['question'], r1_response)
        if r1_tokens['sft_total'] > WINDOW:
            raise RuntimeError(f"R1 trace for {row['sample_id']} exceeds the window ({r1_tokens['sft_total']})")
        r1_row = {**base, 'teacher': 'r1', 'response': r1_response, 'tokens': r1_tokens,
                  'teacher_detail': {'from': row['r1']['from']}, 'final_text': row['r1']['final'].strip(),
                  'reflections': reflection_count(row['r1']['reasoning'])}
        first, judged = ds_selection(row, counter)
        if first is None:
            gaps.append(row['sample_id'])
            if args.paired_exclusion:  # neither teacher trains on it, so both sets keep the same questions
                excluded.append({'sample_id': row['sample_id'], 'index': row['index'],
                                 'r1_sft_tokens': r1_tokens['sft_total'],
                                 'ds_attempts': [j[0] + ':' + j[3]['reason'] for j in judged]})
            else:
                variants['r1'].append(r1_row)
            continue
        variants['r1'].append(r1_row)
        tag, path, candidate, verdict = first
        detail = {'attempt': tag, 'path': rel(path), 'provider': candidate.get('provider'),
                  'model_returned': candidate.get('model_returned'),
                  'effort': _effort(tag, candidate),
                  'prompt_variant': candidate.get('prompt_variant', 'v66_original'),
                  'meta_edits': verdict['meta_edits'], 'usage': candidate.get('usage'),
                  'attempts_before': [j[0] + ':' + j[3]['reason'] for j in judged[:judged.index(first)]]}
        ds_row = {**base, 'teacher': 'ds', 'response': verdict['response'], 'tokens': verdict['tokens'],
                  'teacher_detail': detail, 'final_text': verdict['final'],
                  'reflections': reflection_count(verdict['reasoning'])}
        variants['ds'].append(ds_row)
        if args.ds_verified:
            later = [j for j in judged[judged.index(first) + 1:] if j[3]['usable']]
            variants['ds_verified'].append({**ds_row, '_alternatives': later})
    if gaps and not (args.allow_gaps or args.paired_exclusion):
        raise RuntimeError(f'{len(gaps)} questions still lack a usable DeepSeek trace; run `teacher fill`, or pass '
                           '--paired-exclusion to drop them from both teachers (or --allow-gaps for a preview that '
                           f'cannot be uploaded). First: {gaps[:5]}')
    # Grade every teacher final answer once, in parallel.
    for name, rows in variants.items():
        for r in rows:
            pending_grades.append((name, r))
    with Grader(workers=args.workers) as grader:
        results = grader.grade_many([(r['final_text'], r['reference_answer']) for _, r in pending_grades])
        for (name, r), g in zip(pending_grades, results):
            r['teacher_answer_grade'] = g['status'] if r['reference_answer'] else 'unverifiable'
        if args.ds_verified:
            swapped = 0
            for r in variants['ds_verified']:
                alternatives = r.pop('_alternatives')
                if r['teacher_answer_grade'] != 'wrong' or not alternatives:
                    continue
                grades = grader.grade_many([(alt[3]['final'], r['reference_answer']) for alt in alternatives])
                for alt, g in zip(alternatives, grades):
                    if g['status'] == 'correct':
                        tag, path, candidate, verdict = alt
                        r.update(response=verdict['response'], tokens=verdict['tokens'], final_text=verdict['final'],
                                 reflections=reflection_count(verdict['reasoning']), teacher_answer_grade='correct')
                        r['teacher_detail'] = {**r['teacher_detail'], 'verified_swap_to': tag,
                                               'path': rel(path), 'provider': candidate.get('provider'),
                                               'model_returned': candidate.get('model_returned'),
                                               'effort': _effort(tag, candidate),
                                               'prompt_variant': candidate.get('prompt_variant', 'v66_original')}
                        swapped += 1
                        break
            log(f'ds_verified: swapped {swapped} checkably-wrong traces for later correct usable attempts')
    manifest = {'version': 'v93', 'created': now_iso(), 'window': WINDOW, 'plan_sha256': sha256_file(PLAN),
                'acceptance_policy': __doc__.split('Acceptance rule')[1].split('\n\n')[0].strip(),
                'formats': 'Both teachers share the response layout reasoning + "\\n<|im_start|>answer\\n" + final; '
                           'SFT/QFFT encodings are applied at training time (experiments/qfft_v93/formats.py).',
                'variants': {},
                # gaps: DS is missing these while R1 keeps them (preview builds only; pipeline.sh d refuses them).
                # paired_exclusions: dropped from every variant, so all training sets share the same questions.
                'gaps': [] if args.paired_exclusion else gaps, 'paired_exclusions': excluded,
                'decontamination': {
                    k: v for k, v in read_json(PLAN)['decontamination'].items() if k != 'cleared_not_duplicate'}}
    for name, rows in variants.items():
        rows.sort(key=lambda r: r['index'])
        path = OUT / name / 'train.jsonl'
        write_jsonl(path, rows)
        manifest['variants'][name] = {'path': rel(path), 'rows': len(rows),
                                      'sha256': sha256_file(path), 'questions_sha256': sha256_json(
                                          [r['sample_id'] for r in rows]), **_teacher_summary(rows)}
        if name != 'r1':
            manifest['variants'][name]['ds_provenance'] = _ds_provenance(rows)
        log(f'{name}: {len(rows)} rows -> {path}')
    if excluded:
        log(f'paired exclusion: {len(excluded)} questions dropped from every variant:',
            ', '.join(e['sample_id'] for e in excluded))
    if variants['ds'] and len(variants['ds']) == len(variants['r1']):
        manifest['agreement_r1_vs_ds'] = _agreement(variants['r1'], variants['ds'])
    write_json(OUT / 'manifest.json', manifest)
    write_teacher_report(manifest)
    log('Manifest:', OUT / 'manifest.json', '| report:', OUT / 'teacher_report.md')


def write_teacher_report(manifest: dict) -> None:
    lines = ['# 教师轨迹对比报告（v93，自动生成）', '',
             f"生成时间 {manifest['created']}；训练窗口 {manifest['window']} token；准入规则见 manifest.json。", '',
             '| 变体 | 条数 | 可核对题 | 教师答案正确率 | response 均值/中位/p90/最大 token | >16K 条数 | 反思词/千词 |',
             '|---|---|---|---|---|---|---|']
    for name, v in manifest['variants'].items():
        t = v['response_tokens']
        acc = v['accuracy_on_checkable']
        lines.append(f"| {name} | {v['rows']} | {v['checkable']} | {acc:.1%} | {t.get('mean')}/{t.get('median')}/"
                     f"{t.get('p90')}/{t.get('max')} | {t.get('over_16k')} | {v['reflections_per_1k_words']} |"
                     if acc is not None else f"| {name} | {v['rows']} | 0 | – | – | – | – |")
    if manifest.get('gaps'):
        lines += ['', f"**预览构建（--allow-gaps）**：DS 缺 {len(manifest['gaps'])} 题而 R1 不缺，两位教师的题目不同，"
                      '上表正确率不可直接比较；这份数据不能上传训练（pipeline.sh d 会拒收）。']
    if manifest.get('paired_exclusions'):
        excluded = manifest['paired_exclusions']
        lines += ['', f"成对剔除 {len(excluded)} 题（DS 在全部尝试后仍没有可用轨迹，两位教师都不进训练集，"
                      f"所有训练臂题目相同）：{', '.join(e['sample_id'] for e in excluded)}。"]
    sources = [(name, v['ds_provenance']) for name, v in manifest['variants'].items() if v.get('ds_provenance')]
    if sources:
        lines += ['', '## DeepSeek 轨迹来源（每行一条）', '']

        def fmt(counts):
            return '，'.join(f'{k} {n}' for k, n in counts.items())
        for name, p in sources:
            lines.append(f"- {name}：轮次 {fmt(p['rounds'])}；reasoning_effort {fmt(p['efforts'])}；"
                         f"提示词 {fmt(p['prompt_variants'])}；接口 {fmt(p['providers'])}")
    if 'agreement_r1_vs_ds' in manifest:
        a = manifest['agreement_r1_vs_ds']
        lines += ['', '## R1 与 DeepSeek-V4.1-Flash 在同一批可核对题上的对错矩阵', '',
                  f"共同可核对 {a['common_checkable']} 题：两者都对 {a['both_correct']}，只有 R1 对 {a['only_first']}，"
                  f"只有 DS 对 {a['only_second']}，都错 {a['both_wrong']}。", '',
                  '说明：判分只看最终回答的最后一个 \\boxed（Math-Verify 0.9.0）；证明题/无短答案题记为不可核对，'
                  '不计入正确率。教师答案正确率高不等于学生训练后更强，最终以训练后评测为准。']
    (OUT / 'teacher_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--replacement', action='append', default=[], metavar='OLD=RAW_ID',
                      help='Override the automatic raw-pool replacement, e.g. s1k11/row/0918=<raw sample id>')
    plan.add_argument('--workers', type=int, default=4)
    sub.add_parser('status')
    build = sub.add_parser('build')
    build.add_argument('--allow-gaps', action='store_true',
                       help='Preview only: skip questions without a DS trace in ds/ (R1 keeps them); cannot be uploaded')
    build.add_argument('--paired-exclusion', action='store_true',
                       help='Final build without the questions that still lack a DS trace: drop them from every '
                            'variant (both teachers), recorded in manifest.json and the teacher report')
    build.add_argument('--ds-verified', action='store_true', help='Also write the ds_verified variant')
    build.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    if a.cmd == 'plan':
        a.replacement = dict(item.split('=', 1) for item in a.replacement)
        cmd_plan(a)
    elif a.cmd == 'status':
        cmd_status(a)
    else:
        cmd_build(a)


if __name__ == '__main__':
    main()
