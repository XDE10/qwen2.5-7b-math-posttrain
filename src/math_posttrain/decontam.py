"""Train/benchmark contamination check: normalized 13-gram candidates + sequence-similarity gate.

Follows the Qwen2.5-Math report recipe (13-gram matching after text normalization, plus a similarity
threshold of 0.6) and adds a reviewed decision file, because math benchmarks share boilerplate such as
"can be written as m/n where m and n are relatively prime" that produces template-level hits.

Every flagged pair must carry a human decision in configs/decontam_review_v93.json:
``duplicate`` (the training question must be removed or replaced) or ``not_duplicate`` (kept, reason
recorded). Unreviewed pairs make `check()` fail, so a new benchmark or training row cannot slip through.

CLI (prints flagged pairs side by side for review):
    python -m math_posttrain.decontam --train data/qfft1000_v93/question_plan.json
"""
from __future__ import annotations

import argparse
import difflib
import re
from collections import defaultdict
from pathlib import Path

from math_posttrain.common import CONFIG_DIR, read_json, read_jsonl, sha256_text

N = 13
THRESHOLD = 0.6
REVIEW_FILE = CONFIG_DIR / 'decontam_review.json'


def words(text: str) -> list:
    text = str(text).lower()
    text = re.sub(r'\\[a-z]+', ' ', text)      # drop LaTeX command names
    text = re.sub(r'[^a-z0-9]+', ' ', text)     # drop punctuation and symbols
    return text.split()


def question_key(text: str) -> str:
    """Short fingerprint of a normalized question, stored with review decisions."""
    return sha256_text(' '.join(words(text)))[:16]


def ngrams(ws: list, n: int = N) -> set:
    return {' '.join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def similarity(a_words: list, b_words: list) -> float:
    return difflib.SequenceMatcher(None, ' '.join(a_words), ' '.join(b_words), autojunk=False).ratio()


def find_flags(train: list, bench: list, n: int = N, threshold: float = THRESHOLD) -> list:
    """train/bench: [(id, text)]. Returns pairs sharing an n-gram (or identical) with similarity >= threshold."""
    index = defaultdict(set)
    train_words = {}
    for tid, text in train:
        ws = words(text)
        train_words[tid] = ws
        grams = ngrams(ws, n) or {' '.join(ws)}
        for g in grams:
            index[g].add(tid)
    flags = []
    for bid, text in bench:
        bw = words(text)
        grams = ngrams(bw, n) or {' '.join(bw)}
        candidates = set()
        for g in grams:
            candidates |= index.get(g, set())
        for tid in sorted(candidates):
            ratio = similarity(bw, train_words[tid])
            if ratio >= threshold:
                flags.append({'bench_id': bid, 'train_id': tid, 'ratio': round(ratio, 3),
                              'bench_key': question_key(text)})
    return flags


def load_review(path: Path = REVIEW_FILE) -> dict:
    if not Path(path).exists():
        return {}
    return {(r['bench_id'], r['train_id']): r for r in read_json(path)['pairs']}


def check(train: list, benchmarks: dict, review_path: Path = REVIEW_FILE) -> dict:
    """benchmarks: name -> [(id, text)]. Raises only via the caller; returns a full report."""
    review = load_review(review_path)
    flagged, unreviewed, duplicates, cleared = [], [], [], []
    for name, rows in sorted(benchmarks.items()):
        for flag in find_flags(train, rows):
            flag['benchmark'] = name
            flagged.append(flag)
            decision = review.get((flag['bench_id'], flag['train_id']))
            if decision is None or decision.get('bench_key') != flag['bench_key']:
                unreviewed.append(flag)
            elif decision['decision'] == 'duplicate':
                duplicates.append({**flag, 'reason': decision.get('reason')})
            else:
                cleared.append({**flag, 'reason': decision.get('reason')})
    return {'method': f'normalized {N}-gram candidates + difflib ratio >= {THRESHOLD}; reviewed decisions',
            'benchmarks': {k: len(v) for k, v in benchmarks.items()}, 'train_questions': len(train),
            'flagged': len(flagged), 'duplicates': duplicates, 'cleared_not_duplicate': cleared,
            'unreviewed': unreviewed}


def load_benchmarks(bench_dir) -> dict:
    out = {}
    for path in sorted(Path(bench_dir).glob('*.jsonl')):
        rows = read_jsonl(path)
        out[path.stem] = [(r['sample_id'], r['problem']) for r in rows]
    return out


def main() -> None:
    from math_posttrain.common import ROOT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train', required=True, help='question_plan.json or a JSONL with sample_id/question')
    p.add_argument('--bench-dir', default=str(ROOT / 'data/benchmarks_v93'))
    a = p.parse_args()
    path = Path(a.train)
    rows = read_json(path)['questions'] if path.suffix == '.json' else read_jsonl(path)
    train = [(r['sample_id'], r.get('question') or r.get('problem')) for r in rows]
    texts = dict(train)
    benchmarks = load_benchmarks(a.bench_dir)
    bench_texts = {bid: t for rows in benchmarks.values() for bid, t in rows}
    report = check(train, benchmarks)
    print(f"flagged={report['flagged']} duplicates={len(report['duplicates'])} "
          f"cleared={len(report['cleared_not_duplicate'])} unreviewed={len(report['unreviewed'])}")
    for flag in report['unreviewed']:
        print(f"\n[UNREVIEWED {flag['bench_id']} vs {flag['train_id']} ratio={flag['ratio']} key={flag['bench_key']}]")
        print('  BENCH:', bench_texts[flag['bench_id']][:400].replace('\n', ' '))
        print('  TRAIN:', texts[flag['train_id']][:400].replace('\n', ' '))


if __name__ == '__main__':
    main()
