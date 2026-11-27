"""Train/benchmark contamination check: normalized 13-gram candidates + sequence-similarity gate.

Follows the Qwen2.5-Math report recipe (13-gram matching after text normalization plus difflib similarity >= 0.6)
combined with reviewed non-duplicate template decisions in `configs/decontam_review.json`.
"""
from __future__ import annotations

import argparse
import difflib
import re
from collections import defaultdict
from pathlib import Path

from math_posttrain.common import get_config_dir, read_json, read_jsonl, sha256_text, workspace

N = 13
THRESHOLD = 0.6


def default_review_file() -> Path:
    return get_config_dir() / 'decontam_review.json'


def words(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r'\\[a-z]+', ' ', text)
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return text.split()


def question_key(text: str) -> str:
    return sha256_text(' '.join(words(text)))[:16]


def ngrams(ws: list[str], n: int = N) -> set[str]:
    return {' '.join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def similarity(a_words: list[str], b_words: list[str]) -> float:
    return difflib.SequenceMatcher(None, ' '.join(a_words), ' '.join(b_words), autojunk=False).ratio()


def find_flags(train: list, bench: list, n: int = N, threshold: float = THRESHOLD) -> list[dict]:
    """Return pairs sharing a normalized n-gram (or identical text) with sequence similarity >= threshold."""
    index = defaultdict(set)
    train_words = {}
    for tid, text in train:
        ws = words(text)
        train_words[tid] = ws
        for g in (ngrams(ws, n) or {' '.join(ws)}):
            index[g].add(tid)
    flags = []
    for bid, text in bench:
        bw = words(text)
        candidates = set()
        for g in (ngrams(bw, n) or {' '.join(bw)}):
            candidates |= index.get(g, set())
        for tid in sorted(candidates):
            ratio = similarity(bw, train_words[tid])
            if ratio >= threshold:
                flags.append({
                    'bench_id': bid, 'train_id': tid,
                    'ratio': round(ratio, 3), 'bench_key': question_key(text),
                })
    return flags


def load_review(path: Path | None = None) -> dict:
    review_path = Path(path) if path is not None else default_review_file()
    if not review_path.exists():
        return {}
    return {(r['bench_id'], r['train_id']): r for r in read_json(review_path)['pairs']}


def check(train: list, benchmarks: dict, review_path: Path | None = None) -> dict:
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
    return {
        'method': f'normalized {N}-gram candidates + difflib ratio >= {THRESHOLD}',
        'benchmarks': {k: len(v) for k, v in benchmarks.items()},
        'train_questions': len(train),
        'flagged': len(flagged),
        'duplicates': duplicates,
        'cleared_not_duplicate': cleared,
        'unreviewed': unreviewed,
    }


def load_benchmarks(bench_dir) -> dict:
    out = {}
    for path in sorted(Path(bench_dir).glob('*.jsonl')):
        out[path.stem] = [(r['sample_id'], r['problem']) for r in read_jsonl(path)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train', required=True, help='Training JSONL with sample_id and question/problem')
    p.add_argument('--bench-dir', help='Benchmark JSONL directory (default: data/benchmarks)')
    a = p.parse_args()
    rows = read_jsonl(a.train)
    train = [(r['sample_id'], r.get('question') or r.get('problem')) for r in rows]
    benchmarks = load_benchmarks(a.bench_dir or workspace().benchmarks)
    report = check(train, benchmarks)
    print(f"flagged={report['flagged']} duplicates={len(report['duplicates'])} "
          f"cleared={len(report['cleared_not_duplicate'])} unreviewed={len(report['unreviewed'])}")


if __name__ == '__main__':
    main()
