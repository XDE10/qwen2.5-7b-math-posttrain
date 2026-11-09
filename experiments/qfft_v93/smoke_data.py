"""Worst-case-length rows for the GPU smoke test: every row fills the training window exactly.

The real data has a long tail (R1 traces reach 26.8K tokens, DeepSeek traces up to the 32,768 window), and
a smoke test on randomly drawn rows would usually miss the peak. These rows concatenate s1K-1.1 R1
responses in the training layout, longest first, and cut each row so that its encoding in the requested
format is the full window. They are only for measuring memory and speed, never for learning. The source
is the public s1K-1.1 parquet at its pinned revision, so the smoke does not need the local data build.

    python -m experiments.qfft_v93.smoke_data --tokenizer MODEL_DIR --out smoke.jsonl [--format qfft]
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

from experiments.qfft_v93.common import drive_layout, log, sha256_bytes, write_json, write_jsonl
from experiments.qfft_v93.fetch_benchmarks import SOURCES, download
from experiments.qfft_v93.formats import canonical_response, encode, prompt_ids, response_ids


def s1k_rows(cache: Path) -> list:
    spec = SOURCES['s1K-1.1.parquet']
    data = cache.read_bytes() if cache.exists() else None
    if data is None or sha256_bytes(data) != spec['sha256']:
        data = download(spec)  # pinned revision + SHA-256
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    import pandas as pd
    frame = pd.read_parquet(io.BytesIO(data))
    rows = []
    for r in frame.itertuples():
        try:
            rows.append((str(r.question), canonical_response(r.deepseek_thinking_trajectory, r.deepseek_attempt)))
        except ValueError:
            continue
    return rows


def fit(tokenizer, fmt: str, question: str, ids: list, length: int):
    """Longest prefix of `ids` whose decoded text encodes to at most `length` tokens in `fmt`."""
    budget = length - (len(prompt_ids(tokenizer, question)) + 1 if fmt == 'sft' else 0)
    cut = budget
    while cut > 0:
        text = tokenizer.decode(ids[:cut], skip_special_tokens=False)
        n = len(encode(tokenizer, fmt, question, text, 10 ** 9)['input_ids'])
        if n <= length:
            return text, n
        cut -= max(1, n - length)  # decode/encode round trips can merge or split a few tokens
    raise RuntimeError('could not fit a row into the window')


def build(tokenizer, fmt: str, length: int, count: int, cache: Path) -> list:
    pool = s1k_rows(cache)
    lengths = [len(response_ids(tokenizer, response)) for _, response in pool]
    order = sorted(range(len(pool)), key=lambda i: -lengths[i])
    out = []
    for k in range(count):
        question, pieces, total, j = pool[order[k]][0], [], 0, k
        while total < length + 64:  # a little extra, then cut to the window
            pieces.append(pool[order[j % len(order)]][1])
            total += lengths[order[j % len(order)]] + 2
            j += 1
        ids = response_ids(tokenizer, '\n\n'.join(pieces))
        text, n = fit(tokenizer, fmt, question, ids, length)
        out.append({'index': k, 'sample_id': f'smoke_peak/{k:03d}', 'question': question, 'response': text,
                    'tokens': n, 'traces_joined': len(pieces), 'source': 's1K-1.1 R1 (smoke only)'})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tokenizer', required=True, help='Base model directory (its tokenizer is used)')
    p.add_argument('--out', required=True)
    p.add_argument('--format', choices=['qfft', 'sft'], default='qfft')
    p.add_argument('--length', type=int, default=32768)
    p.add_argument('--rows', type=int, default=32, help='32 rows = 2 optimizer steps at effective batch 16')
    p.add_argument('--cache', help='s1K-1.1 parquet cache (default: Drive data/sources_v93/)')
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    cache = Path(a.cache) if a.cache else drive_layout().data / 'sources_v93/s1K-1.1.parquet'
    rows = build(tokenizer, a.format, a.length, a.rows, cache)
    write_jsonl(Path(a.out), rows)
    summary = {'rows': len(rows), 'format': a.format, 'window': a.length,
               'min_tokens': min(r['tokens'] for r in rows), 'max_tokens': max(r['tokens'] for r in rows)}
    write_json(Path(a.out).with_suffix('.summary.json'), summary)
    log(f"smoke rows: {summary['rows']} x {summary['min_tokens']}-{summary['max_tokens']} tokens ({a.format}) -> {a.out}")


if __name__ == '__main__':
    main()
