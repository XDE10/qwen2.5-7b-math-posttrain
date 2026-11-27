"""Download pinned math benchmarks and build decontaminated SFT/QFFT & RL datasets from public Hugging Face sources.

Usage:
    math-posttrain prepare-data --benchmarks         # 5 pinned benchmarks -> data/benchmarks/
    math-posttrain prepare-data --sft                # decontaminated s1K-1.1 R1 traces -> data/train/r1_qfft.jsonl
    math-posttrain prepare-data --rl --rl-limit 400  # decontaminated OpenR1 RL pool -> data/train/rl_pool.jsonl
    math-posttrain prepare-data --all                # prepare everything in one command
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import urllib.request
from pathlib import Path

from math_posttrain import decontam
from math_posttrain.common import log, now_iso, sha256_bytes, sha256_file, sha256_json, workspace, write_json, write_jsonl
from math_posttrain.formats import ANSWER_TAG, canonical_response
from math_posttrain.grading import reference_for_s1

SEALED_MATH500 = '2d87783f5679c481cbeeb475faf064f6ff08d4ad76c1e5fea5ca383a939ab4c0'

BENCHMARKS = {
    'math500': {
        'repo': 'HuggingFaceH4/MATH-500',
        'revision': '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',
        'file': 'test.jsonl',
        'sha256': '35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132',
    },
    'aime24': {
        'repo': 'HuggingFaceH4/aime_2024',
        'revision': '2fe88a2f1091d5048c0f36abc874fb997b3dd99a',
        'file': 'data/train-00000-of-00001.parquet',
        'sha256': '26139847601a5037c237d5928b195e7260ca8074cf4f264b794af42847f79ccf',
    },
    'aime25': {
        'repo': 'math-ai/aime25',
        'revision': '563bb8404243c5f09de6ec262f2db674fe5bce9b',
        'file': 'test.jsonl',
        'sha256': 'b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331',
    },
    'amc23': {
        'repo': 'math-ai/amc23',
        'revision': '80815d37005feb82cd7f8fbc6901d5d3eff43057',
        'file': 'test-00000-of-00001.parquet',
        'sha256': 'b696e87ba47be4e879a60fd4ef1d4aa522ba78c5ae013c6aff5cc9788a397c5e',
    },
    'gsm8k': {
        'repo': 'openai/gsm8k',
        'revision': '740312add88f781978c0658806c59bc2815b9866',
        'file': 'main/test-00000-of-00001.parquet',
        'sha256': 'ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59',
    },
}

S1K_SPEC = {
    'repo': 'simplescaling/s1K-1.1',
    'revision': '96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d',
    'file': 'data/train-00000-of-00001.parquet',
    'sha256': '10352d1f67d21b95385a364e8cc5a111a82603071acd4e36b0cbfbd6fc628c81',
}
OPENR1_RL_REPO = 'Elliott/Openr1-Math-46k-8192'


def download_pinned(spec: dict) -> bytes:
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    prefix = '' if spec.get('kind') == 'model' else 'datasets/'
    url = f"{endpoint}/{prefix}{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
    with urllib.request.urlopen(url, timeout=300) as response:
        data = response.read()
    digest = sha256_bytes(data)
    if digest != spec['sha256']:
        raise RuntimeError(f"{spec['repo']}/{spec['file']}: sha256 {digest} != pinned {spec['sha256']}")
    return data


def _records(data: bytes, filename: str) -> list[dict]:
    if filename.endswith('.jsonl'):
        return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    import pandas as pd
    return pd.read_parquet(io.BytesIO(data)).to_dict('records')


def _clean_number(answer) -> str:
    s = str(answer).strip()
    if s.endswith('.0') and s[:-2].lstrip('-').isdigit():
        s = s[:-2]
    return s


def normalize_benchmark(name: str, raw: list[dict]) -> list[dict]:
    rows = []
    for i, r in enumerate(raw):
        if name == 'math500':
            row = {'sample_id': f'math500/test/{i:05d}', 'problem': str(r['problem']), 'answer': str(r['answer']),
                   'level': int(r['level']), 'subject': r['subject'], 'unique_id': r['unique_id']}
        elif name == 'aime24':
            row = {'sample_id': f'aime24/{i:02d}', 'problem': str(r['problem']), 'answer': _clean_number(r['answer']),
                   'source_id': str(r['id']), 'url': r.get('url')}
        elif name == 'aime25':
            row = {'sample_id': f'aime25/{i:02d}', 'problem': str(r['problem']), 'answer': _clean_number(r['answer']),
                   'source_id': str(r['id'])}
        elif name == 'amc23':
            row = {'sample_id': f'amc23/{i:02d}', 'problem': str(r['question']), 'answer': _clean_number(r['answer']),
                   'source_id': str(r['id']), 'url': r.get('url')}
        elif name == 'gsm8k':
            answer = str(r['answer']).split('####')[-1].strip().replace(',', '')
            row = {'sample_id': f'gsm8k/test/{i:04d}', 'problem': str(r['question']), 'answer': answer}
        else:
            raise KeyError(name)
        row['source_index'] = i
        rows.append(row)
    return rows


def fetch_benchmarks(out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else workspace().benchmarks
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'created': now_iso(), 'benchmarks': {}}
    for name, spec in BENCHMARKS.items():
        raw = _records(download_pinned(spec), spec['file'])
        rows = normalize_benchmark(name, raw)
        if name == 'math500':
            digest = sha256_json([(r['sample_id'], r['problem'], str(r['answer'])) for r in rows])
            if digest != SEALED_MATH500:
                raise RuntimeError('MATH-500 sealed digest mismatch')
        path = out / f'{name}.jsonl'
        write_jsonl(path, rows)
        manifest['benchmarks'][name] = {**spec, 'rows': len(rows), 'jsonl_sha256': sha256_file(path)}
        log(f'benchmark {name}: {len(rows)} rows -> {path}')
    write_json(out / 'manifest.json', manifest)
    return out


def _ensure_benchmarks(bench_dir: Path) -> dict:
    missing = [name for name in BENCHMARKS if not (bench_dir / f'{name}.jsonl').is_file()]
    if missing:
        log(f'Missing benchmarks ({", ".join(missing)}); downloading pinned benchmarks first...')
        fetch_benchmarks(bench_dir)
    return decontam.load_benchmarks(bench_dir)


def _parse_meta(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value) if isinstance(value, str) else {}
    except (ValueError, SyntaxError):
        return {}


def build_sft_dataset(out_path: Path | None = None, bench_dir: Path | None = None) -> Path:
    """Build decontaminated DeepSeek-R1 QFFT/SFT training JSONL directly from pinned simplescaling/s1K-1.1."""
    ws = workspace()
    out = Path(out_path) if out_path else ws.train_data / 'r1_qfft.jsonl'
    benchmarks = _ensure_benchmarks(Path(bench_dir) if bench_dir else ws.benchmarks)

    log(f"Downloading {S1K_SPEC['repo']}@{S1K_SPEC['revision'][:12]} ...")
    raw = _records(download_pinned(S1K_SPEC), S1K_SPEC['file'])
    candidates = []
    for i, r in enumerate(raw):
        question = str(r.get('question') or '').strip()
        reasoning = str(r.get('deepseek_thinking_trajectory') or '').strip()
        final = str(r.get('deepseek_attempt') or '').strip()
        if not question or not reasoning or not final:
            continue
        if ANSWER_TAG in reasoning or ANSWER_TAG in final:
            continue
        ref, ref_kind = reference_for_s1(_parse_meta(r.get('metadata')), r.get('solution'))
        candidates.append({
            'sample_id': f's1k11/row/{i:04d}',
            'question': question,
            'response': canonical_response(reasoning, final),
            'reference_answer': ref,
            'reference_kind': ref_kind,
            'source': str(r.get('source_type') or 's1K-1.1'),
        })

    report = decontam.check([(r['sample_id'], r['question']) for r in candidates], benchmarks)
    drop_ids = {d['train_id'] for d in report['duplicates'] + report['unreviewed']}
    kept = []
    for idx, row in enumerate(r for r in candidates if r['sample_id'] not in drop_ids):
        kept.append({'index': idx, **row})

    write_jsonl(out, kept)
    write_json(out.with_suffix('.manifest.json'), {
        'created': now_iso(),
        'source': S1K_SPEC,
        'rows': len(kept),
        'dropped_contaminated': sorted(drop_ids),
        'sha256': sha256_file(out),
    })
    log(f'Built SFT/QFFT dataset: {len(kept)} clean rows (dropped {len(drop_ids)} flagged) -> {out}')
    return out


def build_rl_pool(out_path: Path | None = None, bench_dir: Path | None = None,
                  limit: int = 400, seed: int = 2025) -> Path:
    """Build a decontaminated RL prompt pool directly from public Elliott/Openr1-Math-46k-8192."""
    from datasets import load_dataset

    ws = workspace()
    out = Path(out_path) if out_path else ws.train_data / 'rl_pool.jsonl'
    benchmarks = _ensure_benchmarks(Path(bench_dir) if bench_dir else ws.benchmarks)

    log(f'Loading {OPENR1_RL_REPO} from Hugging Face...')
    ds = load_dataset(OPENR1_RL_REPO, split='train').shuffle(seed=seed)

    rows, seen = [], set()
    for i, r in enumerate(ds):
        question = str(r.get('problem') or r.get('question') or '').strip()
        answer = str(r.get('answer') or r.get('solution') or '').strip()
        if not question or not answer:
            continue
        key = ' '.join(decontam.words(question))
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append({'sample_id': f'openr1/{i:05d}', 'question': question, 'reference_answer': answer})
        if len(rows) >= max(limit * 2, limit + 100):
            break

    flagged = set()
    for items in benchmarks.values():
        flagged |= {f['train_id'] for f in decontam.find_flags([(r['sample_id'], r['question']) for r in rows], items)}
    clean = [r for r in rows if r['sample_id'] not in flagged][:limit]
    write_jsonl(out, clean)
    log(f'Built RL pool: {len(clean)} decontaminated prompts (removed {len(flagged)} flagged) -> {out}')
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmarks', action='store_true', help='Download the 5 pinned math benchmarks')
    p.add_argument('--sft', action='store_true', help='Build decontaminated R1 QFFT/SFT dataset from s1K-1.1')
    p.add_argument('--rl', action='store_true', help='Build decontaminated RL prompt pool from OpenR1-Math-46k')
    p.add_argument('--all', action='store_true', help='Prepare benchmarks, SFT dataset, and RL pool')
    p.add_argument('--rl-limit', type=int, default=400, help='Number of prompts to include in RL pool (default: 400)')
    p.add_argument('--seed', type=int, default=2025)
    a = p.parse_args()

    do_bench = a.all or a.benchmarks or not (a.sft or a.rl)
    if do_bench:
        fetch_benchmarks()
    if a.all or a.sft:
        build_sft_dataset()
    if a.all or a.rl:
        build_rl_pool(limit=a.rl_limit, seed=a.seed)


if __name__ == '__main__':
    main()
