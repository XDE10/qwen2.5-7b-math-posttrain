"""Download the v93 benchmark suite at pinned Hugging Face revisions and normalize it to JSONL.

Every source file is pinned by repository revision and SHA-256; the normalized files and their hashes
are recorded in data/benchmarks_v93/manifest.json. MATH-500 is additionally checked against the
project's sealed question/answer digest, so v93 numbers stay comparable with earlier MATH-500 runs.

    python -m experiments.qfft_v93.fetch_benchmarks            # benchmarks only
    python -m experiments.qfft_v93.fetch_benchmarks --sources  # also s1K-1.1 and official S1_QFFT data
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

from experiments.qfft_v93.common import ROOT, log, now_iso, sha256_bytes, sha256_file, write_json, write_jsonl

OUT = ROOT / 'data/benchmarks_v93'
SEALED_MATH500 = '2d87783f5679c481cbeeb475faf064f6ff08d4ad76c1e5fea5ca383a939ab4c0'

BENCHMARKS = {
    'math500': {'repo': 'HuggingFaceH4/MATH-500', 'revision': '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',
                'file': 'test.jsonl',
                'sha256': '35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132'},
    'aime24': {'repo': 'HuggingFaceH4/aime_2024', 'revision': '2fe88a2f1091d5048c0f36abc874fb997b3dd99a',
               'file': 'data/train-00000-of-00001.parquet',
               'sha256': '26139847601a5037c237d5928b195e7260ca8074cf4f264b794af42847f79ccf'},
    'aime25': {'repo': 'math-ai/aime25', 'revision': '563bb8404243c5f09de6ec262f2db674fe5bce9b',
               'file': 'test.jsonl',
               'sha256': 'b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331'},
    'amc23': {'repo': 'math-ai/amc23', 'revision': '80815d37005feb82cd7f8fbc6901d5d3eff43057',
              'file': 'test-00000-of-00001.parquet',
              'sha256': 'b696e87ba47be4e879a60fd4ef1d4aa522ba78c5ae013c6aff5cc9788a397c5e'},
    'gsm8k': {'repo': 'openai/gsm8k', 'revision': '740312add88f781978c0658806c59bc2815b9866',
              'file': 'main/test-00000-of-00001.parquet',
              'sha256': 'ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59'},
}
SOURCES = {
    's1K-1.1.parquet': {'repo': 'simplescaling/s1K-1.1', 'revision': '96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d',
                        'file': 'data/train-00000-of-00001.parquet',
                        'sha256': '10352d1f67d21b95385a364e8cc5a111a82603071acd4e36b0cbfbd6fc628c81'},
    'S1_qfft_official.json': {'repo': 'lwl-uestc/S1_QFFT', 'revision': '4a1b8ad0a6c319b40b56ee8c8c8873ad67bc8e0c',
                              'file': 'S1_qfft.json',
                              'sha256': '12bf2987d9b4e45249474aab037b5d875c3e83a989e84e3fa815a53531932a63'},
    # Base-model tokenizer for local token counting (data_build falls back to it when the copy under
    # data/tokenizers/ is missing or differs).
    'Qwen2.5-7B-Instruct-tokenizer.json': {'repo': 'Qwen/Qwen2.5-7B-Instruct', 'kind': 'model',
                                           'revision': 'a09a35458c702b33eeacc393d103063234e8bc28',
                                           'file': 'tokenizer.json',
                                           'sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'},
}
SOURCES_DIR = OUT / '_sources'


def download(spec: dict) -> bytes:
    prefix = '' if spec.get('kind') == 'model' else 'datasets/'
    url = f"https://huggingface.co/{prefix}{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
    with urllib.request.urlopen(url, timeout=300) as response:
        data = response.read()
    digest = sha256_bytes(data)
    if digest != spec['sha256']:
        raise RuntimeError(f"{spec['repo']}/{spec['file']}: sha256 {digest} != pinned {spec['sha256']}")
    return data


def _records(name: str, data: bytes, spec: dict) -> list:
    if spec['file'].endswith('.jsonl'):
        return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    import pandas as pd
    return pd.read_parquet(io.BytesIO(data)).to_dict('records')


def _clean_number(answer) -> str:
    s = str(answer).strip()
    if s.endswith('.0') and s[:-2].lstrip('-').isdigit():
        s = s[:-2]
    return s


def normalize(name: str, raw: list) -> list:
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
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise RuntimeError(f'{name}: duplicate sample ids')
    return rows


def sealed_math500_digest(rows: list) -> str:
    from experiments.qfft_v93.common import sha256_json
    return sha256_json([(r['sample_id'], r['problem'], str(r['answer'])) for r in rows])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(OUT))
    p.add_argument('--sources', action='store_true', help='Also fetch s1K-1.1 and the official S1_QFFT data')
    a = p.parse_args()
    out = Path(a.out)
    manifest = {'created': now_iso(), 'benchmarks': {}, 'sources': {}}
    for name, spec in BENCHMARKS.items():
        rows = normalize(name, _records(name, download(spec), spec))
        if name == 'math500' and sealed_math500_digest(rows) != SEALED_MATH500:
            raise RuntimeError('MATH-500 differs from the project-sealed questions/answers')
        path = out / f'{name}.jsonl'
        write_jsonl(path, rows)
        manifest['benchmarks'][name] = {**spec, 'rows': len(rows), 'jsonl_sha256': sha256_file(path)}
        log(f'{name}: {len(rows)} rows -> {path}')
    if a.sources:
        for filename, spec in SOURCES.items():
            path = out / '_sources' / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(download(spec))
            manifest['sources'][filename] = {**spec, 'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}
            log(f'source {filename} -> {path}')
    write_json(out / 'manifest.json', manifest)
    log('MATH-500 sealed digest verified; manifest written to', out / 'manifest.json')


if __name__ == '__main__':
    main()
