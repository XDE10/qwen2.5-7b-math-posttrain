"""Freeze the authorized capacity probe from the existing local s1K-1.1 snapshot.

This prepares only 64 training examples. The larger pool remains a candidate,
and its statistics are used only to estimate the cost of a future full run.
"""
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = 'Please reason step by step, and put your final answer within \\boxed{}.'


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def question_hash(text):
    return digest(''.join(text.split()).encode('utf-8'))


def main():
    import pandas as pd
    from tokenizers import Tokenizer
    output = ROOT / 'tools'
    report = ROOT / 'reports/s1_full_sft_probe_20261027_v46'
    report.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'artifacts/research_s1_7b/s1K-1.1.parquet'
    tokenizer_dir = ROOT / 'data/tokenizers/Qwen2.5-7B-Instruct'
    tok = Tokenizer.from_file(str(tokenizer_dir / 'tokenizer.json'))
    excluded_hashes = set()
    for name in ['math_dev.jsonl', 'math500_test.jsonl']:
        for line in (ROOT / 'data/math' / name).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            excluded_hashes.add(question_hash(row['problem']))
    frame = pd.read_parquet(source)
    encoded, excluded = [], []
    for i, row in frame.iterrows():
        sid = f's1k11/row/{i:04d}'
        question = str(row['question'])
        if question_hash(question) in excluded_hashes:
            excluded.append(sid)
            continue
        prompt = ('<|im_start|>system\n' + SYSTEM + '<|im_end|>\n'
            '<|im_start|>user\n' + question + '<|im_end|>\n<|im_start|>assistant\n')
        answer = str(row['deepseek_thinking_trajectory']) + '<|im_start|>answer\n' + str(row['deepseek_attempt'])
        source_ids = tok.encode(prompt, add_special_tokens=False).ids
        target_ids = tok.encode(answer + '<|im_end|>\n', add_special_tokens=False).ids
        # LLaMA-Factory infer_seqlen: retain short prompt, right-truncate target.
        if len(source_ids) * 2 >= 16384:
            raise ValueError('Unexpected long prompt; explicitly implement upstream infer_seqlen first')
        target = target_ids[:16384-len(source_ids)]
        encoded.append({'sample_id': sid, 'input_ids': source_ids + target,
            'labels': [-100] * len(source_ids) + target, 'prompt_tokens': len(source_ids),
            'original_tokens': len(source_ids) + len(target_ids),
            'truncated_at_16384': len(source_ids) + len(target_ids) > 16384,
            'probe_training_approved': True})
    ordered = sorted(encoded, key=lambda r: (len(r['input_ids']), r['sample_id']))
    selected = [ordered[round(i * (len(ordered)-1) / 63)] for i in range(64)]
    # Test the maximum-length case before spending time on smaller examples.
    selected = [selected[-1]] + selected[:-1]
    if len({r['sample_id'] for r in selected}) != 64:
        raise ValueError('Length quantiles selected duplicate rows')
    raw = ''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in selected).encode()
    blob = gzip.compress(raw, mtime=0)
    asset = output / 's1_full_sft_probe_v46.jsonl.gz'
    asset.write_bytes(blob)
    manifest = {
        'stage': 'probe_only', 'authorization': 'User authorized trying single-A100 full SFT capacity and speed.',
        'dataset': 'simplescaling/s1K-1.1', 'dataset_revision': '96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d',
        'dataset_parquet_sha256': digest(source.read_bytes()),
        'recipe_reference': 'RomEscape/Research-Question_Free@bea48878e67f189324fe45e08dfeb2bf15e79478',
        'model': 'Qwen/Qwen2.5-7B-Instruct', 'model_revision': 'a09a35458c702b33eeacc393d103063234e8bc28',
        'tokenizer_sha256': {name: digest((tokenizer_dir / name).read_bytes()) for name in ['tokenizer.json', 'tokenizer_config.json']},
        'approved_training_files': [asset.name], 'probe_file_sha256': digest(blob),
        'approved_probe_sample_ids': [r['sample_id'] for r in selected],
        'selection': '64 evenly spaced length-order quantiles, maximum first; no benchmark outputs used.',
        'candidate_pool_count': len(encoded), 'candidate_pool_cutoff_tokens': sum(len(r['input_ids']) for r in encoded),
        'candidate_pool_original_mean_tokens': statistics.mean(r['original_tokens'] for r in encoded),
        'candidate_pool_truncated_at_16384_count': sum(r['truncated_at_16384'] for r in encoded),
        'excluded_exact_dev_math500_rows': excluded,
        'probe_original_lengths': [r['original_tokens'] for r in selected],
        'probe_cutoff_lengths': [len(r['input_ids']) for r in selected],
        'full_training_approved_by_this_manifest': False,
        'notes': ['Capacity probe only, eight updates; no trained-model quality result.',
            '16384 cutoff mirrors reference right truncation; truncation counts explicitly retained.',
            'Exact whitespace-normalized overlap check against current dev and MATH500 only, not a new global audit.',
            'No source answer filtering or rewriting. No prior AIMO data/manifest modified.']}
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    (output / 's1_full_sft_probe_v46.json').write_text(text, encoding='utf-8')
    (report / 'manifest.json').write_text(text, encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ['candidate_pool_count', 'candidate_pool_cutoff_tokens',
        'candidate_pool_truncated_at_16384_count', 'excluded_exact_dev_math500_rows']}))
    print('probe asset bytes:', len(blob))


if __name__ == '__main__':
    main()
