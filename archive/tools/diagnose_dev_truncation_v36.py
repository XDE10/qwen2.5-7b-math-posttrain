"""CPU-only examination of saved dev outputs; no generation or API calls."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('review_helper', ROOT / 'tools/continue_final_eval_v33.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def main():
    directory = ROOT / 'artifacts/final_eval_local_snapshot/evaluations/final_dev256_32k_v32'
    raw = tool.base.journal(directory / 'predictions.jsonl')
    reviewed = {r['sample_id']: r for r in tool.base.journal(directory / 'native_ds_v33/results.jsonl')}
    original, _ = tool.sources('review-dev', ROOT, directory)
    golds = {r['sample_id']: r for r in original}
    from transformers import AutoTokenizer
    from src.training.math_reward_v2 import category
    tok = AutoTokenizer.from_pretrained(ROOT / 'data/tokenizers/Qwen2.5-7B-Instruct', local_files_only=True)
    checker = tool.base.PooledVerifier(4)
    def strict(row):
        value = checker.check(row['text'], golds[row['sample_id']]['answer'])
        return row['sample_id'], category(value)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            strict_status = dict(pool.map(strict, raw))
    finally:
        checker.close()
    selected = [r for r in raw if strict_status[r['sample_id']] == 'parse_failure']
    records = []
    for index, row in enumerate(selected, 1):
        sid = row['sample_id']
        gold = golds[sid]['answer']
        text = row['text']
        tail_ids = row['token_ids'][-8192:]
        windows = Counter(tuple(tail_ids[i:i + 32]) for i in range(0, max(0, len(tail_ids) - 31), 4))
        result = {'sample_id': sid, 'reference_answer': gold, 'finish_reason': row['finish_reason'],
            'has_open_think': '<think>' in text, 'has_close_think': '</think>' in text,
            'has_box': r'\boxed' in text, 'max_identical_tail_32_token_window_count': max(windows.values(), default=0),
            'native_full_correct': reviewed[sid]['native_math_verify']['correct'],
            'reviewed_full_verdict': reviewed[sid]['final_verdict']}
        for budget in (2000, 6000):
            prefix = tok.decode(row['token_ids'][:budget], skip_special_tokens=False)
            value = tool.judge.grade_bounded(prefix, gold, ROOT)
            integer = bool(re.fullmatch(r'[+-]?\d+', gold.strip()))
            result[f'prefix_{budget}'] = {'native_equivalent': value['correct'], 'native_status': value['status'],
                'parsed_prediction': value.get('parsed_prediction'),
                'integer_gold_mentioned': bool(re.search(r'(?<![\w.\-])' + re.escape(gold.strip()) + r'(?![\w.])', prefix)) if integer else None,
                'has_box': r'\boxed' in prefix}
        records.append(result)
        if index % 20 == 0:
            print(f'CPU prefix diagnostic: {index}/{len(selected)}', flush=True)
    joint = Counter((r['finish_reason'], strict_status[r['sample_id']]) for r in raw)
    result = {'count': len(raw), 'strict_joint_counts': {f'{k[0]}/{k[1]}': v for k, v in joint.items()},
        'strict_parse_failure_count': len(selected),
        'parse_failure_finish_reasons': dict(Counter(r['finish_reason'] for r in selected)),
        'parse_failure_has_open_think': sum(r['has_open_think'] for r in records),
        'parse_failure_has_close_think': sum(r['has_close_think'] for r in records),
        'parse_failure_has_box': sum(r['has_box'] for r in records),
        'parse_failure_repeated_tail_windows_at_least_4': sum(r['max_identical_tail_32_token_window_count'] >= 4 for r in records),
        'parse_failure_native_full_correct': sum(r['native_full_correct'] for r in records),
        'parse_failure_reviewed_verdicts': dict(Counter(r['reviewed_full_verdict'] for r in records)),
        'prefix_native_equivalent': {str(b): sum(r[f'prefix_{b}']['native_equivalent'] for r in records) for b in (2000, 6000)},
        'prefix_integer_gold_mentions': {str(b): sum(r[f'prefix_{b}']['integer_gold_mentioned'] is True for r in records) for b in (2000, 6000)},
        'integer_gold_mention_eligible': sum(r['prefix_2000']['integer_gold_mentioned'] is not None for r in records),
        'limits': ['A mentioned correct number is not evidence of a solved problem.',
                   'Math-Verify can extract intermediate expressions; prefix equivalence is only a candidate clue.',
                   'No GPU ablation was run, so YaRN/decoding/training causality is unresolved.',
                   'Repetition window criterion is an operational symptom, not a proof of an infinite loop.']}
    output = directory / 'cpu_diagnosis_v36'
    output.mkdir(exist_ok=True)
    tool.base.save(output / 'summary.json', result)
    tool.base.save(output / 'per_sample.json', records)
    examples = []
    for row in selected[:2]:
        examples.append({'sample_id': row['sample_id'], 'original_problem': golds[row['sample_id']]['problem'],
            'reference_answer': golds[row['sample_id']]['answer'], 'first_1000_chars': row['text'][:1000],
            'last_1000_chars': row['text'][-1000:]})
    tool.base.save(output / 'examples_first_last_1000.json', examples)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(examples, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
