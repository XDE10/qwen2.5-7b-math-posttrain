"""Fixed dev-only E0 protocol and preserved training-side hard-candidate probe."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path

from src.training.frozen_verifier_v4 import sha, grader_identity, digest
from src.training.rollout_pipeline_v4 import jsonl, read, write_new, validate_model, SYSTEM
from src.training.math_reward_v2 import category

DEV_SHA = 'b5b3c90eeea953ae954cc2aa652caff4bd9c6edcb95f1b896215b84772225023'


def hard_ids(records, n=4):
    groups = defaultdict(list)
    for record in records:
        groups[record['sample_id']].append(record)
    return sorted(sid for sid, rs in groups.items()
                  if len(rs) == n and {r['trial'] for r in rs} == set(range(n))
                  and all(r['category'] == 'mismatch' and r['finish_reason'] == 'stop' for r in rs))


def messages(row):
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row['problem']}]


def summary(records):
    statuses = Counter(r['category'] for r in records)
    scorable = statuses['equivalent'] + statuses['mismatch']
    correct = statuses['equivalent']
    return {'count': len(records), 'correct': correct,
            'correct_over_all': correct / len(records),
            'scorable_count': scorable,
            'correct_over_scorable': correct / scorable if scorable else None,
            'status_counts': dict(statuses),
            'finish_reason_counts': dict(Counter(r['finish_reason'] for r in records)),
            'mean_generated_tokens': sum(r['generated_tokens'] for r in records) / len(records),
            'scope': 'development set only; unsupported statuses are reported separately; not a final-test score'}


def save_rows(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    p.add_argument('--model-identity', required=True)
    p.add_argument('--sampling-root', required=True)
    p.add_argument('--output-root', required=True)
    a = p.parse_args()
    root = Path(a.output_root)
    if root.exists():
        raise ValueError('Use a new output-root; preserve prior evaluation')
    data = Path('data/math/math_dev.jsonl')
    if sha(data) != DEV_SHA:
        raise ValueError('Dev content changed')
    rows = jsonl(data)
    if len(rows) != 256 or len({r['sample_id'] for r in rows}) != 256 or any(r['split'] != 'dev' for r in rows):
        raise ValueError('Expected original 256 unique dev rows')
    protected = set(read('data/portable_release/reserved_dev_ids_never_train.json')['sample_ids'])
    if not {r['sample_id'] for r in rows} <= protected:
        raise ValueError('Dev IDs must remain protected')
    source = Path(a.sampling_root)
    protocol = read(source / 'sampling_identity.json')
    done = read(source / 'sampling_complete.json')
    if protocol['num_samples'] != 4 or not done['complete'] or done['identity_sha256'] != sha(source / 'sampling_identity.json'):
        raise ValueError('Require completed four-sample screening')
    if protocol['grader_identity_sha256'] != digest(grader_identity()):
        raise ValueError('Frozen grader changed')
    records = []
    for shard in range(protocol['num_shards']):
        relative = f'graded/shard-{shard:05d}.jsonl'
        if sha(source / relative) != done['files'][relative]:
            raise ValueError('Grading journal changed')
        records.extend(jsonl(source / relative))
    ids = hard_ids(records)
    if len(ids) != 637:
        raise ValueError('Expected observed 637 hard candidates')
    candidates = jsonl(source / 'candidate_pool.jsonl')
    if sha(source / 'candidate_pool.jsonl') != protocol['candidate_sha256']:
        raise ValueError('Candidate content changed')
    selected = [r for r in candidates if r['sample_id'] in set(ids)]
    if len(selected) != 637 or any(r['training_approved'] is not False for r in selected):
        raise ValueError('Hard candidate approval must remain false')
    identity = validate_model(a.model_path, a.model_identity)
    root.mkdir(parents=True)
    save_rows(root / 'hard_candidates_e0_0of4.jsonl', selected)
    write_new(root / 'hard_probe_identity.json', {
        'sample_ids': ids, 'count': 637, 'training_approved': False,
        'role': 'training-side candidate probe; not dev/test; not approved for training',
        'source_sampling_identity_sha256': sha(source / 'sampling_identity.json'),
        'output_sha256': sha(root / 'hard_candidates_e0_0of4.jsonl'),
        'future_comparison': 'E0 vs formal E1 raw/compact; same four per-ID seeds, temperature/top-p and 2048 completion budget as source screening'})
    from transformers import AutoTokenizer
    from src.training.sft_encoding_v6 import token_ids
    tok_path = Path('data/tokenizers/Qwen2.5-7B-Instruct')
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    prompts = [token_ids(tok.apply_chat_template(messages(r), tokenize=True, add_generation_prompt=True,
                        return_dict=False, return_tensors=None, padding=False, truncation=False)) for r in rows]
    config = {'experiment': 'E0_dev_v9', 'model_identity': identity, 'dev_sha256': DEV_SHA,
              'seed': 42, 'temperature': 0, 'top_p': 1, 'top_k': -1, 'max_new_tokens': 2048,
              'max_model_len': max(map(len, prompts)) + 2048, 'max_num_seqs': 128,
              'gpu_memory_utilization': 0.90, 'grader_identity': grader_identity(),
              'source_sha256': sha(__file__),
              'tokenizer_sha256': {p.name: sha(p) for p in tok_path.iterdir() if p.is_file()},
              'versions': {n: version(n) for n in ['torch', 'transformers', 'vllm', 'tokenizers']}}
    write_new(root / 'evaluation_identity.json', config)
    from vllm import LLM, SamplingParams
    model = LLM(model=str(Path(a.model_path).resolve()), tokenizer=str(tok_path), dtype='bfloat16',
                tensor_parallel_size=1, max_model_len=config['max_model_len'], max_num_seqs=128,
                gpu_memory_utilization=0.90, enable_prefix_caching=True, enable_chunked_prefill=True,
                max_num_batched_tokens=8192, seed=42, generation_config='vllm')
    outputs = model.generate([{'prompt_token_ids': ids} for ids in prompts],
                             SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=2048, seed=42), use_tqdm=True)
    if len(outputs) != len(rows):
        raise ValueError('Generation coverage changed')
    raw = []
    for row, output in zip(rows, outputs):
        if len(output.outputs) != 1:
            raise ValueError('Expected one greedy prediction')
        completion = output.outputs[0]
        raw.append({'sample_id': row['sample_id'], 'text': completion.text,
                    'token_ids': list(completion.token_ids), 'generated_tokens': len(completion.token_ids),
                    'finish_reason': completion.finish_reason})
    save_rows(root / 'predictions.jsonl', raw)
    from src.training.runner_support_v4 import PooledVerifier
    checker = PooledVerifier(4)
    def grade(pair):
        record, row = pair
        result = checker.check(record['text'], row['reference_answer'])
        return {**record, 'verification': result, 'category': category(result)}
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(grade, zip(raw, rows)))
    finally:
        checker.close()
    save_rows(root / 'graded_predictions.jsonl', results)
    report = summary(results)
    report['predictions_sha256'] = sha(root / 'predictions.jsonl')
    write_new(root / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
