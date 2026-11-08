"""Continue AFTER final dev256: CPU/DS review, MATH-500, then hard637 x4.

All generation uses the final merged SFT, AIMO SFT prompt, 32768 budget.
No other checkpoints or E0 are generated. This file does not modify v32.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT = Path(__file__).resolve()


def sibling(name):
    path = SCRIPT.with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = sibling('evaluate_final_dev256_v32')
judge = sibling('evaluate_sft_ds_v29')


def sources(stage, project, runs):
    if stage == 'review-dev':
        _, _, data = base.protocol_data(project)
        return [{'sample_id': r['sample_id'], 'problem': r['problem'], 'answer': str(r['reference_answer'])}
                for r in data], {'samples': 1, 'temperature': 0, 'top_p': 1, 'seed': 42}
    if stage == 'math500':
        path = project / 'reports/aimo_sft_20261026_v26/math500.jsonl'
        data = base.journal(path)
        canonical = [(r['sample_id'], r['problem'], str(r['answer'])) for r in data]
        import hashlib
        value = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(value).hexdigest() != '2d87783f5679c481cbeeb475faf064f6ff08d4ad76c1e5fea5ca383a939ab4c0':
            raise RuntimeError('MATH-500 differs from sealed original questions/golds')
        return data, {'samples': 1, 'temperature': 0.6, 'top_p': 0.95, 'seed': 42}
    found = sorted(p for p in runs.glob('E0_dev_v9_*') if (p / 'hard_probe_identity.json').exists()
                   and (p / 'hard_candidates_e0_0of4.jsonl').exists())
    if not found:
        raise RuntimeError('Existing 637-question probe not found; no new screening performed')
    baseline = found[-1]
    origin = base.read(baseline / 'hard_probe_identity.json')
    path = baseline / 'hard_candidates_e0_0of4.jsonl'
    sampling_path = runs / 'rl_screening_v8_n4/sampling_identity.json'
    if base.sha(path) != origin['output_sha256'] or base.sha(sampling_path) != origin['source_sampling_identity_sha256']:
        raise RuntimeError('Hard probe identity changed')
    data = base.journal(path)
    if len(data) != 637 or {r['sample_id'] for r in data} != set(origin['sample_ids']):
        raise RuntimeError('Expected original 637 probe questions')
    if any(r['training_approved'] is not False for r in data):
        raise RuntimeError('Hard probe must remain unapproved for training')
    original_sampling = base.read(sampling_path)
    return [{'sample_id': r['sample_id'], 'problem': r['problem'], 'answer': str(r['reference_answer'])}
            for r in data], {'samples': 4, 'temperature': original_sampling['temperature'],
            'top_p': original_sampling['top_p'], 'seed': original_sampling['seed'],
            'historical_probe_identity': origin,
            'note': 'Same conditional question pool and trial seeds; new AIMO prompt and 32768 budget. Not an independent test or exact old-protocol comparison.'}


def key(row):
    return row['sample_id'], row['draw']


def draw_seed(settings, sid, draw):
    if settings['samples'] == 1:
        import hashlib
        return (settings['seed'] + int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)) % 2**31
    import hashlib
    raw = json.dumps([settings['seed'], sid, draw], sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) & 0x7fffffff


def inspect_raw(raw, data, samples):
    expected = {(r['sample_id'], draw) for r in data for draw in range(samples)}
    seen = {key(r) for r in raw}
    if len(seen) != len(raw) or not seen <= expected:
        raise RuntimeError('Duplicate or unexpected generated responses')
    return expected - seen


def review(raw, data, settings, directory, project, api_key):
    if inspect_raw(raw, data, settings['samples']):
        raise RuntimeError('Generation incomplete; do not publish a full score')
    by_id = {r['sample_id']: r for r in data}
    review_dir = directory / 'native_ds_v33'
    review_dir.mkdir(exist_ok=True)
    identity = {'generator_identity_sha256': base.sha(directory / 'identity.json'),
        'input_sha256': judge.digest(raw), 'source_sha256': judge.digest(data),
        'script_sha256': base.sha(SCRIPT), 'judge_helper_sha256': base.sha(judge.__file__),
        'model': 'deepseek-flash', 'system_sha256': judge.digest(judge.SYSTEM), 'thinking': 'disabled'}
    if (review_dir / 'identity.json').exists() and base.read(review_dir / 'identity.json') != identity:
        raise RuntimeError('Review inputs/settings changed; preserve old records')
    base.save(review_dir / 'identity.json', identity)
    completed = base.journal(review_dir / 'results.jsonl')
    cache = {key(r): r for r in completed}
    if len(cache) != len(completed) or not set(cache) <= {key(r) for r in raw}:
        raise RuntimeError('Invalid review journal')
    attempts_dir = review_dir / 'api_attempts'
    attempts_dir.mkdir(exist_ok=True)
    def persist(result):
        base.append(review_dir / 'results.jsonl', result)
        cache[key(result)] = result
        if len(cache) % 10 == 0 or len(cache) == len(raw):
            print(f'CPU/DS review: {len(cache)}/{len(raw)} responses saved', flush=True)
    # Parser transport is single-threaded; only independent DS requests overlap.
    # Each request has its own append journal, avoiding concurrent file writes.
    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending_reviews = {}
        for index, row in enumerate(raw, 1):
            if key(row) in cache:
                continue
            gold = by_id[row['sample_id']]
            native = judge.grade_bounded(row['text'], gold['answer'], project)
            result = {**row, 'original_problem': gold['problem'], 'original_reference_answer': gold['answer'],
                'native_math_verify': native, 'ds_review': None, 'final_verdict': 'correct'}
            if native['correct']:
                persist(result)
            else:
                payload = judge.judge_payload(gold['problem'], gold['answer'], row['text'], native, row['finish_reason'])
                future = pool.submit(judge.call_ds, payload, api_key, 'deepseek-flash',
                                     attempts_dir / (judge.digest(key(row)) + '.jsonl'))
                pending_reviews[future] = result
            if index % 20 == 0:
                print(f'Native Math-Verify: {index}/{len(raw)} checked; DS uses up to 4 requests in parallel.', flush=True)
        for future in as_completed(pending_reviews):
            result = pending_reviews[future]
            try:
                result['ds_review'] = future.result()
                result['final_verdict'] = result['ds_review']['judgment']['verdict']
                persist(result)
            except Exception as error:
                errors.append(type(error).__name__)
    if errors:
        raise RuntimeError(f'{len(errors)} DS requests failed; successful reviews saved. Repeat to retry only unfinished records.')
    records = list(cache.values())
    counts = Counter(r['final_verdict'] for r in records)
    auto = sum(r['native_math_verify']['correct'] for r in records)
    total = len(raw)
    result = {'responses': total, 'questions': len(data), 'samples_per_question': settings['samples'],
        'native_math_verify_correct': auto, 'native_math_verify_accuracy': auto / total,
        'reviewed_correct': counts['correct'], 'reviewed_incorrect': counts['incorrect'], 'uncertain': counts['uncertain'],
        'reviewed_accuracy_lower_bound': counts['correct'] / total,
        'reviewed_accuracy_upper_bound': (counts['correct'] + counts['uncertain']) / total,
        'ds_reviewed': total - auto, 'ds_corrected_to_correct': counts['correct'] - auto,
        'finish_reasons': dict(Counter(r['finish_reason'] for r in raw)),
        'native_statuses': dict(Counter(r['native_math_verify']['status'] for r in records)),
        'mean_output_tokens': sum(r['output_tokens'] for r in raw) / total,
        'length_finish_rate': sum(r['finish_reason'] == 'length' for r in raw) / total,
        'scope': 'Final model only. Native automatic and model-assisted answer scores separately; not proof certification.'}
    if settings['samples'] == 4:
        groups = defaultdict(list)
        for row in records:
            groups[row['sample_id']].append(row)
        result['native_questions_with_any_correct'] = sum(any(r['native_math_verify']['correct'] for r in rs) for rs in groups.values())
        result['reviewed_questions_with_any_correct'] = sum(any(r['final_verdict'] == 'correct' for r in rs) for rs in groups.values())
        result['reviewed_middle_groups_no_uncertainty_or_truncation'] = sum(
            all(r['final_verdict'] != 'uncertain' and r['finish_reason'] == 'stop' for r in rs)
            and 1 <= sum(r['final_verdict'] == 'correct' for r in rs) <= 3 for rs in groups.values())
        result['scope'] = 'Conditional E0 0/4 difficulty probe; NOT independent test accuracy or training approval.'
    usage = Counter()
    attempts = base.journal(review_dir / 'api_attempts.jsonl')
    for path in sorted(attempts_dir.glob('*.jsonl')):
        attempts.extend(base.journal(path))
    for attempt in attempts:
        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            usage[name] += attempt.get('response', {}).get('usage', {}).get(name, 0) or 0
    result['api_usage_including_rejected_replies'] = dict(usage)
    base.save(review_dir / 'summary.json', result)
    base.save(review_dir / 'COMPLETE.json', {'results_sha256': base.sha(review_dir / 'results.jsonl')})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['review-dev', 'math500', 'hard637', 'next'], default='review-dev')
    parser.add_argument('--project-root', type=Path, default=Path('/content/workspaces/aimo-sft-v26'))
    parser.add_argument('--run-root', type=Path, default=Path('/content/drive/MyDrive/runs/project1/aimo_sft_v26'))
    parser.add_argument('--credentials', type=Path, default=Path('/content/drive/MyDrive/.project1_credentials/deepseek.json'))
    parser.add_argument('--review-only', action='store_true', help='Never load GPU models')
    parser.add_argument('--generate-only', action='store_true', help='GPU generation only; review separately on local CPU/API')
    parser.add_argument('--generate-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.generate_only and args.review_only:
        parser.error('--generate-only and --review-only cannot be combined')
    if args.generate_only and args.stage == 'review-dev':
        parser.error('review-dev has no generation; choose math500 or hard637')
    if args.stage == 'next':
        for stage in (['math500', 'hard637'] if args.generate_only else ['review-dev', 'math500', 'hard637']):
            command = [sys.executable, '-u', str(SCRIPT), '--stage', stage, '--project-root', str(args.project_root),
                '--run-root', str(args.run_root), '--credentials', str(args.credentials)]
            if args.review_only:
                command.append('--review-only')
            if args.generate_only:
                command.append('--generate-only')
            subprocess.run(command, check=True)
        if args.generate_only:
            print('ALL_GENERATION_COMPLETE: MATH-500 and hard637 raw saved; local CPU/API review still pending. GPU can be stopped.', flush=True)
        else:
            print('ALL_FINAL_EVALUATIONS_COMPLETE: dev review, MATH-500 and hard637 review saved. GPU can be stopped.', flush=True)
        return
    project, run = args.project_root.resolve(), args.run_root.resolve()
    os.chdir(project)
    sys.path.insert(0, str(project))
    from experiments.aimo_sft_v26.common import prompt_ids, SUFFIX
    if not (run / 'evaluations/final_dev256_32k_v32/COMPLETE.json').exists():
        raise RuntimeError('Wait for the currently running dev256 v32 to finish first; no second task started')
    api_key = None if args.generate_worker or args.generate_only else judge.credential(args.credentials)
    data, settings = sources(args.stage, project, run.parent)
    directory = run / 'evaluations' / ('final_dev256_32k_v32' if args.stage == 'review-dev' else f'final_{args.stage}_32k_v33')
    directory.mkdir(parents=True, exist_ok=True)
    if args.stage == 'review-dev':
        original = base.journal(directory / 'predictions.jsonl')
        raw = [{'sample_id': r['sample_id'], 'draw': 0, 'text': r['text'],
                'output_tokens': r['generated_tokens'], 'finish_reason': r['finish_reason']} for r in original]
        with judge.locked(directory / 'native_ds_v33.lock'):
            review(raw, data, settings, directory, project, api_key)
        print('DEV_REVIEW_COMPLETE: CPU/API task complete; no GPU was loaded.', flush=True)
        return
    model_identity = run / 'merged/sft_identity.json'
    generation_identity = {'script_sha256': base.sha(SCRIPT), 'source_sha256': judge.digest(data),
        'model': base.read(model_identity), 'settings': settings, 'prompt_messages': [{'role': 'user', 'content': '<problem>' + SUFFIX}],
        'prompt_function_sha256': base.sha(project / 'experiments/aimo_sft_v26/common.py'),
        'max_new_tokens': 32768, 'max_model_len': 40960, 'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768},
        'max_num_seqs': 16, 'max_num_batched_tokens': 8192, 'gpu_memory_utilization': 0.90}
    if (directory / 'identity.json').exists() and base.read(directory / 'identity.json') != generation_identity:
        raise RuntimeError('Generation identity differs; do not mix protocol outputs')
    base.save(directory / 'identity.json', generation_identity)
    raw_path = directory / 'raw.jsonl'
    raw = base.journal(raw_path)
    pending = inspect_raw(raw, data, settings['samples'])
    if args.generate_worker:
        model = Path('/content/models') / ('aimo_sft_v26_merged--' + base.sha(model_identity)[:12])
        model.mkdir(parents=True, exist_ok=True)
        stored = run / 'merged'
        needed = [p for p in stored.iterdir() if p.is_file() and
                  (not (model / p.name).exists() or p.stat().st_size != (model / p.name).stat().st_size)]
        if shutil.disk_usage(model).free < sum(p.stat().st_size for p in needed) + 2_000_000_000:
            raise RuntimeError('Insufficient SSD space for the existing Drive merged model')
        for source in needed:
            temporary = model / (source.name + '.copying')
            shutil.copyfile(source, temporary)
            temporary.replace(model / source.name)
        index = base.read(model / 'model.safetensors.index.json')
        if any(not (model / name).is_file() for name in set(index['weight_map'].values())):
            raise RuntimeError('Final merged model shard missing')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(project / 'data/tokenizers/Qwen2.5-7B-Instruct', local_files_only=True)
        expected_tokens = base.read(project / 'configs/project1_aimo_sft_v26.json')['tokenizer_sha256']
        if any(base.sha(project / 'data/tokenizers/Qwen2.5-7B-Instruct' / name) != value for name, value in expected_tokens.items()):
            raise RuntimeError('Pinned SFT tokenizer differs')
        tokens = {r['sample_id']: prompt_ids(tok, r['problem']) for r in data}
        if any(len(t) + 32768 > 40960 for t in tokens.values()):
            raise RuntimeError('Prompt exceeds context; no truncation')
        from vllm import LLM, SamplingParams
        engine = LLM(model=str(model), tokenizer=str(project / 'data/tokenizers/Qwen2.5-7B-Instruct'),
            dtype='bfloat16', tensor_parallel_size=1, max_model_len=40960, max_num_seqs=16,
            max_num_batched_tokens=8192, gpu_memory_utilization=0.90, enable_prefix_caching=True,
            enable_chunked_prefill=True, hf_overrides={'rope_scaling': generation_identity['rope_scaling']}, generation_config='vllm', seed=42)
        # 64 pending responses per durable batch; engine dynamically schedules 16.
        work = [(r, draw) for r in data for draw in range(settings['samples']) if (r['sample_id'], draw) in pending]
        for start in range(0, len(work), 64):
            batch = work[start:start + 64]
            params = [SamplingParams(n=1, max_tokens=32768, temperature=settings['temperature'], top_p=settings['top_p'],
                top_k=-1, seed=draw_seed(settings, r['sample_id'], draw)) for r, draw in batch]
            outputs = engine.generate([{'prompt_token_ids': tokens[r['sample_id']]} for r, _ in batch], params, use_tqdm=True)
            if len(outputs) != len(batch):
                raise RuntimeError('Generation count mismatch')
            for (row, draw), output in zip(batch, outputs):
                value = output.outputs[0]
                base.append(raw_path, {'sample_id': row['sample_id'], 'draw': draw, 'text': value.text,
                    'output_tokens': len(value.token_ids), 'finish_reason': value.finish_reason})
            print(f'{args.stage}: saved {len(base.journal(raw_path))}/{len(data) * settings["samples"]} responses', flush=True)
        print('GENERATION_COMPLETE: GPU child exits; CPU/API review follows.', flush=True)
        return
    with judge.locked(directory / 'pipeline_v33.lock'):
        if pending:
            if args.review_only:
                raise RuntimeError('Generation incomplete; --review-only will not load GPU')
            lock_path = run.parent / 'projection_pilot_v21/running.lock'
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with judge.locked(lock_path):
                subprocess.run([sys.executable, str(project / 'experiments/aimo_sft_v26/install_missing.py'), '--eval'], check=True)
                subprocess.run([sys.executable, '-u', str(SCRIPT), '--stage', args.stage,
                    '--project-root', str(project), '--run-root', str(run), '--generate-worker'],
                    env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn'), check=True)
        if args.generate_only:
            print(args.stage.upper() + '_GENERATION_ONLY_COMPLETE: raw saved; review on local CPU/API. No DS calls made.', flush=True)
            return
        review(base.journal(raw_path), data, settings, directory, project, api_key)
    print(args.stage.upper() + '_REVIEW_COMPLETE: this stage finished; no GPU model remains loaded.', flush=True)


if __name__ == '__main__':
    main()
