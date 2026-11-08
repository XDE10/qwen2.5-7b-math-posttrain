"""16-question diagnostic: native/YaRN4 x greedy/sampling; GPU generation only.

Four arms share final merged weights, AIMO prompt, 8192 output budget, context,
request seeds, repetition penalty and scheduler. Local scoring is separate.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve()
ARMS = [('native', 'greedy'), ('native', 'sample'), ('yarn4', 'greedy'), ('yarn4', 'sample')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def seed_for(sid):
    return (42 + int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)) % 2**31


def sampling_config(mode, sid):
    return {'n': 1, 'max_tokens': 8192, 'temperature': 0.0 if mode == 'greedy' else 0.6,
            'top_p': 1.0 if mode == 'greedy' else 0.95, 'top_k': -1,
            'repetition_penalty': 1.0, 'seed': seed_for(sid)}


def validate_selection(selection):
    data = selection['rows']
    if len(data) != 16 or len({r['sample_id'] for r in data}) != 16:
        raise ValueError('Expected 16 unique frozen questions')
    if Counter(r['prior_finish_reason'] for r in data) != {'length': 8, 'stop': 8}:
        raise ValueError('Expected frozen 8 length + 8 stop strata')
    if any(not isinstance(r.get('problem'), str) or not str(r.get('answer', '')).strip() for r in data):
        raise ValueError('Question/gold missing')
    return data


def tail_repeat(token_ids):
    value = token_ids[-8192:]
    counts = Counter(tuple(value[i:i + 32]) for i in range(0, max(0, len(value) - 31), 4))
    return max(counts.values(), default=0)


def score_local(args, selection, data, output):
    identity = read(output / 'identity.json')
    if identity['selection_sha256'] != sha(args.selection):
        raise RuntimeError('Local selection differs from the actual GPU pilot')
    helper = SCRIPT.with_name('evaluate_sft_ds_v29.py')
    if not helper.exists():
        raise RuntimeError('Local native Math-Verify helper missing')
    spec = importlib.util.spec_from_file_location('pilot_native_grader', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    all_scores = {}
    golds = {r['sample_id']: r for r in data}
    for rope, mode in ARMS:
        name = rope + '_' + mode
        directory = output / name
        raw = rows(directory / 'raw.jsonl')
        if sha(directory / 'raw.jsonl') != read(directory / 'GENERATION_COMPLETE.json')['raw_sha256']:
            raise RuntimeError('Saved pilot raw hash differs: ' + name)
        if len(raw) != 16 or {r['sample_id'] for r in raw} != set(golds):
            raise RuntimeError('Incomplete arm: ' + name)
        scores = []
        for row in raw:
            value = module.grade_bounded(row['text'], str(golds[row['sample_id']]['answer']), args.project_root)
            scores.append({**row, 'native_grade': value, 'tail_repeat_max': tail_repeat(row['token_ids'])})
        result = {'count': 16, 'native_correct': sum(r['native_grade']['correct'] for r in scores),
            'native_accuracy': sum(r['native_grade']['correct'] for r in scores) / 16,
            'stop': sum(r['finish_reason'] == 'stop' for r in scores),
            'length': sum(r['finish_reason'] == 'length' for r in scores),
            'no_box': sum(r'\boxed' not in r['text'] for r in scores),
            'repeated_tail_window_at_least_4': sum(r['tail_repeat_max'] >= 4 for r in scores),
            'mean_output_tokens': sum(len(r['token_ids']) for r in scores) / 16,
            'native_statuses': dict(Counter(r['native_grade']['status'] for r in scores)),
            'timing': read(directory / 'GENERATION_COMPLETE.json'),
            'by_prior_stratum': {stratum: {
                'count': 8, 'length': sum(r['finish_reason'] == 'length' for r in scores if golds[r['sample_id']]['prior_finish_reason'] == stratum),
                'native_correct': sum(r['native_grade']['correct'] for r in scores if golds[r['sample_id']]['prior_finish_reason'] == stratum)}
                for stratum in ('length', 'stop')}}
        save(directory / 'native_scores.json', scores)
        save(directory / 'summary.json', result)
        all_scores[name] = result
    save(output / 'comparison.json', {'arms': all_scores, 'selection': selection,
        'limit': '16 stratified diagnostic questions, one sample per arm; not a full test accuracy or conclusive causal attribution.'})
    print(json.dumps(all_scores, ensure_ascii=False, indent=2), flush=True)
    print('PILOT16_LOCAL_SCORING_COMPLETE: no GPU or DS API used.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('/content/workspaces/aimo-sft-v26'))
    parser.add_argument('--run-root', type=Path, default=Path('/content/drive/MyDrive/runs/project1/aimo_sft_v26'))
    parser.add_argument('--selection', type=Path, default=SCRIPT.with_name('pilot16_rope_decode_v39.json'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--score-local', action='store_true')
    parser.add_argument('--worker-rope', choices=['native', 'yarn4'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    selection = read(args.selection)
    data = validate_selection(selection)
    output = args.output_root or args.run_root / 'evaluations/pilot16_rope_decode_v39'
    output.mkdir(parents=True, exist_ok=True)
    if args.score_local:
        score_local(args, selection, data, output)
        return
    if not args.project_root.is_dir():
        subprocess.run([sys.executable, '/content/drive/MyDrive/projects/project1/start_aimo_sft_v28.py', '--prepare-only'], check=True)
    project = args.project_root.resolve()
    os.chdir(project)
    sys.path.insert(0, str(project))
    from experiments.aimo_sft_v26.common import prompt_ids, SUFFIX
    completed = read(args.run_root / 'TRAIN_COMPLETE.json')
    if completed['steps'] != 429:
        raise RuntimeError('Expected the completed final step-429 model')
    model_identity = args.run_root / 'merged/sft_identity.json'
    identity = {'selection_sha256': sha(args.selection), 'script_sha256': sha(SCRIPT),
        'source_predictions_sha256': selection['source_predictions_sha256'], 'model_identity': read(model_identity),
        'common_sha256': sha(project / 'experiments/aimo_sft_v26/common.py'),
        'prompt': [{'role': 'user', 'content': '<problem>' + SUFFIX}],
        'max_new_tokens': 8192, 'max_model_len': 16384, 'max_num_seqs': 16,
        'max_num_batched_tokens': 8192, 'repetition_penalty': 1.0,
        'comparison': 'same final model; RoPE setting x decoding recipe only; no budget forcing or repetition-penalty ablation'}
    if (output / 'identity.json').exists() and read(output / 'identity.json') != identity:
        raise RuntimeError('Pilot identity differs; preserve existing run')
    save(output / 'identity.json', identity)
    if args.worker_rope:
        from transformers import AutoTokenizer
        stored = args.run_root / 'merged'
        model = Path('/content/models') / ('aimo_sft_v26_merged--' + sha(model_identity)[:12])
        model.mkdir(parents=True, exist_ok=True)
        files = [p for p in stored.iterdir() if p.is_file()]
        missing = [p for p in files if not (model / p.name).exists() or p.stat().st_size != (model / p.name).stat().st_size]
        if shutil.disk_usage(model).free < sum(p.stat().st_size for p in missing) + 2_000_000_000:
            raise RuntimeError('Insufficient SSD space for existing Drive merged model')
        for source in missing:
            temporary = model / (source.name + '.copying')
            shutil.copyfile(source, temporary)
            temporary.replace(model / source.name)
        index = read(model / 'model.safetensors.index.json')
        if any(not (model / name).is_file() for name in set(index['weight_map'].values())):
            raise RuntimeError('Missing merged shard')
        tok_path = project / 'data/tokenizers/Qwen2.5-7B-Instruct'
        expected = read(project / 'configs/project1_aimo_sft_v26.json')['tokenizer_sha256']
        if any(sha(tok_path / name) != value for name, value in expected.items()):
            raise RuntimeError('Tokenizer changed')
        tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        tokens = {r['sample_id']: prompt_ids(tok, r['problem']) for r in data}
        if any(len(ids) + 8192 > 16384 for ids in tokens.values()):
            raise RuntimeError('Pilot prompt+output exceeds common context')
        rope = None if args.worker_rope == 'native' else {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
        from vllm import LLM, SamplingParams
        from importlib.metadata import version
        init_started = time.monotonic()
        engine = LLM(model=str(model), tokenizer=str(tok_path), dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=16384, max_num_seqs=16, max_num_batched_tokens=8192,
            gpu_memory_utilization=0.90, enable_chunked_prefill=True, enable_prefix_caching=True,
            hf_overrides={'rope_scaling': rope}, generation_config='vllm', seed=42)
        init_seconds = time.monotonic() - init_started
        model_config = getattr(engine.llm_engine, 'model_config', None)
        if model_config is None:
            model_config = getattr(getattr(engine.llm_engine, 'vllm_config', None), 'model_config', None)
        hf_config = getattr(model_config, 'hf_config', None)
        if hf_config is None:
            raise RuntimeError('Cannot read effective RoPE configuration; do not label this a valid ablation')
        effective_config = hf_config.to_dict()
        effective_rope = {name: effective_config.get(name) for name in ('rope_scaling', 'rope_parameters', 'rope_theta')}
        is_yarn = 'yarn' in json.dumps(effective_rope).lower()
        if is_yarn != (args.worker_rope == 'yarn4'):
            raise RuntimeError('Effective RoPE differs from requested pilot arm')
        save(output / (args.worker_rope + '_environment.json'), {
            'versions': {n: version(n) for n in ('torch', 'transformers', 'vllm', 'tokenizers')},
            'requested_rope_scaling': rope, 'effective_rope': effective_rope, 'prompt_token_ids_sha256': digest(tokens),
            'model_config_sha256': sha(model / 'config.json'), 'engine_initialization_seconds': init_seconds,
            'timing_note': 'Each RoPE engine is reused for greedy then sampling; prefix-cache warming affects timing. Not a pure kernel-speed comparison.'})
        for mode in ('greedy', 'sample'):
            name = args.worker_rope + '_' + mode
            directory = output / name
            if (directory / 'GENERATION_COMPLETE.json').exists():
                print('Already complete, skipped: ' + name, flush=True)
                continue
            directory.mkdir(exist_ok=True)
            params = [SamplingParams(**sampling_config(mode, r['sample_id'])) for r in data]
            print(f'START {name}: 16 questions, max_new_tokens=8192', flush=True)
            started = time.monotonic()
            responses = engine.generate([{'prompt_token_ids': tokens[r['sample_id']]} for r in data], params, use_tqdm=True)
            seconds = time.monotonic() - started
            if len(responses) != 16:
                raise RuntimeError('Pilot response count mismatch')
            raw = []
            for row, response in zip(data, responses):
                if len(response.outputs) != 1:
                    raise RuntimeError('Expected one response per question')
                value = response.outputs[0]
                raw.append({'sample_id': row['sample_id'], 'prior_finish_reason': row['prior_finish_reason'],
                    'text': value.text, 'token_ids': list(value.token_ids), 'output_tokens': len(value.token_ids),
                    'finish_reason': value.finish_reason, 'sampling_params': sampling_config(mode, row['sample_id'])})
            temporary = directory / 'raw.jsonl.writing'
            temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in raw), encoding='utf-8')
            temporary.replace(directory / 'raw.jsonl')
            save(directory / 'GENERATION_COMPLETE.json', {'count': 16, 'generation_seconds': seconds,
                'generated_tokens': sum(r['output_tokens'] for r in raw), 'raw_sha256': sha(directory / 'raw.jsonl')})
            print(f'COMPLETE {name}: 16/16 saved; {seconds:.1f} seconds', flush=True)
        return
    import fcntl
    lock_path = args.run_root.parent / 'projection_pilot_v21/running.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another project GPU task is running; no second engine started') from None
        subprocess.run([sys.executable, str(project / 'experiments/aimo_sft_v26/install_missing.py'), '--eval'], check=True)
        for rope in ('native', 'yarn4'):
            if all((output / (rope + '_' + mode) / 'GENERATION_COMPLETE.json').exists() for mode in ('greedy', 'sample')):
                continue
            subprocess.run([sys.executable, '-u', str(SCRIPT), '--project-root', str(project), '--run-root', str(args.run_root),
                '--selection', str(args.selection.resolve()), '--output-root', str(output), '--worker-rope', rope],
                env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn'), check=True)
        save(output / 'ALL_GENERATION_COMPLETE.json', {'responses': 64, 'arms': [r + '_' + d for r, d in ARMS]})
        print('PILOT16_ALL_GENERATION_COMPLETE: 64 answers saved; GPU can be stopped. Local scoring still pending.', flush=True)


if __name__ == '__main__':
    main()
