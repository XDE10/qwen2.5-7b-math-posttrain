"""RomEscape public s1.1 QFFT MATH-500, 7936 initial + optional 256 final-answer tokens.

GPU generation only. Keeps pre-intervention output and forced continuation.
No gold answer is sent to the model. This is a distinct inference protocol.
"""
import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import math
import statistics

SCRIPT = Path(__file__).resolve()
TOTAL = 8192
ANSWER = 256
REASON = TOTAL - ANSWER
CONTEXT = 16384
DIRECT = ('\n\nStop reasoning now. Based on the work above, give your single final answer immediately. '
          'Do not explain or repeat the derivation. Put the final answer in \\boxed{}.\n')
MODEL_ID = 'PJiNH/QFFT-repro-S1.1-QFFT-7B'
MODEL_REVISION = 'e8a801fb66a54b86f4219689a6d9973d91330d25'
ROPE = None
SYSTEM = r'Please reason step by step, and put your final answer within \boxed{}.'



SAMPLING = {'n': 1, 'temperature': 0.6, 'top_p': 0.95, 'top_k': -1,
            'repetition_penalty': 1.0, 'skip_special_tokens': False}
EXPECTED_SHARDS = {'model-00001-of-00004.safetensors': 4877660776,
                   'model-00002-of-00004.safetensors': 4932751008,
                   'model-00003-of-00004.safetensors': 4330865200,
                   'model-00004-of-00004.safetensors': 1089994880}


def sampling_params(factory, sid, cap):
    return factory(**SAMPLING, max_tokens=cap, seed=seed(sid))


def token_statistics(original, final):
    if not original or len(original) != len(final):
        raise RuntimeError('Token statistics require matched nonempty outputs')
    before = {r['sample_id']: r for r in original}
    after = {r['sample_id']: r for r in final}
    if len(before) != len(original) or len(after) != len(final) or set(before) != set(after):
        raise RuntimeError('Token statistics IDs differ or repeat')
    for sid, row in after.items():
        if row['output_tokens'] != len(row['token_ids']) or before[sid]['output_tokens'] != len(before[sid]['token_ids']):
            raise RuntimeError('Saved token count differs')
        expected = before[sid]['output_tokens'] + row['output_tokens'] if row['forced'] else before[sid]['output_tokens']
        if row['total_generated_tokens'] != expected or expected > TOTAL or before[sid]['output_tokens'] > REASON or (row['forced'] and row['output_tokens'] > ANSWER):
            raise RuntimeError('Saved output violates the 7936+256 budget')
    lengths = sorted(r['output_tokens'] for r in original)
    totals = sorted(r['total_generated_tokens'] for r in final)
    return {'count': len(original), 'samples_per_problem': 1,
            'original_mean_tokens': statistics.mean(lengths), 'original_median_tokens': statistics.median(lengths),
            'original_p95_tokens': lengths[math.ceil(0.95 * len(lengths))-1], 'original_max_tokens': max(lengths),
            'original_total_tokens': sum(lengths), 'total_generated_mean_tokens': statistics.mean(totals),
            'total_generated_tokens_including_answer_continuation': sum(totals),
            'original_length_count': sum(r['finish_reason'] == 'length' for r in original),
            'forced_count': sum(r['forced'] for r in final),
            'original_without_box': sum(r'\boxed' not in r['text'] for r in original),
            'final_without_box': sum(r'\boxed' not in r['text'] for r in final),
            'note': 'Lengths are over all500 outputs, including failures and truncations. Final total includes original reasoning plus optional continuation; never just continuation length.'}


def verify_model_shards(model):
    names = set(read(model / 'model.safetensors.index.json')['weight_map'].values())
    if names != set(EXPECTED_SHARDS):
        raise RuntimeError('QFFT checkpoint index differs from the pinned public model')
    for name, size in EXPECTED_SHARDS.items():
        if not (model / name).is_file() or (model / name).stat().st_size != size:
            raise RuntimeError('Missing or incomplete QFFT shard: ' + name)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    tmp = Path(str(path) + '.writing')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def rows(path):
    if not path.exists():
        return []
    value = path.read_text(encoding='utf-8')
    if value and not value.endswith('\n'):
        raise RuntimeError('Partial journal tail; preserve and repair before resume: ' + str(path))
    return [json.loads(line) for line in value.splitlines() if line.strip()]


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def by_id(values, expected):
    result = {r['sample_id']: r for r in values}
    if len(result) != len(values) or not set(result) <= expected:
        raise RuntimeError('Duplicate or unexpected saved sample IDs')
    return result


def seed(sid):
    return (42 + int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)) % 2**31


def prompt_token_ids(tok, problem):
    value = tok.apply_chat_template([{'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': problem}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    # Transformers releases differ: normalize list/BatchEncoding/tensor here,
    # before context length checking, JSON hashing and loading the GPU engine.
    if isinstance(value, Mapping):
        value = value['input_ids']
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value or any(type(x) is not int or x < 0 for x in value):
        raise RuntimeError('Tokenizer did not return a single nonempty integer token sequence')
    return list(value)


def prepare_identity(output, identity):
    path = output / 'identity.json'
    if path.exists() and read(path) != identity:
        raise RuntimeError('Identity differs; preserve existing evaluation')
    save(path, identity)


def assert_gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                             '--format=csv,noheader'], capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError('GPU compute process already exists; no duplicate task started: ' + result.stdout.strip())


def needs_answer(row):
    return row['finish_reason'] == 'length' or r'\boxed' not in row['text']


def answer_suffix(text):
    # s1 trains on Qwen role markers, not XML <think> tags.
    # If already in answer mode, do not insert another answer marker.
    marker = '' if '<|im_start|>answer' in text else '\n<|im_start|>answer\n'
    return marker + DIRECT + 'Final Answer: '


def continuation_ids(prompt_ids, row, tok):
    # Keep exact generated IDs: decoding and retokenizing 32k text can change it.
    return list(prompt_ids) + row['token_ids'] + tok.encode(answer_suffix(row['text']), add_special_tokens=False)


def math_data(project):
    data = rows(project / 'reports/aimo_sft_20261026_v26/math500.jsonl')
    triples = [(r['sample_id'], r['problem'], str(r['answer'])) for r in data]
    digest = hashlib.sha256(json.dumps(triples, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if len(data) != 500 or len({r['sample_id'] for r in data}) != 500 or digest != '2d87783f5679c481cbeeb475faf064f6ff08d4ad76c1e5fea5ca383a939ab4c0':
        raise RuntimeError('MATH-500 differs from sealed original questions/golds')
    return data, digest


def score_local(project, output, data):
    spec = importlib.util.spec_from_file_location('native_v41', SCRIPT.with_name('evaluate_sft_ds_v29.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    complete = read(output / 'GENERATION_COMPLETE.json')
    if complete['raw_sha256'] != sha(output / 'raw.jsonl') or complete['reasoning_sha256'] != sha(output / 'reasoning.jsonl'):
        raise RuntimeError('Saved raw hash differs')
    expected = {r['sample_id'] for r in data}
    original = by_id(rows(output / 'reasoning.jsonl'), expected)
    final = by_id(rows(output / 'raw.jsonl'), expected)
    if set(original) != expected or set(final) != expected:
        raise RuntimeError('Generation incomplete')
    scored = []
    for index, item in enumerate(data, 1):
        sid = item['sample_id']
        before = helper.grade_bounded(original[sid]['text'], str(item['answer']), project)
        after = helper.grade_bounded(final[sid]['text'], str(item['answer']), project)
        scored.append({'sample_id': sid, 'forced': final[sid]['forced'], 'original_grade': before,
                       'final_grade': after, 'final_has_box': r'\boxed' in final[sid]['text']})
        if index % 25 == 0:
            print(f'LOCAL_SCORE {index}/500', flush=True)
    save(output / 'native_scores.json', scored)
    summary = {'count': 500, 'original_native_correct': sum(r['original_grade']['correct'] for r in scored),
               'final_native_correct': sum(r['final_grade']['correct'] for r in scored),
               'final_native_correct_with_box': sum(r['final_grade']['correct'] and r['final_has_box'] for r in scored),
               'forced_count': sum(r['forced'] for r in scored),
               'original_finish_reasons': dict(Counter(r['finish_reason'] for r in original.values())),
               'answer_continuation_finish_reasons': dict(Counter(r['finish_reason'] for r in final.values() if r['forced'])),
               'note': 'Native answer-equivalence scores, not proof verification. Forced rows score only the new answer continuation; original output kept and scored separately. No DS used.'}
    summary['token_statistics'] = token_statistics(list(original.values()), list(final.values()))
    save(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('ROMESCAPE_S1_QFFT_MATH500_V58_LOCAL_SCORING_COMPLETE', flush=True)


def worker(args, project, output, data):
    sys.path.insert(0, str(project))
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from huggingface_hub import snapshot_download
    from importlib.metadata import version
    stored = args.run_root.parents[2] / 'models' / ('RomEscape-s1-QFFT-7B--' + MODEL_REVISION[:12])
    # The standard run-root is MyDrive/runs/project1/aimo_sft_v26.
    # parents[2] is MyDrive: do not store this downloaded model in a run directory.
    print('MODEL_DOWNLOAD_OR_REUSE: pinned RomEscape S1.1 QFFT (~15.23GB BF16); inference only.', flush=True)
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(stored), token=False,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    model = Path('/content/models') / ('RomEscape-s1-QFFT-7B--' + MODEL_REVISION[:12])
    model.mkdir(parents=True, exist_ok=True)
    missing = [p for p in stored.iterdir() if p.is_file() and
               (not (model / p.name).exists() or (model / p.name).stat().st_size != p.stat().st_size)]
    if shutil.disk_usage(model).free < sum(p.stat().st_size for p in missing) + 2_000_000_000:
        raise RuntimeError('Insufficient SSD space for public QFFT model')
    print('MODEL_SSD_COPY: missing files=' + str(len(missing)), flush=True)
    for source in missing:
        temporary = model / (source.name + '.copying')
        shutil.copyfile(source, temporary)
        temporary.replace(model / source.name)
    verify_model_shards(model)
    tok_path = model
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    prompts = {r['sample_id']: prompt_token_ids(tok, r['problem']) for r in data}
    prompt_digest = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    if any(len(ids) + TOTAL + 256 > CONTEXT for ids in prompts.values()):
        raise RuntimeError('Prompt plus generation and intervention exceeds context')
    engine = LLM(model=str(model), tokenizer=str(tok_path), dtype='bfloat16', tensor_parallel_size=1,
                 max_model_len=CONTEXT, max_num_seqs=16, max_num_batched_tokens=8192,
                 gpu_memory_utilization=0.90, enable_chunked_prefill=True, enable_prefix_caching=True,
                 hf_overrides={'rope_scaling': ROPE}, generation_config='vllm', seed=42)
    config = getattr(engine.llm_engine, 'model_config', None)
    if config is None:
        config = getattr(getattr(engine.llm_engine, 'vllm_config', None), 'model_config', None)
    hf = getattr(config, 'hf_config', None)
    if hf is None:
        raise RuntimeError('Cannot read effective public model configuration')
    if 'yarn' in json.dumps(hf.to_dict()).lower():
        raise RuntimeError('Unexpected YaRN in public model native-context evaluation')
    save(output / 'environment.json', {'versions': {n: version(n) for n in ('torch', 'transformers', 'vllm')},
                                    'effective_hf_config': hf.to_dict(), 'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                                    'tokenizer_sha256': sha(model / 'tokenizer.json'),
                                    'prompt_token_ids_sha256': prompt_digest})
    expected = set(prompts)
    original = by_id(rows(output / 'reasoning.jsonl'), expected)
    final = by_id(rows(output / 'raw.jsonl'), expected)
    if not set(final) <= set(original):
        raise RuntimeError('Final output without original journal')
    # Persist every 16 questions, so a disconnect never loses the whole500.
    for start in range(0, 500, 16):
        batch = data[start:start + 16]
        pending = [r for r in batch if r['sample_id'] not in original]
        if pending:
            print(f'REASONING batch {start + 1}-{min(start + 16, 500)}/500; saved={len(original)}; cap={REASON}', flush=True)
            responses = engine.generate([{'prompt_token_ids': prompts[r['sample_id']]} for r in pending],
                [sampling_params(SamplingParams, r['sample_id'], REASON) for r in pending], use_tqdm=True)
            if len(responses) != len(pending):
                raise RuntimeError('Generation count mismatch')
            for item, response in zip(pending, responses):
                value = response.outputs[0]
                row = {'sample_id': item['sample_id'], 'text': value.text, 'token_ids': list(value.token_ids),
                       'output_tokens': len(value.token_ids), 'finish_reason': value.finish_reason}
                append(output / 'reasoning.jsonl', row)
                original[item['sample_id']] = row
        force = []
        for item in batch:
            sid = item['sample_id']
            if sid in final:
                continue
            if needs_answer(original[sid]):
                force.append(item)
            else:
                row = {**original[sid], 'forced': False, 'total_generated_tokens': original[sid]['output_tokens']}
                append(output / 'raw.jsonl', row)
                final[sid] = row
        if force:
            print(f'DIRECT_ANSWER {len(force)} questions; cap={ANSWER}; no gold supplied', flush=True)
            continuation = [continuation_ids(prompts[r['sample_id']], original[r['sample_id']], tok) for r in force]
            if any(len(ids) + ANSWER > CONTEXT for ids in continuation):
                raise RuntimeError('Forced-answer context overflow; no prompt truncated')
            responses = engine.generate([{'prompt_token_ids': ids} for ids in continuation],
                [sampling_params(SamplingParams, r['sample_id'], ANSWER) for r in force], use_tqdm=True)
            if len(responses) != len(force):
                raise RuntimeError('Continuation count mismatch')
            for item, response in zip(force, responses):
                sid = item['sample_id']
                value = response.outputs[0]
                row = {'sample_id': sid, 'forced': True, 'text': value.text, 'token_ids': list(value.token_ids),
                       'output_tokens': len(value.token_ids), 'finish_reason': value.finish_reason,
                       'intervention_text': answer_suffix(original[sid]['text']),
                       'original_finish_reason': original[sid]['finish_reason'],
                       'total_generated_tokens': original[sid]['output_tokens'] + len(value.token_ids)}
                if row['total_generated_tokens'] > TOTAL:
                    raise RuntimeError('Generated token budget exceeded')
                append(output / 'raw.jsonl', row)
                final[sid] = row
        print(f'MATH500_PROGRESS {len(final)}/500 saved; forced={sum(r["forced"] for r in final.values())}', flush=True)
    if set(original) != expected or set(final) != expected:
        raise RuntimeError('Incomplete MATH-500 outputs')
    save(output / 'GENERATION_COMPLETE.json', {'count': 500, 'raw_sha256': sha(output / 'raw.jsonl'),
                                              'reasoning_sha256': sha(output / 'reasoning.jsonl')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('/content/workspaces/aimo-sft-v26'))
    parser.add_argument('--run-root', type=Path, default=Path('/content/drive/MyDrive/runs/project1/aimo_sft_v26'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--score-local', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.score_local:
        print('QFFT_PROTOCOL: one sample per question; 7936+256 output; native RoPE; temp0.6/top_p0.95/rp1.0; generation_config=vllm; no DS or training.', flush=True)
        assert_gpu_idle()
    if not args.project_root.is_dir() and not args.score_local:
        subprocess.run([sys.executable, str(SCRIPT.with_name('start_aimo_sft_v28.py')), '--prepare-only'], check=True)
    project = args.project_root.resolve()
    args.run_root = args.run_root.resolve()
    output = args.output_root or args.run_root / 'evaluations/public_romescape_s1_qfft_7b_math500_v58'
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data, data_sha = math_data(project)
    if args.score_local:
        score_local(project, output, data)
        return
    identity = {'script_sha256': sha(SCRIPT), 'math500_sha256': data_sha,
                'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'dtype': 'bfloat16',
                'prompt': 'published Qwen chat template; system boxed instruction + user original problem',
                'total_generated_token_cap': TOTAL, 'reasoning_cap': REASON, 'answer_cap': ANSWER,
                'max_model_len': CONTEXT, 'rope': ROPE, 'temperature': 0.6, 'top_p': 0.95,
                'repetition_penalty': 1.0, 'seed_recipe': '42+sha256(sample_id)[:8] mod2**31',
                'max_num_seqs': 16, 'batch_size_for_persistence': 16, 'direct_answer_prompt': DIRECT,
                'force_when': 'length or no boxed marker',
                'continuation': 'exact original prompt+generated token_ids+suffix; s1 answer marker when needed; no Wait extensions',
                'samples_per_problem': 1,
                'comparison_note': 'Same500/messages/seed/8K budget/sampling/native RoPE/scorer as RomEscape SFT v52 sample_00. QFFT training excluded questions, but inference receives the original question normally. Not author 32K/n16 reproduction.',
                'scoring': 'original and answer continuation separately; no repeated generation or gold in prompt'}
    prepare_identity(output, identity)
    if args.worker:
        assert_gpu_idle()
        worker(args, project, output, data)
        return
    if not (output / 'GENERATION_COMPLETE.json').exists():
        import fcntl
        lock = args.run_root.parent / 'projection_pilot_v21/running.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another project GPU task is active; no second task started') from None
            assert_gpu_idle()
            subprocess.run([sys.executable, str(project / 'experiments/aimo_sft_v26/install_missing.py'), '--eval'], check=True)
            subprocess.run([sys.executable, '-u', str(SCRIPT), '--worker', '--project-root', str(project),
                            '--run-root', str(args.run_root), '--output-root', str(output)],
                           env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn'), check=True)
    complete = read(output / 'GENERATION_COMPLETE.json')
    if complete['count'] != 500 or complete['raw_sha256'] != sha(output / 'raw.jsonl') or complete['reasoning_sha256'] != sha(output / 'reasoning.jsonl'):
        raise RuntimeError('Completion file integrity check failed')
    stats = token_statistics(rows(output / 'reasoning.jsonl'), rows(output / 'raw.jsonl'))
    save(output / 'generation_token_statistics.json', stats)
    print('QFFT_TOKEN_STATISTICS ' + json.dumps(stats), flush=True)
    print('ROMESCAPE_S1_QFFT_MATH500_V58_ALL_GENERATION_COMPLETE: 500 answers saved; GPU can be stopped. Local scoring pending.', flush=True)


if __name__ == '__main__':
    main()
