"""Only final SFT MATH-500, matching the historical E0 v15 protocol."""
import argparse
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import queue
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT = Path(__file__).resolve()
# Verbatim historical prompt, including its original backslash spelling.
SYSTEM = r'Please reason step by step, and put your final answer within \boxed{}.'
HISTORICAL_PROTOCOL = {
    'experiment': 'MATH500_three_models_v15', 'models': ['E0', 'raw', 'compact'],
    'data': 'data/math/math500_test.jsonl',
    'data_sha256': '4a8ada6bff87e172a59d78467764e781ae92460a4c6a3ec23f11e99ecd97c04e',
    'count': 500, 'seed': 42, 'temperature': 0, 'top_p': 1, 'top_k': -1,
    'max_new_tokens': 2048, 'max_num_seqs': 128, 'gpu_memory_utilization': 0.9,
    'grader_identity_sha256': '4b3c204fccf89eb377e9f4dee5326be04d417248262c87ba2732573a13dc1b2a',
    'purpose': 'User-authorized fixed test snapshot for E0 and completed E1 raw/compact; not for selecting models, tuning training or changing grading rules',
    'unsupported_policy': 'Report rejection categories separately; do not remove test rows or change the denominator',
}


class PooledVerifier:
    def __init__(self, count=4):
        from src.training.frozen_verifier_v4 import Worker
        self.workers = [Worker() for _ in range(count)]
        self.available = queue.Queue()
        for worker in self.workers:
            self.available.put(worker)

    def check(self, text, answer):
        worker = self.available.get()
        try:
            return worker.check(text, answer)
        finally:
            self.available.put(worker)

    def close(self):
        for worker in self.workers:
            worker.close()


def protocol_data(project):
    """The portable v26 bundle omits historical metadata, but seals the same 500 rows."""
    protocol_path = project / 'configs/project1_math500_protocol_v15.json'
    protocol = read(protocol_path) if protocol_path.exists() else HISTORICAL_PROTOCOL
    if protocol != HISTORICAL_PROTOCOL:
        raise RuntimeError('Historical E0 protocol differs')
    path = project / protocol['data']
    if path.exists():
        if sha(path) != protocol['data_sha256']:
            raise RuntimeError('Historical MATH-500 data hash mismatch')
        return protocol, path, journal(path)
    path = project / 'reports/aimo_sft_20261026_v26/math500.jsonl'
    data = [{**r, 'reference_answer': r['answer']} for r in journal(path)]
    # Packaging can change newline bytes. Check all original IDs, problems, golds
    # in order against a digest independently computed from the old E0 source.
    canonical = [(r['sample_id'], r['problem'], str(r['reference_answer'])) for r in data]
    serialized = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(serialized).hexdigest() != '2d87783f5679c481cbeeb475faf064f6ff08d4ad76c1e5fea5ca383a939ab4c0':
        raise RuntimeError('Portable questions/answers differ from historical E0')
    return protocol, path, data


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def journal(path):
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    if text and not text.endswith('\n'):
        raise RuntimeError('Partial journal tail; preserve and repair before resume')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def messages(problem, system):
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': problem}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('/content/workspaces/aimo-sft-v26'))
    parser.add_argument('--run-root', type=Path, default=Path('/content/drive/MyDrive/runs/project1/aimo_sft_v26'))
    parser.add_argument('--grade-only', action='store_true', help='CPU scoring only; never load GPU model')
    parser.add_argument('--worker', choices=['generate', 'grade'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    project = args.project_root.resolve()
    if not project.exists():
        if args.grade_only:
            raise RuntimeError('Restore project first with v28 --prepare-only')
        subprocess.run([sys.executable, '/content/drive/MyDrive/projects/project1/start_aimo_sft_v28.py',
                        '--prepare-only'], check=True)
    os.chdir(project)
    sys.path.insert(0, str(project))
    from src.training.frozen_verifier_v4 import grader_identity, digest
    protocol, data_path, data = protocol_data(project)
    frozen = grader_identity()
    if digest(frozen) != protocol['grader_identity_sha256']:
        raise RuntimeError('Historical E0 scorer identity differs; do not change frozen hash. Report this error for targeted repair.')
    by_id = {r['sample_id']: r for r in data}
    if len(data) != 500 or len(by_id) != 500:
        raise RuntimeError('Expected original 500 unique test questions')
    run = args.run_root.resolve()
    complete = read(run / 'TRAIN_COMPLETE.json')
    model_origin = read(run / 'merged/sft_identity.json')
    target = run / 'evaluations/final_math500_e0_protocol_v30'
    target.mkdir(parents=True, exist_ok=True)
    identity = {'script_sha256': sha(SCRIPT), 'protocol': protocol,
        'protocol_content_digest': digest(protocol), 'grader_identity_sha256': digest(frozen),
        'actual_data_sha256': sha(data_path),
        'model_origin': model_origin, 'model_steps': complete['steps'],
        'prompt_messages': messages('<original problem>', SYSTEM),
        'scope': 'Final merged SFT only; no dev selection, no E0 regeneration, no DS',
        'e0_historical_reference': {'correct': 382, 'count': 500, 'accuracy': 0.764}}
    if (target / 'identity.json').exists() and read(target / 'identity.json') != identity:
        raise RuntimeError('Existing evaluation identity differs; preserve previous output')
    save(target / 'identity.json', identity)
    raw_path = target / 'predictions.jsonl'
    raw = journal(raw_path)
    if len({r['sample_id'] for r in raw}) != len(raw) or any(r['sample_id'] not in by_id for r in raw):
        raise RuntimeError('Duplicate or unknown generated rows')
    if args.worker == 'generate':
        from transformers import AutoTokenizer
        stored = run / 'merged'
        model = Path('/content/models') / ('aimo_sft_v26_merged--' + sha(stored / 'sft_identity.json')[:12])
        model.mkdir(parents=True, exist_ok=True)
        files = [p for p in stored.iterdir() if p.is_file()]
        pending_files = [p for p in files if not (model / p.name).exists() or (model / p.name).stat().st_size != p.stat().st_size]
        if shutil.disk_usage(model).free < sum(p.stat().st_size for p in pending_files) + 2_000_000_000:
            raise RuntimeError('Insufficient SSD space for final merged model')
        for source in pending_files:
            temporary = model / (source.name + '.copying')
            shutil.copyfile(source, temporary)
            temporary.replace(model / source.name)
        index = read(model / 'model.safetensors.index.json')
        if any(not (model / name).is_file() for name in set(index['weight_map'].values())):
            raise RuntimeError('Merged model shard missing')
        tokenizer_path = project / 'data/tokenizers/Qwen2.5-7B-Instruct'
        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        expected = read(project / 'configs/project1_aimo_sft_v26.json')['tokenizer_sha256']
        if any(sha(tokenizer_path / name) != value for name, value in expected.items()):
            raise RuntimeError('Historical tokenizer changed')
        def encode(row):
            value = tok.apply_chat_template(messages(row['problem'], SYSTEM), tokenize=True,
                add_generation_prompt=True, return_dict=False, return_tensors=None, padding=False, truncation=False)
            return list(value['input_ids'] if isinstance(value, Mapping) else value)
        encoded = {r['sample_id']: encode(r) for r in data}
        from importlib.metadata import version
        save(target / 'generation_environment.json', {
            'versions': {name: version(name) for name in ('torch', 'transformers', 'vllm', 'tokenizers')},
            'max_model_len': max(map(len, encoded.values())) + 2048,
            'max_num_seqs': 128, 'max_num_batched_tokens': 8192,
            'prompt_token_ids_sha256': digest(encoded),
            'tokenizer_sha256': {name: sha(tokenizer_path / name) for name in expected}})
        done = {r['sample_id'] for r in raw}
        pending = [r for r in data if r['sample_id'] not in done]
        from vllm import LLM, SamplingParams
        engine = LLM(model=str(model), tokenizer=str(tokenizer_path), dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=max(map(len, encoded.values())) + 2048,
            max_num_seqs=128, gpu_memory_utilization=0.90,
            enable_prefix_caching=True, enable_chunked_prefill=True, max_num_batched_tokens=8192,
            seed=42, generation_config='vllm')
        # All remaining requests submitted together; vLLM fills freed slots dynamically.
        outputs = engine.generate([{'prompt_token_ids': encoded[r['sample_id']]} for r in pending],
            SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=2048, seed=42), use_tqdm=True)
        if len(outputs) != len(pending):
            raise RuntimeError('Generation response count mismatch')
        for row, output in zip(pending, outputs):
            if len(output.outputs) != 1:
                raise RuntimeError('Expected exactly one greedy answer')
            response = output.outputs[0]
            append(raw_path, {'sample_id': row['sample_id'], 'text': response.text,
                'token_ids': list(response.token_ids), 'generated_tokens': len(response.token_ids),
                'finish_reason': response.finish_reason})
        print('GENERATION_COMPLETE: 500 answers saved. GPU child will exit; CPU scoring follows.', flush=True)
        return
    if args.worker == 'grade':
        if len(raw) != 500:
            raise RuntimeError('Generation incomplete; no full MATH-500 score yet')
        from src.training.math_reward_v2 import category
        path = target / 'graded_predictions.jsonl'
        graded = journal(path)
        done = {r['sample_id'] for r in graded}
        if len(done) != len(graded) or not done <= set(by_id):
            raise RuntimeError('Invalid scored journal')
        checker = PooledVerifier(4)
        def grade(record):
            result = checker.check(record['text'], by_id[record['sample_id']]['reference_answer'])
            return {**record, 'verification': result, 'category': category(result)}
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for record in pool.map(grade, [r for r in raw if r['sample_id'] not in done]):
                    if record['category'] in ('timeout', 'verifier_error'):
                        raise RuntimeError('Scorer infrastructure failed; raw retained. Repeat command to resume CPU grading.')
                    append(path, record)
                    graded.append(record)
        finally:
            checker.close()
        counts = Counter(r['category'] for r in graded)
        finish = Counter(r['finish_reason'] for r in raw)
        correct = counts['equivalent']
        report = {'count': 500, 'correct': correct, 'correct_over_all': correct / 500,
            'status_counts': dict(counts), 'finish_reason_counts': dict(finish),
            'length_finish_rate': finish.get('length', 0) / 500,
            'unscorable_rate': (500 - counts['equivalent'] - counts['mismatch']) / 500,
            'mean_generated_tokens': sum(r['generated_tokens'] for r in raw) / 500,
            'historical_E0_correct': 382, 'historical_E0_accuracy': 0.764,
            'difference_percentage_points': (correct / 500 - 0.764) * 100,
            'protocol_content_digest': digest(protocol), 'grader_identity_sha256': digest(frozen),
            'prediction_sha256': sha(raw_path), 'model_steps': complete['steps'],
            'note': 'Historical E0 protocol matched; current inference library/hardware versions may differ. Final model, not dev-selected best.'}
        save(target / 'summary.json', report)
        save(target / 'COMPLETE.json', {'summary_sha256': sha(target / 'summary.json')})
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print('MATH500_E0_PROTOCOL_COMPLETE: all generation and scoring finished; GPU can be stopped.', flush=True)
        return
    import fcntl
    lock = (Path('/content/drive/MyDrive/runs/project1/projection_pilot_v21/running.lock'))
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another GPU task is still running; do not start a second evaluation') from None
        if (target / 'COMPLETE.json').exists():
            print(json.dumps(read(target / 'summary.json'), ensure_ascii=False, indent=2))
            print('MATH500_E0_PROTOCOL_COMPLETE: already complete; no GPU model loaded.')
            return
        command = [sys.executable, '-u', str(SCRIPT), '--project-root', str(project), '--run-root', str(run)]
        if len(raw) < 500:
            if args.grade_only:
                raise RuntimeError('Not all 500 outputs exist; generation still needs GPU')
            subprocess.run([sys.executable, str(project / 'experiments/aimo_sft_v26/install_missing.py'), '--eval'], check=True)
            env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn')
            subprocess.run(command + ['--worker', 'generate'], env=env, check=True)
        subprocess.run(command + ['--worker', 'grade'], check=True)


if __name__ == '__main__':
    main()
