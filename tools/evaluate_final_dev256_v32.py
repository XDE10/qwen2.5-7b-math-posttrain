"""Final SFT dev256 only, 32768 output tokens; same prompt function as AIMO SFT."""
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
HISTORICAL_PROTOCOL = {
    'experiment': 'final_dev256_32k_v32', 'models': ['final_sft'],
    'data': 'data/math/math_dev.jsonl',
    'data_sha256': 'b5b3c90eeea953ae954cc2aa652caff4bd9c6edcb95f1b896215b84772225023',
    'count': 256, 'seed': 42, 'temperature': 0, 'top_p': 1, 'top_k': -1,
    'max_new_tokens': 32768, 'max_num_seqs': 16, 'gpu_memory_utilization': 0.9,
    'grader_identity_sha256': '4b3c204fccf89eb377e9f4dee5326be04d417248262c87ba2732573a13dc1b2a',
    'purpose': 'Final SFT dev256 only; same question set and frozen scorer, AIMO SFT prompt and 32768 generation budget',
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
    protocol = dict(HISTORICAL_PROTOCOL, experiment='final_dev256_32k_v32',
                    data='data/math/math_dev.jsonl',
                    data_sha256='b5b3c90eeea953ae954cc2aa652caff4bd9c6edcb95f1b896215b84772225023',
                    count=256, max_new_tokens=32768, max_num_seqs=16)
    path = project / protocol['data']
    if path.exists():
        if sha(path) != protocol['data_sha256']:
            raise RuntimeError('Historical dev256 data hash mismatch')
        return protocol, path, journal(path)
    path = project / 'reports/aimo_sft_20261026_v26/dev.jsonl'
    data = [{**r, 'reference_answer': r['answer']} for r in journal(path)]
    canonical = [(r['sample_id'], r['problem'], str(r['reference_answer'])) for r in data]
    serialized = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(serialized).hexdigest() != 'f4e5e522198147594988c20202f5be00577cdd1520072a4c7a88249618d8a5ac':
        raise RuntimeError('Portable dev256 differs from historical E0')
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
    from experiments.aimo_sft_v26.common import prompt_ids, SUFFIX
    protocol, data_path, data = protocol_data(project)
    frozen = grader_identity()
    if digest(frozen) != protocol['grader_identity_sha256']:
        raise RuntimeError('Historical E0 scorer identity differs; do not change frozen hash. Report this error for targeted repair.')
    by_id = {r['sample_id']: r for r in data}
    if len(data) != 256 or len(by_id) != 256:
        raise RuntimeError('Expected original 256 unique dev questions')
    run = args.run_root.resolve()
    complete = read(run / 'TRAIN_COMPLETE.json')
    model_origin = read(run / 'merged/sft_identity.json')
    target = run / 'evaluations/final_dev256_32k_v32'
    target.mkdir(parents=True, exist_ok=True)
    identity = {'script_sha256': sha(SCRIPT), 'protocol': protocol,
        'protocol_content_digest': digest(protocol), 'grader_identity_sha256': digest(frozen),
        'actual_data_sha256': sha(data_path),
        'model_origin': model_origin, 'model_steps': complete['steps'],
        'prompt_messages': [{'role': 'user', 'content': '<original problem>' + SUFFIX}],
        'prompt_function_sha256': sha(project / 'experiments/aimo_sft_v26/common.py'),
        'scope': 'Final merged SFT dev256 only; 32768 max generation; no checkpoint selection or other benchmarks; no DS',
        'context': {'max_model_len': 40960, 'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}},
        'e0_historical_reference': {'correct': 200, 'count': 256, 'accuracy': 0.78125}}
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
            return prompt_ids(tok, row['problem'])
        encoded = {r['sample_id']: encode(r) for r in data}
        from importlib.metadata import version
        save(target / 'generation_environment.json', {
            'versions': {name: version(name) for name in ('torch', 'transformers', 'vllm', 'tokenizers')},
            'max_model_len': 40960,
            'max_num_seqs': 16, 'max_num_batched_tokens': 8192,
            'prompt_token_ids_sha256': digest(encoded),
            'tokenizer_sha256': {name: sha(tokenizer_path / name) for name in expected}})
        done = {r['sample_id'] for r in raw}
        pending = [r for r in data if r['sample_id'] not in done]
        from vllm import LLM, SamplingParams
        engine = LLM(model=str(model), tokenizer=str(tokenizer_path), dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=40960,
            max_num_seqs=16, gpu_memory_utilization=0.90,
            enable_prefix_caching=True, enable_chunked_prefill=True, max_num_batched_tokens=8192,
            seed=42, generation_config='vllm',
            hf_overrides={'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}})
        # All remaining requests submitted together; vLLM fills freed slots dynamically.
        outputs = engine.generate([{'prompt_token_ids': encoded[r['sample_id']]} for r in pending],
            SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=32768, seed=42), use_tqdm=True)
        if len(outputs) != len(pending):
            raise RuntimeError('Generation response count mismatch')
        for row, output in zip(pending, outputs):
            if len(output.outputs) != 1:
                raise RuntimeError('Expected exactly one greedy answer')
            response = output.outputs[0]
            append(raw_path, {'sample_id': row['sample_id'], 'text': response.text,
                'token_ids': list(response.token_ids), 'generated_tokens': len(response.token_ids),
                'finish_reason': response.finish_reason})
        print('GENERATION_COMPLETE: 256 answers saved. GPU child will exit; CPU scoring follows.', flush=True)
        return
    if args.worker == 'grade':
        if len(raw) != 256:
            raise RuntimeError('Generation incomplete; no full dev256 score yet')
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
        report = {'count': 256, 'correct': correct, 'correct_over_all': correct / 256,
            'status_counts': dict(counts), 'finish_reason_counts': dict(finish),
            'length_finish_rate': finish.get('length', 0) / 256,
            'unscorable_rate': (256 - counts['equivalent'] - counts['mismatch']) / 256,
            'mean_generated_tokens': sum(r['generated_tokens'] for r in raw) / 256,
            'historical_E0_correct': 200, 'historical_E0_accuracy': 0.78125,
            'difference_percentage_points': (correct / 256 - 0.78125) * 100,
            'protocol_content_digest': digest(protocol), 'grader_identity_sha256': digest(frozen),
            'prediction_sha256': sha(raw_path), 'model_steps': complete['steps'],
            'note': 'Same dev questions/scorer; prompt matches AIMO SFT, max generation 32768 and YaRN4 enabled. Old E0 used a different message format and 2048 budget; reference only. Final checkpoint, not dev-selected best.'}
        save(target / 'summary.json', report)
        save(target / 'COMPLETE.json', {'summary_sha256': sha(target / 'summary.json')})
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print('FINAL_DEV256_COMPLETE: all generation and scoring finished; GPU can be stopped.', flush=True)
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
            print('FINAL_DEV256_COMPLETE: already complete; no GPU model loaded.')
            return
        command = [sys.executable, '-u', str(SCRIPT), '--project-root', str(project), '--run-root', str(run)]
        if len(raw) < 256:
            if args.grade_only:
                raise RuntimeError('Not all 256 outputs exist; generation still needs GPU')
            subprocess.run([sys.executable, str(project / 'experiments/aimo_sft_v26/install_missing.py'), '--eval'], check=True)
            env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn')
            subprocess.run(command + ['--worker', 'generate'], env=env, check=True)
        subprocess.run(command + ['--worker', 'grade'], check=True)


if __name__ == '__main__':
    main()
