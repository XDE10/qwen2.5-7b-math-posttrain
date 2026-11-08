"""v22: batched, inference-only scheduling benchmark. Never edits v21 outputs."""
import argparse
from collections import Counter
from importlib.metadata import version
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.projection_pilot_v21.pilot import (
    Backend, ContextOverflow, PROPOSAL, QUESTION_SUFFIX, read, rows, save, sha,
)


def choose_rows(selected):
    """Two fixed rows per original length bin; no correctness-based selection."""
    groups = {}
    for row in selected:
        groups.setdefault(row['length_bin'], []).append(row)
    if len(groups) != 4 or any(len(group) < 2 for group in groups.values()):
        raise ValueError('Expected four frozen length bins')
    # Interleave bins so all batch sizes see both short and long expert inputs.
    return [group[i] for i in range(2) for group in groups.values()]


def trajectory(row, config):
    """Same transitions/RNG order as v21 sample(updates=2), yielding engine work."""
    rng = random.Random(config['seed'] + int(row['sample_id'][:8], 16))
    ids, events = [], []
    width = config['max_new_tokens'] // config['block_num']
    score, status = None, 'length'
    for block in range(config['block_num']):
        ids = yield ('propose', row, ids, min(width, config['max_new_tokens']-len(ids)), rng.randrange(2**31))
        if not ids:
            status = 'empty_output'
            break
        score = yield ('score', row['question'], ids)
        for update in range(2):
            cut = rng.randrange(len(ids))
            candidate = yield ('propose', row, ids[:cut], len(ids)-cut, rng.randrange(2**31))
            candidate_score = yield ('score', row['question'], candidate)
            accepted = candidate_score > score
            events.append({'block': block, 'update': update, 'cut': cut,
                           'old_mean_logprob': score, 'candidate_mean_logprob': candidate_score,
                           'accepted': accepted, 'candidate_tokens': len(candidate)})
            if accepted:
                ids, score = candidate, candidate_score
        if ids[-1] == config['eos']:
            status = 'stop'
            break
    return {'sample_id': row['sample_id'], 'token_ids': ids, 'trace': events,
            'finish_reason': status, 'final_mean_logprob': score, 'training_approved': False}


def schedule(selected, config, concurrency, execute, completed):
    """One outstanding request per question; batch independent questions only."""
    pending = iter(selected)
    active = []
    def fill():
        while len(active) < concurrency:
            row = next(pending, None)
            if row is None:
                return
            gen = trajectory(row, config)
            active.append([gen, next(gen)])
    fill()
    while active:
        current = list(active)
        values = execute([job[1] for job in current])
        if len(values) != len(current):
            raise RuntimeError('Engine result count mismatch')
        for job, value in zip(current, values):
            try:
                job[1] = job[0].send(value)
            except StopIteration as stop:
                completed(stop.value)
                active.remove(job)
        fill()


class BatchBackend(Backend):
    def __init__(self, model, config, concurrency):
        import torch
        from vllm import LLM, SamplingParams
        self.config, self.SamplingParams = config, SamplingParams
        self.llm = LLM(model=str(model), tokenizer=str(model), dtype='bfloat16',
                       tensor_parallel_size=1, max_model_len=config['max_model_len'],
                       max_num_seqs=concurrency, max_num_batched_tokens=2048,
                       gpu_memory_utilization=config['gpu_memory_utilization'],
                       enable_chunked_prefill=True, enable_prefix_caching=True,
                       generation_config='vllm', seed=config['seed'])
        self.tok, self.torch = self.llm.get_tokenizer(), torch
        self.eos = self.tok.eos_token_id
        self.generated = self.scored = self.input_tokens = self.calls = 0

    def execute(self, requests):
        prompts, params, metadata = [], [], []
        for req in requests:
            if req[0] == 'score':
                _, question, ids = req
                prefix = self.prompt(question + QUESTION_SUFFIX)
                full = prefix + ids
                if not ids or len(full)+1 > self.config['max_model_len']:
                    raise ContextOverflow('Scoring input empty or exceeds context')
                params.append(self.SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0))
                metadata.append(('score', len(prefix), ids))
            else:
                _, row, prefix, limit, seed = req
                text = PROPOSAL.format(question=row['question'], solution=row['solution'],
                                       partial=self.tok.decode(prefix, skip_special_tokens=False))
                full = self.prompt(text) + prefix
                if len(full)+limit > self.config['max_model_len']:
                    raise ContextOverflow('Proposal exceeds context; no silent truncation')
                params.append(self.SamplingParams(max_tokens=limit, temperature=0.6, top_p=1.0,
                              repetition_penalty=1.1, stop_token_ids=[self.eos], seed=seed))
                metadata.append(('propose', prefix, None))
            prompts.append({'prompt_token_ids': full})
            self.input_tokens += len(full)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError('Engine result count mismatch')
        values = []
        for output, meta in zip(outputs, metadata):
            kind, prefix, ids = meta
            generated = list(output.outputs[0].token_ids)
            self.generated += len(generated)
            if kind == 'propose':
                if output.outputs[0].finish_reason == 'stop' and (not generated or generated[-1] != self.eos):
                    generated.append(self.eos)
                values.append(prefix + generated)
            else:
                lps = output.prompt_logprobs[prefix:]
                if len(lps) != len(ids):
                    raise RuntimeError('Score coverage mismatch')
                scores = [float(item[token].logprob) for token, item in zip(ids, lps)]
                if not all(math.isfinite(x) for x in scores):
                    raise RuntimeError('Nonfinite score')
                values.append(sum(scores)/len(scores))
                self.scored += len(ids)
        self.calls += 1
        return values


class DeviceMeter:
    """Whole-device sampled occupancy/utilization, NOT per-worker allocated memory."""
    def __init__(self):
        self.samples, self.stop = [], threading.Event()
    def __enter__(self):
        def poll():
            while not self.stop.is_set():
                try:
                    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits', '-i', '0'], capture_output=True, text=True, timeout=4)
                    if r.returncode == 0:
                        self.samples.append([float(x.strip()) for x in r.stdout.strip().split(',')])
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                self.stop.wait(1)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)


def worker(args, model, config, selected, destination):
    from src.training.frozen_verifier_v4 import Worker
    from src.training.math_reward_v2 import category
    start = time.monotonic()
    backend = BatchBackend(model, config, args.worker)
    load_seconds = time.monotonic()-start
    # Fixed non-experimental warmup, excluded from throughput/token counters.
    backend.llm.generate([{'prompt_token_ids': backend.prompt('Compute 1+1.')}],
                         backend.SamplingParams(max_tokens=32, temperature=0), use_tqdm=False)
    backend.llm.reset_prefix_cache()
    config = dict(config, eos=backend.eos)
    completed = []
    def collect(result):
        completed.append(result)
        save(destination/'partial.json', completed)
        print(f"parallel={args.worker} completed={len(completed)}/{len(selected)}", flush=True)
    with DeviceMeter() as meter:
        start = time.monotonic()
        schedule(selected, config, args.worker, backend.execute, collect)
        seconds = time.monotonic()-start
    # Grading excluded from inference time, identical existing frozen verifier.
    verifier = Worker()
    by_id = {r['sample_id']: r for r in selected}
    try:
        for record in completed:
            record['text'] = backend.decode(record['token_ids'])
            record['grade'] = verifier.check(record['text'], by_id[record['sample_id']]['reference_answer'])
            record['grade_category'] = category(record['grade'])
    finally:
        verifier.close()
    save(destination/'results.json', completed)
    summary = {'concurrency': args.worker, 'questions': len(completed), 'seconds': seconds,
        'questions_per_minute': len(completed)*60/seconds, 'model_load_seconds': load_seconds,
        'generated_tokens': backend.generated, 'generated_tokens_per_second': backend.generated/seconds,
        'input_tokens': backend.input_tokens, 'scored_tokens': backend.scored, 'batch_calls': backend.calls,
        'grade_categories': dict(Counter(r['grade_category'] for r in completed)),
        'finish_reasons': dict(Counter(r['finish_reason'] for r in completed)),
        'device_peak_used_mib_sampled': max((s[0] for s in meter.samples), default=None),
        'device_mean_utilization_pct_sampled': sum(s[1] for s in meter.samples)/len(meter.samples) if meter.samples else None,
        'meter_samples': len(meter.samples), 'versions': {n: version(n) for n in ['torch','vllm','transformers']},
        'gpu': backend.torch.cuda.get_device_name(0), 'training_approved': False}
    save(destination/'summary.json', summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', help='Existing pinned LOCAL SSD model directory')
    p.add_argument('--output-root', default='/content/drive/MyDrive/runs/project1/parallel_bench_v22')
    p.add_argument('--concurrency', nargs='+', type=int, default=[1,2,4])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--timeout-minutes', type=float, default=20, help='Hard limit PER setting; partial result kept')
    p.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if any(c not in (1,2,4) for c in a.concurrency) or a.repeats < 1 or a.timeout_minutes <= 0:
        p.error('Use concurrency 1/2/4, positive repeats and timeout')
    c = read(ROOT/'configs/project1_projection_pilot_v21.json')
    if sha(ROOT/c['data_file']) != c['data_sha256']:
        raise ValueError('Frozen data changed')
    selected = choose_rows(rows(ROOT/c['data_file']))
    name = 'DeepSeek-R1-Distill-Qwen-7B--'+c['model_revision']
    model = Path(a.model_path) if a.model_path else Path('/content/models')/name
    if not model.is_dir():
        raise RuntimeError('Local model missing. Reuse the v21 local SSD model via --model-path; no download performed.')
    if read(model/'projection_model_identity.json')['revision'] != c['model_revision']:
        raise ValueError('Wrong model revision')
    for file, expected in c['tokenizer_sha256'].items():
        if sha(model/file) != expected:
            raise ValueError('Tokenizer identity mismatch')
    index = read(model/'model.safetensors.index.json')
    if any(not (model/file).is_file() for file in set(index['weight_map'].values())):
        raise ValueError('Missing model weights')
    out = Path(a.output_root)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    if a.worker:
        worker(a, model, c, selected, out)
        return
    import fcntl
    # Hold the existing default pilot lock: refuse overlapping GPU runs.
    locks = []
    old = Path('/content/drive/MyDrive/runs/project1/projection_pilot_v21/running.lock')
    for path in [old, out/'running.lock']:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another default pilot/benchmark is running. Wait for it to finish.')
        locks.append(handle)
    from src.training.frozen_verifier_v4 import grader_identity
    identity = {'config': c, 'selected_ids': [r['sample_id'] for r in selected],
        'grader': grader_identity(),
        'code_sha256': sha(__file__), 'base_sampler_sha256': sha(ROOT/'experiments/projection_pilot_v21/pilot.py'),
        'model_identity': read(model/'projection_model_identity.json'), 'concurrency': a.concurrency,
        'repeats': a.repeats, 'timeout_minutes': a.timeout_minutes}
    if (out/'identity.json').exists() and read(out/'identity.json') != identity:
        raise ValueError('Protocol changed; choose a new output-root')
    save(out/'identity.json', identity)
    records = []
    for repeat in range(a.repeats):
        # Reverse on odd repeats to reduce a consistent ordering advantage.
        for concurrency in (a.concurrency if repeat%2 == 0 else list(reversed(a.concurrency))):
            dest = out/f'c{concurrency}_r{repeat+1}'
            if not (dest/'summary.json').exists():
                dest.mkdir(exist_ok=True)
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(concurrency),
                           '--model-path', str(model), '--output-root', str(dest)]
                print(f'Start concurrency={concurrency}, repeat={repeat+1}; isolated engine', flush=True)
                with (dest/'console.log').open('w', encoding='utf-8') as log:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = proc.wait(timeout=a.timeout_minutes*60)
                    except subprocess.TimeoutExpired:
                        import signal
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        code = -9
                    finally:
                        # Parent Ctrl+C must not leave a detached GPU engine running.
                        if proc.poll() is None:
                            import signal
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                if code != 0:
                    records.append({'concurrency': concurrency, 'repeat': repeat+1, 'failed_exit': code,
                                    'log': str(dest/'console.log')})
                    save(out/'comparison.json', records)
                    # Stop after engine/OOM/timeout failure; do not compound GPU failures.
                    print('Setting failed; stopped. Inspect '+str(dest/'console.log'), flush=True)
                    return
            records.append(dict(read(dest/'summary.json'), repeat=repeat+1))
            save(out/'comparison.json', records)
    rates = {n: sum(r['questions_per_minute'] for r in records if r['concurrency']==n)/a.repeats
             for n in a.concurrency}
    best = max(rates.values())
    suggested = min(n for n, rate in rates.items() if rate >= best*0.95)
    save(out/'COMPLETE.json', {'provisional_concurrency': suggested, 'rates_questions_per_minute': rates,
        'rule': 'Smallest concurrency within 95% of fastest mean throughput; not quality or stability certification',
        'warning': '8 questions only; batching may change token outputs. Inspect quality and logs before scaling.'})
    print('Benchmark complete; provisional concurrency='+str(suggested)+'; see '+str(out), flush=True)


if __name__ == '__main__':
    main()
