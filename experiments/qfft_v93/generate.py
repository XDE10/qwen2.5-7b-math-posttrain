"""Resumable vLLM generation for the v93 evaluation suite.

    python -m experiments.qfft_v93.generate --model run:qfft_ds_v93 --suite full

* Every model receives identical prompt token ids: formats.prompt_ids() with the base Qwen2.5 tokenizer.
* One request per (problem, draw) with a deterministic seed, so any subset can be regenerated exactly
  and a crashed run resumes at the draw level.
* Continuous batching: requests are fed to the vLLM engine and each finished one is appended to a local
  JSONL immediately; the JSONL is copied to Drive every minute and at the end.
* Special tokens are kept in the text so graders can locate <|im_start|>answer.
* Generation never grades; scoring runs afterwards on CPU (score.py), so a pathological output can
  never stall the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

from experiments.qfft_v93.common import (JsonlAppender, drive_layout, gpu_profile, load_config, local_root, log,
                                         now_iso, read_json, read_jsonl, sha256_file, sha256_json, sync_file,
                                         write_json, write_jsonl)
from experiments.qfft_v93.formats import SYSTEM_PROMPT, prompt_ids

STOP_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>


def draw_seed(base: int, sample_id: str, draw: int) -> int:
    return (base + int(hashlib.sha256(f'{sample_id}#{draw}'.encode()).hexdigest()[:8], 16)) % (2 ** 31)


def suite_spec(name: str, cfg: dict) -> dict:
    suites = cfg['eval']['suites']
    if name not in suites:
        raise KeyError(f'Unknown suite {name}; choose from {sorted(suites)}')
    return suites[name]


class VLLMEngine:
    def __init__(self, model_path, max_model_len, seed, gpu_memory_utilization, max_num_seqs, attention_backend=None):
        os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
        if attention_backend:
            os.environ['VLLM_ATTENTION_BACKEND'] = attention_backend
        from vllm import LLM
        self.llm = LLM(model=str(model_path), tokenizer=str(model_path), dtype='bfloat16',
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       max_num_seqs=max_num_seqs, enable_prefix_caching=True, seed=seed,
                       generation_config='vllm')
        self.max_num_seqs = max_num_seqs

    @staticmethod
    def params(job, temperature, top_p):
        from vllm import SamplingParams
        return SamplingParams(n=1, temperature=temperature, top_p=top_p, max_tokens=job['max_tokens'],
                              seed=job['seed'], skip_special_tokens=False, stop_token_ids=STOP_IDS)

    def run(self, jobs, temperature, top_p, on_finish):
        engine = getattr(self.llm, 'llm_engine', None)
        if all(hasattr(engine, name) for name in ('add_request', 'step', 'has_unfinished_requests')):
            pending = list(reversed(jobs))
            first = pending.pop()
            try:  # API probe: a signature mismatch raises here, before anything is queued
                engine.add_request(self._rid(first), {'prompt_token_ids': first['prompt_ids']},
                                   self.params(first, temperature, top_p))
            except TypeError as error:
                log(f'vLLM add_request signature differs ({error}); using block-wise generate')
                return self._blocks(jobs, temperature, top_p, on_finish)
            return self._stream(engine, pending, {self._rid(first): first}, temperature, top_p, on_finish)
        log('vLLM engine-step API unavailable; using block-wise generate')
        return self._blocks(jobs, temperature, top_p, on_finish)

    @staticmethod
    def _rid(job):
        return f"{job['benchmark']}|{job['sample_id']}|{job['draw']}"

    def _stream(self, engine, pending, by_id, temperature, top_p, on_finish):
        """Keep ~2x max_num_seqs requests queued; persist each one the moment it finishes."""
        window = 2 * self.max_num_seqs
        while pending or engine.has_unfinished_requests():
            while pending and len(by_id) < window:
                job = pending.pop()
                engine.add_request(self._rid(job), {'prompt_token_ids': job['prompt_ids']},
                                   self.params(job, temperature, top_p))
                by_id[self._rid(job)] = job
            for output in engine.step():
                if output.finished:
                    # vLLM >= 0.30 reports the caller's id; older builds may return the internal "<id>-<8 chars>".
                    key = output.request_id if output.request_id in by_id else output.request_id.rsplit('-', 1)[0]
                    on_finish(by_id.pop(key), output.outputs[0])

    def _blocks(self, jobs, temperature, top_p, on_finish):
        size = 4 * self.max_num_seqs
        for start in range(0, len(jobs), size):
            block = jobs[start:start + size]
            outputs = self.llm.generate([{'prompt_token_ids': j['prompt_ids']} for j in block],
                                        [self.params(j, temperature, top_p) for j in block], use_tqdm=False)
            for job, output in zip(block, outputs):
                on_finish(job, output.outputs[0])


class FakeEngine:
    """CPU stand-in that answers correctly on even draws: tests resume/scoring without a GPU."""

    class _Out:
        def __init__(self, text, n):
            self.text, self.token_ids, self.finish_reason, self.stop_reason = text, list(range(n)), 'stop', None

    def __init__(self, answers):
        self.answers = answers

    def run(self, jobs, temperature, top_p, on_finish):
        for job in jobs:
            answer = self.answers[job['sample_id']] if job['draw'] % 2 == 0 else '0'
            text = (f'Let me think. Wait, check again: \\boxed{{-1}}.\n<|im_start|>answer\n'
                    f'The answer is $\\boxed{{{answer}}}$.<|im_end|>')
            on_finish(job, self._Out(text, 40 + job['draw']))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='Registry name (base, qwen_math, ...) or run:<run_name>[/merged]')
    p.add_argument('--tag', help='Output folder name under evals/ (default: derived from --model)')
    p.add_argument('--suite', default='full')
    p.add_argument('--benchmarks', help='Comma list overriding the suite benchmark selection')
    p.add_argument('--bench-dir', help='Default: Drive data/benchmarks_v93')
    p.add_argument('--evals-root', help='Separate output root for a development evaluation')
    p.add_argument('--model-path', help='Override local model directory')
    p.add_argument('--tokenizer-path', help='Prompt tokenizer (default: local base model)')
    p.add_argument('--engine', choices=['vllm', 'fake'], default='vllm')
    p.add_argument('--gpu-memory-utilization', type=float)
    p.add_argument('--max-num-seqs', type=int)
    p.add_argument('--attention-backend')
    a = p.parse_args()
    cfg = load_config('pipeline.json')
    ev = cfg['eval']
    layout = drive_layout()
    tag = a.tag or a.model.replace('run:', '').replace('/', '_')
    suite = suite_spec(a.suite, cfg)
    names = a.benchmarks.split(',') if a.benchmarks else list(suite)
    bench_dir = Path(a.bench_dir) if a.bench_dir else layout.data / 'benchmarks_v93'
    from experiments.qfft_v93 import models
    model_path = Path(a.model_path) if a.model_path else models.ensure_local(a.model, layout)
    max_len = min(models.max_model_len(a.model) if not a.model_path else ev['max_model_len'], ev['max_model_len'])
    from transformers import AutoTokenizer
    tok_path = a.tokenizer_path or models.ensure_tokenizer('base', layout)
    tok = AutoTokenizer.from_pretrained(tok_path)
    identity_file = model_path / 'p1_model_identity.json'
    model_identity = read_json(identity_file) if identity_file.exists() else {'spec': a.model}
    jobs, answers, plans = [], {}, {}
    for name in names:
        spec = {**suite[name]} if name in suite else {'n': 1}
        rows = read_jsonl(bench_dir / f'{name}.jsonl')
        if spec.get('limit'):
            rows = rows[:spec['limit']]
        max_tokens = spec.get('max_tokens')
        out_dir = (Path(a.evals_root) if a.evals_root else layout.evals) / tag / name
        protocol = {'model': a.model, 'model_identity': model_identity, 'benchmark': name,
                    'benchmark_sha256': sha256_file(bench_dir / f'{name}.jsonl'), 'n': spec['n'],
                    'limit': spec.get('limit'), 'temperature': ev['temperature'], 'top_p': ev['top_p'],
                    'max_model_len': max_len, 'max_tokens_cap': max_tokens, 'seed': ev['seed'],
                    'system_prompt': SYSTEM_PROMPT, 'engine': a.engine}
        id_path = out_dir / 'identity.json'
        if id_path.exists() and read_json(id_path)['protocol'] != protocol:
            raise RuntimeError(f'{id_path}: protocol differs from this request; use another --tag')
        write_json(id_path, {'protocol': protocol, 'protocol_sha256': sha256_json(protocol), 'updated': now_iso()})
        local_raw = local_root() / 'evals' / sha256_json(protocol) / tag / name / 'raw.jsonl'
        drive_raw = out_dir / 'raw.jsonl'
        merged = {}
        for record in read_jsonl(drive_raw) + read_jsonl(local_raw):  # union: Drive may be ahead or behind
            merged.setdefault((record['sample_id'], record['draw']), record)
        done = set(merged)
        write_jsonl(local_raw, list(merged.values()))
        expected = 0
        for row in rows:
            answers[row['sample_id']] = row['answer']
            ids = prompt_ids(tok, row['problem'])
            budget = max_len - len(ids)
            if max_tokens:
                budget = min(budget, max_tokens)
            if budget < 256:
                raise RuntimeError(f"{row['sample_id']}: prompt leaves only {budget} tokens")
            for draw in range(spec['n']):
                expected += 1
                if (row['sample_id'], draw) in done:
                    continue
                jobs.append({'benchmark': name, 'sample_id': row['sample_id'], 'draw': draw,
                             'seed': draw_seed(ev['seed'], row['sample_id'], draw), 'prompt_ids': ids,
                             'max_tokens': budget})
        plans[name] = {'expected': expected, 'done_before': len(done), 'local': local_raw, 'drive': drive_raw,
                       'out_dir': out_dir}
        log(f'{tag}/{name}: {expected} generations planned, {len(done)} already saved')
    session = None
    if not jobs:
        log('Nothing to generate; all requested generations exist')
    else:
        profile = gpu_profile()
        engine_t0 = time.monotonic()
        if a.engine == 'fake':
            engine = FakeEngine(answers)
        else:
            engine = VLLMEngine(model_path, max_len, ev['seed'],
                                a.gpu_memory_utilization or profile.get('vllm_gpu_memory_utilization', 0.9),
                                a.max_num_seqs or profile.get('vllm_max_num_seqs', 128), a.attention_backend)
        engine_start_s = time.monotonic() - engine_t0
        writers = {name: JsonlAppender(plan['local']) for name, plan in plans.items()}
        counters = {'done': 0, 'tokens': 0, 'started': time.monotonic(), 'last_sync': time.monotonic()}

        def sync_all():
            for name, plan in plans.items():
                with writers[name].lock:
                    if not writers[name].handle.closed:
                        writers[name].handle.flush()
                sync_file(plan['local'], plan['drive'])
            counters['last_sync'] = time.monotonic()

        def on_finish(job, out):
            n = len(out.token_ids)
            writers[job['benchmark']].write({
                'sample_id': job['sample_id'], 'draw': job['draw'], 'seed': job['seed'], 'text': out.text,
                'tokens': n, 'finish_reason': out.finish_reason, 'stop_reason': str(out.stop_reason),
                'prompt_tokens': len(job['prompt_ids']), 'time': now_iso()})
            counters['done'] += 1
            counters['tokens'] += n
            if counters['done'] % 200 == 0 or time.monotonic() - counters['last_sync'] > 60:
                elapsed = time.monotonic() - counters['started']
                log(f"{tag}: {counters['done']}/{len(jobs)} new generations, "
                    f"{counters['tokens'] / max(elapsed, 1e-6):.0f} output tok/s")
                sync_all()

        try:
            engine.run(jobs, ev['temperature'], ev['top_p'], on_finish)
        finally:
            for w in writers.values():
                w.close()
            sync_all()
            seconds = time.monotonic() - counters['started']
            session = {'new_generations': counters['done'], 'output_tokens': counters['tokens'],
                       'seconds': round(seconds, 1), 'engine_start_s': round(engine_start_s, 1),
                       'output_tokens_per_s': round(counters['tokens'] / max(seconds, 1e-6), 1),
                       'gpu': profile.get('name'), 'ended': now_iso()}
            log(f"{tag}: {session['new_generations']} new generations, {session['output_tokens']:,} output tokens in "
                f"{seconds:.0f} s ({session['output_tokens_per_s']:.0f} tok/s; engine start {engine_start_s:.0f} s)")
    for name, plan in plans.items():
        rows = {(r['sample_id'], r['draw']) for r in read_jsonl(plan['drive'])}
        complete = len(rows) >= plan['expected']
        status_path = plan['out_dir'] / 'generation_status.json'
        previous = read_json(status_path) if status_path.exists() else {}
        write_json(status_path, {
            'expected': plan['expected'], 'saved': len(rows), 'complete': complete, 'updated': now_iso(),
            # throughput of the latest session that generated anything (all benchmarks of the call together)
            'last_session': session or previous.get('last_session')})
        if not complete:
            raise RuntimeError(f'{tag}/{name}: only {len(rows)}/{plan["expected"]} generations saved; re-run to resume')
    log(f'{tag}: generation complete for {", ".join(names)}')


if __name__ == '__main__':
    main()
