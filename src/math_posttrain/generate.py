"""Resumable vLLM 32K evaluation generation.

Usage:
    math-posttrain generate --model qfft_r1 --tag qfft_r1 --suite full
    math-posttrain generate --model XDE10/Qwen2.5-7B-Math-QFFT-R1 --tag qfft_r1 --suite competition
    math-posttrain generate --model base --tag smoke_test --suite smoke --engine fake

* Every model receives identical prompt token IDs from the base Qwen2.5 tokenizer.
* Deterministic per-(problem, draw) seeds allow exact resumption at the draw level.
* Special tokens are kept so graders locate `<|im_start|>answer` accurately.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

from math_posttrain.common import (
    JsonlAppender, gpu_profile, load_config, log, now_iso, read_json, read_jsonl,
    sha256_file, sha256_json, workspace, write_json, write_jsonl,
)
from math_posttrain.formats import SYSTEM_PROMPT, prompt_ids
from math_posttrain.models import max_model_len, resolve_model, resolve_tokenizer

STOP_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>


def draw_seed(base: int, sample_id: str, draw: int) -> int:
    return (base + int(hashlib.sha256(f'{sample_id}#{draw}'.encode()).hexdigest()[:8], 16)) % (2 ** 31)


def suite_spec(name: str, cfg: dict) -> dict:
    suites = cfg['eval']['suites']
    if name not in suites:
        raise KeyError(f'Unknown suite {name!r}; choose from {sorted(suites)}')
    return suites[name]


class VLLMEngine:
    def __init__(self, model_path: Path, max_len: int, seed: int, gpu_memory_utilization: float,
                 max_num_seqs: int, attention_backend: str | None = None):
        os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
        if attention_backend:
            os.environ['VLLM_ATTENTION_BACKEND'] = attention_backend
        from vllm import LLM

        self.llm = LLM(
            model=str(model_path),
            tokenizer=str(model_path),
            dtype='bfloat16',
            max_model_len=max_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=True,
            seed=seed,
            generation_config='vllm',
        )
        self.max_num_seqs = max_num_seqs

    @staticmethod
    def params(job: dict, temperature: float, top_p: float):
        from vllm import SamplingParams
        return SamplingParams(
            n=1, temperature=temperature, top_p=top_p, max_tokens=job['max_tokens'],
            seed=job['seed'], skip_special_tokens=False, stop_token_ids=STOP_IDS,
        )

    def run(self, jobs: list[dict], temperature: float, top_p: float, on_finish) -> None:
        engine = getattr(self.llm, 'llm_engine', None)
        if all(hasattr(engine, name) for name in ('add_request', 'step', 'has_unfinished_requests')):
            pending = list(reversed(jobs))
            first = pending.pop()
            try:
                engine.add_request(
                    self._rid(first), {'prompt_token_ids': first['prompt_ids']},
                    self.params(first, temperature, top_p),
                )
            except TypeError as error:
                log(f'vLLM add_request signature differs ({error}); using block-wise generate')
                return self._blocks(jobs, temperature, top_p, on_finish)
            return self._stream(engine, pending, {self._rid(first): first}, temperature, top_p, on_finish)
        return self._blocks(jobs, temperature, top_p, on_finish)

    @staticmethod
    def _rid(job: dict) -> str:
        return f"{job['benchmark']}|{job['sample_id']}|{job['draw']}"

    def _stream(self, engine, pending, by_id, temperature, top_p, on_finish) -> None:
        window = 2 * self.max_num_seqs
        while pending or engine.has_unfinished_requests():
            while pending and len(by_id) < window:
                job = pending.pop()
                engine.add_request(
                    self._rid(job), {'prompt_token_ids': job['prompt_ids']},
                    self.params(job, temperature, top_p),
                )
                by_id[self._rid(job)] = job
            for output in engine.step():
                if output.finished:
                    key = output.request_id if output.request_id in by_id else output.request_id.rsplit('-', 1)[0]
                    on_finish(by_id.pop(key), output.outputs[0])

    def _blocks(self, jobs, temperature, top_p, on_finish) -> None:
        size = 4 * self.max_num_seqs
        for start in range(0, len(jobs), size):
            block = jobs[start:start + size]
            outputs = self.llm.generate(
                [{'prompt_token_ids': j['prompt_ids']} for j in block],
                [self.params(j, temperature, top_p) for j in block],
                use_tqdm=False,
            )
            for job, output in zip(block, outputs):
                on_finish(job, output.outputs[0])


class FakeEngine:
    """Zero-GPU stand-in answering correctly on even draws to test end-to-end eval & scoring."""

    class _Out:
        def __init__(self, text: str, n: int):
            self.text, self.token_ids, self.finish_reason, self.stop_reason = text, list(range(n)), 'stop', None

    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def run(self, jobs, temperature, top_p, on_finish) -> None:
        for job in jobs:
            answer = self.answers[job['sample_id']] if job['draw'] % 2 == 0 else '0'
            text = (
                f'Let me think. Wait, check again: \\boxed{{-1}}.\n<|im_start|>answer\n'
                f'The answer is $\\boxed{{{answer}}}$.<|im_end|>'
            )
            on_finish(job, self._Out(text, 40 + job['draw']))


def _default_tag(model_spec: str) -> str:
    clean = model_spec.replace('run:', '').strip('/').replace('\\', '/')
    return clean.replace('/', '_')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='Registry alias, HF repo ID, `run:<name>`, or local path')
    p.add_argument('--tag', help='Output folder name under outputs/evals/ (default: derived from --model)')
    p.add_argument('--suite', default='full', help='Suite name from configs/eval.json (default: full)')
    p.add_argument('--benchmarks', help='Comma-separated list overriding suite benchmarks')
    p.add_argument('--bench-dir', help='Benchmark JSONL directory (default: data/benchmarks)')
    p.add_argument('--evals-root', help='Evaluation output root (default: outputs/evals)')
    p.add_argument('--tokenizer', help='Prompt tokenizer path/spec (default: base Qwen2.5 tokenizer)')
    p.add_argument('--engine', choices=['vllm', 'fake'], default='vllm')
    p.add_argument('--gpu-memory-utilization', type=float)
    p.add_argument('--max-num-seqs', type=int)
    p.add_argument('--attention-backend')
    a = p.parse_args()

    cfg = load_config('eval.json')
    ev = cfg['eval']
    ws = workspace()
    tag = a.tag or _default_tag(a.model)
    suite = suite_spec(a.suite, cfg)
    names = a.benchmarks.split(',') if a.benchmarks else list(suite)
    bench_dir = Path(a.bench_dir) if a.bench_dir else ws.benchmarks
    evals_root = Path(a.evals_root) if a.evals_root else ws.evals

    missing = [n for n in names if not (bench_dir / f'{n}.jsonl').is_file()]
    if missing:
        raise FileNotFoundError(
            f'Missing benchmark files in {bench_dir}: {", ".join(missing)}. '
            f'Run `math-posttrain prepare-data --benchmarks` first.'
        )

    model_path = None
    if a.engine == 'vllm':
        model_path = resolve_model(a.model)
    max_len = min(max_model_len(a.model, model_path), int(ev['max_model_len']))

    tok = None
    if a.engine == 'vllm' or a.tokenizer:
        from transformers import AutoTokenizer
        tok_path = resolve_tokenizer(a.tokenizer or 'base')
        tok = AutoTokenizer.from_pretrained(tok_path)

    jobs, answers, plans = [], {}, {}
    for name in names:
        spec = {**suite[name]} if name in suite else {'n': 1}
        rows = read_jsonl(bench_dir / f'{name}.jsonl')
        if spec.get('limit'):
            rows = rows[:spec['limit']]
        max_tokens = spec.get('max_tokens')
        out_dir = evals_root / tag / name
        protocol = {
            'model': a.model,
            'benchmark': name,
            'benchmark_sha256': sha256_file(bench_dir / f'{name}.jsonl'),
            'n': spec['n'],
            'limit': spec.get('limit'),
            'temperature': ev['temperature'],
            'top_p': ev['top_p'],
            'max_model_len': max_len,
            'max_tokens_cap': max_tokens,
            'seed': ev['seed'],
            'system_prompt': SYSTEM_PROMPT,
            'engine': a.engine,
        }
        id_path = out_dir / 'identity.json'
        if id_path.exists() and read_json(id_path)['protocol'] != protocol:
            raise RuntimeError(f'{id_path}: protocol differs from existing outputs; choose another --tag')
        write_json(id_path, {'protocol': protocol, 'protocol_sha256': sha256_json(protocol), 'updated': now_iso()})

        raw_path = out_dir / 'raw.jsonl'
        existing = {(r['sample_id'], r['draw']): r for r in read_jsonl(raw_path)}
        if existing:
            write_jsonl(raw_path, list(existing.values()))
        done = set(existing)
        expected = 0
        for row in rows:
            answers[row['sample_id']] = row['answer']
            ids = prompt_ids(tok, row['problem']) if tok is not None else [ord(c) for c in row['problem'][:64]]
            budget = max_len - len(ids)
            if max_tokens:
                budget = min(budget, max_tokens)
            if budget < 256:
                raise RuntimeError(f"{row['sample_id']}: prompt leaves only {budget} tokens")
            for draw in range(spec['n']):
                expected += 1
                if (row['sample_id'], draw) in done:
                    continue
                jobs.append({
                    'benchmark': name, 'sample_id': row['sample_id'], 'draw': draw,
                    'seed': draw_seed(ev['seed'], row['sample_id'], draw),
                    'prompt_ids': ids, 'max_tokens': budget,
                })
        plans[name] = {'expected': expected, 'raw': raw_path, 'out_dir': out_dir}
        log(f'{tag}/{name}: {expected} generations planned, {len(done)} already saved')

    session = None
    if not jobs:
        log('Nothing to generate; all requested generations already exist')
    else:
        profile = gpu_profile()
        t0 = time.monotonic()
        if a.engine == 'fake':
            engine = FakeEngine(answers)
        else:
            engine = VLLMEngine(
                model_path, max_len, ev['seed'],
                a.gpu_memory_utilization or profile.get('vllm_gpu_memory_utilization', 0.9),
                a.max_num_seqs or profile.get('vllm_max_num_seqs', 128),
                a.attention_backend,
            )
        engine_start_s = time.monotonic() - t0
        writers = {name: JsonlAppender(plan['raw']) for name, plan in plans.items()}
        counters = {'done': 0, 'tokens': 0, 'started': time.monotonic()}

        def on_finish(job, out):
            n = len(out.token_ids)
            writers[job['benchmark']].write({
                'sample_id': job['sample_id'], 'draw': job['draw'], 'seed': job['seed'],
                'text': out.text, 'tokens': n, 'finish_reason': out.finish_reason,
                'stop_reason': str(out.stop_reason), 'prompt_tokens': len(job['prompt_ids']),
                'time': now_iso(),
            })
            counters['done'] += 1
            counters['tokens'] += n
            if counters['done'] % 200 == 0:
                elapsed = max(time.monotonic() - counters['started'], 1e-6)
                log(f"{tag}: {counters['done']}/{len(jobs)} generated ({counters['tokens'] / elapsed:.0f} tok/s)")

        try:
            engine.run(jobs, ev['temperature'], ev['top_p'], on_finish)
        finally:
            for w in writers.values():
                w.close()
            seconds = max(time.monotonic() - counters['started'], 1e-6)
            session = {
                'new_generations': counters['done'],
                'output_tokens': counters['tokens'],
                'seconds': round(seconds, 1),
                'engine_start_s': round(engine_start_s, 1),
                'output_tokens_per_s': round(counters['tokens'] / seconds, 1),
                'gpu': profile.get('name'),
                'ended': now_iso(),
            }
            log(f"{tag}: generated {session['new_generations']} samples ({session['output_tokens']:,} tokens) in {seconds:.1f}s")

    for name, plan in plans.items():
        saved = {(r['sample_id'], r['draw']) for r in read_jsonl(plan['raw'])}
        complete = len(saved) >= plan['expected']
        write_json(plan['out_dir'] / 'generation_status.json', {
            'expected': plan['expected'], 'saved': len(saved), 'complete': complete,
            'updated': now_iso(), 'last_session': session,
        })
        if not complete:
            raise RuntimeError(f'{tag}/{name}: only {len(saved)}/{plan["expected"]} saved; re-run to resume')
    log(f'{tag}: generation complete for {", ".join(names)}')


if __name__ == '__main__':
    main()
