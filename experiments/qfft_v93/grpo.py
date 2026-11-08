"""LoRA-GRPO on top of a merged full-FT SFT/QFFT checkpoint (TRL + colocated vLLM), resumable across VMs.

    python -m experiments.qfft_v93.grpo --config grpo/grpo_qfft_ds.json

Design choices (see docs/interview/面试追问与参考答案.md for the reasoning):
* The SFT/QFFT checkpoint is merged into full weights first and becomes the base; a fresh LoRA is the
  policy. With beta = 0 no reference model is loaded at all.
* loss_type "dapo" (token-level normalization, TRL default), clip-higher epsilon_high = 0.28,
  truncated completions masked out of the loss, binary final-answer reward from grading.py.
* The reward decodes completion ids with special tokens kept, so only the text after
  <|im_start|>answer is graded, exactly as in evaluation.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from experiments.qfft_v93 import decontam
from experiments.qfft_v93.checkpointing import CheckpointMirror, resolve_resume
from experiments.qfft_v93.common import (Heartbeat, JsonlAppender, copy_tree_throttled, drive_layout, gpu_profile,
                                         load_config, local_root, log, now_iso, read_json, read_jsonl, sha256_json,
                                         write_json)
from experiments.qfft_v93.formats import messages
from experiments.qfft_v93.grading import Grader
from experiments.qfft_v93.telemetry import ProgressCallback, SwanLab


def build_pool(cfg: dict, layout, bench_dir: Path | None, sft_questions: set):
    """RL prompts: listed files + optional globs (relative to the Drive root), deduplicated and decontaminated."""
    files = [layout.root / f for f in cfg['pool']['files']]
    for pattern in cfg['pool'].get('optional_globs', []):
        files += [Path(p) for p in sorted(glob.glob(str(layout.root / pattern)))]
    rows, seen = [], set()
    for path in files:
        if not path.exists():
            if str(path.relative_to(layout.root)) in cfg['pool']['files']:
                raise FileNotFoundError(f'RL pool file missing: {path}')
            continue
        for r in read_jsonl(path):
            question = r.get('problem') or r.get('question')
            answer = r.get('reference_answer', r.get('answer'))
            if not question or answer is None or not str(answer).strip():
                continue
            key = ' '.join(decontam.words(question))
            if key in seen:
                continue
            seen.add(key)
            rows.append({'sample_id': r.get('sample_id', f'pool/{len(rows):05d}'), 'question': question,
                         'reference_answer': str(answer), 'pool_file': path.name})
    report = {'files': [str(p) for p in files if p.exists()], 'unique_questions': len(rows)}
    if bench_dir and Path(bench_dir).exists():
        benchmarks = decontam.load_benchmarks(bench_dir)
        flagged = set()
        for name, items in benchmarks.items():
            flagged |= {f['train_id'] for f in decontam.find_flags([(r['sample_id'], r['question']) for r in rows], items)}
        rows = [r for r in rows if r['sample_id'] not in flagged]
        report['removed_benchmark_similar'] = len(flagged)
    report['overlap_with_sft_questions'] = sum(' '.join(decontam.words(r['question'])) in sft_questions for r in rows)
    report['final_questions'] = len(rows)
    log('RL pool:', report)
    return rows, report


class FinalAnswerReward:
    """Binary reward: 1 if the final answer segment matches the reference (grading.py policy)."""

    def __init__(self, tokenizer, grader: Grader, journal: JsonlAppender, max_failure_rate: float = 0.05):
        self.__name__ = 'final_answer'
        self.tokenizer, self.grader, self.journal = tokenizer, grader, journal
        self.max_failure_rate = max_failure_rate
        self.calls = 0

    def __call__(self, prompts, completions, completion_ids=None, reference_answer=None, trainer_state=None, **kwargs):
        if completion_ids is not None:
            texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
            texts = [t.split('<|im_end|>')[0] for t in texts]
        else:
            texts = [c[-1]['content'] if isinstance(c, list) else c for c in completions]
        results = self.grader.grade_many(list(zip(texts, reference_answer)))
        counts = Counter(r['status'] for r in results)
        failures = counts['timeout'] + counts['error']
        self.calls += 1
        step = getattr(trainer_state, 'global_step', None)
        self.journal.write({'step': step, 'call': self.calls, 'time': now_iso(), 'counts': dict(counts)})
        if failures > self.max_failure_rate * len(results):
            raise RuntimeError(f'Grader infrastructure failures {failures}/{len(results)}; aborting instead of '
                               'turning them into wrong answers')
        log_metric = kwargs.get('log_metric')
        if callable(log_metric):
            for status in ('correct', 'wrong', 'no_answer'):
                try:
                    log_metric(f'reward/{status}_frac', counts[status] / len(results))
                except Exception:
                    break
        return [1.0 if r['status'] == 'correct' else 0.0 for r in results]


def grpo_config(cfg: dict, out: Path, smoke: bool, max_steps: int | None):
    import inspect
    from trl import GRPOConfig
    fields = set(inspect.signature(GRPOConfig).parameters)
    profile = gpu_profile()
    kind = profile.get('kind', 'generic')
    vllm = cfg['vllm']
    kwargs = dict(
        output_dir=str(out), max_steps=max_steps or cfg['max_steps'], learning_rate=cfg['learning_rate'],
        lr_scheduler_type=cfg['lr_scheduler_type'], warmup_steps=cfg['warmup_steps'],
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'], num_generations=cfg['num_generations'],
        max_completion_length=cfg['max_completion_length'], temperature=cfg['temperature'], top_p=cfg['top_p'],
        beta=cfg['beta'], epsilon=cfg['epsilon'], epsilon_high=cfg['epsilon_high'], loss_type=cfg['loss_type'],
        scale_rewards=cfg['scale_rewards'], mask_truncated_completions=cfg['mask_truncated_completions'],
        num_iterations=1, disable_dropout=True, shuffle_dataset=True, remove_unused_columns=False,
        bf16=not smoke, gradient_checkpointing=not smoke,
        gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=1, save_strategy='steps',
        save_steps=cfg['save_steps'], save_total_limit=None, seed=cfg['seed'], report_to='none',
        log_completions=False, disable_tqdm=True)
    if smoke:
        kwargs.update(use_vllm=False)
        if 'use_cpu' in fields:
            kwargs['use_cpu'] = True
    else:
        kwargs.update(use_vllm=True, vllm_mode='colocate',
                      vllm_gpu_memory_utilization=vllm['gpu_memory_utilization'].get(kind, vllm['gpu_memory_utilization']['generic']),
                      vllm_enable_sleep_mode=bool(vllm['enable_sleep_mode'].get(kind, False)),
                      vllm_max_model_length=cfg['max_prompt_tokens'] + cfg['max_completion_length'])
        # Extra vLLM SamplingParams (TRL merges them in colocate mode). The GPU smoke sets
        # min_tokens = max_completion_length so every completion has the worst-case length.
        generation = dict(cfg.get('generation_kwargs') or {})
        if generation.get('min_tokens') == 'max_completion_length':
            generation['min_tokens'] = cfg['max_completion_length']
        if generation:
            kwargs['generation_kwargs'] = generation
    unsupported = set(kwargs) - fields
    if unsupported:
        raise TypeError('Unsupported GRPOConfig fields for this TRL version: ' + ', '.join(sorted(unsupported)))
    return GRPOConfig(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--model-path', help='Override the merged init model directory')
    p.add_argument('--pool', help='Override: JSONL with question/problem and reference_answer')
    p.add_argument('--bench-dir', help='Benchmarks for decontaminating the RL pool (default: Drive data dir)')
    p.add_argument('--max-steps', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--no-swanlab', action='store_true')
    p.add_argument('--no-merge', action='store_true')
    a = p.parse_args()
    cfg = load_config(a.config)
    layout = drive_layout()
    run = cfg['run_name']
    drive_run, local_run = layout.runs / run, local_root() / 'runs' / run
    drive_run.mkdir(parents=True, exist_ok=True)
    local_run.mkdir(parents=True, exist_ok=True)
    if (drive_run / 'GRPO_COMPLETE.json').exists() and ((drive_run / 'merged').exists() or a.no_merge):
        log(f'{run}: already complete')
        return
    if a.model_path:
        model_path = Path(a.model_path)
    else:
        from experiments.qfft_v93.models import ensure_local
        model_path = ensure_local(cfg['init_model'], layout)
    bench_dir = Path(a.bench_dir) if a.bench_dir else layout.data / 'benchmarks_v93'
    sft_questions = set()
    init_identity = Path(model_path) / 'p1_model_identity.json'
    if a.pool:
        pool = [{'sample_id': r.get('sample_id', f'pool/{i:05d}'), 'question': r.get('problem') or r.get('question'),
                 'reference_answer': str(r.get('reference_answer', r.get('answer')))}
                for i, r in enumerate(read_jsonl(a.pool))]
        pool_report = {'override': a.pool, 'final_questions': len(pool)}
    else:
        pool, pool_report = build_pool(cfg, layout, bench_dir, sft_questions)
    write_json(drive_run / 'rl_pool_report.json', pool_report)
    identity = {'run': run, 'init_model': cfg['init_model'], 'pool_sha256': sha256_json([r['sample_id'] for r in pool]),
                'recipe': {k: v for k, v in cfg.items() if k not in ('run_name', 'pool', 'vllm')},
                'init_identity': read_json(init_identity) if init_identity.exists() else None,
                'smoke': a.smoke, 'max_steps_override': a.max_steps}
    id_path = drive_run / 'run_identity.json'
    if id_path.exists() and read_json(id_path)['identity'] != identity:
        raise RuntimeError(f'{id_path} describes a different experiment; choose a new run_name')
    write_json(id_path, {'identity': identity, 'updated': now_iso(), 'code_bundle': os.environ.get('P1_CODE_SHA')})

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOTrainer
    tok = AutoTokenizer.from_pretrained(model_path)
    dataset = Dataset.from_list([{'prompt': messages(r['question']), 'reference_answer': r['reference_answer'],
                                  'sample_id': r['sample_id']} for r in pool])
    args = grpo_config(cfg, local_run, a.smoke, a.max_steps)
    lora = cfg['lora']
    peft_config = LoraConfig(r=lora['r'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
                             target_modules=lora['target_modules'], task_type='CAUSAL_LM')
    swan = SwanLab(drive_run, run, {'identity': identity, 'pool': pool_report}, tags=['v93', 'grpo', 'lora'],
                   enabled=not (a.no_swanlab or a.smoke), log_dir=local_run / 'swanlog')
    journal = JsonlAppender(drive_run / 'reward_status.jsonl')
    mirror = CheckpointMirror(local_run, drive_run, every=1, keep=cfg.get('mirror_keep', 3),
                              watch_path=str(local_root()), min_free_gib=8, mirror_final=True)
    with Grader(workers=cfg.get('reward_workers', 8), timeout=cfg.get('reward_timeout', 30)) as grader:
        reward = FinalAnswerReward(tok, grader, journal)
        # Strings keep GRPOConfig JSON-serializable; TRL maps them to torch dtypes.
        args.model_init_kwargs = {**(args.model_init_kwargs or {}),
                                  'dtype': 'float32' if a.smoke else 'bfloat16', 'attn_implementation': 'sdpa'}
        trainer = GRPOTrainer(model=str(model_path), reward_funcs=[reward], args=args, train_dataset=dataset,
                              processing_class=tok, peft_config=peft_config,
                              callbacks=[ProgressCallback(local_run, drive_run, swan, {'run': run}), mirror])
        resume = resolve_resume(local_run, drive_run)
        beat = Heartbeat(lambda: read_json(drive_run / 'status.json').get('step')
                         if (drive_run / 'status.json').exists() else 'starting', interval=300)
        beat.start()
        started = time.monotonic()
        try:
            trainer.train(resume_from_checkpoint=resume)
        finally:
            beat.stop()
            journal.close()
        adapter_local = local_run / 'adapter'
        trainer.save_model(str(adapter_local))
        tok.save_pretrained(str(adapter_local))
        mirror.wait()
    copy_tree_throttled(adapter_local, drive_run / 'adapter', watch_path=str(local_root()))
    summary = {'steps': trainer.state.global_step, 'resumed_from': resume, 'adapter': str(drive_run / 'adapter'),
               'wall_hours_this_session': round((time.monotonic() - started) / 3600, 3), 'completed': now_iso()}
    write_json(drive_run / 'GRPO_COMPLETE.json', summary)
    write_json(drive_run / 'status.json', {'state': 'complete', **summary})
    swan.finish()
    if not a.no_merge:
        from experiments.qfft_v93.merge_lora import merge
        merge(model_path, adapter_local, drive_run / 'merged', smoke=a.smoke)
    shutil.rmtree(local_run, ignore_errors=True)
    log(f'{run}: COMPLETE -> {drive_run}')


if __name__ == '__main__':
    main()
