"""Single-GPU LoRA-GRPO / MMR-DR-GRPO training (TRL + colocated vLLM).

Usage:
    math-posttrain train-grpo --config grpo/mmr_dr50.json
    math-posttrain train-grpo --config grpo/dapo_16k.json
"""
from __future__ import annotations

import argparse
import inspect
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from math_posttrain.common import (
    JsonlAppender, latest_checkpoint, load_config, log, now_iso, read_jsonl, workspace, write_json,
)
from math_posttrain.formats import messages
from math_posttrain.grading import Grader, last_box_without_final_tag
from math_posttrain.models import resolve_model


class FinalAnswerReward:
    """Binary final-answer reward using Math-Verify with optional last-box fallback."""

    def __init__(self, tokenizer, grader: Grader, journal: JsonlAppender, last_box_fallback: bool = False):
        self.__name__ = 'final_answer'
        self.tokenizer = tokenizer
        self.grader = grader
        self.journal = journal
        self.last_box_fallback = last_box_fallback
        self.calls = 0

    def _grade(self, items):
        grades = self.grader.grade_many(items, strict=True)
        failed = [i for i, g in enumerate(grades) if g['status'] in ('timeout', 'error')]
        if failed:
            for i, g in zip(failed, self.grader.grade_many([items[i] for i in failed], strict=True), strict=True):
                grades[i] = g
        return grades

    def __call__(self, prompts, completions, completion_ids=None, reference_answer=None, trainer_state=None, **kwargs):
        if completion_ids is not None:
            raw_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
            texts = [t.split('<|im_end|>')[0] for t in raw_texts]
        else:
            texts = [c[-1]['content'] if isinstance(c, list) else c for c in completions]
        strict_grades = self._grade(list(zip(texts, reference_answer, strict=True)))
        results = strict_grades.copy()
        if self.last_box_fallback:
            candidates = []
            for idx, grade in enumerate(strict_grades):
                if grade.get('how') == 'missing_answer_tag':
                    text = last_box_without_final_tag(texts[idx])
                    if text is not None:
                        candidates.append((idx, text, reference_answer[idx]))
            if candidates:
                fb_grades = self._grade([(text, ref) for _, text, ref in candidates])
                for (idx, _, _), grade in zip(candidates, fb_grades, strict=True):
                    results[idx] = {**grade, 'how': 'last_box_fallback'}
        counts = Counter(r['status'] for r in results)
        self.calls += 1
        step = getattr(trainer_state, 'global_step', None)
        self.journal.write({'step': step, 'call': self.calls, 'time': now_iso(), 'counts': dict(counts)})
        log_metric = kwargs.get('log_metric')
        if callable(log_metric):
            for status in ('correct', 'wrong', 'no_answer'):
                log_metric(f'reward/{status}_frac', counts[status] / len(results))
        return [1.0 if r['status'] == 'correct' else 0.0 for r in results]


def build_grpo_config(cfg: dict, out_dir: Path, *, smoke: bool, max_steps: int | None):
    from trl import GRPOConfig

    fields = set(inspect.signature(GRPOConfig).parameters)
    vllm = cfg.get('vllm', {})
    kwargs = dict(
        output_dir=str(out_dir),
        max_steps=max_steps or cfg['max_steps'],
        learning_rate=cfg['learning_rate'],
        lr_scheduler_type=cfg['lr_scheduler_type'],
        warmup_steps=cfg['warmup_steps'],
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        num_generations=cfg['num_generations'],
        max_completion_length=cfg['max_completion_length'],
        temperature=cfg['temperature'],
        top_p=cfg.get('top_p', 0.95),
        beta=cfg['beta'],
        epsilon=cfg.get('epsilon', 0.2),
        epsilon_high=cfg.get('epsilon_high'),
        loss_type=cfg['loss_type'],
        scale_rewards=cfg.get('scale_rewards', 'none'),
        mask_truncated_completions=cfg.get('mask_truncated_completions', True),
        num_iterations=1,
        disable_dropout='mmr_dr_grpo' not in cfg,
        shuffle_dataset='mmr_dr_grpo' not in cfg,
        remove_unused_columns=False,
        bf16=not smoke,
        gradient_checkpointing=not smoke,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1,
        save_strategy='steps',
        save_steps=cfg['save_steps'],
        save_total_limit=cfg.get('save_total_limit', 2),
        seed=cfg['seed'],
        report_to='none',
        log_completions=False,
    )
    if 'optim' in cfg:
        kwargs['optim'] = cfg['optim']
    if 'lr_scheduler_kwargs' in cfg:
        kwargs['lr_scheduler_kwargs'] = cfg['lr_scheduler_kwargs']
    if 'vllm_importance_sampling_mode' in cfg and 'vllm_importance_sampling_mode' in fields:
        kwargs['vllm_importance_sampling_mode'] = cfg['vllm_importance_sampling_mode']
    if smoke:
        kwargs['use_vllm'] = False
        if 'use_cpu' in fields:
            kwargs['use_cpu'] = True
    else:
        kwargs.update(
            use_vllm=True,
            vllm_mode='colocate',
            vllm_gpu_memory_utilization=float(vllm.get('gpu_memory_utilization', 0.35)),
            vllm_enable_sleep_mode=bool(vllm.get('enable_sleep_mode', True)),
            vllm_max_model_length=cfg['max_prompt_tokens'] + cfg['max_completion_length'],
        )
    return GRPOConfig(**{k: v for k, v in kwargs.items() if k in fields})


def load_pool(pool_path: Path) -> list[dict]:
    if not pool_path.is_file():
        raise FileNotFoundError(
            f'RL prompt pool not found: {pool_path}. Run `math-posttrain prepare-data --rl` first.'
        )
    rows = []
    for i, r in enumerate(read_jsonl(pool_path)):
        q = str(r.get('question') or r.get('problem') or '').strip()
        ans = str(r.get('reference_answer') or r.get('answer') or '').strip()
        if q and ans:
            rows.append({'sample_id': r.get('sample_id', f'pool/{i:05d}'), 'question': q, 'reference_answer': ans})
    if not rows:
        raise ValueError(f'RL prompt pool is empty: {pool_path}')
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='grpo/mmr_dr50.json', help='Config path under configs/ or filesystem')
    p.add_argument('--model', help='Override init_model (alias, HF repo ID, run:<name>, or local path)')
    p.add_argument('--pool', help='Override RL prompt pool JSONL (default: data/<config.pool_file>)')
    p.add_argument('--out', help='Override output directory (default: outputs/runs/<run_name>)')
    p.add_argument('--max-steps', type=int, help='Override total RL optimizer steps')
    p.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint in output dir')
    p.add_argument('--no-merge', action='store_true', help='Export LoRA adapter only without merging into base')
    p.add_argument('--smoke', action='store_true', help='CPU/fp32 plumbing test without vLLM')
    a = p.parse_args()

    cfg = load_config(a.config)
    ws = workspace()
    run_name = cfg['run_name']
    out_dir = Path(a.out) if a.out else ws.runs / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_path = Path(a.pool) if a.pool else ws.data_dir / cfg['pool_file']
    pool = load_pool(pool_path)
    model_spec = a.model or cfg['init_model']
    model_path = resolve_model(model_spec)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOTrainer

    if not a.smoke and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for GRPO training (pass --smoke for CPU check).')

    mmr_cfg = cfg.get('mmr_dr_grpo')
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if mmr_cfg:
        from math_posttrain.mmr_grpo import JinaEmbeddings, MmrDRGRPOTrainer, MmrReward, training_messages
        prompt_fn = training_messages
    else:
        prompt_fn = messages

    dataset = Dataset.from_list([
        {'prompt': prompt_fn(r['question']), 'reference_answer': r['reference_answer'], 'sample_id': r['sample_id']}
        for r in pool
    ])
    args = build_grpo_config(cfg, out_dir, smoke=a.smoke, max_steps=a.max_steps)
    args.model_init_kwargs = {
        **(args.model_init_kwargs or {}),
        'dtype': 'float32' if a.smoke else 'bfloat16',
        'attn_implementation': 'sdpa',
    }
    lora = cfg['lora']
    peft_config = LoraConfig(
        r=lora['r'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
        target_modules=lora['target_modules'], task_type='CAUSAL_LM',
    )

    journal = JsonlAppender(out_dir / 'reward_status.jsonl')
    grading_ctx = nullcontext() if mmr_cfg else Grader(workers=cfg.get('reward_workers', 4))
    t0 = time.monotonic()
    with grading_ctx as grader:
        if mmr_cfg:
            reward = MmrReward(tok, cfg, JinaEmbeddings(mmr_cfg))
            trainer_cls = MmrDRGRPOTrainer
            extra = {'prompt_limit': cfg['max_prompt_tokens'], 'scheduler_steps': mmr_cfg['scheduler_steps']}
        else:
            reward = FinalAnswerReward(tok, grader, journal, last_box_fallback=cfg.get('last_box_fallback', True))
            trainer_cls = GRPOTrainer
            extra = {}

        trainer = trainer_cls(
            model=str(model_path),
            reward_funcs=[reward],
            args=args,
            train_dataset=dataset,
            processing_class=tok,
            peft_config=peft_config,
            **extra,
        )
        if mmr_cfg:
            reward.eos.update(trainer.eos_token_ids)
        try:
            resume_ckpt = latest_checkpoint(out_dir) if a.resume else None
            trainer.train(resume_from_checkpoint=resume_ckpt)
        finally:
            journal.close()

        adapter_dir = out_dir / 'adapter'
        trainer.save_model(str(adapter_dir))
        tok.save_pretrained(str(adapter_dir))

    merge_adapter = cfg.get('merge_adapter', True) and not a.no_merge
    if merge_adapter:
        from math_posttrain.merge_lora import merge
        merge(model_path, adapter_dir, out_dir / 'merged', smoke=a.smoke)

    write_json(out_dir / 'grpo_summary.json', {
        'run_name': run_name,
        'steps': trainer.state.global_step,
        'adapter_dir': str(adapter_dir),
        'merged_dir': str(out_dir / 'merged') if merge_adapter else None,
        'wall_hours': round((time.monotonic() - t0) / 3600, 3),
        'completed': now_iso(),
    })
    log(f'{run_name}: GRPO complete -> {out_dir}')


if __name__ == '__main__':
    main()
