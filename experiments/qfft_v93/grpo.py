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
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from experiments.qfft_v93 import decontam, drive_io
from experiments.qfft_v93.checkpointing import CheckpointMirror, read_state, resolve_resume
from experiments.qfft_v93.common import (CONFIG_DIR, Heartbeat, JsonlAppender, drive_layout, gpu_profile,
                                         load_config, local_root, log, now_iso, read_json, read_jsonl, sha256_file, sha256_json,
                                         write_json, write_jsonl)
from experiments.qfft_v93.formats import messages
from experiments.qfft_v93.grading import Grader, last_box_without_final_tag
from experiments.qfft_v93.telemetry import ProgressCallback, SwanLab


def approved_pool_sources(cfg: dict, root: Path):
    manifest = read_json(CONFIG_DIR / cfg['pool']['approval_manifest'])
    if cfg['run_name'] not in manifest['run_names']:
        raise ValueError('RL pool approval belongs to a different run')
    for name in manifest['approved_training_files']:
        path = root / name
        spec = manifest['files'][name]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != spec['sha256']:
            raise ValueError(f'RL pool file hash differs from the approval manifest: {path}')
        rows = [json.loads(line) for line in content.decode('utf-8-sig').splitlines() if line.strip()]
        if len(rows) != spec['rows'] or any(r.get('training_approved') is not spec['source_training_approved'] for r in rows):
            raise ValueError(f'RL pool rows differ from the approval manifest: {path}')
        yield path, rows


def build_pool(cfg: dict, layout, bench_dir: Path | None, sft_questions: set):
    """RL prompts: listed files + optional globs (relative to the Drive root), deduplicated and decontaminated."""
    manifest = cfg['pool'].get('approval_manifest')
    if manifest:
        sources = approved_pool_sources(cfg, layout.root)
    else:
        files = [layout.root / f for f in cfg['pool']['files']]
        for path in files:
            if not path.exists():
                raise FileNotFoundError(f'RL pool file missing: {path}')
        for pattern in cfg['pool'].get('optional_globs', []):
            files += [Path(p) for p in sorted(glob.glob(str(layout.root / pattern)))]
        sources = ((path, read_jsonl(path)) for path in files if path.exists())
    rows, seen = [], set()
    used_files = []
    for path, source_rows in sources:
        used_files.append(str(path))
        for r in source_rows:
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
    report = {'files': used_files, 'unique_questions': len(rows)}
    comparison_ids = cfg['pool'].get('comparison_sample_ids')
    if manifest:
        report['approval_manifest'] = manifest
    if bench_dir and Path(bench_dir).exists():
        benchmarks = decontam.load_benchmarks(bench_dir)
        if comparison_ids is not None and any(
            not benchmarks.get(name) for name in ('aime24', 'aime25', 'amc23', 'math500', 'gsm8k')
        ):
            raise ValueError('The length comparison requires all five benchmark files for decontamination')
        flagged = set()
        for name, items in benchmarks.items():
            flagged |= {f['train_id'] for f in decontam.find_flags([(r['sample_id'], r['question']) for r in rows], items)}
        rows = [r for r in rows if r['sample_id'] not in flagged]
        report['removed_benchmark_similar'] = len(flagged)
    if comparison_ids is not None:
        if not manifest or 'removed_benchmark_similar' not in report:
            raise ValueError('The length comparison requires the approved pool and benchmark decontamination')
        by_id = {row['sample_id']: row for row in rows}
        if (len(by_id) != len(rows) or len(set(comparison_ids)) != len(comparison_ids)
                or set(comparison_ids) - by_id.keys()):
            raise ValueError('The fixed comparison sample IDs must be unique and present in the cleaned pool')
        rows = [by_id[sid] for sid in comparison_ids]
        report['comparison_sample_ids'] = comparison_ids
        report['comparison_sample_ids_sha256'] = sha256_json(comparison_ids)
        report['comparison_source_counts'] = dict(Counter(r['pool_file'] for r in rows))
    report['overlap_with_sft_questions'] = sum(' '.join(decontam.words(r['question'])) in sft_questions for r in rows)
    report['final_questions'] = len(rows)
    log('RL pool:', report)
    return rows, report


def frozen_mmr_pool(cfg, data_root):
    # The approved file contains the exact ordered draw, including the reviewed repairs.
    (path, rows), = approved_pool_sources(cfg, data_root)
    completions_per_step = cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']
    questions_per_step, remainder = divmod(completions_per_step, cfg['num_generations'])
    extension = cfg.get('extend_from', {})
    draw_steps = (load_config(extension['config'])['max_steps']
                  if extension and extension.get('data_mode') != 'append_frozen_prefix' else cfg['max_steps'])
    if remainder or len(rows) != draw_steps * questions_per_step:
        raise ValueError('Frozen MMR draw must contain exactly one prompt per rollout group for all steps')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Frozen MMR pool has duplicate sample IDs')
    if any(not row['question'].strip() or not row['reference_answer'].strip() for row in rows):
        raise ValueError('Frozen MMR pool contains a blank question or reference')
    pool = [{key: row[key] for key in ('sample_id', 'question', 'reference_answer')} for row in rows]
    return pool, {'files': [str(path)], 'final_questions': len(pool),
                  'approval_manifest': cfg['pool']['approval_manifest']}


class RolloutDiagnostics:
    """Capture colocated vLLM outputs before TRL drops the stop metadata."""

    def __init__(self, trainer, journal=None, comparison_ids=None, *, mmr_dir=None, identity=None):
        if trainer.accelerator.num_processes != 1:
            raise ValueError('Rollout diagnostics require single-process colocated vLLM')
        self.journal = journal
        self.comparison_ids = comparison_ids
        self.num_generations = getattr(trainer, 'num_generations', None)
        self.eos_and_pad = {*trainer.eos_token_ids, trainer._tokenizer.pad_token_id}
        self.batches = []
        self.mmr_dir = mmr_dir
        self.identity = identity
        self._mmr_batch = None
        llm = trainer.vllm_generation.llm
        generate = llm.generate

        def capture(*args, **kwargs):
            outputs = generate(*args, **kwargs)
            self.batches.extend(output for request in outputs for output in request.outputs)
            return outputs

        llm.generate = capture

    def record_mmr(self, step, sample_ids, completion_ids, texts, records):
        outputs, self.batches = self.batches, []
        if self._mmr_batch is not None:
            raise RuntimeError('Prior MMR diagnostic batch has no Trainer advantages')
        rows = []
        for index, (output, sid, ids, text, record) in enumerate(zip(
                outputs, sample_ids, completion_ids, texts, records, strict=True)):
            generated = list(output.token_ids)
            empty_placeholder = not generated and len(ids) == 1 and record['trl_clipped']
            if generated != list(ids) and not empty_placeholder:
                raise RuntimeError(f'MMR rollout token mismatch at step {step}, index {index}')
            rows.append({**record, 'step': step, 'index': index, 'sample_id': sid,
                         'text': text if generated else '', 'token_ids_sha256': sha256_json(generated),
                         'tokens': len(generated), 'finish_reason': output.finish_reason,
                         'stop_reason': output.stop_reason, **self.identity})
        self._mmr_batch = step, rows

    def finish_mmr(self, advantages):
        step, rows = self._mmr_batch
        for row, advantage in zip(rows, advantages, strict=True):
            row['advantage'] = advantage
        # Store each rollout batch once. A retry of an uncheckpointed step replaces that step's trace.
        write_jsonl(self.mmr_dir / f'step-{step:04d}.jsonl', rows)
        self._mmr_batch = None

    def record(self, step, call, sample_ids, completion_ids, raw_texts, reward_texts, references, strict_grades, grades):
        outputs, self.batches = self.batches, []
        if len(outputs) != len(grades) or len(sample_ids) != len(grades):
            raise RuntimeError('Rollout diagnostics could not align vLLM outputs with rewards')
        diagnostics = []
        for index, (output, sid, ids, raw, text, ref, strict_grade, grade) in enumerate(
            zip(outputs, sample_ids, completion_ids, raw_texts, reward_texts, references, strict_grades, grades, strict=True)
        ):
            if list(output.token_ids) != list(ids):
                raise RuntimeError(f'Rollout diagnostics token mismatch for call {call}, index {index}')
            row = {
                'step': step, 'call': call, 'index': index, 'sample_id': sid,
                'text': raw, 'vllm_text': output.text, 'reward_text': text,
                'tokens': len(ids), 'reference_answer': ref,
                'finish_reason': output.finish_reason, 'stop_reason': output.stop_reason,
                'truncated': output.finish_reason == 'length',
                'trl_clipped': not ids or ids[-1] not in self.eos_and_pad,
                'strict_status': strict_grade['status'], 'how': strict_grade.get('how'),
                'reward_status': grade['status'], 'reward_how': grade.get('how'),
            }
            self.journal.write(row)
            if self.num_generations:
                diagnostics.append(row)
        if not self.num_generations:
            return {}
        size = self.num_generations
        group_ids = sample_ids[::size]
        if self.comparison_ids is not None:
            groups_per_step = len(self.comparison_ids) // 2
            expected = self.comparison_ids[step * groups_per_step:(step + 1) * groups_per_step]
            if group_ids != expected:
                raise RuntimeError(f'Unexpected prompt groups in comparison rollout at step {step + 1}')
        if sample_ids != [sid for sid in group_ids for _ in range(size)] or not group_ids:
            raise RuntimeError(f'Unexpected prompt groups in comparison rollout at step {step + 1}')
        mixed_unmasked = with_signal = 0
        for start in range(0, len(diagnostics), size):
            group = diagnostics[start:start + size]
            unmasked = {row['reward_status'] == 'correct' for row in group if not row['trl_clipped']}
            raw_rewards = {row['reward_status'] == 'correct' for row in group}
            mixed_unmasked += len(unmasked) == 2
            with_signal += bool(unmasked) and len(raw_rewards) == 2
        length_stops = sum(row['truncated'] for row in diagnostics)
        masked = sum(row['trl_clipped'] for row in diagnostics)
        tokens = sum(row['tokens'] for row in diagnostics)
        return {
            'completions': len(diagnostics), 'groups': len(group_ids),
            'length_stops': length_stops, 'length_stop_frac': length_stops / len(diagnostics),
            'loss_masked': masked, 'loss_masked_frac': masked / len(diagnostics),
            'mixed_unmasked_reward_groups': mixed_unmasked, 'groups_with_reward_signal': with_signal,
            'completion_tokens': tokens, 'mean_completion_tokens': tokens / len(diagnostics),
        }


class FinalAnswerReward:
    """Binary reward: 1 if the final answer segment matches the reference (grading.py policy)."""

    def __init__(self, tokenizer, grader: Grader, journal: JsonlAppender,
                 last_box_fallback: bool = False):
        self.__name__ = 'final_answer'
        self.tokenizer, self.grader, self.journal = tokenizer, grader, journal
        self.calls = 0
        self.diagnostics = None
        self.last_box_fallback = last_box_fallback

    def _grade(self, items):
        grades = self.grader.grade_many(items, strict=True)
        failed = [i for i, grade in enumerate(grades) if grade['status'] in ('timeout', 'error')]
        if failed:
            for i, grade in zip(failed, self.grader.grade_many([items[i] for i in failed], strict=True), strict=True):
                grades[i] = grade
        return grades

    def __call__(self, prompts, completions, completion_ids=None, reference_answer=None, trainer_state=None, **kwargs):
        if completion_ids is not None:
            raw_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
            texts = [t.split('<|im_end|>')[0] for t in raw_texts]
        else:
            raw_texts = texts = [c[-1]['content'] if isinstance(c, list) else c for c in completions]
        strict_grades = self._grade(list(zip(texts, reference_answer, strict=True)))
        results = strict_grades.copy()
        if self.last_box_fallback:
            candidates = []
            for index, grade in enumerate(strict_grades):
                if grade.get('how') == 'missing_answer_tag':
                    text = last_box_without_final_tag(texts[index])
                    if text is not None:
                        candidates.append((index, text, reference_answer[index]))
            if candidates:
                fallback_grades = self._grade([(text, ref) for _, text, ref in candidates])
                for (index, _, _), grade in zip(candidates, fallback_grades, strict=True):
                    results[index] = {**grade, 'how': 'last_box_fallback'}
        counts = Counter(r['status'] for r in results)
        self.calls += 1
        step = getattr(trainer_state, 'global_step', None)
        metrics = {}
        if self.diagnostics:
            metrics = self.diagnostics.record(step, self.calls, kwargs['sample_id'], completion_ids, raw_texts,
                                              texts, reference_answer, strict_grades, results)
        self.journal.write({'step': step, 'call': self.calls, 'time': now_iso(), 'counts': dict(counts), **metrics})
        failures = {key: counts[key] for key in ('timeout', 'error', 'ref_unparsed', 'no_reference') if counts[key]}
        if failures:
            # TRL's DAPO token denominator is computed before rewards, including unscorable rows.
            raise RuntimeError(f'GRPO batch could not be scored after retry: {failures}')
        log_metric = kwargs.get('log_metric')
        if callable(log_metric):
            for name, value in metrics.items():
                log_metric(f'comparison/{name}', value)
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
        num_iterations=1, disable_dropout=True, shuffle_dataset=cfg.get('shuffle_dataset', True),
        remove_unused_columns=False,
        bf16=not smoke, gradient_checkpointing=not smoke,
        gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=1, save_strategy='steps',
        save_steps=cfg['save_steps'], save_total_limit=None, seed=cfg['seed'], report_to='none',
        log_completions=False, disable_tqdm=True)
    if 'optim' in cfg:
        kwargs['optim'] = cfg['optim']
    if 'mmr_dr_grpo' in cfg:
        # Seeded selection already happened when the ordered draw was frozen.
        kwargs.update(disable_dropout=False, lr_scheduler_kwargs=cfg['lr_scheduler_kwargs'], shuffle_dataset=False)
    if 'vllm_importance_sampling_mode' in cfg:
        mode = cfg['vllm_importance_sampling_mode']
        if mode not in ('sequence_mask', 'token_mask', 'sequence_truncate', 'token_truncate'):
            raise ValueError(f'Invalid vllm_importance_sampling_mode: {mode!r}')
        kwargs['vllm_importance_sampling_mode'] = mode
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


def run_identity(cfg, pool, model_path, *, smoke, max_steps, merge_adapter):
    from experiments.qfft_v93.models import TOKENIZER_FILES, MARK

    model_files = {path.name: sha256_file(path) for path in sorted(model_path.iterdir())
                   if path.is_file() and (path.name in {*TOKENIZER_FILES, MARK}
                                          or path.suffix in ('.safetensors', '.bin')
                                          or path.name.endswith('.index.json'))}
    approval = cfg['pool'].get('approval_manifest')
    return {
        'run': cfg['run_name'], 'init_model': cfg['init_model'], 'model_files': model_files,
        'code_bundle': os.environ.get('P1_CODE_SHA'),
        'gpu_profile': None if smoke else gpu_profile().get('kind'),
        'pool_sha256': sha256_json(pool), 'recipe_sha256': sha256_json(cfg),
        'approval_sha256': sha256_file(CONFIG_DIR / approval) if approval else None,
        'smoke': smoke, 'max_steps_override': max_steps, 'merge_adapter': merge_adapter,
    }


def extension_source(cfg, identity, runs_dir, *, pool=None, data_root=None):
    """Authenticate the completed source while allowing an explicit increase in the step budget."""
    extension = cfg['extend_from']
    source_cfg = load_config(extension['config'])
    source_step = source_cfg['max_steps']
    mmr = bool(cfg.get('mmr_dr_grpo'))
    allowed = {'run_name', 'max_steps', 'note', 'extend_from', 'retained_checkpoint_steps',
               'save_steps', 'validation'}

    def recipe(value):
        value = {k: v for k, v in value.items() if k not in allowed}
        if mmr:
            value['pool'] = {k: v for k, v in value['pool'].items() if k != 'approval_manifest'}
        return value

    if mmr:
        old_approval = CONFIG_DIR / source_cfg['pool']['approval_manifest']
        new_approval = CONFIG_DIR / cfg['pool']['approval_manifest']
        old_scope, new_scope = read_json(old_approval), read_json(new_approval)
        if extension['approval_sha256'] != sha256_file(old_approval):
            raise RuntimeError('MMR continuation source approval hash changed')
        if extension.get('data_mode') == 'append_frozen_prefix':
            if data_root is None or pool is None:
                raise RuntimeError('MMR prefix continuation requires both validated source and target data')
            (old_file, old_rows), = approved_pool_sources(source_cfg, data_root)
            (new_file, new_rows), = approved_pool_sources(cfg, data_root)
            prefix = new_scope.get('unchanged_prefix', {})
            old_name, = old_scope['approved_training_files']
            if (prefix != {'file': old_name, **old_scope['files'][old_name]}
                    or not new_file.read_bytes().startswith(old_file.read_bytes())
                    or len(new_rows) <= len(old_rows)):
                raise RuntimeError('MMR continuation changed its previously trained byte prefix')
            source_pool = [{key: row[key] for key in ('sample_id', 'question', 'reference_answer')}
                           for row in old_rows]
            if pool[:len(source_pool)] != source_pool:
                raise RuntimeError('MMR continuation changed the ordered source prompt groups')
        elif (extension.get('data_mode') is not None
              or new_scope['approved_training_files'] != old_scope['approved_training_files']
              or new_scope['files'] != old_scope['files']):
            raise RuntimeError('MMR continuation must keep its approved file or declare an approved prefix extension')
        horizon = source_cfg.get('mmr_dr_grpo', {}).get('scheduler_steps', 0)
        scheduler_ok = (source_cfg.get('mmr_dr_grpo')
                        and source_cfg['lr_scheduler_type'] == 'cosine_with_min_lr'
                        and source_step < cfg['max_steps'] <= horizon)
    else:
        scheduler_ok = source_cfg['lr_scheduler_type'] == 'constant_with_warmup'
    if (recipe(cfg) != recipe(source_cfg)
            or cfg['run_name'] == source_cfg['run_name']
            or not source_cfg.get('mirror_optimizer_state')
            or not scheduler_ok
            or not source_cfg['warmup_steps'] <= source_step < cfg['max_steps']):
        raise RuntimeError('Extension must increase the step budget without changing the training recipe')
    source = runs_dir / source_cfg['run_name']
    source_identity = read_json(source / 'run_identity.json')['identity']
    expected = {**identity, 'run': source_cfg['run_name'], 'recipe_sha256': sha256_json(source_cfg),
                'code_bundle': extension['code_bundle'], 'approval_sha256': extension['approval_sha256']}
    if mmr and extension.get('data_mode') == 'append_frozen_prefix':
        expected['pool_sha256'] = sha256_json(source_pool)
    if source_identity != expected or read_json(source / 'GRPO_COMPLETE.json')['steps'] != source_step:
        raise RuntimeError('Extension source identity or completed step does not match')
    return source, source_step, sha256_json(source_identity)


def reload_mmr_checkpoint(trainer, checkpoint: Path) -> None:
    """Exercise the normal Trainer resume loaders after the isolated smoke."""
    trainer.create_optimizer_and_scheduler(num_training_steps=trainer.args.max_steps)
    # Release the optimizer tensors already on the device before loading their replacements.
    trainer.optimizer.state.clear()
    path = str(checkpoint)
    trainer._load_from_checkpoint(path)
    trainer._load_optimizer_and_scheduler(path)
    trainer._load_rng_state(path)


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
    p.add_argument('--pause-after', type=int, help='Exit after publishing a full mirror at this step')
    p.add_argument('--require-resume', action='store_true', help='Require a complete checkpoint before training')
    p.add_argument('--check', action='store_true', help='Validate the frozen MMR recipe and local data on CPU')
    a = p.parse_args()
    cfg = load_config(a.config)
    mmr_cfg = cfg.get('mmr_dr_grpo')
    if mmr_cfg and any((a.pool, a.model_path, a.bench_dir, a.smoke, a.max_steps is not None,
                        a.pause_after is not None, a.no_merge, a.no_swanlab)):
        p.error('The approved MMR run uses its frozen config, model, data, evaluation export and final step')
    if a.check:
        if not mmr_cfg or a.require_resume:
            p.error('--check is available for the frozen MMR recipe on local data')
        from experiments.qfft_v93.mmr_grpo import check_local
        check_local(cfg, frozen_mmr_pool, grpo_config)
        return
    if mmr_cfg and gpu_profile().get('kind') != 'g4':
        p.error('The frozen MMR recipe requires one G4. Use --check for a local CPU check.')
    extending = 'extend_from' in cfg
    if extending and (a.max_steps is not None or a.pause_after is not None):
        p.error('An extension uses the step budget recorded in its config')
    comparison_ids = cfg['pool'].get('comparison_sample_ids')
    completions_per_step = cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']
    if comparison_ids is not None and (
        a.pool or a.model_path or a.bench_dir or a.smoke or a.max_steps is not None or cfg['max_steps'] != 2
        or cfg.get('shuffle_dataset', True) or not cfg.get('log_rollout_diagnostics')
        or len(comparison_ids) * cfg['num_generations'] != 2 * completions_per_step
    ):
        p.error('The length comparison requires the fixed two-step, single-GPU model and pool recipe')
    if a.pause_after is not None and (
        not cfg.get('mirror_optimizer_state')
        or not 0 < a.pause_after < (a.max_steps or cfg['max_steps'])
        or a.pause_after % cfg['save_steps']
    ):
        p.error('--pause-after requires a scheduled full checkpoint before the final step')
    merge_adapter = cfg.get('merge_adapter', True) and not a.no_merge
    layout = drive_layout()
    run = cfg['run_name']
    drive_run, local_run = layout.runs / run, local_root() / 'runs' / run
    drive_run.mkdir(parents=True, exist_ok=True)
    local_run.mkdir(parents=True, exist_ok=True)
    if (drive_run / 'GRPO_COMPLETE.json').exists() and ((drive_run / 'merged').exists() or not merge_adapter):
        log(f'{run}: already complete')
        return
    if a.model_path:
        model_path = Path(a.model_path)
    else:
        from experiments.qfft_v93.models import ensure_local
        model_path = ensure_local(cfg['init_model'], layout)
    if not a.smoke:
        model_context = read_json(model_path / 'config.json')['max_position_embeddings']
        if cfg['max_prompt_tokens'] + cfg['max_completion_length'] > model_context:
            p.error(f'Prompt plus completion budget exceeds the model window of {model_context} tokens')
    bench_dir = Path(a.bench_dir) if a.bench_dir else layout.data / 'benchmarks_v93'
    sft_questions = set()
    if mmr_cfg:
        pool, pool_report = frozen_mmr_pool(cfg, layout.data)
    elif a.pool:
        pool = [{'sample_id': r.get('sample_id', f'pool/{i:05d}'), 'question': r.get('problem') or r.get('question'),
                 'reference_answer': str(r.get('reference_answer', r.get('answer')))}
                for i, r in enumerate(read_jsonl(a.pool))]
        pool_report = {'override': a.pool, 'final_questions': len(pool)}
    else:
        pool, pool_report = build_pool(cfg, layout, bench_dir, sft_questions)
    identity = run_identity(cfg, pool, model_path, smoke=a.smoke, max_steps=a.max_steps, merge_adapter=merge_adapter)
    if extending:
        source, source_step, source_identity = extension_source(cfg, identity, layout.runs,
                                                               pool=pool, data_root=layout.data)
        identity['extension'] = {'source_run': source.name, 'source_step': source_step,
                                 'source_identity_sha256': source_identity}
    id_path = drive_run / 'run_identity.json'
    if id_path.exists() and read_json(id_path)['identity'] != identity:
        raise RuntimeError(f'{id_path} describes a different experiment; choose a new run_name')
    if not id_path.exists() and (any(local_run.glob('checkpoint-*')) or (drive_run / 'ckpt_mirror').exists()):
        raise RuntimeError(f'{id_path} is missing for a run with checkpoints')
    write_json(drive_run / 'rl_pool_report.json', pool_report)
    write_json(id_path, {'identity': identity, 'updated': now_iso(), 'code_bundle': os.environ.get('P1_CODE_SHA')})
    resume = resolve_resume(local_run, drive_run)
    if extending and resume is None:
        resume = resolve_resume(local_run, source, exact_drive_step=source_step)
    if (a.require_resume or extending) and resume is None:
        raise RuntimeError('No complete GRPO checkpoint is available for the requested resume')
    if extending:
        resumed_step = read_state(resume)['global_step']
        if resumed_step < source_step:
            raise RuntimeError('Extension checkpoint predates its declared source step')
        log(f'{run}: extending from step {resumed_step} to {cfg["max_steps"]} with full training state')
    mirror = CheckpointMirror(local_run, drive_run, every=1, keep=cfg.get('mirror_keep', 3),
                              watch_path=str(local_root()), min_free_gib=8, mirror_final=True,
                              mirror_optimizer_state=cfg.get('mirror_optimizer_state', False),
                              retain_steps=cfg.get('retained_checkpoint_steps', ()))
    if a.pause_after is not None and mirror.has_full_mirror(a.pause_after):
        log(f'{run}: paused after step {a.pause_after}; full Drive mirror confirmed')
        return
    if a.pause_after is not None and resume and read_state(resume)['global_step'] >= a.pause_after:
        raise RuntimeError(f'{run}: local progress already reached step {a.pause_after} without the requested Drive mirror')

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOTrainer
    tok = AutoTokenizer.from_pretrained(model_path)
    prompt_messages = messages
    if mmr_cfg:
        from experiments.qfft_v93.mmr_grpo import (
            JinaEmbeddings, MmrDRGRPOTrainer, MmrRecoveryCheckpoints, MmrReward, training_messages,
        )
        prompt_messages = training_messages
    if not a.smoke and not mmr_cfg:
        lengths = [len(tok.apply_chat_template(messages(row['question']), add_generation_prompt=True,
                                               tokenize=True, return_dict=False)) for row in pool]
        if max(lengths) > cfg['max_prompt_tokens']:
            raise ValueError(f'Longest GRPO prompt is {max(lengths)} tokens, above the configured prompt budget')
    dataset = Dataset.from_list([{'prompt': prompt_messages(r['question']), 'reference_answer': r['reference_answer'],
                                  'sample_id': r['sample_id']} for r in pool])
    args = grpo_config(cfg, local_run, a.smoke, a.max_steps)
    lora = cfg['lora']
    peft_config = LoraConfig(r=lora['r'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
                             target_modules=lora['target_modules'], task_type='CAUSAL_LM')
    swan = SwanLab(drive_run, run, {'identity': identity, 'pool': pool_report}, tags=['v93', 'grpo', 'lora'],
                   enabled=not (a.no_swanlab or a.smoke), log_dir=local_run / 'swanlog')
    journal = JsonlAppender(local_run / 'reward_status.jsonl', mirror=drive_run / 'reward_status.jsonl')
    callbacks = [ProgressCallback(local_run, drive_run, swan, {'run': run})]
    if mmr_cfg:
        callbacks.append(MmrRecoveryCheckpoints(cfg.get('retained_checkpoint_steps', ())))
    callbacks.append(mirror)
    milestone_worker = None
    if cfg.get('retained_checkpoint_steps'):
        if not extending or not mmr_cfg or not merge_adapter:
            raise RuntimeError('Retained MMR milestones require a full-state continuation and model exports')
        from experiments.qfft_v93.milestones import start_worker
        milestone_worker = start_worker(cfg, identity, model_path, drive_run, source)
    comparison = None
    if comparison_ids is not None or cfg.get('measure_gpu'):
        from experiments.qfft_v93.grpo_telemetry import ComparisonTelemetry
        comparison = ComparisonTelemetry()
        callbacks.insert(0, comparison)
    if a.pause_after is not None:
        from transformers import TrainerCallback

        class PauseAfterSave(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                if state.global_step == a.pause_after:
                    mirror.wait()
                    if not mirror.has_full_mirror(state.global_step):
                        raise RuntimeError(f'Full mirror for step {state.global_step} could not be confirmed: {mirror.error}')
                    control.should_training_stop = True
                return control

        callbacks.append(PauseAfterSave())
    grading = nullcontext() if mmr_cfg else Grader(workers=cfg.get('reward_workers', 8),
                                                   timeout=cfg.get('reward_timeout', 30))
    with grading as grader:
        if mmr_cfg:
            reward = MmrReward(tok, cfg, JinaEmbeddings(mmr_cfg))
            trainer_class = MmrDRGRPOTrainer
            trainer_kwargs = {'prompt_limit': cfg['max_prompt_tokens'], 'scheduler_steps': mmr_cfg['scheduler_steps']}
        else:
            reward = FinalAnswerReward(tok, grader, journal, last_box_fallback=cfg.get('last_box_fallback', False))
            trainer_class = GRPOTrainer
            trainer_kwargs = {}
        # Strings keep GRPOConfig JSON-serializable; TRL maps them to torch dtypes.
        args.model_init_kwargs = {**(args.model_init_kwargs or {}),
                                  'dtype': 'float32' if a.smoke else 'bfloat16', 'attn_implementation': 'sdpa'}
        trainer = trainer_class(model=str(model_path), reward_funcs=[reward], args=args, train_dataset=dataset,
                                processing_class=tok, peft_config=peft_config, callbacks=callbacks, **trainer_kwargs)
        if mmr_cfg:
            reward.eos.update(trainer.eos_token_ids)
        if mmr_cfg and reward.exclude_masked_completions:
            reward.diagnostics = RolloutDiagnostics(
                trainer, mmr_dir=drive_run / 'rollout_diagnostics', identity={
                    'run': run, 'run_identity_sha256': sha256_json(identity),
                    'embedding_revision': mmr_cfg['embedding_revision'], 'code_bundle': identity['code_bundle'],
                },
            )
        elif cfg.get('log_rollout_diagnostics'):
            if not args.use_vllm or args.vllm_mode != 'colocate':
                raise ValueError('Rollout diagnostics require colocated vLLM')
            diagnostic_file = 'rollout_diagnostics.jsonl'
            reward.diagnostics = RolloutDiagnostics(
                trainer, JsonlAppender(local_run / diagnostic_file, mirror=drive_run / diagnostic_file), comparison_ids)
        beat = Heartbeat(lambda: read_json(drive_run / 'status.json').get('step')
                         if (drive_run / 'status.json').exists() else 'starting', interval=300)
        beat.start()
        started = time.monotonic()
        try:
            if comparison is not None:
                comparison.attach(trainer)
                comparison.start()
            if mmr_cfg and resume and read_state(resume)['global_step'] == cfg['max_steps']:
                from transformers import TrainerState

                # Recover an interrupted final export without another optimizer update.
                trainer._load_from_checkpoint(resume)
                trainer.state = TrainerState.load_from_json(Path(resume) / 'trainer_state.json')
                if not mirror.has_full_mirror(trainer.state.global_step):
                    mirror.on_save(args, trainer.state, trainer.control)
            else:
                trainer.train(resume_from_checkpoint=resume)
        finally:
            beat.stop()
            if comparison is not None:
                comparison.close()
            journal.close()
            if getattr(reward, 'diagnostics', None) and reward.diagnostics.journal is not None:
                reward.diagnostics.journal.close()
        if a.pause_after is not None:
            if trainer.state.global_step != a.pause_after:
                raise RuntimeError(f'{run}: expected to pause at step {a.pause_after}, reached {trainer.state.global_step}')
            swan.finish()
            log(f'{run}: paused after step {a.pause_after}; full Drive mirror confirmed')
            return
        if mmr_cfg and trainer.state.global_step != cfg['max_steps']:
            raise RuntimeError(f'{run}: expected to finish at step {cfg["max_steps"]}, got {trainer.state.global_step}')
        if cfg.get('verify_final_checkpoint'):
            checkpoint = local_run / f'checkpoint-{trainer.state.global_step}'
            if read_state(checkpoint)['global_step'] != trainer.state.global_step:
                raise RuntimeError(f'{run}: local checkpoint has the wrong step')
            reload_mmr_checkpoint(trainer, checkpoint)
            log(f'{run}: reloaded full local {checkpoint.name}')
        adapter_local = local_run / 'adapter'
        trainer.save_model(str(adapter_local))
        tok.save_pretrained(str(adapter_local))
        mirror.wait()
        if cfg.get('mirror_optimizer_state') and not mirror.has_full_mirror(trainer.state.global_step):
            raise RuntimeError(f'Final full checkpoint mirror could not be confirmed: {mirror.error}')
    drive_io.upload_tree(adapter_local, drive_run / 'adapter', watch_path=str(local_root()))
    summary = {'steps': trainer.state.global_step, 'resumed_from': resume, 'adapter': str(drive_run / 'adapter'),
               'wall_hours_this_session': round((time.monotonic() - started) / 3600, 3), 'completed': now_iso()}
    if cfg.get('verify_final_checkpoint'):
        summary['local_checkpoint_reload_verified'] = True
    if extending:
        summary['extension'] = identity['extension']
    if milestone_worker:
        from experiments.qfft_v93.milestones import wait_worker
        log('Waiting for all retained milestone model exports ...')
        summary['milestones'] = wait_worker(*milestone_worker)
    elif mmr_cfg and merge_adapter:
        from experiments.qfft_v93.merge_lora import merge
        merge(model_path, adapter_local, drive_run / 'merged', smoke=a.smoke)
    write_json(drive_run / 'GRPO_COMPLETE.json', summary)
    write_json(drive_run / 'status.json', {'state': 'complete', **summary})
    swan.finish()
    if merge_adapter and not mmr_cfg:
        from experiments.qfft_v93.merge_lora import merge
        merge(model_path, adapter_local, drive_run / 'merged', smoke=a.smoke)
    shutil.rmtree(local_run, ignore_errors=True)
    log(f'{run}: COMPLETE -> {drive_run}')


if __name__ == '__main__':
    main()
