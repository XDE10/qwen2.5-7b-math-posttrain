"""Capture one fixed R1 rollout batch and compare vLLM IS weights offline."""
from __future__ import annotations

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from experiments.qfft_v93 import drive_io
from experiments.qfft_v93.common import drive_layout, load_config, local_root, log, read_json, read_jsonl, write_jsonl

RUN = 'r1_is_diagnostic'
CAPTURE = 'logprobs.jsonl'
TRL_VERSION = '1.17.1'


def compare_weights(rows, upper=3.0):
    """Summarize the TRL correction on completion tokens admitted by its loss mask."""
    import torch

    groups = {mode: {} for mode in ('sequence_mask', 'token_mask')}
    seen = set()
    missing_sampling_logps = 0
    for row in rows:
        sid, draw = row['sample_id'], row['draw']
        if (sid, draw) in seen:
            raise ValueError(f'Duplicate completion: {sid} draw {draw}')
        seen.add((sid, draw))
        tokens = len(row['completion_ids'])
        sampling = row['sampling_logps']
        training = row['training_logps']
        mask = row['completion_mask']
        if not tokens or not (tokens == len(sampling) == len(training) == len(mask)):
            raise ValueError(f'Misaligned token data: {sid} draw {draw}')
        mask = torch.tensor(mask, dtype=torch.bool)
        sampling = torch.tensor([float('nan') if value is None else value for value in sampling],
                                dtype=torch.float32)
        training = torch.tensor(training, dtype=torch.float32)
        missing_sampling_logps += int((sampling.isnan() & mask).sum())
        # Missing sampling logprobs get zero log-ratio, as in TRL 1.17.1.
        delta = torch.nan_to_num((training - sampling) * mask, nan=0.0)
        sequence = delta.sum().exp()
        token = delta.exp()
        sequence = sequence.masked_fill(sequence > upper, 0)
        if 'trl_sequence_weight' in row and not math.isclose(
            float(sequence), row['trl_sequence_weight'], rel_tol=5e-4, abs_tol=1e-8
        ):
            raise ValueError(f'Recomputed sequence weight disagrees with TRL: {sid} draw {draw}')
        weights = {
            'sequence_mask': sequence.expand_as(delta)[mask],
            'token_mask': token.masked_fill(token > upper, 0)[mask],
        }
        advantage = abs(row['advantage'])
        length = '<4K' if tokens < 4096 else '4K–8K' if tokens < 8192 else '8K+'
        reward = row['reward_status']
        for mode, weight in weights.items():
            nonzero = int(weight.count_nonzero())
            for key in ('all', f'length/{length}', f'reward/{reward}'):
                totals = groups[mode].setdefault(key, {
                    'completions': 0, 'loss_tokens': 0, 'zero_weight_tokens': 0,
                    'zero_weight_completions': 0, 'nonzero_advantage_tokens': 0,
                    'weighted_nonzero_advantage_tokens': 0, 'is_weight_sum': 0.0,
                    'absolute_advantage_weight_sum': 0.0,
                })
                totals['completions'] += 1
                totals['loss_tokens'] += weight.numel()
                totals['zero_weight_tokens'] += weight.numel() - nonzero
                totals['zero_weight_completions'] += int(bool(weight.numel()) and nonzero == 0)
                totals['nonzero_advantage_tokens'] += weight.numel() if advantage else 0
                totals['weighted_nonzero_advantage_tokens'] += nonzero if advantage else 0
                totals['is_weight_sum'] += float(weight.sum())
                totals['absolute_advantage_weight_sum'] += advantage * float(weight.sum())
    if not seen:
        raise ValueError('No completions to compare')
    return {'completions': len(seen), 'upper': upper,
            'missing_sampling_logps_on_loss_tokens': missing_sampling_logps, 'modes': groups}


def capture_rows(result, diagnostics, sample_ids, draws):
    """Align the exact padded tensors returned by TRL with its scored rollouts."""
    names = ('prompt_ids', 'prompt_mask', 'completion_ids', 'completion_mask',
             'sampling_per_token_logps', 'old_per_token_logps', 'advantages')
    batch = {name: result[name].detach().cpu() for name in names}
    expected = [sid for sid in sample_ids for _ in range(draws)]
    if [row['sample_id'] for row in diagnostics] != expected:
        raise RuntimeError('Unexpected prompt groups in diagnostic rollout')
    if any(len(tensor) != len(expected) for tensor in batch.values()):
        raise RuntimeError('Unexpected TRL diagnostic tensor batch size')
    for i, row in enumerate(diagnostics):
        n = row['tokens']
        mask = batch['completion_mask'][i, :n]
        if (n <= 0 or n > batch['completion_ids'].shape[1] or int(mask.sum()) != (0 if row['trl_clipped'] else n)
                or batch['completion_mask'][i, n:].any().item()):
            raise RuntimeError(f'Unexpected TRL completion mask at index {i}')
        sampling = batch['sampling_per_token_logps'][i, :n].tolist()
        training = batch['old_per_token_logps'][i, :n].tolist()
        if any(math.isinf(value) for value in sampling) or not all(math.isfinite(value) for value in training):
            raise RuntimeError(f'Non-finite training or infinite sampling logprob at index {i}')
        yield {
            'sample_id': row['sample_id'], 'draw': i % draws,
            'prompt_ids': batch['prompt_ids'][i][batch['prompt_mask'][i].bool()].tolist(),
            'completion_ids': batch['completion_ids'][i, :n].tolist(),
            'completion_mask': mask.tolist(),
            'sampling_logps': [None if math.isnan(value) else value for value in sampling],
            'training_logps': training,
            'advantage': float(batch['advantages'][i]),
            'text': row['text'], 'finish_reason': row['finish_reason'], 'stop_reason': row['stop_reason'],
            'trl_clipped': row['trl_clipped'], 'reward_status': row['reward_status'],
            'trl_sequence_weight': float(result['importance_sampling_ratio'][i].item()),
        }


def capture(out):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOTrainer

    from experiments.qfft_v93.formats import messages
    from experiments.qfft_v93.grading import Grader
    from experiments.qfft_v93.grpo import FinalAnswerReward, RolloutDiagnostics, build_pool, grpo_config
    from experiments.qfft_v93.models import MARK, ensure_local

    if version('trl') != TRL_VERSION:
        raise RuntimeError(f'This capture uses the TRL {TRL_VERSION} rollout interface')
    if not torch.cuda.is_available():
        raise RuntimeError('Capture requires CUDA. The compare subcommand runs on CPU.')
    layout = drive_layout()
    # Reuse the already-approved 16K comparison panel, taking its first batch.
    cfg = load_config('grpo/compare_grpo_qfft_r1_16k.json')
    n_per_step = cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']
    draws = cfg['num_generations']
    pool, _ = build_pool(cfg, layout, layout.data / 'benchmarks_v93', set())
    pool = pool[:n_per_step // draws]
    ids = [row['sample_id'] for row in pool]
    model_path = ensure_local(cfg['init_model'], layout)
    identity = read_json(model_path / MARK)
    if identity.get('steps') != 750 or identity.get('identity', {}).get('run') != cfg['init_model'].removeprefix('run:'):
        raise RuntimeError('Expected the original R1 QFFT step-750 export')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    args = grpo_config(cfg, out, smoke=False, max_steps=1)
    if (not args.vllm_importance_sampling_correction or args.vllm_importance_sampling_mode != 'sequence_mask'
            or args.vllm_importance_sampling_clip_min is not None or args.vllm_importance_sampling_clip_max != 3.0):
        raise RuntimeError('Capture expected the original TRL vLLM importance sampling settings')
    args.model_init_kwargs = {'dtype': 'bfloat16', 'attn_implementation': 'sdpa'}
    lora = cfg['lora']
    peft = LoraConfig(r=lora['r'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
                      target_modules=lora['target_modules'], task_type='CAUSAL_LM')
    dataset = Dataset.from_list([{
        'prompt': messages(row['question']), 'reference_answer': row['reference_answer'],
        'sample_id': row['sample_id'],
    } for row in pool])
    inputs = [row for row in dataset for _ in range(draws)]
    diagnostics = []
    with Grader(workers=cfg['reward_workers'], timeout=cfg['reward_timeout']) as grader:
        reward = FinalAnswerReward(tokenizer, grader, SimpleNamespace(write=lambda _: None),
                                   last_box_fallback=cfg['last_box_fallback'])
        trainer = GRPOTrainer(model=str(model_path), reward_funcs=[reward], args=args,
                              train_dataset=dataset, processing_class=tokenizer, peft_config=peft)
        reward.diagnostics = RolloutDiagnostics(trainer, SimpleNamespace(write=diagnostics.append))
        # Fresh LoRA has a zero update. Use train-mode sampling and scoring without an optimizer or backward.
        trainer.model.train()
        with torch.no_grad():
            result = trainer._generate_and_score_completions(inputs)
        if len(diagnostics) != n_per_step or int(result['completion_mask'].sum()) != int(result['num_items_in_batch']):
            raise RuntimeError('Unexpected rollout size or DAPO loss-token denominator')
        rows = list(capture_rows(result, diagnostics, ids, draws))
    write_jsonl(out / CAPTURE, rows)
    return out / CAPTURE


def run_capture():
    local = local_root() / 'diagnostics' / RUN
    remote = drive_layout().runs / RUN / CAPTURE
    if remote.is_file():
        data = remote
    else:
        data = local / CAPTURE
        if not data.is_file():
            data = capture(local)
    rows = read_jsonl(data, tolerate_torn_tail=False)
    cfg = load_config('grpo/compare_grpo_qfft_r1_16k.json')
    batch_size = cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']
    ids = cfg['pool']['comparison_sample_ids'][:batch_size // cfg['num_generations']]
    if [(row['sample_id'], row['draw']) for row in rows] != [
        (sid, draw) for sid in ids for draw in range(cfg['num_generations'])
    ]:
        raise RuntimeError('Capture does not contain the complete fixed R1 batch in order')
    summary = compare_weights(rows)
    if data != remote:
        drive_io.upload_file(data, remote)
    log(f'{RUN}: saved {len(rows)} completions to {remote}')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('capture', help='Generate and score one fixed R1 batch on a GPU')
    compare = commands.add_parser('compare', help='Recompute both corrections on CPU')
    compare.add_argument('capture', type=Path)
    args = parser.parse_args()
    if args.command == 'capture':
        run_capture()
    else:
        print(json.dumps(compare_weights(read_jsonl(args.capture, tolerate_torn_tail=False)),
                         indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
