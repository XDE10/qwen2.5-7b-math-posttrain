"""Plan an explicit weights-only continuation in a separate run."""
from __future__ import annotations

import math
import re
from pathlib import Path

from experiments.qfft_v93.checkpointing import latest_warm_start_weights
from experiments.qfft_v93.common import read_json, sha256_json


def run_name(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError(f'Invalid run name: {value!r}')
    return value


def plan_warm_start(runs: Path, source_run: str, identity: dict, total: int) -> dict:
    source = runs / run_name(source_run)
    if source_run == identity['run']:
        raise ValueError('Warm-start requires a separate destination run')
    source_identity = read_json(source / 'run_identity.json')['identity']
    source_recipe = {k: v for k, v in source_identity.items() if k not in ('run', 'warm_start')}
    if source_identity['run'] != source_run or source_recipe != {k: v for k, v in identity.items() if k != 'run'}:
        raise ValueError('Warm-start source and destination must use the same data, base model and training recipe')
    recipe = identity['recipe']
    if recipe['lr_scheduler_type'] != 'cosine':
        raise ValueError('Weights-only warm-start currently supports the cosine schedule')
    step, path, state, files = latest_warm_start_weights(source)
    previous = source_identity.get('warm_start', {})
    previous_offset = previous.get('original_start_step', 0)
    offset = previous_offset + step
    if state.get('max_steps') != total - previous_offset or not 0 < offset < total:
        raise ValueError(f'Checkpoint {path} does not have remaining steps in the original {total}-step run')
    warmup = max(1, math.ceil(total * recipe['warmup_fraction']))
    return {
        'mode': 'weights_only_warm_start',
        'source_run': source_run,
        'source_checkpoint': path.name,
        'source_checkpoint_step': step,
        'source_identity_sha256': sha256_json(source_identity),
        'source_weight_files': files,
        'original_start_step': offset,
        'original_total_steps': total,
        'remaining_steps': total - offset,
        'optimizer': 'rebuilt_from_config',
        'sampler': 'restarted_from_configured_seed',
        'learning_rate': {
            'policy': 'original_cosine_at_checkpoint_offset',
            'warmup_steps': warmup,
            'base_lr': recipe['learning_rate'],
            'first_update_lr': recipe['learning_rate'] * cosine_multiplier(offset, warmup, total),
        },
    }


def cosine_multiplier(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / warmup
    progress = (min(step, total) - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def warm_start_scheduler(optimizer, plan: dict):
    from torch.optim.lr_scheduler import LambdaLR

    offset = plan['original_start_step']
    warmup = plan['learning_rate']['warmup_steps']
    total = plan['original_total_steps']
    return LambdaLR(optimizer, lambda step: cosine_multiplier(offset + step, warmup, total))
