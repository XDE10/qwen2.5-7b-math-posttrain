"""Full-parameter SFT or QFFT of Qwen2.5-7B-Instruct on one GPU, resumable across VMs.

    python -m torch.distributed.run --standalone --nproc_per_node=1 \
        -m experiments.qfft_v93.train_sft --config train/qfft_ds.json

Memory plan (7.62B params, one GPU): bf16 weights + bf16 grads on GPU (~30 GB), fp32 master weights and
AdamW moments on the CPU through DeepSpeed ZeRO-2 offload (~122 GB host RAM), gradient checkpointing
and Liger fused linear cross-entropy so a 32K-token example never materializes 32K x 152K logits.

The SFT and QFFT runs of one teacher read the same rows in the same order (same seed, same dataset
length) and differ only in the encoding from formats.py.
"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import shutil
import time
from pathlib import Path

from experiments.qfft_v93 import drive_io
from experiments.qfft_v93.checkpointing import CheckpointMirror, prune_all_mirrors, resolve_resume
from experiments.qfft_v93.common import (CONFIG_DIR, Heartbeat, drive_layout, local_root, log,
                                         load_config, now_iso, read_json, read_jsonl, sha256_file, sha256_json,
                                         write_json)
from experiments.qfft_v93.formats import IGNORE_INDEX, encode
from experiments.qfft_v93.telemetry import ProgressCallback, SwanLab


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--model-path', help='Override the local model directory (default: prepared base model)')
    p.add_argument('--data', help='Override the training JSONL path')
    p.add_argument('--max-steps', type=int, help='Override for smoke tests')
    p.add_argument('--smoke', action='store_true', help='CPU/fp32/no DeepSpeed plumbing test with a tiny model')
    p.add_argument('--no-swanlab', action='store_true')
    p.add_argument('--stop-after', type=int, help='Test hook: exit after this optimizer step (simulated crash)')
    p.add_argument('--new-run', action='store_true', help='Allow a different identity in an existing run dir')
    p.add_argument('--no-export', action='store_true', help='Smoke tests: skip exporting final weights')
    return p.parse_args()


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, rows):
        import torch
        width = max(len(r['input_ids']) for r in rows)
        pad = lambda seq, value: seq + [value] * (width - len(seq))  # noqa: E731
        return {'input_ids': torch.tensor([pad(r['input_ids'], self.pad_id) for r in rows]),
                'attention_mask': torch.tensor([pad([1] * len(r['input_ids']), 0) for r in rows]),
                'labels': torch.tensor([pad(r['labels'], IGNORE_INDEX) for r in rows])}


def stop_after_callback(step: int, mirror: CheckpointMirror):
    """Test hook: simulate losing the VM right after checkpoint `step` reached Drive."""
    from transformers import TrainerCallback

    class _Stop(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step >= step:
                mirror.wait()
                log(f'--stop-after {step}: simulating a VM loss after the save')
                os._exit(17)
    return _Stop()


def training_arguments(cfg: dict, out: Path, rows: int, smoke: bool, max_steps: int | None):
    from transformers import TrainingArguments
    fields = set(inspect.signature(TrainingArguments).parameters)
    per_epoch = math.ceil(rows / (cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']))
    total = max_steps or per_epoch * cfg['num_train_epochs']
    kwargs = dict(
        output_dir=str(out), num_train_epochs=cfg['num_train_epochs'], max_steps=max_steps or -1,
        learning_rate=cfg['learning_rate'], lr_scheduler_type=cfg['lr_scheduler_type'],
        warmup_steps=max(1, math.ceil(total * cfg['warmup_fraction'])),
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        weight_decay=cfg['weight_decay'], adam_beta1=cfg['adam_beta1'], adam_beta2=cfg['adam_beta2'],
        max_grad_norm=cfg['max_grad_norm'], bf16=not smoke, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1, save_strategy='steps', save_steps=cfg['save_steps'], save_total_limit=None,
        seed=cfg['seed'], data_seed=cfg['seed'], dataloader_num_workers=0, remove_unused_columns=False,
        report_to='none', disable_tqdm=True)
    if 'include_num_input_tokens_seen' in fields:
        kwargs['include_num_input_tokens_seen'] = True
    if not smoke:
        if cfg.get('deepspeed'):
            kwargs['deepspeed'] = str(CONFIG_DIR / cfg['deepspeed'])
        if cfg.get('liger'):
            kwargs['use_liger_kernel'] = True
    elif 'use_cpu' in fields:
        kwargs['use_cpu'] = True
    unsupported = set(kwargs) - fields
    if unsupported:
        raise TypeError('Unsupported TrainingArguments fields for this Transformers version: ' + ', '.join(sorted(unsupported)))
    return TrainingArguments(**kwargs), total


def main() -> None:
    a = parse_args()
    cfg = load_config(a.config)
    layout = drive_layout()
    run = cfg['run_name']
    drive_run = layout.runs / run
    local_run = local_root() / 'runs' / run
    drive_run.mkdir(parents=True, exist_ok=True)
    local_run.mkdir(parents=True, exist_ok=True)
    if (drive_run / 'TRAIN_COMPLETE.json').exists():
        log(f'{run}: already complete ({drive_run / "TRAIN_COMPLETE.json"}); nothing to do')
        return
    data_path = Path(a.data) if a.data else layout.data / cfg['data']
    data_sha = sha256_file(data_path)
    if a.model_path:
        model_path = Path(a.model_path)
    else:
        from experiments.qfft_v93.models import ensure_local, verify_base_tokenizer
        model_path = ensure_local(cfg.get('init_model', 'base'), layout)
        verify_base_tokenizer(model_path)
    identity = {'run': run, 'format': cfg['format'], 'teacher': cfg['teacher'], 'data_sha256': data_sha,
                'init_model': cfg.get('init_model', 'base'),
                'recipe': {k: cfg[k] for k in ('max_length', 'num_train_epochs', 'learning_rate', 'warmup_fraction',
                                              'lr_scheduler_type', 'per_device_train_batch_size',
                                              'gradient_accumulation_steps', 'weight_decay', 'adam_beta1',
                                              'adam_beta2', 'max_grad_norm', 'seed')},
                'smoke': a.smoke, 'max_steps_override': a.max_steps}
    id_path = drive_run / 'run_identity.json'
    if id_path.exists() and read_json(id_path)['identity'] != identity and not a.new_run:
        raise RuntimeError(f'{id_path} describes a different experiment; use a new run_name or --new-run')
    write_json(id_path, {'identity': identity, 'identity_sha256': sha256_json(identity),
                         'code_bundle': os.environ.get('P1_CODE_SHA'), 'updated': now_iso()})

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
    if not a.smoke and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required (use --smoke for a CPU plumbing test)')
    tok = AutoTokenizer.from_pretrained(model_path)
    rows = sorted(read_jsonl(data_path), key=lambda r: r['index'])
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate sample ids in training data')
    encoded = [encode(tok, cfg['format'], r['question'], r['response'], cfg['max_length']) for r in rows]
    stats = {'examples': len(encoded), 'format': cfg['format'],
             'total_tokens': sum(len(e['input_ids']) for e in encoded),
             'supervised_tokens': sum(e['supervised_tokens'] for e in encoded),
             'max_tokens': max(len(e['input_ids']) for e in encoded),
             'loss': 'assistant response + <|im_end|>' if cfg['format'] == 'sft' else 'every response token, no EOS'}
    write_json(drive_run / 'tokenization.json', stats)
    log(f"{run}: {stats['examples']} examples, {stats['total_tokens']:,} tokens "
        f"({stats['supervised_tokens']:,} supervised), longest {stats['max_tokens']}")

    args, total_steps = training_arguments(cfg, local_run, len(encoded), a.smoke, a.max_steps)
    dtype = torch.float32 if a.smoke else torch.bfloat16
    load_kwargs = {'attn_implementation': cfg.get('attn_implementation', 'sdpa')}
    major, minor = (int(x) for x in transformers.__version__.split('.')[:2])
    load_kwargs['dtype' if (major, minor) >= (4, 56) else 'torch_dtype'] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    if not a.smoke and sum(p.numel() for p in model.parameters() if p.requires_grad) != n_params:
        raise RuntimeError('Expected full-parameter training')

    swan = SwanLab(drive_run, run, {'identity': identity, 'tokenization': stats, 'params': n_params,
                                    'transformers': transformers.__version__, 'torch': torch.__version__},
                   tags=['v93', cfg['format'], cfg['teacher'], 'full-ft'], enabled=not (a.no_swanlab or a.smoke),
                   log_dir=local_run / 'swanlog')
    mirror = CheckpointMirror(local_run, drive_run, every=cfg.get('mirror_every_saves', 1),
                              keep=cfg.get('mirror_keep', 2), watch_path=str(local_root()))
    callbacks = [ProgressCallback(local_run, drive_run, swan, {'run': run}), mirror]
    if a.stop_after:
        callbacks.append(stop_after_callback(a.stop_after, mirror))
    from torch.utils.data import Dataset

    class Rows(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, i):
            return encoded[i]

    trainer = Trainer(model=model, args=args, train_dataset=Rows(), data_collator=Collator(tok.pad_token_id),
                      callbacks=callbacks)
    resume = resolve_resume(local_run, drive_run)
    beat = Heartbeat(lambda: read_json(drive_run / 'status.json').get('step') if (drive_run / 'status.json').exists()
                     else 'starting', interval=300)
    beat.start()
    started = time.monotonic()
    try:
        result = trainer.train(resume_from_checkpoint=resume)
    finally:
        beat.stop()
        swan.log({'lifecycle/train_loop_exit': 1}, trainer.state.global_step)
    if a.no_export:
        mirror.wait()
        write_json(drive_run / 'status.json', {'state': 'smoke_done', 'steps': trainer.state.global_step,
                                               'metrics': result.metrics, 'updated': now_iso()})
        swan.finish()
        shutil.rmtree(local_run, ignore_errors=True)
        log(f'{run}: smoke finished at step {trainer.state.global_step} (no export)')
        return
    log(f'{run}: training loop finished at step {trainer.state.global_step}; exporting final weights')
    final_local = local_run / 'final'
    trainer.save_model(str(final_local))
    tok.save_pretrained(str(final_local))
    write_json(final_local / 'p1_model_identity.json', {'identity': identity, 'steps': trainer.state.global_step,
                                                        'base': cfg.get('init_model', 'base'), 'exported': now_iso()})
    mirror.wait()
    drive_io.upload_tree(final_local, drive_run / 'final', watch_path=str(local_root()))
    summary = {'steps': trainer.state.global_step, 'metrics': result.metrics, 'tokenization': stats,
               'wall_hours_this_session': round((time.monotonic() - started) / 3600, 3),
               'final': str(drive_run / 'final'), 'resumed_from': resume, 'completed': now_iso()}
    write_json(drive_run / 'TRAIN_COMPLETE.json', summary)
    write_json(drive_run / 'status.json', {'state': 'complete', **summary})
    swan.log({'lifecycle/export_complete': 1}, trainer.state.global_step)
    swan.finish()
    if not cfg.get('keep_mirrors_after_complete', False):
        prune_all_mirrors(drive_run)
    # Keep the exported weights on local SSD where `eval run:<name>` looks first (no 15 GB re-copy).
    from experiments.qfft_v93.models import local_cache_path
    cache = local_cache_path(f'run:{run}', layout)
    shutil.rmtree(cache, ignore_errors=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(final_local), str(cache))
    shutil.rmtree(local_run, ignore_errors=True)
    log(f'{run}: COMPLETE -> {drive_run / "final"}')


if __name__ == '__main__':
    main()
