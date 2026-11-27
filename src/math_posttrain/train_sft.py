"""Single-GPU full-parameter QFFT or SFT training for Qwen2.5-7B-Instruct.

Usage:
    torchrun --standalone --nnodes=1 --nproc_per_node=1 \
        -m math_posttrain.train_sft --config train/qfft_r1.json

Memory plan (7.62B params, single GPU): bf16 weights + bf16 grads on GPU (~30 GB), fp32 master
weights and AdamW moments offloaded to host RAM via DeepSpeed ZeRO-2 offload, gradient checkpointing,
and Liger fused linear cross-entropy.
"""
from __future__ import annotations

import argparse
import inspect
import math
import time
from pathlib import Path

from math_posttrain.common import (
    get_config_dir, latest_checkpoint, load_config, log, now_iso, read_jsonl, sha256_file, workspace, write_json,
)
from math_posttrain.formats import IGNORE_INDEX, encode
from math_posttrain.models import resolve_model


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, rows):
        import torch
        width = max(len(r['input_ids']) for r in rows)
        pad = lambda seq, value: seq + [value] * (width - len(seq))  # noqa: E731
        return {
            'input_ids': torch.tensor([pad(r['input_ids'], self.pad_id) for r in rows]),
            'attention_mask': torch.tensor([pad([1] * len(r['input_ids']), 0) for r in rows]),
            'labels': torch.tensor([pad(r['labels'], IGNORE_INDEX) for r in rows]),
        }


def training_steps(cfg: dict, rows: int) -> int:
    eff_batch = cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']
    return math.ceil(rows / eff_batch) * cfg['num_train_epochs']


def build_training_args(cfg: dict, out_dir: Path, rows: int, *, smoke: bool, max_steps: int | None):
    from transformers import TrainingArguments

    fields = set(inspect.signature(TrainingArguments).parameters)
    total = training_steps(cfg, rows)
    warmup_steps = max(1, math.ceil((max_steps or total) * cfg['warmup_fraction']))
    kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=cfg['num_train_epochs'],
        max_steps=max_steps or -1,
        learning_rate=cfg['learning_rate'],
        lr_scheduler_type=cfg['lr_scheduler_type'],
        warmup_steps=warmup_steps,
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        weight_decay=cfg['weight_decay'],
        adam_beta1=cfg['adam_beta1'],
        adam_beta2=cfg['adam_beta2'],
        max_grad_norm=cfg['max_grad_norm'],
        bf16=not smoke,
        fp16=False,
        gradient_checkpointing=not smoke,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1,
        save_strategy='steps',
        save_steps=cfg['save_steps'],
        save_total_limit=cfg.get('save_total_limit', 2),
        seed=cfg['seed'],
        data_seed=cfg['seed'],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to='none',
    )
    if not smoke:
        if cfg.get('deepspeed'):
            kwargs['deepspeed'] = str(get_config_dir() / cfg['deepspeed'])
        if cfg.get('liger'):
            kwargs['use_liger_kernel'] = True
    elif 'use_cpu' in fields:
        kwargs['use_cpu'] = True
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in fields})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='train/qfft_r1.json', help='Config path under configs/ or filesystem')
    p.add_argument('--model', help='Override init_model (alias, HF repo ID, or local directory)')
    p.add_argument('--data', help='Override training JSONL path (default: data/<config.data>)')
    p.add_argument('--out', help='Override output directory (default: outputs/runs/<run_name>)')
    p.add_argument('--max-steps', type=int, help='Override total optimizer steps')
    p.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint in output dir')
    p.add_argument('--smoke', action='store_true', help='CPU/fp32 plumbing test without DeepSpeed/Liger')
    a = p.parse_args()

    cfg = load_config(a.config)
    ws = workspace()
    run_name = cfg['run_name']
    out_dir = Path(a.out) if a.out else ws.runs / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(a.data) if a.data else ws.data_dir / cfg['data']
    if not data_path.is_file():
        raise FileNotFoundError(
            f'Training data not found: {data_path}. Run `math-posttrain prepare-data --sft` first.'
        )

    rows = sorted(read_jsonl(data_path), key=lambda r: r.get('index', 0))
    model_spec = a.model or cfg.get('init_model', 'base')
    model_path = resolve_model(model_spec)

    import torch
    import transformers
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

    if not a.smoke and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for full-parameter training (pass --smoke for CPU check).')

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    truncate = bool(cfg.get('truncate', False))
    encoded = [
        encode(tok, cfg['format'], r['question'], r['response'], cfg['max_length'], truncate=truncate)
        for r in rows
    ]
    stats = {
        'examples': len(encoded),
        'format': cfg['format'],
        'total_tokens': sum(len(e['input_ids']) for e in encoded),
        'supervised_tokens': sum(e['supervised_tokens'] for e in encoded),
        'max_tokens': max(len(e['input_ids']) for e in encoded),
        'data_sha256': sha256_file(data_path),
    }
    write_json(out_dir / 'tokenization.json', stats)
    log(f"{run_name}: {stats['examples']} examples, {stats['total_tokens']:,} tokens ({stats['supervised_tokens']:,} supervised)")

    args = build_training_args(cfg, out_dir, len(encoded), smoke=a.smoke, max_steps=a.max_steps)
    dtype = torch.float32 if a.smoke else torch.bfloat16
    major, minor = (int(x) for x in transformers.__version__.split('.')[:2])
    load_kwargs = {'attn_implementation': cfg.get('attn_implementation', 'sdpa')}
    load_kwargs['dtype' if (major, minor) >= (4, 56) else 'torch_dtype'] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.config.use_cache = False

    class _Rows(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, i):
            return encoded[i]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_Rows(),
        data_collator=Collator(tok.pad_token_id),
    )
    resume_ckpt = latest_checkpoint(out_dir) if a.resume else None
    t0 = time.monotonic()
    result = trainer.train(resume_from_checkpoint=resume_ckpt)

    # Restore KV-cache for fast downstream inference before exporting final weights.
    model.config.use_cache = True
    final_dir = out_dir / 'final'
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    write_json(out_dir / 'train_summary.json', {
        'run_name': run_name,
        'steps': trainer.state.global_step,
        'metrics': result.metrics,
        'wall_hours': round((time.monotonic() - t0) / 3600, 3),
        'final_dir': str(final_dir),
        'completed': now_iso(),
    })
    log(f'{run_name}: training complete -> {final_dir}')


if __name__ == '__main__':
    main()
