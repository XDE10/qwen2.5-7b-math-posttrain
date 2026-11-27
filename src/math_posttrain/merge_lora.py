"""Merge a LoRA adapter into its base checkpoint and save bf16 safetensors (GPU if present, else CPU).

Usage:
    math-posttrain merge-lora --base qfft_r1 --adapter outputs/runs/grpo_mmr_dr50/adapter --out outputs/merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

from math_posttrain.common import log, now_iso, write_json
from math_posttrain.models import resolve_model


def merge(base: str | Path, adapter: str | Path, out: str | Path, *, smoke: bool = False) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path = resolve_model(str(base))
    adapter_path = resolve_model(str(adapter))
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    dtype = torch.float32 if smoke else torch.bfloat16
    device = 'cuda' if torch.cuda.is_available() and not smoke else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, device_map=device)
    model = PeftModel.from_pretrained(model, str(adapter_path)).merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.save_pretrained(out_path, safe_serialization=True, max_shard_size='5GB')
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_path)
    write_json(out_path / 'merge_info.json', {
        'merged': now_iso(),
        'base': str(base),
        'adapter': str(adapter),
    })
    log(f'Merged weights saved to {out_path}')
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', required=True, help='Base model alias, HF repo ID, or local path')
    p.add_argument('--adapter', required=True, help='LoRA adapter HF repo ID or local path')
    p.add_argument('--out', required=True, help='Output directory for merged weights')
    p.add_argument('--smoke', action='store_true', help='Run on CPU in fp32')
    a = p.parse_args()
    merge(a.base, a.adapter, a.out, smoke=a.smoke)


if __name__ == '__main__':
    main()
