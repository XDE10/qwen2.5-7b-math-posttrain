"""Merge a LoRA adapter into its base and export bf16 safetensors for vLLM (GPU if present, else CPU).

    python -m experiments.qfft_v93.merge_lora --base <dir> --adapter <dir> --out <drive dir>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from experiments.qfft_v93 import drive_io
from experiments.qfft_v93.common import local_root, log, now_iso, read_json, sha256_file, write_json


def merge(base: Path, adapter: Path, out: Path, smoke: bool = False, *, provenance=None) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float32 if smoke else torch.bfloat16
    device = 'cuda' if torch.cuda.is_available() and not smoke else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map=device)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload(safe_merge=True)
    staging = local_root() / 'merge_staging' / Path(out).name
    shutil.rmtree(staging, ignore_errors=True)
    model.save_pretrained(staging, safe_serialization=True, max_shard_size='4GB')
    AutoTokenizer.from_pretrained(base).save_pretrained(staging)
    base_identity = Path(base) / 'p1_model_identity.json'
    write_json(staging / 'p1_model_identity.json', {
        'merged': now_iso(), 'base': str(base), 'adapter': str(adapter),
        'base_identity': read_json(base_identity) if base_identity.exists() else None})
    if provenance is not None:
        write_json(staging / 'p1_export_manifest.json', {
            'provenance': provenance, 'files': {
                p.name: {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                for p in sorted(staging.iterdir()) if p.is_file()}})
    drive_io.upload_tree(staging, out, watch_path=str(local_root()), verify_full=provenance is not None)
    shutil.rmtree(staging, ignore_errors=True)
    log(f'Merged weights -> {out}')
    return Path(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    merge(Path(a.base), Path(a.adapter), Path(a.out), smoke=a.smoke)


if __name__ == '__main__':
    main()
