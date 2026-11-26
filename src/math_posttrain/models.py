"""Model registry and Drive -> local SSD preparation (download once from Hugging Face, reuse forever).

Model specs (configs/pipeline.json -> "models") are pinned by revision. A spec string can also be
`run:<run_name>` (final SFT/QFFT weights) or `run:<run_name>/merged` (GRPO, merged into its base).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from math_posttrain import drive_io
from math_posttrain.common import (TOKENIZER_SHA256, drive_layout, human_bytes, load_config, local_root, log,
                                         now_iso, read_json, sha256_file, tree_bytes, write_json)

WEIGHT_PATTERNS = ['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.tiktoken', 'merges.txt', 'vocab.json']
MARK = 'p1_model_identity.json'


def registry() -> dict:
    return load_config('pipeline.json')['models']


def _complete(folder: Path) -> bool:
    if not folder.exists():
        return False
    index = folder / 'model.safetensors.index.json'
    if index.exists():
        shards = set(read_json(index)['weight_map'].values())
        return all((folder / s).is_file() and (folder / s).stat().st_size > 0 for s in shards)
    return (folder / 'model.safetensors').is_file()


def drive_path(spec: str, layout=None) -> Path:
    layout = layout or drive_layout()
    if spec.startswith('run:'):
        run = spec[4:]
        return layout.runs / (run if '/' in run else run + '/final')
    return layout.models / registry()[spec]['drive_dir']


def max_model_len(spec: str) -> int:
    if spec.startswith('run:'):
        return 32768
    return int(registry()[spec].get('max_model_len', 32768))


def download_to_drive(spec: str, layout=None) -> Path:
    """snapshot_download to local SSD, then publish the folder on Drive (drive_io: rclone CLI or throttled copy)."""
    layout = layout or drive_layout()
    entry = registry()[spec]
    target = drive_path(spec, layout)
    if _complete(target):
        log(f'{spec}: already on Drive at {target}')
        return target
    from huggingface_hub import snapshot_download
    staging = local_root() / 'downloads' / entry['drive_dir']
    log(f"{spec}: downloading {entry['repo']}@{entry['revision'][:12]} ...")
    snapshot_download(repo_id=entry['repo'], revision=entry['revision'], local_dir=str(staging),
                      allow_patterns=WEIGHT_PATTERNS)
    shutil.rmtree(staging / '.cache', ignore_errors=True)  # huggingface_hub bookkeeping, not model files
    size = tree_bytes(staging)
    write_json(staging / MARK, {'repo': entry['repo'], 'revision': entry['revision'], 'downloaded': now_iso(),
                                'bytes': size})
    drive_io.upload_tree(staging, target)
    shutil.rmtree(staging, ignore_errors=True)
    log(f'{spec}: stored on Drive ({human_bytes(size)})')
    return target


def local_cache_path(spec: str, layout=None) -> Path:
    layout = layout or drive_layout()
    src = drive_path(spec, layout)
    return local_root() / 'models' / src.as_posix().replace(layout.root.as_posix(), '').strip('/').replace('/', '__')


def ensure_local(spec: str, layout=None) -> Path:
    """Copy a model from Drive to local SSD (skipping files that already match in size)."""
    layout = layout or drive_layout()
    src = drive_path(spec, layout)
    if not _complete(src):
        if spec.startswith('run:'):
            raise FileNotFoundError(f'{spec}: no complete weights at {src}')
        download_to_drive(spec, layout)
    dst = local_cache_path(spec, layout)
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    if drive_io.rclone_remote():  # multi-threaded download, skips files that already match in size
        drive_io.download_tree(src, dst, atomic=False)
    for path in sorted(p for p in src.iterdir() if p.is_file()):
        out = dst / path.name
        if out.exists() and out.stat().st_size == path.stat().st_size:
            continue
        shutil.copyfile(path, out.with_name(out.name + '.copying'))
        out.with_name(out.name + '.copying').replace(out)
        copied += path.stat().st_size
    if not _complete(dst):
        raise RuntimeError(f'{spec}: local copy incomplete at {dst}')
    if copied:
        log(f'{spec}: copied {human_bytes(copied)} from Drive to {dst}')
    return dst


TOKENIZER_FILES = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json',
                   'added_tokens.json', 'chat_template.jinja', 'generation_config.json', 'config.json']


def ensure_tokenizer(spec: str = 'base', layout=None) -> Path:
    """Local copy of just the tokenizer files (prompt ids for every evaluated model come from the base)."""
    layout = layout or drive_layout()
    src = drive_path(spec, layout)
    if not (src / 'tokenizer.json').exists():
        download_to_drive(spec, layout)
    dst = local_root() / 'tokenizers' / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        if (src / name).exists() and not (dst / name).exists():
            shutil.copyfile(src / name, dst / name)
    if spec == 'base':
        verify_base_tokenizer(dst)
    return dst


def verify_base_tokenizer(model_dir: Path) -> None:
    for name, expected in TOKENIZER_SHA256.items():
        if sha256_file(Path(model_dir) / name) != expected:
            raise RuntimeError(f'{name} differs from Qwen2.5-7B-Instruct@a09a354')
