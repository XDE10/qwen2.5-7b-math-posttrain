"""Model and tokenizer resolution via local paths, training runs, registry aliases, or Hugging Face repo IDs."""
from __future__ import annotations

from pathlib import Path

from math_posttrain.common import load_config, log, read_json, workspace

WEIGHT_PATTERNS = ['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.tiktoken', 'merges.txt', 'vocab.json']
TOKENIZER_PATTERNS = [
    'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
    'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja', 'config.json',
]


def registry() -> dict:
    return load_config('eval.json')['models']


def resolve_model(spec: str) -> Path:
    """Resolve a model spec to a local directory, downloading once via Hugging Face Hub cache when needed.

    Accepted formats:
    - Existing local directory path (e.g. `./outputs/runs/qfft_r1_16k_b8/final`)
    - `run:<run_name>` or `run:<run_name>/<subdir>` under `workspace().runs`
    - Registry alias from `configs/eval.json` (e.g. `base`, `qfft_r1`, `qfft_official`)
    - Any Hugging Face repository ID (`owner/repo`)
    """
    local = Path(spec)
    if local.exists():
        return local.resolve()

    if spec.startswith('run:'):
        rel = spec[4:]
        target = workspace().runs / (rel if '/' in rel else f'{rel}/final')
        if not target.is_dir():
            raise FileNotFoundError(f'No trained weights found for {spec!r} at {target}')
        return target.resolve()

    models = registry()
    if spec in models:
        entry = models[spec]
        repo, rev = entry['repo'], entry.get('revision')
        log(f'Resolving model {spec} -> {repo}@{rev[:12] if rev else "main"}')
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=repo, revision=rev, allow_patterns=WEIGHT_PATTERNS))

    if '/' in spec:
        log(f'Resolving Hugging Face model {spec}')
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=spec, allow_patterns=WEIGHT_PATTERNS))

    raise ValueError(
        f'Unknown model spec {spec!r}. Pass a local directory, Hugging Face repo ID (owner/repo), '
        f'`run:<name>`, or one of {sorted(models)}.'
    )


def resolve_tokenizer(spec: str = 'base') -> Path:
    """Resolve only tokenizer files (~11 MB for Qwen2.5) without downloading full 14 GB weights."""
    local = Path(spec)
    if local.exists():
        return local.resolve()
    if spec.startswith('run:'):
        return resolve_model(spec)
    models = registry()
    if spec in models:
        entry = models[spec]
        repo, rev = entry['repo'], entry.get('revision')
    elif '/' in spec:
        repo, rev = spec, None
    else:
        raise ValueError(f'Unknown tokenizer spec {spec!r}')
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo, revision=rev, allow_patterns=TOKENIZER_PATTERNS))


def max_model_len(spec: str, model_path: Path | None = None) -> int:
    models = registry()
    if spec in models and 'max_model_len' in models[spec]:
        return int(models[spec]['max_model_len'])
    if model_path is not None:
        cfg_file = Path(model_path) / 'config.json'
        if cfg_file.is_file():
            return int(read_json(cfg_file).get('max_position_embeddings', 32768))
    return 32768
