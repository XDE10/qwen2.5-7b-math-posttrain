"""Identity/checkpoint guards for the single-device project-one v2 runner."""
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from src.data.approved_training_inputs_v2 import load_approved

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ('torch', 'transformers', 'trl', 'peft', 'datasets', 'accelerate', 'tokenizers')
GRADER_FILES = ('scripts/math_verify_v3_strict_worker.py', 'scripts/math_verify_pilot_worker.py', 'src/training/math_reward_v2.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def expand_config(value):
    if isinstance(value, dict):
        return {k: expand_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_config(v) for v in value]
    if isinstance(value, str):
        result = os.path.expandvars(value)
        if '${' in result:
            raise ValueError('Set required configuration environment variables')
        return result
    return value


def validate_checkpoint(path, identity):
    path = Path(path)
    required = ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'run_identity.json')
    missing = [name for name in required if not (path / name).is_file() or not (path / name).stat().st_size]
    if missing:
        raise ValueError('Incomplete checkpoint: ' + ', '.join(missing))
    if read(path / 'run_identity.json') != identity:
        raise ValueError('Checkpoint identity mismatch; reinstall pinned versions, never edit identity')
    if read(path / 'trainer_state.json').get('global_step', 0) <= 0:
        raise ValueError('Checkpoint has no positive global_step')


def validate_merged(config):
    base = Path(config['model']).resolve()
    if sha(base / 'weight_manifest.json') != config['expected_sft']['weight_manifest_sha256']:
        raise ValueError('Merged weight manifest identity changed')
    origin = read(base / 'merged_sft_identity.json')
    checks = read(base / 'weight_manifest.json')
    if not checks or not any(name.endswith('.safetensors') or name.endswith('.bin') for name in checks):
        raise ValueError('Merged weight manifest contains no weights')
    for name, digest in checks.items():
        path = (base / name).resolve()
        if not path.is_relative_to(base) or not path.is_file() or sha(path) != digest:
            raise ValueError('Merged weight digest/path changed')
    expected = config['expected_sft']
    if expected['arm'] != config['arm']:
        raise ValueError('Declared RL arm differs from expected SFT arm')
    oc = origin['config']
    if oc.get('stage') != 'sft' or oc.get('arm') != expected['arm']:
        raise ValueError('Merged SFT arm mismatch')
    if origin.get('data_sha256') != expected['data_sha256'] or oc.get('output_dir') != expected['output_dir']:
        raise ValueError('Merged SFT origin mismatch')
    if origin.get('schema') != 'project1-run-v2':
        raise ValueError('A v2 RL run requires a v2 SFT origin; legacy remains preserved')
    if sha(base / 'merged_sft_identity.json') != expected['identity_sha256']:
        raise ValueError('Pin exact merged SFT identity after SFT export')
    return sha(base / 'weight_manifest.json')


def rl_lengths(config):
    prompt = config['max_prompt_length']
    completion = config['max_new_tokens']
    if prompt <= 0 or completion <= 0 or config['max_length'] != prompt + completion:
        raise ValueError('RL max_length must equal max_prompt_length + max_new_tokens')
    return prompt, completion


def approved_config(path):
    c = expand_config(read(path))
    if c.get('draft', False):
        raise ValueError('Draft, unfrozen configuration cannot train')
    rows = load_approved(c['data_manifest'], c['data_file'])
    if not c.get('model_revision') or c['model_revision'] == 'main':
        raise ValueError('Pin model revision')
    if c.get('vision'):
        raise ValueError('v2 runner is project-one text only')
    if c['arm'] not in ('raw', 'compact'):
        raise ValueError('Explicit raw/compact arm required')
    if c['stage'] == 'sft' and c['data_file'] != 'project1_' + c['arm'] + '_paired.jsonl':
        raise ValueError('SFT data file differs from declared arm')
    if c['stage'] == 'rl':
        rl_lengths(c)
    merged = validate_merged(c) if c['stage'] == 'rl' and c['initial_state'] == 'sft' else None
    tokenizer = Path(c['tokenizer_path'])
    tc = read(tokenizer / 'tokenizer_config.json')
    identity = {
        'schema': 'project1-run-v2',
        'config': {k: v for k, v in c.items() if k != 'resume_from_checkpoint'},
        'versions': {name: version(name) for name in VERSIONS},
        'manifest_sha256': sha(c['data_manifest']),
        'data_sha256': sha(Path(c['data_manifest']).parent / c['data_file']),
        'tokenizer_sha256': {p.name: sha(p) for p in sorted(tokenizer.iterdir()) if p.is_file()},
        'template_sha256': hashlib.sha256(json.dumps(tc['chat_template'], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        'grader_sha256': {name: sha(ROOT / name) for name in GRADER_FILES},
        'bundled_grader_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / '.pilot_deps').rglob('*.py'))},
        'merged_weight_manifest_sha256': merged,
        'runner_sha256': sha(ROOT / 'src/training/approved_hf_v2.py'),
    }
    if c.get('resume_from_checkpoint'):
        validate_checkpoint(c['resume_from_checkpoint'], identity)
    return c, rows, identity
