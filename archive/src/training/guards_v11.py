"""Extend v10 identity for the cloud logger without exposing credentials."""
import json
from pathlib import Path
import tempfile
from src.training.guards_v10 import approved_config as base_config
from src.training.guards_v2 import sha, validate_checkpoint


def approved_config(path):
    supplied = json.loads(Path(path).read_text(encoding='utf-8'))
    resume = supplied.pop('resume_from_checkpoint', None)
    with tempfile.TemporaryDirectory(prefix='math7b-v11-') as directory:
        temporary = Path(directory) / 'config.json'
        temporary.write_text(json.dumps(supplied), encoding='utf-8')
        config, rows, identity = base_config(temporary)
    identity['runner_sha256'] = sha(Path(__file__).with_name('approved_hf_v11.py'))
    identity['guard_v11_sha256'] = sha(__file__)
    identity['logger_v11_sha256'] = sha(Path(__file__).with_name('swanlab_online_v11.py'))
    if resume:
        config['resume_from_checkpoint'] = resume
        validate_checkpoint(resume, identity)
    return config, rows, identity
