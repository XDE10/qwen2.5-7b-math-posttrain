"""Bind explicit epoch scheduling and new runner before resume comparison."""
import json
from pathlib import Path
import tempfile

from src.training.guards_v6 import approved_config as base_config
from src.training.guards_v2 import sha, validate_checkpoint
from src.training.sft_epoch_v10 import training_schedule


def approved_config(path):
    supplied = json.loads(Path(path).read_text(encoding='utf-8'))
    schedule = training_schedule(supplied)
    resume = supplied.pop('resume_from_checkpoint', None)
    with tempfile.TemporaryDirectory(prefix='math7b-v10-') as folder:
        temporary = Path(folder) / 'config.json'
        temporary.write_text(json.dumps(supplied), encoding='utf-8')
        config, rows, identity = base_config(temporary)
    if len(rows) != 1500:
        raise ValueError('Formal paired exposure requires 1500 rows')
    identity['runner_sha256'] = sha(Path(__file__).with_name('approved_hf_v10.py'))
    identity['guard_v10_sha256'] = sha(__file__)
    identity['epoch_schedule_source_sha256'] = sha(Path(__file__).with_name('sft_epoch_v10.py'))
    identity['actual_trainer_schedule'] = schedule
    if resume:
        config['resume_from_checkpoint'] = resume
        validate_checkpoint(resume, identity)
    return config, rows, identity
