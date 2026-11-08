"""Extend v4 identities for the v6 SFT runner before checking resume state."""
import json
import tempfile
from pathlib import Path

from src.training.guards_v4 import approved_config as base_config
from src.training.guards_v2 import expand_config, sha, validate_checkpoint


def approved_config(path):
    supplied = expand_config(json.loads(Path(path).read_text(encoding='utf-8')))
    if supplied.get('stage') != 'sft' or supplied.get('vision'):
        raise ValueError('v6 is text SFT only; RL/vision compatibility is unverified')
    resume = supplied.pop('resume_from_checkpoint', None)
    # v4's complete data/E0/tokenizer/teacher guards are retained. Delay its
    # resume comparison until the new implementation identity is attached.
    with tempfile.TemporaryDirectory(prefix='math7b-v6-identity-') as folder:
        temporary = Path(folder) / 'config.json'
        temporary.write_text(json.dumps(supplied, ensure_ascii=False), encoding='utf-8')
        config, rows, identity = base_config(temporary)
    directory = Path(__file__).parent
    identity['runner_sha256'] = sha(directory / 'approved_hf_v6.py')
    identity['guard_v6_sha256'] = sha(__file__)
    identity['sft_encoder_v6_sha256'] = sha(directory / 'sft_encoding_v6.py')
    identity['sft_encoder_original_sha256'] = sha(directory / 'approved_hf.py')
    if resume:
        config['resume_from_checkpoint'] = resume
        validate_checkpoint(resume, identity)
    return config, rows, identity
