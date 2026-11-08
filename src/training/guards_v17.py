"""Retain approved data/E0 checks, bind the user-authorized compact-only recipe."""
import json
from pathlib import Path
import tempfile
from src.training.guards_v4 import approved_config as base_config
from src.training.guards_v2 import sha, validate_checkpoint
from src.training.sft_schedule_v17 import training_schedule


def approved_config(path):
    supplied=json.loads(Path(path).read_text(encoding='utf-8'))
    if supplied.get('stage')!='sft' or supplied.get('vision') or supplied.get('arm')!='compact':
        raise ValueError('v17 is compact-only text SFT')
    resume=supplied.pop('resume_from_checkpoint',None)
    with tempfile.TemporaryDirectory(prefix='math7b-v17-') as directory:
        temporary=Path(directory)/'config.json';temporary.write_text(json.dumps(supplied),encoding='utf-8')
        config,rows,identity=base_config(temporary)
    if len(rows)!=1500:raise ValueError('Preserve the approved paired1500 release')
    identity.update(runner_sha256=sha(Path(__file__).with_name('approved_hf_v17.py')),
                    guard_v17_sha256=sha(__file__),
                    schedule_source_sha256=sha(Path(__file__).with_name('sft_schedule_v17.py')),
                    logger_source_sha256=sha(Path(__file__).with_name('swanlab_online_v11.py')),
                    sft_encoder_v6_sha256=sha(Path(__file__).with_name('sft_encoding_v6.py')),
                    sft_encoder_original_sha256=sha(Path(__file__).with_name('approved_hf.py')),
                    actual_trainer_schedule=training_schedule(config,len(rows)))
    if resume:
        config['resume_from_checkpoint']=resume
        validate_checkpoint(resume,identity)
    return config,rows,identity
