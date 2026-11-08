"""Version4 guard: new rollout approval evidence, unchanged v2 train/dev checks."""
import json
import tempfile
from pathlib import Path
from src.training.guards_v2 import approved_config as base_config,sha,expand_config,validate_checkpoint
from src.training.frozen_verifier_v4 import digest,grader_identity
from src.training.rollout_pipeline_v4 import validate_model


def approved_config(path):
    supplied=expand_config(json.loads(Path(path).read_text(encoding='utf-8')))
    resume=supplied.pop('resume_from_checkpoint',None)
    with tempfile.TemporaryDirectory(prefix='math7b-v4-identity-') as directory:
        temporary=Path(directory)/'config.json'
        temporary.write_text(json.dumps(supplied,ensure_ascii=False),encoding='utf-8')
        c,rows,identity=base_config(temporary)
    if c['stage']=='rl':
        manifest_path=Path(c['data_manifest']).resolve()
        m=json.loads(manifest_path.read_text(encoding='utf-8'))
        if m.get('schema')!='project1-rollout-approved-v4':raise ValueError('v4 formal RL requires new conditional rollout approval')
        if m['grader_identity_sha256']!=digest(grader_identity()):raise ValueError('Approved scorer changed')
        for name,h in m['files'].items():
            p=(manifest_path.parent/name).resolve()
            if not p.is_relative_to(manifest_path.parent) or sha(p)!=h:raise ValueError('Approved evidence/config changed')
        if c['initial_state']=='instruct':validate_model(c['model'],manifest_path.parent/'e0_model_identity.json')
        if c['initial_state']=='sft':
            origin=json.loads((Path(c['model'])/'merged_sft_identity.json').read_text(encoding='utf-8'))
            for key,value in c['expected_sft_protocol'].items():
                if origin['config'].get(key)!=value:raise ValueError('E3/E4 requires frozen formal E1, not smoke')
    else:
        if not c.get('e0_model_identity_path'):raise ValueError('v4 SFT requires a verified local E0 path/identity; no implicit downloads')
        validate_model(c['model'],c['e0_model_identity_path'])
    identity['runner_sha256']=sha(Path(__file__).with_name('approved_hf_v4.py'))
    identity['guard_v4_sha256']=sha(__file__)
    identity['grader_v4_identity']=grader_identity()
    if resume:
        c['resume_from_checkpoint']=resume
        validate_checkpoint(resume,identity)
    return c,rows,identity
