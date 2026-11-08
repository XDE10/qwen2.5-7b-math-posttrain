"""Pure protocol validation and durable trial journals; no model import."""
import json
import math
import os
from pathlib import Path,PurePosixPath
from src.training.frozen_verifier_v4 import digest,sha,grader_identity

BASE='Qwen/Qwen2.5-7B-Instruct'
REV='a09a35458c702b33eeacc393d103063234e8bc28'
SYSTEM=r'Please reason step by step, and put your final answer within \boxed{}.'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    path=Path(path)
    if not path.exists():return []
    data=path.read_bytes()
    if data and not data.endswith(b'\n'):raise ValueError('Incomplete journal tail; preserve it and repair explicitly before resume')
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def append(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def trial_seed(seed,sid,trial):return int(digest([seed,sid,trial])[:8],16)&0x7fffffff


def indexed(records):
    result={}
    for row in records:
        key=(row['sample_id'],row['trial'])
        if key in result:raise ValueError('Duplicate trial record')
        result[key]=row
    return result


def selection(records,num_samples):
    if num_samples not in (4,8):raise ValueError('Only separate 4/8-sample protocols supported')
    by=indexed(records)
    if len(records)!=num_samples or {r['trial'] for r in records}!=set(range(num_samples)):
        return {'eligible':False,'reason':'incomplete_trials'}
    if len({r['sample_id'] for r in records})!=1:raise ValueError('Mixed sample IDs')
    if any(r['category'] not in ('equivalent','mismatch') for r in records):
        return {'eligible':False,'reason':'unscorable_trial'}
    if any(r['finish_reason']!='stop' for r in records):return {'eligible':False,'reason':'truncated_or_aborted'}
    correct=sum(r['category']=='equivalent' for r in records)
    return {'eligible':num_samples//4<=correct<=3*num_samples//4,'correct':correct,'num_samples':num_samples,
            'empirical_rate':correct/num_samples,'reason':'within_range' if num_samples//4<=correct<=3*num_samples//4 else 'outside_range'}


def load_prepared(candidates,seal_path):
    seal=read(seal_path)
    if sha(candidates)!=seal['final_candidate_sha256']:raise ValueError('Prepared candidate digest changed')
    if digest(grader_identity())!=seal['grader_identity_sha256']:raise ValueError('Frozen scorer changed')
    rows=jsonl(candidates)
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Empty/duplicate candidates')
    if seal['candidate_count']!=len(rows):raise ValueError('Prepared candidate count changed')
    if any(r['training_approved'] is not False or r['split']!='train' for r in rows):raise ValueError('Candidate approval/split changed')
    return rows,seal


def validate_model(path,identity_path):
    identity=read(identity_path);root=Path(path).resolve()
    if identity.get('repo')!=BASE or identity.get('revision')!=REV or identity.get('verified_against_hf_lfs') is not True:
        raise ValueError('Require verified official pinned E0 snapshot; no mock/SFT origin')
    if (root/'merged_sft_identity.json').exists():raise ValueError('Selection must start from E0, not merged SFT')
    if not identity.get('files') or not any(n.endswith('.safetensors') for n in identity['files']):raise ValueError('No pinned E0 weights')
    for name,h in identity['files'].items():
        logical=PurePosixPath(name)
        if logical.is_absolute() or '..' in logical.parts or '\\' in name:raise ValueError('E0 filename escapes snapshot')
        p=root.joinpath(*logical.parts)
        if sha(p)!=h:raise ValueError('E0 file identity changed')
    return identity


class Metrics:
    def __init__(self,root,mode='offline',name='rl-screening',config=None):
        self.path=Path(root)/'metrics.jsonl';self.swan=None;self.closed=False
        if mode!='disabled':
            if mode=='online' and not os.environ.get('SWANLAB_API_KEY'):raise ValueError('Set SWANLAB_API_KEY securely before online logging')
            import swanlab
            self.swan=swanlab
            swanlab.init(project='project1-7b-math',name=name,mode=mode,log_dir=str(Path(root)/'swanlog'),config=config or {},reinit=True)

    def log(self,values,step):
        safe={k:float(v) for k,v in values.items() if isinstance(v,(float,int)) and math.isfinite(float(v))}
        append(self.path,{'step':step,'metrics':safe})
        if self.swan:self.swan.log(safe,step=step)

    def finish(self):
        if not self.closed and self.swan:self.swan.finish()
        self.closed=True
