"""Resolve Drive variables and pin a completed SFT export into a new config."""
import argparse
import json
from pathlib import Path
from src.training.guards_v2 import expand_config, read, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--template',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--resume')
    a=p.parse_args()
    c=expand_config(read(a.template))
    if c.get('draft'):
        raise ValueError('Do not materialize formal drafts as approved runs')
    if c['stage']=='rl' and c['initial_state']=='sft':
        c['expected_sft']['identity_sha256']=sha(Path(c['model'])/'merged_sft_identity.json')
        c['expected_sft']['weight_manifest_sha256']=sha(Path(c['model'])/'weight_manifest.json')
    if a.resume:c['resume_from_checkpoint']=a.resume
    with Path(a.output).open('x',encoding='utf-8') as f:
        json.dump(c,f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
