"""Explicitly back up and remove only a torn final trial-journal fragment."""
import argparse
import json
import re
from pathlib import Path
from src.training.frozen_verifier_v4 import sha
from src.training.rollout_pipeline_v4 import indexed


def repair(path,output_root):
    path=Path(path).resolve();out=Path(output_root).resolve();out.mkdir(parents=True,exist_ok=True)
    if path.parent.name not in ('raw','graded') or not re.fullmatch(r'shard-\d{5}\.jsonl',path.name):raise ValueError('Only sampling trial journals may be repaired')
    if (path.parent.parent/'sampling_complete.json').exists():raise ValueError('Completed evidence must never be repaired in place')
    data=path.read_bytes()
    if not data or data.endswith(b'\n'):return {'changed':False}
    end=data.rfind(b'\n')+1;valid=data[:end]
    indexed([json.loads(line) for line in valid.splitlines() if line.strip()])
    old=sha(path);backup=out/(path.parent.name+'_'+path.name+'.'+old+'.backup')
    with backup.open('xb') as f:f.write(data)
    # Only remove a fragment; even a complete JSON record without its commit
    # newline remains uncommitted, preserved in the backup for inspection.
    path.write_bytes(valid)
    return {'changed':True,'backup':str(backup),'previous_sha256':old,'new_sha256':sha(path),'removed_tail_bytes':len(data)-end}


def main():
    p=argparse.ArgumentParser();p.add_argument('--journal',required=True);p.add_argument('--output-root',required=True);a=p.parse_args()
    from filelock import FileLock
    root=Path(a.journal).resolve().parent.parent
    with FileLock(str(root/'.sampling.lock'),timeout=0):print(json.dumps(repair(a.journal,a.output_root)))


if __name__=='__main__':main()
