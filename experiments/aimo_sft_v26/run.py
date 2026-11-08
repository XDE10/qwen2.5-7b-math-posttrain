"""Recover model from Drive and launch direct SFT or resumable evaluation."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.aimo_sft_v26.common import config,read,save,sha


def prepare_model(c):
    drive=Path('/content/drive/MyDrive')
    if not drive.is_dir():raise RuntimeError('Mount Drive in notebook first')
    name='Qwen2.5-7B-Instruct--'+c['model_revision']
    stored=drive/'models'/name;local=Path('/content/models')/name
    index=stored/'model.safetensors.index.json'
    marker=stored/'projection_model_identity.json'
    complete=index.exists() and all((stored/f).is_file() for f in set(read(index)['weight_map'].values()))
    if not complete or not marker.exists():
        from huggingface_hub import snapshot_download
        print('Preparing pinned model; completed files reused:',stored,flush=True)
        snapshot_download(repo_id=c['model'],revision=c['model_revision'],local_dir=str(stored),
            allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])
    if marker.exists():
        identity=read(marker)
        if identity['repo']!=c['model'] or identity['revision']!=c['model_revision']:
            raise ValueError('Existing Drive model identity differs')
    else:
        save(marker,{'repo':c['model'],'revision':c['model_revision'],
                     'origin':'huggingface_hub fixed revision snapshot_download'})
    local.mkdir(parents=True,exist_ok=True)
    files=[f for f in stored.iterdir() if f.is_file()]
    needed=sum(f.stat().st_size for f in files if not (local/f.name).exists() or (local/f.name).stat().st_size!=f.stat().st_size)
    if shutil.disk_usage(local).free < needed+2_000_000_000:raise RuntimeError('Insufficient SSD space')
    print('Restoring Qwen2.5-7B-Instruct model to local SSD...',flush=True)
    for source in files:
        dest=local/source.name
        if dest.exists() and dest.stat().st_size==source.stat().st_size:continue
        temporary=dest.with_name(dest.name+'.copying');shutil.copyfile(source,temporary);temporary.replace(dest)
    for name,expected in c['tokenizer_sha256'].items():
        if sha(local/name)!=expected:raise ValueError('Tokenizer differs: '+name)
    return local


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=['train','eval'])
    p.add_argument('--variant',choices=['base','sft'],default='base')
    p.add_argument('--benchmark',nargs='+',choices=['dev','aime2024','aime2025','math500'],default=['aime2024','aime2025'])
    p.add_argument('--time-budget-minutes',type=float,default=90)
    p.add_argument('--checkpoint',help='Evaluate an epoch checkpoint directory name, e.g. checkpoint-150')
    p.add_argument('--batch-size',type=int,choices=[1,2,4],help='Training physical batch; default 2, effective batch 16')
    p.add_argument('--output-root',default='/content/drive/MyDrive/runs/project1/aimo_sft_v26')
    a=p.parse_args();c=config()
    if a.checkpoint and (a.task!='eval' or a.variant!='sft' or Path(a.checkpoint).name!=a.checkpoint):
        p.error('--checkpoint requires eval --variant sft and a checkpoint directory name')
    for name,expected in read(ROOT/'reports/aimo_sft_20261026_v26/bundle_checksums.json').items():
        if sha(ROOT/name)!=expected:raise ValueError('Bundle changed: '+name)
    out=Path(a.output_root)
    if a.task=='train' and (out/'TRAIN_COMPLETE.json').exists():
        print('Training already complete; no model restored or loaded.',flush=True);return
    import fcntl
    shared=Path('/content/drive/MyDrive/runs/project1/projection_pilot_v21/running.lock')
    shared.parent.mkdir(parents=True,exist_ok=True)
    lock=shared.open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another project GPU task is running')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Connect A100 GPU first')
    if a.task=='train' or a.variant=='base':
        model=prepare_model(c)
    else:
        if a.checkpoint:
            checkpoint=out/a.checkpoint
            if not (checkpoint/'adapter_config.json').exists():raise ValueError('Checkpoint adapter missing')
            base=prepare_model(c)
            stored=Path('/content/models')/('v26-'+a.checkpoint+'--'+sha(checkpoint/'run_identity.json')[:12])
            subprocess.run([sys.executable,str(ROOT/'experiments/aimo_sft_v26/export_checkpoint.py'),
                '--base',str(base),'--checkpoint',str(checkpoint),'--output',str(stored)],check=True)
        else:
            if not (out/'TRAIN_COMPLETE.json').exists():raise RuntimeError('Training/export has not completed')
            stored=out/'merged'
        model=Path('/content/models')/('aimo_sft_v26_merged--'+sha(stored/'sft_identity.json')[:12]+('-'+a.checkpoint if a.checkpoint else ''))
        if a.checkpoint:model=stored
        index=read(stored/'model.safetensors.index.json')
        files=[f for f in stored.iterdir() if f.is_file()]
        model.mkdir(parents=True,exist_ok=True)
        needed=sum(f.stat().st_size for f in files if not (model/f.name).exists() or (model/f.name).stat().st_size!=f.stat().st_size)
        if shutil.disk_usage(model).free<needed+2_000_000_000:raise RuntimeError('Insufficient SSD space for merged evaluation model')
        for source in files:
            dest=model/source.name
            if dest.exists() and dest.stat().st_size==source.stat().st_size:continue
            temporary=dest.with_name(dest.name+'.copying');shutil.copyfile(source,temporary);temporary.replace(dest)
        if any(not (model/f).exists() for f in set(index['weight_map'].values())):raise ValueError('Merged shards missing')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    if a.task=='train':
        cmd=[sys.executable,'-u',str(ROOT/'experiments/aimo_sft_v26/train.py'),'--model-path',str(model),'--output-root',str(out)]
        if a.batch_size:cmd+=['--batch-size',str(a.batch_size)]
    else:
        cmd=[sys.executable,'-u',str(ROOT/'experiments/aimo_sft_v26/evaluate.py'),'--model-path',str(model),
             '--output-root',str(out/'evaluations'/(a.checkpoint or a.variant)),'--variant',a.variant,
             '--benchmark',*a.benchmark,'--time-budget-minutes',str(a.time_budget_minutes)]
    subprocess.run(cmd,cwd=ROOT,check=True)


if __name__=='__main__':main()
