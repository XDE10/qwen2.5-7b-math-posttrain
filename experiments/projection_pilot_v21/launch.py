"""Colab launcher: reuse/download a fixed snapshot, cache in Drive, run pilot."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.projection_pilot_v21.pilot import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',help='Existing complete local pinned snapshot; skips downloading')
    p.add_argument('--output-root')
    p.add_argument('--time-budget-minutes',type=float,default=90)
    p.add_argument('--download-only',action='store_true',help='Download to Drive on a CPU runtime, then exit without GPU or vLLM')
    a=p.parse_args()
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    c=read(ROOT/'configs/project1_projection_pilot_v21.json')
    seal=read(ROOT/'reports/projection_pilot_20261026_v21/bundle_checksums.json')
    for name,expected in seal.items():
        if sha(ROOT/name)!=expected:raise ValueError(f'Bundle file changed: {name}')
    if not a.download_only:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU: connect Colab first')
        if not importlib.util.find_spec('vllm'):
            raise RuntimeError('vLLM is missing; run bash experiments/projection_pilot_v21/install_missing.sh once')
        print('GPU:',torch.cuda.get_device_name(0),flush=True)
    drive=Path('/content/drive/MyDrive')
    if not drive.is_dir() and (not a.output_root or not a.model_path):
        raise RuntimeError('Mount Drive in notebook, or explicitly supply both --output-root and --model-path')
    output=Path(a.output_root) if a.output_root else drive/'runs/project1/projection_pilot_v21'
    if a.model_path:
        model=Path(a.model_path).resolve()
    else:
        from huggingface_hub import snapshot_download
        name='DeepSeek-R1-Distill-Qwen-7B--'+c['model_revision']
        stored=drive/'models'/name
        print('Preparing fixed model in Drive; existing downloaded files are reused:',stored,flush=True)
        snapshot_download(repo_id=c['model'],revision=c['model_revision'],local_dir=str(stored),
            allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])
        if a.download_only:
            print('模型已完整下载到Drive；本次没有加载GPU或启动采样：'+str(stored),flush=True)
            return
        model=Path('/content/models')/name
        model.mkdir(parents=True,exist_ok=True)
        files=[f for f in stored.iterdir() if f.is_file()]
        required=sum(f.stat().st_size for f in files if not (model/f.name).exists())
        if shutil.disk_usage(model).free < required+2_000_000_000:
            raise RuntimeError('Not enough local SSD space for model copy')
        from tqdm import tqdm
        for source in tqdm(files,desc='Copy model to local SSD'):
            dest=model/source.name
            if dest.exists() and dest.stat().st_size==source.stat().st_size:continue
            temporary=dest.with_name(dest.name+'.copying')
            shutil.copyfile(source,temporary);temporary.replace(dest)
    # Existing snapshots must contain actual weights; a tokenizer-only cache is insufficient.
    weights=list(model.glob('*.safetensors'))
    if not weights:raise ValueError('No model weights found')
    index=model/'model.safetensors.index.json'
    if index.exists():
        missing=set(read(index)['weight_map'].values())-{f.name for f in weights}
        if missing:raise ValueError('Missing model shards: '+str(missing))
    marker=model/'projection_model_identity.json'
    if marker.exists():
        ident=read(marker)
        if ident['revision']!=c['model_revision']:raise ValueError('Existing model identity mismatch')
    else:
        if a.model_path:
            raise ValueError('Existing model needs projection_model_identity.json with verified fixed Hub revision; omit --model-path to prepare it automatically')
        save(marker,{'repo':c['model'],'revision':c['model_revision'],
            'origin':'huggingface_hub fixed-revision snapshot_download; copied unchanged to local SSD',
            'weights':{f.name:f.stat().st_size for f in weights}})
    output.mkdir(parents=True,exist_ok=True)
    print('Starting inference-only pilot. Reuse this same output-root to resume:',output,flush=True)
    subprocess.run([sys.executable,'-u',str(ROOT/'experiments/projection_pilot_v21/pilot.py'),
        '--model-path',str(model),'--output-root',str(output),
        '--time-budget-minutes',str(a.time_budget_minutes)],cwd=ROOT,check=True)


if __name__=='__main__':main()
