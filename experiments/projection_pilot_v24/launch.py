"""Prepare the pinned Qwen model and launch v24; preserve installed packages."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.projection_pilot_v21.pilot import read, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',help='Verified pinned local snapshot with projection_model_identity.json')
    p.add_argument('--output-root',default='/content/drive/MyDrive/runs/project1/projection_pilot_v24')
    p.add_argument('--download-only',action='store_true')
    p.add_argument('--time-budget-minutes',type=float,default=90)
    args = p.parse_args()
    c = read(ROOT/'configs/project1_projection_pilot_v24.json')
    for name,expected in read(ROOT/'reports/projection_pilot_20261026_v24/bundle_checksums.json').items():
        if sha(ROOT/name)!=expected:
            raise ValueError('Bundle changed: '+name)
    if not args.download_only:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required for sampling')
        if not importlib.util.find_spec('vllm'):
            raise RuntimeError('vLLM missing; reuse v21 install_missing.sh. No automatic upgrades.')
    if args.model_path:
        model = Path(args.model_path).resolve()
        identity = read(model/'projection_model_identity.json')
        if identity['repo']!=c['model'] or identity['revision']!=c['model_revision']:
            raise ValueError('Existing model identity mismatch')
    else:
        drive = Path('/content/drive/MyDrive')
        if not drive.is_dir():
            raise RuntimeError('Mount Drive first')
        name = 'Qwen2.5-7B-Instruct--'+c['model_revision']
        stored = drive/'models'/name
        # Existing weights in this exact cache are reused, no R1 weights are used.
        from huggingface_hub import snapshot_download
        print('Preparing pinned Qwen snapshot in Drive:',stored,flush=True)
        snapshot_download(repo_id=c['model'],revision=c['model_revision'],local_dir=str(stored),
            allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])
        identity = {'repo':c['model'],'revision':c['model_revision'],
                    'origin':'huggingface_hub fixed-revision snapshot_download'}
        save(stored/'projection_model_identity.json',identity)
        if args.download_only:
            print('Qwen model downloaded; no GPU model loaded or sampling started.',flush=True)
            return
        model = Path('/content/models')/name
        model.mkdir(parents=True,exist_ok=True)
        files = [f for f in stored.iterdir() if f.is_file()]
        required = sum(f.stat().st_size for f in files
                       if not (model/f.name).exists() or (model/f.name).stat().st_size!=f.stat().st_size)
        if shutil.disk_usage(model).free < required+2_000_000_000:
            raise RuntimeError('Insufficient SSD space; keep existing caches and select another model-path')
        for source in files:
            dest = model/source.name
            if dest.exists() and dest.stat().st_size==source.stat().st_size:
                continue
            temporary = dest.with_name(dest.name+'.copying')
            shutil.copyfile(source,temporary)
            temporary.replace(dest)
    index = read(model/'model.safetensors.index.json')
    if any(not (model/name).is_file() for name in set(index['weight_map'].values())):
        raise ValueError('Model shards missing')
    for name,expected in c['tokenizer_sha256'].items():
        if sha(model/name)!=expected:
            raise ValueError('Tokenizer mismatch: '+name)
    if args.download_only:
        print('Existing pinned model validated; no sampling started.',flush=True)
        return
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    subprocess.run([sys.executable,'-u',str(ROOT/'experiments/projection_pilot_v24/pilot.py'),
        '--model-path',str(model),'--output-root',args.output_root,
        '--time-budget-minutes',str(args.time_budget_minutes)],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
