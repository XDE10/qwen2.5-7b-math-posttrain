"""Preserve installed core versions; install only missing training/eval packages."""
import argparse
from importlib.metadata import PackageNotFoundError,version
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval',action='store_true',help='Also require vLLM for evaluation')
    a=p.parse_args()
    constraints=[]
    for name in ['torch','transformers','tokenizers','peft','accelerate','torchao','datasets','huggingface-hub']:
        try:constraints.append(f'{name}=={version(name)}')
        except PackageNotFoundError:pass
    path=Path('/content/aimo-sft-v25-core-constraints.txt')
    path.write_text('\n'.join(constraints)+'\n',encoding='utf-8')
    needed={'transformers':'transformers','peft':'peft','accelerate':'accelerate',
            'datasets':'datasets','huggingface_hub':'huggingface-hub','swanlab':'swanlab'}
    if a.eval:needed['vllm']='vllm'
    missing=[package for module,package in needed.items() if importlib.util.find_spec(module) is None]
    print('Preserved core versions:\n'+'\n'.join(constraints),flush=True)
    if missing:
        env=dict(os.environ);env.pop('PIP_CONSTRAINT',None)
        cmd=[sys.executable,'-m','pip','install','-c',str(path),*missing]
        if 'vllm' in missing:cmd.insert(4,'--only-binary=vllm')
        subprocess.run(cmd,env=env,check=True)
    else:print('Required packages already installed; no installation.',flush=True)
    import torch
    print('torch:',torch.__version__,'CUDA:',torch.cuda.is_available(),flush=True)
    if torch.cuda.is_available():print('GPU:',torch.cuda.get_device_name(0),flush=True)


if __name__=='__main__':main()
