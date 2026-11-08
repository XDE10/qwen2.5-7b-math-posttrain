"""Launch only compact from original E0; automatic SwanLab credentials."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from scripts.run_e1_autoauth_v12 import authenticated_environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/project1_compact_cosine_v17.json')
    p.add_argument('--output-root',default='/content/drive/MyDrive/runs/project1/compact_cosine_v17')
    p.add_argument('--e0-model-path',default='/content/models/Qwen2.5-7B-Instruct--a09a35458c702b33eeacc393d103063234e8bc28')
    p.add_argument('--e0-model-identity',default='/content/project1-env-evidence/e0-identity/e0_model_identity.json')
    p.add_argument('--resume-from-checkpoint')
    a=p.parse_args();os.chdir(Path(__file__).resolve().parents[1])
    env=authenticated_environment(os.environ)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    args=[sys.executable,'-u','-m','scripts.train_project1_sft_v17','--config',a.config,
          '--output-root',a.output_root,'--e0-model-path',a.e0_model_path,
          '--e0-model-identity',a.e0_model_identity,'--swanlab-mode','online']
    if a.resume_from_checkpoint:args+=['--resume-from-checkpoint',a.resume_from_checkpoint]
    print('Starting compact-only SFT from original E0: cosine + 3% warmup; SwanLab online.',flush=True)
    output=Path(a.output_root);output.mkdir(parents=True,exist_ok=True)
    with (output/'train.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=process.wait()
        if code:raise subprocess.CalledProcessError(code,args)


if __name__=='__main__':main()
