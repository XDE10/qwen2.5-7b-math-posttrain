"""Sync completed offline SFT, then evaluate both arms online without retraining."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.run_e1_autoauth_v12 import authenticated_environment


def sync_logs(training, environment):
    receipt=training/'swanlab_sync_receipt_v13.json'
    done=json.loads(receipt.read_text()) if receipt.exists() else {}
    command=shutil.which('swanlab')
    if not command:raise RuntimeError('SwanLab executable missing')
    for arm in ('raw','compact'):
        runs=sorted((training/f'E1_{arm}_equal_exposure_v10'/'swanlog').glob('run-*'))
        if not runs:raise ValueError('No offline training log for '+arm)
        for path in runs:
            if done.get(str(path)):continue
            args=[command,'sync',str(path),'--project','project1-7b-math']
            if environment.get('SWANLAB_WORKSPACE'):args+=['--workspace',environment['SWANLAB_WORKSPACE']]
            print('Sync completed SFT log: '+str(path),flush=True)
            subprocess.run(args,env=environment,check=True)
            done[str(path)]=True
            receipt.write_text(json.dumps(done,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs-root',default='/content/drive/MyDrive/runs/project1')
    parser.add_argument('--output-root')
    parser.add_argument('--cache-root',default='/content/models')
    parser.add_argument('--baseline-root')
    parser.add_argument('--sync-only',action='store_true')
    a=parser.parse_args()
    project=Path(__file__).resolve().parents[1]; os.chdir(project)
    env=authenticated_environment(os.environ)
    runs=Path(a.runs_root); training=runs/'E1_equal_exposure_v10'
    sync_logs(training,env)
    if a.sync_only:return
    if a.baseline_root:baseline=Path(a.baseline_root)
    else:
        candidates=sorted(p for p in runs.glob('E0_dev_v9_*') if (p/'summary.json').is_file() and (p/'hard_probe_identity.json').is_file())
        if not candidates:raise ValueError('Completed E0 dev directory not found; specify --baseline-root')
        baseline=candidates[-1]
    print('Use E0 baseline: '+str(baseline),flush=True)
    out=Path(a.output_root) if a.output_root else runs/'E1_eval_v13'
    out.mkdir(parents=True,exist_ok=True)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    for arm in ('raw','compact'):
        target=out/arm
        if (target/'dev_summary.json').exists() and (target/'hard_summary.json').exists():
            identity=json.loads((target/'evaluation_identity.json').read_text())
            if identity['model_run']!=str(training/f'E1_{arm}_equal_exposure_v10'):
                raise ValueError('Existing evaluation model differs')
            print('Already complete; preserve '+str(target),flush=True);continue
        args=[sys.executable,'-u','-m','scripts.evaluate_e1_v13','--arm',arm,
              '--model-run',str(training/f'E1_{arm}_equal_exposure_v10'),
              '--baseline-root',str(baseline),'--sampling-root',str(runs/'rl_screening_v8_n4'),
              '--output-root',str(target),'--cache-root',a.cache_root]
        subprocess.run(args,env=env,check=True)
    comparison={'E0_dev':json.loads((baseline/'summary.json').read_text()),
                'scope':'Dev comparison and conditional training-side hard probe; no new training approval'}
    for arm in ('raw','compact'):
        comparison[arm]={'dev':json.loads((out/arm/'dev_summary.json').read_text()),
                         'hard':json.loads((out/arm/'hard_summary.json').read_text())}
    path=out/'comparison.json'
    if not path.exists():path.write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(comparison,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
