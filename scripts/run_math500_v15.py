"""Sequential fixed E0/raw/compact test; automatic auth; never trains."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_e1_autoauth_v12 import authenticated_environment
from src.training.frozen_verifier_v4 import sha, grader_identity, digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-root',default='/content/drive/MyDrive/runs/project1')
    p.add_argument('--output-root');p.add_argument('--cache-root',default='/content/models')
    p.add_argument('--e0-model-path',default='/content/models/Qwen2.5-7B-Instruct--a09a35458c702b33eeacc393d103063234e8bc28')
    p.add_argument('--e0-model-identity',default='/content/project1-env-evidence/e0-identity/e0_model_identity.json')
    a=p.parse_args();project=Path(__file__).resolve().parents[1];os.chdir(project)
    runs=Path(a.runs_root);root=Path(a.output_root) if a.output_root else runs/'MATH500_v15'
    env=authenticated_environment(os.environ)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    protocol=json.loads(Path('configs/project1_math500_protocol_v15.json').read_text(encoding='utf-8'))
    if digest(grader_identity())!=protocol['grader_identity_sha256']:raise ValueError('Use the frozen scorer; do not change it using test outcomes')
    frozen={'protocol':protocol,'versions':{n:version(n) for n in ['vllm','torch','transformers','tokenizers']},
            'source_sha256':sha('scripts/evaluate_math500_v15.py')}
    root.mkdir(parents=True,exist_ok=True);path=root/'frozen_protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=frozen:raise ValueError('Frozen test protocol changed')
    else:path.write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    identities=[]
    for name in protocol['models']:
        target=root/name
        if not (target/'summary.json').exists():
            args=[sys.executable,'-u','-m','scripts.evaluate_math500_v15','--model-key',name,
                  '--runs-root',str(runs),'--output-root',str(target),'--cache-root',a.cache_root,
                  '--e0-model-path',a.e0_model_path,'--e0-model-identity',a.e0_model_identity]
            subprocess.run(args,env=env,check=True)
        report=json.loads((target/'summary.json').read_text())
        identity=json.loads((target/'evaluation_identity.json').read_text())
        if report['model_key']!=name or report['count']!=500 or identity['source_sha256']!=frozen['source_sha256']:
            raise ValueError('Existing test output identity mismatch')
        identities.append((report['protocol_sha256'],report['grader_identity_sha256'],report['tokenizer_sha256']))
    if any(identity!=identities[0] for identity in identities[1:]):raise ValueError('Do not compare different test/scorer/tokenizer protocols')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    output=runs/'summary_v15'/stamp
    subprocess.run([sys.executable,'-u','-m','scripts.summarize_project1_v15','--runs-root',str(runs),
                    '--math500-root',str(root),'--output-root',str(output)],env=env,check=True)
    print('Summary saved: '+str(output),flush=True)


if __name__=='__main__':main()
