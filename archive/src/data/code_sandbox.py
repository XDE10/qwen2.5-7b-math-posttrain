"""Invoke a restricted Linux container; dataset code never runs on the host."""
import json
import subprocess
from pathlib import Path
from src.data.screening_io import json_write,sha

ENVIRONMENT=Path('reports/code_v2_sandbox_environment.json')


def environment():
    if ENVIRONMENT.exists():return json.loads(ENVIRONMENT.read_text(encoding='utf-8'))
    image=subprocess.check_output(['docker','image','inspect','gdrl-cpu:local','--format','{{.Id}}'],text=True).strip()
    worker=Path('scripts/code_sandbox_worker.py').resolve()
    command=['docker','run','--rm','-i','--network','none','--read-only','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--pids-limit','64','--memory','512m','--cpus','1',
        '--user','65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
        '--mount',f'type=bind,source={worker},target=/worker.py,readonly',
        '--entrypoint','python',image,'-I','/worker.py']
    meta={'image_id':image,'worker_sha256':sha(worker),'command':command,
          'boundary':'No network, no host secrets/workspace mounts; only read-only worker script mounted; input via stdin; per-child CPU/AS/file/process/time bounds.'}
    json_write(ENVIRONMENT,meta)
    return meta


def run_jobs(jobs):
    meta=environment()
    if meta['worker_sha256']!=sha('scripts/code_sandbox_worker.py'):raise ValueError('Sandbox worker version changed')
    process=subprocess.run(meta['command'],input=json.dumps(jobs,ensure_ascii=False),encoding='utf-8',
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=max(60,len(jobs)*30))
    if process.returncode:raise RuntimeError('Sandbox infrastructure failure: '+process.stderr[-2000:])
    response=json.loads(process.stdout)
    return response


def function_job(row):
    return {'kind':'function','sample_id':row['sample_id'],'reference_code':row['reference_code'],
            'entry_point':row['entry_point'],'assertions':[t['assertion'] for t in row['test_cases']],
            'constant_expected':row['test_cases'][0]['expected_repr']}
