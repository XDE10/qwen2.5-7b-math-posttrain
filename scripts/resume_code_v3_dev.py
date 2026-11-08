"""Complete pending v3 dev only after fresh restricted-container tests succeed."""
import argparse
import json
from pathlib import Path
from src.data.screening_io import rows,sha,json_write
from src.data.prepare_math_data import write_jsonl
from src.data.code_sandbox import function_job,run_jobs,environment


def run(live=False):
    source=Path('data/project2_code_v3');out=Path('data/project2_code_v3_dev_verified')
    m=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for v in m['files'].values():assert sha(v['path'])==v['sha256']
    pending=list(rows(source/'dev_replacement_plan_pending.jsonl'))
    if not live:print(f'{len(pending)} AceCode dev replacements require fresh Docker tests; use --execute after engine recovery');return
    if out.exists():raise FileExistsError('verified release exists; do not overwrite')
    verified=[]
    for r in pending:
        result=run_jobs([function_job(r)])['results'][0]
        if not result.get('passed') or any(result.get('constant_mutant_passes',[True])):raise ValueError('Source test or constant-mutant gate failed: '+r['sample_id'])
        verified.append({**r,'execution':result,'execution_status':'all_source_tests_passed_in_restricted_container'})
    dev=list(rows(source/'dev_verification.jsonl'))+verified
    assert len(dev)==200
    out.mkdir()
    train=list(rows(source/'train_prompts.jsonl'));write_jsonl(out/'train_prompts.jsonl',train)
    write_jsonl(out/'dev_verification.jsonl',dev)
    write_jsonl(out/'dev_metadata.jsonl',[{k:v for k,v in r.items() if k not in ['reference_code','test_cases','verification_info','execution']} for r in dev])
    json_write(out/'manifest.json',{'version':'code-v3-fresh-dev-completed','parent_manifest_sha256':sha(source/'manifest.json'),
        'fresh_dev_ids':[r['sample_id'] for r in verified],'sandbox':environment(),'train':2500,'dev':200,
        'training_approved':False,'human_review':'pending','scripts':{__file__:sha(__file__)},
        'files':{p.name:{'path':p.as_posix(),'sha256':sha(p)} for p in out.glob('*.jsonl')}})
    print('Fresh dev tests passed; new separate release created:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();run(a.execute)
