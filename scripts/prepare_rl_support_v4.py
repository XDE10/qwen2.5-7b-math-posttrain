"""CPU support-domain selfcheck, resumable; never modifies sealed source."""
import argparse
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from src.training.frozen_verifier_v4 import Worker,selfcheck,grader_identity,digest,sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='reports/handover_20261024_v3/rl_candidates_4630_lexically_screened_v1.jsonl')
    p.add_argument('--output-root',required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();out=Path(a.output_root).resolve();out.mkdir(parents=True,exist_ok=True)
    identity={'source_sha256':sha(a.input),'grader':grader_identity()}
    ip=out/'support_identity.json'
    if ip.exists() and json.loads(ip.read_text(encoding='utf-8'))!=identity:raise ValueError('Support resume identity changed')
    ip.write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=Path(a.input).read_bytes().splitlines(keepends=True)
    rows=[json.loads(line) for line in lines if line.strip()]
    cache=out/'support_checks.jsonl';results={}
    if cache.exists():
        for line in cache.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r=json.loads(line)
                if r['sample_id'] in results:raise ValueError('Duplicate cached selfcheck')
                results[r['sample_id']]=r
    local=threading.local();workers=[];lock=threading.Lock()
    def task(row):
        if not hasattr(local,'worker'):
            local.worker=Worker()
            with lock:workers.append(local.worker)
        return selfcheck(row,local.worker)
    try:
        with cache.open('a',encoding='utf-8') as log,ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(task,row) for row in rows if row['sample_id'] not in results]
            for future in as_completed(futures):
                r=future.result();results[r['sample_id']]=r
                log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
                if len(results)%100==0:print(json.dumps({'checked':len(results),'total':len(rows)}),flush=True)
    finally:
        for worker in workers:worker.close()
    if set(results)!={r['sample_id'] for r in rows}:raise ValueError('Selfcheck coverage mismatch')
    keep={sid for sid,r in results.items() if r['supported_and_consistent']}
    supported=out/'rl_candidates_verifier_supported_v2.jsonl'
    quarantine=out/'rl_candidates_verifier_quarantine_v2.jsonl'
    supported.write_bytes(b''.join(line for line in lines if line.strip() and json.loads(line)['sample_id'] in keep))
    quarantine.write_bytes(b''.join(line for line in lines if line.strip() and json.loads(line)['sample_id'] not in keep))
    summary={'input_count':len(rows),'supported_count':len(keep),'quarantined_count':len(rows)-len(keep),
             'answer_categories':dict(Counter(r['answer_category'] for r in results.values())),
             'solution_categories':dict(Counter(r['solution_category'] for r in results.values())),
             'supported_file_sha256':sha(supported),'quarantine_file_sha256':sha(quarantine),'grader_identity_sha256':digest(identity['grader']),
             'training_approved':False,'scope':'provided answer and reference solution passed this frozen grader; does not guarantee every future generated answer is parseable or certify reasoning'}
    (out/'support_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
