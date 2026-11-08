"""Bounded pilot: 50 rows per source, contracts, protected split checks, sandbox."""
import json
import random
from collections import Counter
from pathlib import Path
from src.data.code_v2_sources import source_rows,metadata
from src.data.code_function_screen import extract,InterfaceError
from src.data.code_v2_dedup import Protected,function_body_hash
from src.data.code_sandbox import run_jobs,function_job
from src.data.screening_io import rows,sha,json_write
from src.data.prepare_math_data import write_jsonl,fingerprint

BASE=Path('data/code_v2_pilot')


def decorate(candidate,raw,source):
    source_id=raw['id'] if source=='ace' else str(raw['raw_index'])
    return {**candidate,'sample_id':source+'/'+source_id,'source':metadata(source)['repo'],
            'source_key':source,'source_revision':metadata(source)['revision'],'source_split':'train',
            'source_id':source_id,'problem_hash':fingerprint(candidate['problem']),
            'reference_body_sha256':function_body_hash(candidate['reference_code'],candidate['entry_point']),
            'family_id':'entry/'+candidate['entry_point']}


def run():
    BASE.mkdir(exist_ok=True);protected=Protected();all_candidates=[];audit=[];stats={}
    for source in ['ace','magicoder']:
        pool=[];scanned=0
        for raw in source_rows(source):
            scanned+=1
            if source=='magicoder' and raw.get('lang')!='python':continue
            pool.append(raw)
            if len(pool)>=2000:break
        selected=random.Random(42).sample(pool,50)
        write_jsonl(BASE/(source+'_raw_50.jsonl'),selected)
        counts=Counter()
        for raw in selected:
            source_id=raw.get('id',raw.get('raw_index'));evidence=None
            try:
                candidate=decorate(extract(raw,source),raw,source)
                evidence=protected.match(candidate)
                if evidence:raise InterfaceError(evidence['reason'])
                all_candidates.append(candidate);reason='static_candidate'
            except InterfaceError as exc:reason=str(exc)
            counts[reason]+=1
            audit.append({'source':source,'source_id':source_id,'reason':reason,'evidence':evidence})
        stats[source]={'source_rows_scanned_to_sample_pool':scanned,'sample_pool_rows':len(pool),
                       'sample_rows':50,'counts':dict(counts)}
    ace=[r for r in all_candidates if r['source_key']=='ace']
    response=run_jobs([function_job(r) for r in ace])
    outcomes={r['sample_id']:r for r in response['results']}
    for r in all_candidates:
        if r['source_key']=='ace':r['pilot_execution']=outcomes[r['sample_id']]
    write_jsonl(BASE/'static_candidates.jsonl',all_candidates)
    write_jsonl(BASE/'source_audit.jsonl',audit)
    json_write(BASE/'sandbox_results.json',response)
    report={'sample_rule':'seed42 sample 50 from first 2000 Ace rows in first shard and first 2000 Python Magicoder rows; bounded convenience pilot, not population prevalence.',
        'stats':stats,'protected_files':protected.files,'sandbox_python':response['python'],
        'ace_reference_passed':sum(r['passed'] for r in response['results']),
        'ace_executed':len(ace),'agent_review':'PENDING','bulk_gate_passed':False,
        'script_hashes':{p:sha(p) for p in ['src/data/code_function_screen.py','src/data/code_v2_dedup.py',
            'src/data/code_sandbox.py','scripts/code_sandbox_worker.py','src/data/audit_code_v2_pilot.py',
            'data/code_v2_sources/pilot_quarantine.json']}}
    json_write(BASE/'report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
