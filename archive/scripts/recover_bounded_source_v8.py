"""Deterministically retry local resource timeouts, never paid semantic decisions."""
import json,sys,subprocess,time
from scripts.prepare_strict_inequalities_v8 import infer_profile,source_filter,replay
from scripts.strict_v8_common import *

def run():
    folder=DATA/'inequalities'
    while not (folder/'manifest.json').exists():time.sleep(3)
    failed={r['sample_id'] for r in rows(folder/'source_audit.jsonl') if r.get('reason')=='bounded_raw_probe_error'}
    proposals=[]
    with (ROOT/'reports/openr1_full_audit_final_20261023/questions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            q=json.loads(line)
            if q['question_hash'] not in failed:continue
            ts=[t for t in q['trajectories'] if t.get('verified') and t.get('complete') and t.get('answer_tokens') is not None]
            n=min(t['answer_tokens'] for t in ts)
            t=next(t for t in ts if t['answer_tokens']==n and t.get('quality',{}).get('automatic_pass') and t.get('total_tokens',99999)<=4096)
            proposals.append({'question_hash':q['question_hash'],'problem_type':'Inequalities','source':q['source'],'trajectory':t})
    replayed={r['sample_id']:r for r in replay(proposals)};source={};jobs=[]
    for p in proposals:
        r=replayed[p['question_hash']];r={**r,'solution':r['raw_solution']};profile=infer_profile(r)
        if not profile:continue
        a=source_filter(r);tri=a['v4_triage'];blocked=tri['quality_isolation_reasons']+tri['experiment_scope_exclusions']+[x for x in tri['manual_review_flags'] if x not in ['malformed_boxes','distinct_boxes_components_collection_or_intermediate_requires_review']]+[x for x in a['reasons'] if x!='v4_scope_or_source_or_structure_review']
        if blocked:continue
        sid=r['sample_id'];source[sid]={**r,'problem':a['problem'],'source':p['source'],'problem_type':'Inequalities','split':'train','training_approved':False,'answer_profile':profile,'source_answer_tokens':p['trajectory']['answer_tokens'],'source_total_tokens':p['trajectory']['total_tokens'],'local_timeout_recovery_only':True}
        jobs.append({'id':sid,'output':r['solution'],'reference':r['reference_answer'],'profile':profile})
    results={}
    for i in range(0,len(jobs),8):
        chunk=jobs[i:i+8]
        try:
            p=subprocess.run([sys.executable,'-m','scripts.strict_v8_verify_worker'],input=json.dumps(chunk),text=True,encoding='utf-8',capture_output=True,timeout=90,cwd=ROOT)
            results.update(json.loads(p.stdout.strip().splitlines()[-1]))
        except Exception as e:results.update({r['id']:{'correct':False,'status':'bounded_batch_recovery_error','error_type':type(e).__name__} for r in chunk})
        print('Local source timeout replay',min(i+8,len(jobs)),'/',len(jobs),flush=True)
    candidates=[{**source[sid],'raw_answer_check':v} for sid,v in results.items() if v.get('correct')]
    write(folder/'timeout_recovery_candidates.jsonl',candidates);dump(folder/'timeout_recovery_audit.json',{'examined_ids':sorted(failed),'results':results,'new_training_approvals':0,'api_calls':0,'verifier_sha':sha(ROOT/'src/synthesis/typed_answers_v8.py')})
    print('Recovered technical sources',len(candidates),flush=True)
if __name__=='__main__':run()
