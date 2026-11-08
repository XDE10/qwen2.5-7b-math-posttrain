import json,getpass,time,threading,subprocess,sys,base64
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from scripts.strict_v8_common import *
from scripts.compact_distillation_v3 import QUOTAS,tokenizer
from src.synthesis.finalfill_budget import FinalFillClient
from src.synthesis.shared_cost_client import BudgetExhausted
from src.synthesis.compact_v4 import payload,serialize_response,admission
from src.synthesis.data_review_contract import payload as judge_payload,parse as judge_parse
from src.data.perception_v4 import SYSTEM as VISUAL_SYSTEM,parse_review,prune
from src.data.screening_io import ScreeningSimilarityIndex

def run(key):
    client=FinalFillClient(BUDGET,key);slots=threading.BoundedSemaphore(8);lock=threading.Lock()
    if not (REPORT/'before_ledger.json').exists():(REPORT/'before_ledger.json').write_bytes((BUDGET/'ledger.json').read_bytes())
    baseline=rows(ROOT/'data/topup_v6_20261024/release/project1_raw_paired.jsonl');path=REPORT/'approved_new_pairs.jsonl';approved=rows(path) if path.exists() else [];ids={r['sample_id'] for r in baseline+approved};counts=Counter(r['problem_type'] for r in baseline+approved);index=ScreeningSimilarityIndex(.85)
    for r in baseline+approved:index.add(r)
    def call(p,task):
        with slots:
            try:return client.call(p,task)
            except RuntimeError as e:
                if isinstance(e,BudgetExhausted) or 'URLError' not in str(e):raise
                return client.call(p,task+'/transport2')
    def batch(name,records,fn,workers=4):
        out=REPORT/(name+'.jsonl');results=rows(out) if out.exists() else [];seen={r['id'] for r in results};pending=[r for r in records if r['id'] not in seen]
        with out.open('a',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=workers) as pool:
            fs={pool.submit(fn,r):r for r in pending}
            for future in as_completed(fs):
                r=fs[future]
                try:v=future.result()
                except BudgetExhausted:v={'status':'not_dispatched_budget','verdict':'not_reviewed_budget'}
                except Exception as e:v={'status':'uncertain_transport_or_local_error','verdict':'uncertain_transport_or_local_error','error_type':type(e).__name__}
                v.update(id=r['id'],sample_id=r.get('sample_id',r['id']));results.append(v);f.write(json.dumps(v,ensure_ascii=False)+'\n');f.flush()
        return results
    def generate(r):
        with lock:
            if counts[r['problem_type']]>=QUOTAS[r['problem_type']]:return {'status':'not_dispatched_type_target_met'}
        c=call(payload(r['problem']),'v8/generation/'+r['id']);ch=c['response']['choices'][0]
        return {'status':'complete','raw_content':ch['message'].get('content',''),'finish_reason':ch.get('finish_reason'),'request_id':c['response'].get('id')}
    def judge(r):
        with lock:
            if counts[r['problem_type']]>=QUOTAS[r['problem_type']]:return {'verdict':'not_dispatched_type_target_met'}
        req={'profile':'two_full_trajectory_mathematical_audit','problem':r['problem'],'reference_answer':r['reference_answer'],'raw_solution':r['raw_solution'],'compact_solution':r['compact_solution'],'answer_type':r.get('answer_profile'),
            'additional_contract':'Audit BOTH original full unmodified trajectories. Distinguish quoted boxed-format examples, corrected exploration, intermediate results and final solution collections. Verify domain, strict endpoints, excluded poles, all branches and equality cases. Do not accept merely because final sets match. A source-condition ambiguity or uncorrected false step must isolate. No teacher/ref feedback was supplied to blind generation.'}
        c=call(judge_payload(req),'v8/pair_review/'+r['id']);ch=c['response']['choices'][0];v=judge_parse(ch['message'].get('content',''),ch.get('finish_reason'))
        result={**v,'request_id':c['response'].get('id'),'raw_review':ch['message'].get('content','')}
        if v['verdict']=='pass':
            with lock:
                if counts[r['problem_type']]<QUOTAS[r['problem_type']] and r['sample_id'] not in ids and not index.match(r['problem']):
                    approved.append({**r,'training_approved':True,'approval_request_id':c['response'].get('id'),'human_review_performed':False,'approval_method':'v8_typed_or_frozen_scalar_equivalence+unchanged_quality_length+full_two_trajectory_AI_review'});ids.add(r['sample_id']);index.add(r);counts[r['problem_type']]+=1;write(path,approved)
                else:result['release_status']='surplus_or_duplicate_not_training_approved'
        return result
    def local(chunk,generated,folder,tok):
        folder.mkdir(exist_ok=True);write(folder/'source_rows.jsonl',chunk);gen={r['sample_id']:r for r in generated};jobs=[];parsed={}
        for r in chunk:
            g=gen.get(r['sample_id'],{})
            if g.get('status')!='complete':continue
            p=serialize_response(g['raw_content']);parsed[r['sample_id']]=p
            for kind,text in [('raw',r['solution']),('compact',p['text'])]:jobs.append({'id':r['sample_id']+'/'+kind,'output':text,'reference':r['reference_answer'],'profile':r.get('answer_profile',{})})
        verified={}
        def work(j):
            try:
                c=subprocess.run([sys.executable,'-m','scripts.strict_v8_verify_worker'],input=json.dumps(j),text=True,encoding='utf-8',capture_output=True,timeout=90,cwd=ROOT);return json.loads(c.stdout.strip().splitlines()[-1])
            except Exception as e:return {x['id']:{'correct':False,'status':'bounded_verifier_batch_error','error_type':type(e).__name__} for x in j}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for v in pool.map(work,[jobs[i:i+16] for i in range(0,len(jobs),16)]):verified.update(v)
        results=[];common=[]
        for r in chunk:
            sid=r['sample_id'];g=gen.get(sid,{})
            if sid not in parsed:results.append({'sample_id':sid,'status':g.get('status','not_generated'),'common':False});continue
            p=parsed[sid];a=admission(p,r['problem'],g['finish_reason'],tok,verified[sid+'/compact'])
            if __import__('re').search(r'\\frac\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*\{\s*\}',p['text']):a['errors'].append('empty_fraction_denominator');a['accepted_candidate']=False
            ok=a['accepted_candidate'] and verified[sid+'/raw'].get('correct',False)
            raw_full=len(tok.apply_chat_template([{'role':'system','content':r'Please reason step by step, and put your final answer within \boxed{}.'},{'role':'user','content':r['problem']},{'role':'assistant','content':r['solution']}],tokenize=True,add_generation_prompt=False))
            if raw_full>(4096 if r['problem_type']=='Inequalities' else 3800):ok=False
            results.append({'sample_id':sid,'admission':a,'raw_check':verified[sid+'/raw'],'raw_full_tokens':raw_full,'common':ok})
            if ok:common.append({**r,'raw_solution':r['solution'],'compact_solution':p['text'],'answer_tokens':a['answer_tokens'],'raw_full_tokens':raw_full})
        write(folder/'local_checks.jsonl',results);write(folder/'common_pairs.jsonl',common);return common
    def pipeline(kind):
        if kind=='inequalities':
            while not (DATA/'inequalities/manifest.json').exists():time.sleep(3)
            pool=rows(DATA/'inequalities/candidates.jsonl');roundsize=20
        else:pool=rows(ROOT/'data/finalfill_v7_20261024/project1/candidates.jsonl');roundsize=100
        held=ScreeningSimilarityIndex(.85)
        for file in ['data/topup_v6_20261024/release/project2_math_dev.jsonl','data/topup_v6_20261024/release/project2_code_dev.jsonl','data/topup_v6_20261024/project2/candidates.jsonl','data/topup_v6_20261024/project2_dev_reserve/candidates.jsonl']:
            for r in rows(ROOT/file):
                if r.get('split')=='dev':held.add(r)
        pool=[r for r in pool if not held.match(r['problem'])];tok=tokenizer();sent=set();roundno=0
        while client.committed()<11:
            with lock:remaining=[r for r in pool if r['sample_id'] not in sent and counts[r['problem_type']]<QUOTAS[r['problem_type']]]
            if not remaining:break
            chunk=remaining[:roundsize];sent.update(r['sample_id'] for r in chunk);roundno+=1
            folder=DATA/f'{kind}_round_{roundno:02d}';generated=batch(kind+'_generation',[{**r,'id':r['sample_id']} for r in chunk],generate)
            common=local(chunk,generated,folder,tok);batch(kind+'_pair_reviews',[{**r,'id':r['sample_id']} for r in common],judge)
            with lock:progress={'type_counts':dict(counts),'approved_total':sum(counts.values()),'new_approved':len(approved),'round':roundno,'pool':kind,'budget':client.summary()}
            dump(REPORT/(kind+'_progress.json'),progress);print('V8 approved',progress,flush=True)
        dump(REPORT/(kind+'_summary.json'),{'finished':True,'rounds':roundno,'questions_in_rounds':len(sent),'global_type_counts':dict(counts),'target_exact':dict(counts)==QUOTAS})
    def recover_geometry():
        previous=rows(ROOT/'reports/finalfill_v7_20261024/p3_reviews.jsonl');cases={r['sample_id']:r for r in rows(ROOT/'data/finalfill_v7_20261024/project3/dispatched_case_plan.jsonl')};pending=[{**cases[r['sample_id']],'id':r['sample_id']} for r in previous if not r.get('request_id')]
        def audit(r):
            facts=r['canonical_candidates_with_audit_provenance'];q={'problem':r['problem'],'stored_answer_for_audit_only':r['reference_answer'],'candidate_facts':[{'index':i,'fact':f} for i,f in enumerate(facts)],'unparsed_source_entries_for_audit_only':r['uncertain']}
            p={'model':'deepseek-flash','thinking':{'type':'disabled'},'temperature':0,'max_tokens':min(3500,700+65*len(facts)),'response_format':{'type':'json_object'},'messages':[{'role':'system','content':VISUAL_SYSTEM},{'role':'user','content':[{'type':'text','text':json.dumps(q,ensure_ascii=False)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(__import__('pathlib').Path(r['image_path']).read_bytes()).decode(),'detail':'original'}}]}]}
            c=call(p,'v8/geometry_transport_recovery/'+r['id']);ch=c['response']['choices'][0];v=parse_review(ch['message'].get('content',''),len(facts),ch.get('finish_reason'))
            return {'original_id':r['original_id'],'split':'train','review':v,'pruning':prune(r,v),'request_id':c['response'].get('id'),'raw_review':ch['message'].get('content',''),'finish_reason':ch.get('finish_reason'),'recovery_reason':'only previously missing response, never semantic resampling'}
        batch('geometry_transport_recovery',pending,audit,2)
    hashes={p:sha(ROOT/p) for p in ['src/synthesis/typed_answers_v8.py','scripts/run_strict_v8.py','scripts/strict_v8_verify_worker.py','src/synthesis/compact_v4.py','src/synthesis/data_review_contract.py']}
    dump(REPORT/'frozen_protocol.json',{'target_total':1500,'exact_type_targets':QUOTAS,'ineq_source_tokens':[1200,3800],'ineq_raw_full_max':4096,'other_source_max':3200,'other_raw_full_max':3800,'compact_tokens':[150,1500],'compact_full_max':3800,'no_quality_gate_relaxation':True,'first_ineq_batch_is_independent_of27_debug_cases':True,'ineq_step_size':20,'same_teacher_payload':payload('PROBLEM_ONLY'),'shared_total_cap':11,'code_hashes':hashes})
    try:
        recover_geometry()
        with ThreadPoolExecutor(max_workers=2) as threads:
            futures={threads.submit(pipeline,k):k for k in ['other_five','inequalities']}
            for f in as_completed(futures):
                try:f.result();print('V8 pipeline complete',futures[f],flush=True)
                except Exception as e:dump(REPORT/(futures[f]+'_error.json'),{'type':type(e).__name__,'message':str(e)});print('V8 pipeline error',futures[f],type(e).__name__,flush=True)
    finally:dump(REPORT/'budget_after.json',client.summary());client.close()
if __name__=='__main__':run(getpass.getpass('API key (hidden): '))
