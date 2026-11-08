"""Bounded rolling top-up to1500, full train geometry coverage, same guards."""
import getpass,time,threading,json,base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.finalfill_common import *
from src.synthesis.finalfill_budget import FinalFillClient
from src.synthesis.shared_cost_client import BudgetExhausted
from src.synthesis.compact_v4 import payload as gen_payload
from src.synthesis.data_review_contract import payload as review_payload,parse as review_parse
from src.data.perception_v4 import SYSTEM as VISUAL_SYSTEM,parse_review,prune,token_key
from src.data.screening_io import ScreeningSimilarityIndex
from scripts.compact_distillation_v3 import QUOTAS

def run(key):
    snapshot();client=FinalFillClient(BUDGET,key);semaphore=threading.BoundedSemaphore(8)
    def call(payload,task):
        with semaphore:
            try:return client.call(payload,task)
            except RuntimeError as e:
                if isinstance(e,BudgetExhausted) or 'URLError' not in str(e):raise
                return client.call(payload,task+'/transport2')
    def batch(name,records,worker,workers=4):
        path=REPORT/(name+'.jsonl');results=rows(path) if path.exists() else [];ids={r['id'] for r in results}
        pending=[r for r in records if r['id'] not in ids];total=len(results)+len(pending)
        with path.open('a',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(worker,r):r for r in pending}
            for future in as_completed(futures):
                r=futures[future]
                try:x=future.result()
                except BudgetExhausted:x={'status':'not_dispatched_budget','verdict':'not_reviewed_budget'}
                except Exception as e:x={'status':'uncertain_transport_or_local_error','verdict':'uncertain_transport_or_local_error','error_type':type(e).__name__}
                x.update(id=r['id'],sample_id=r.get('sample_id',r['id']));results.append(x);f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush()
                if len(results)%25==0:print(name,len(results),'/',total,client.summary(),flush=True)
        return results
    def p1():
        while (REPORT/'quota_decision.json').exists() and read(REPORT/'quota_decision.json')['mode']=='pending_source_diagnosis':time.sleep(3)
        while not (DATA/'project1/manifest.json').exists():time.sleep(3)
        out=DATA/'project1';candidates=rows(out/'candidates.jsonl')
        held=ScreeningSimilarityIndex(.85)
        for file in ['data/topup_v6_20261024/release/project2_math_dev.jsonl','data/topup_v6_20261024/release/project2_code_dev.jsonl','data/topup_v6_20261024/project2/candidates.jsonl','data/topup_v6_20261024/project2_dev_reserve/candidates.jsonl']:
            for r in rows(ROOT/file):
                if r.get('split')=='dev':held.add(r)
        candidates=[r for r in candidates if not held.match(r['problem'])]
        write(out/'eligible_after_full_dev_protection.jsonl',candidates)
        approved_path=REPORT/'p1_new_approved.jsonl';approved=rows(approved_path) if approved_path.exists() else [];approved_ids={r['sample_id'] for r in approved};baseline=rows(ROOT/'data/topup_v6_20261024/release/project1_raw_paired.jsonl');counts=Counter(r['problem_type'] for r in baseline+approved);stop=threading.Event();lock=threading.Lock()
        def quota_mode():
            q=REPORT/'quota_decision.json'
            return read(q)['mode'] if q.exists() else 'pending'
        def can_approve(kind):return quota_mode()=='available_five_strata' or counts[kind]<QUOTAS[kind]
        def generate(r):
            if stop.is_set():return {'status':'not_dispatched_target_met'}
            response=call(gen_payload(r['problem']),'v7/p1/generate/'+r['id']);ch=response['response']['choices'][0]
            return {'status':'complete','raw_content':ch['message'].get('content',''),'finish_reason':ch.get('finish_reason'),'request_id':response['response'].get('id')}
        def review(r):
            with lock:
                if len(baseline)+len(approved)>=1500:stop.set();return {'verdict':'not_dispatched_target_met'}
            req={'profile':'same_problem_two_complete_solution_audit','problem':r['problem'],'reference_answer':r['reference_answer'],'raw_solution':r['raw_solution'],'compact_solution':r['compact_solution'],'additional_contract':'Audit BOTH full trajectories for invalid intermediate equality, unsupported deduction, missing cases and source ambiguity. A matching answer cannot rescue an uncorrected false derivation. Explicitly corrected exploration is allowed.'}
            response=call(review_payload(req),'v7/p1/pair/'+r['id']);ch=response['response']['choices'][0];v=review_parse(ch['message'].get('content',''),ch.get('finish_reason'))
            result={**v,'request_id':response['response'].get('id'),'raw_review':ch['message'].get('content','')}
            if v['verdict']=='pass':
                with lock:
                    if r['sample_id'] not in approved_ids and len(baseline)+len(approved)<1500 and can_approve(r['problem_type']):
                        approved.append({**r,'approval_request_id':response['response'].get('id'),'training_approved':True,'approval_method':'unchanged_local_gate_plus_two_full_trajectory_AI_audit','human_review_performed':False});approved_ids.add(r['sample_id']);counts[r['problem_type']]+=1;write(approved_path,approved)
                        if len(baseline)+len(approved)==1500:stop.set()
                    else:result['release_status']='qualified_surplus_or_pending_quota_not_training_approved'
            return result
        # Independent 100-question rounds, initial plan1100; remaining frozen
        # reservoir is touched only when still short and cap permits.
        for roundno in range((len(candidates)+99)//100):
            if stop.is_set() or client.committed()>=11:break
            chunk=candidates[roundno*100:(roundno+1)*100];folder=out/f'round_{roundno+1:02d}';folder.mkdir(exist_ok=True)
            write(folder/'verifier_references.jsonl',chunk)
            generated=batch('p1_generation',[{**r,'id':r['sample_id']} for r in chunk],generate)
            current={r['sample_id']:r for r in generated};write(folder/'generation_results.jsonl',[current[r['sample_id']] for r in chunk if r['sample_id'] in current])
            import scripts.project1_batch500 as frozen
            frozen.DATA=folder;frozen.REPORT=folder;frozen.prepare=lambda:None
            if not (folder/'common_candidates.json').exists():frozen.verify_local()
            common=read(folder/'common_candidates.json')
            batch('p1_pair_reviews',[{**r,'id':r['sample_id']} for r in common],review)
            # If the user relaxes only strata quotas, previously audited pass
            # surplus can be released without any new API or semantic retry.
            if quota_mode()=='available_five_strata':
                allreviews={r['sample_id']:r for r in rows(REPORT/'p1_pair_reviews.jsonl')};allcommon=[]
                for p in sorted(out.glob('round_*/common_candidates.json')):allcommon+=read(p)
                for r in allcommon:
                    v=allreviews.get(r['sample_id'],{})
                    if v.get('verdict')=='pass' and r['sample_id'] not in approved_ids and len(baseline)+len(approved)<1500:
                        approved.append({**r,'approval_request_id':v['request_id'],'training_approved':True,'approval_method':'unchanged_checks_quota_only_user_amendment','human_review_performed':False});approved_ids.add(r['sample_id']);counts[r['problem_type']]+=1
                write(approved_path,approved)
                if len(baseline)+len(approved)==1500:stop.set()
            dump(REPORT/'p1_progress.json',{'rounds_complete':roundno+1,'new_approved':len(approved),'total_approved':977+len(approved),'type_counts':dict(counts),'quota_mode':quota_mode(),'shared_budget':client.summary()});print('P1 approved',977+len(approved),dict(counts),flush=True)
        dump(REPORT/'p1_summary.json',{'new_approved':len(approved),'total_approved':977+len(approved),'type_counts':dict(counts),'target_met':977+len(approved)==1500,'quota_mode':quota_mode(),'stopped_at_1500':stop.is_set(),'source_questions_sent':sum(r.get('status')=='complete' for r in rows(REPORT/'p1_generation.jsonl'))})
    def p3():
        while not (DATA/'project3/manifest.json').exists():time.sleep(3)
        records=rows(DATA/'project3/remaining896.jsonl')
        # Recover the one previous transport-only failure as coverage requires a
        # completed audit, not merely a dispatch. No semantic verdict re-sampled.
        lookup={r['sample_id']:r for r in rows(ROOT/'data/topup_v6_20261024/project3/cases.jsonl')}
        for r in rows(ROOT/'reports/topup_v6_20261024/p3_fact_review_reparsed.jsonl'):
            if not r.get('review',{}).get('task_status'):
                c=lookup[r.get('sample_id',r['id'])]
                if c['split']=='train':records.append({**c,'transport_recovery_only':True})
        write(DATA/'project3/dispatched_case_plan.jsonl',records)
        def audit(r):
            facts=r['canonical_candidates_with_audit_provenance'];record={'problem':r['problem'],'stored_answer_for_audit_only':r['reference_answer'],'candidate_facts':[{'index':i,'fact':f} for i,f in enumerate(facts)],'unparsed_source_entries_for_audit_only':r['uncertain']}
            payload={'model':'deepseek-flash','thinking':{'type':'disabled'},'temperature':0,'max_tokens':min(3500,700+len(facts)*65),'response_format':{'type':'json_object'},'messages':[{'role':'system','content':VISUAL_SYSTEM},{'role':'user','content':[{'type':'text','text':json.dumps(record,ensure_ascii=False)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(__import__('pathlib').Path(r['image_path']).read_bytes()).decode(),'detail':'original'}}]}]}
            response=call(payload,'v7/p3/fact_audit/'+r['id']);ch=response['response']['choices'][0];v=parse_review(ch['message'].get('content',''),len(facts),ch.get('finish_reason'))
            return {'original_id':r['original_id'],'split':r['split'],'review':v,'pruning':prune(r,v),'request_id':response['response'].get('id'),'raw_review':ch['message'].get('content',''),'finish_reason':ch.get('finish_reason')}
        results=batch('p3_reviews',[{**r,'id':r['sample_id']} for r in records],audit)
        dump(REPORT/'p3_summary.json',{'new_or_transport_recovery_cases':len(records),'returned':sum(bool(r.get('request_id')) for r in results),'prior_train_attempted':1200,'remaining_unique_source_train':896,'no_test_or_val_generation':True})
    dump(REPORT/'protocol.json',{'total_cap_usd':11,'target_p1_pairs':1500,'quality_and_length_rules_unchanged':True,'p1_round_size':100,'p1_initial_plan':1100,'p3_remaining_train':896,'semantic_retries':False,'uncertain_not_auto_approved':True,'code_hashes':{p:sha(ROOT/p) for p in ['scripts/run_finalfill_v7.py','src/synthesis/finalfill_budget.py','src/synthesis/compact_v4.py','src/synthesis/data_review_contract.py','src/data/perception_v4.py','src/data/perception_text_guards.py']}})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs={pool.submit(fn):name for fn,name in [(p1,'p1'),(p3,'p3')]}
            for f in as_completed(fs):
                try:f.result();print('V7 pipeline finished',fs[f],client.summary(),flush=True)
                except Exception as e:dump(REPORT/(fs[f]+'_pipeline_error.json'),{'type':type(e).__name__,'message':str(e)});print('V7 pipeline error',fs[f],type(e).__name__,flush=True)
    finally:dump(REPORT/'budget_after.json',client.summary());client.close()
if __name__=='__main__':run(getpass.getpass('API key (hidden): '))
