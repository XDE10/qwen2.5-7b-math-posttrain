"""One shared-budget owner, resumable independent pipelines, max 8 in flight."""
import base64,getpass,time,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from scripts.topup_common import *
from src.synthesis.topup_budget import TopupClient
from src.synthesis.shared_cost_client import BudgetExhausted
from src.synthesis.compact_v4 import payload as gen_payload
from src.synthesis.data_review_contract import payload as review_payload,parse as review_parse,SYSTEM as REVIEW_SYSTEM
from src.data.perception_v4 import SYSTEM as VISUAL_SYSTEM,parse_review,prune,token_key

def wait_ready(project):
    while not (DATA/project/'manifest.json').exists():time.sleep(3)
def run(key):
    snapshot();client=TopupClient(BUDGET,key);slots=threading.BoundedSemaphore(8)
    def call(payload,task):
        with slots:
            try:return client.call(payload,task)
            except RuntimeError as e:
                if isinstance(e,BudgetExhausted) or 'URLError' not in str(e):raise
                return client.call(payload,task+'/transport2')
    def execute_batch(name,records,worker,workers=6):
        path=REPORT/(name+'.jsonl');prior={r['id']:r for r in rows(path)} if path.exists() else {};results=list(prior.values())
        with path.open('a',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=workers) as pool:
            fs={pool.submit(worker,r):r for r in records if r['id'] not in prior}
            for future in as_completed(fs):
                r=fs[future]
                try:x=future.result()
                except BudgetExhausted:x={'id':r['id'],'verdict':'not_reviewed_budget','status':'not_dispatched_budget'}
                except Exception as e:x={'id':r['id'],'verdict':'uncertain_transport_or_local_error','status':'uncertain_transport_or_local_error','error_type':type(e).__name__}
                x['id']=r['id'];results.append(x);f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush()
                if len(results)%50==0:print(name,len(results),'/',len(records),client.summary(),flush=True)
        return results
    def p1():
        wait_ready('project1');out=DATA/'project1'
        records=[{**r,'id':r['sample_id']} for r in rows(out/'blind_requests.jsonl')]
        def generate(r):
            response=call(gen_payload(r['problem']),'v6/p1/generate/'+r['id']);ch=response['response']['choices'][0]
            return {'sample_id':r['id'],'status':'complete','raw_content':ch['message'].get('content',''),'finish_reason':ch.get('finish_reason'),'request_id':response['response'].get('id'),'returned_model':response['response'].get('model'),'usage':response['response'].get('usage',{})}
        generated=execute_batch('p1_generation',records,generate,8)
        # Reuse the exact frozen local admission/verifier implementation.
        import scripts.project1_batch500 as previous
        previous.DATA=out;previous.REPORT=REPORT/'p1_local';previous.REPORT.mkdir(exist_ok=True);previous.prepare=lambda:None
        write(out/'verifier_references.jsonl',rows(out/'native_new1500.jsonl'))
        write(previous.REPORT/'generation_results.jsonl',generated)
        previous.verify_local()
        pairs=read(previous.REPORT/'common_candidates.json')
        def audit(r):
            request={'profile':'same_problem_two_complete_solution_audit','problem':r['problem'],'reference_answer':r['reference_answer'],'raw_solution':r['raw_solution'],'compact_solution':r['compact_solution'],'additional_contract':'Check both full trajectories. Reject uncorrected intermediate false equalities, missing cases, unsupported deductions. Explicitly corrected exploration is not itself a defect. Matching final answers cannot rescue invalid reasoning.'}
            response=call(review_payload(request),'v6/p1/pair/'+r['id']);ch=response['response']['choices'][0]
            return {'sample_id':r['sample_id'],**review_parse(ch['message'].get('content',''),ch.get('finish_reason')),'request_id':response['response'].get('id'),'raw_review':ch['message'].get('content','')}
        reviewed=execute_batch('p1_pair_review',[{**r,'id':r['sample_id']} for r in pairs],audit,8)
        dump(REPORT/'p1_summary.json',{'new_questions':len(records),'generation':dict(Counter(r['status'] for r in generated)),'local_common':len(pairs),'pair_review':dict(Counter(r.get('verdict') for r in reviewed)),'local_rule_sha':sha(ROOT/'src/synthesis/compact_v4.py'),'review_contract_sha':sha(ROOT/'src/synthesis/data_review_contract.py')})
    def p2():
        wait_ready('project2');records=[{**r,'id':r['record_key']} for r in rows(DATA/'project2/audit_requests.jsonl')]
        def audit(r):
            if len(json.dumps(r,ensure_ascii=False))>60000:return {'verdict':'uncertain','issues':['full_input_exceeds_bound_no_truncation']}
            response=call(review_payload({k:v for k,v in r.items() if k!='id'}),'v6/p2/review/'+r['id']);ch=response['response']['choices'][0]
            return {'sample_id':r['sample_id'],'record_key':r['record_key'],'split':r['split'],'domain':r['domain'],**review_parse(ch['message'].get('content',''),ch.get('finish_reason')),'request_id':response['response'].get('id'),'raw_review':ch['message'].get('content','')}
        reviewed=execute_batch('p2_review',records,audit,6)
        dump(REPORT/'p2_summary.json',{'reviewed':len(reviewed),'verdicts':dict(Counter(r.get('verdict') for r in reviewed))})
    def p3():
        wait_ready('project3');records=[{**r,'id':r['sample_id']} for r in rows(DATA/'project3/cases.jsonl')]
        def audit(r):
            facts=r['canonical_candidates_with_audit_provenance'];record={'problem':r['problem'],'stored_answer_for_audit_only':r['reference_answer'],'candidate_facts':[{'index':i,'fact':f} for i,f in enumerate(facts)],'unparsed_source_entries_for_audit_only':r['uncertain']}
            raw=Path(r['image_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
            payload={'model':'deepseek-flash','thinking':{'type':'disabled'},'temperature':0,'max_tokens':min(3500,700+len(facts)*65),'response_format':{'type':'json_object'},'messages':[{'role':'system','content':VISUAL_SYSTEM},{'role':'user','content':[{'type':'text','text':json.dumps(record,ensure_ascii=False)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode(),'detail':'original'}}]}]}
            response=call(payload,'v6/p3/fact_audit/'+r['id']);ch=response['response']['choices'][0];review=parse_review(ch['message'].get('content',''),len(facts),ch.get('finish_reason'));result=prune(r,review)
            # Unsupported scoring serialization cannot silently become a reward label.
            if result['accepted']:
                try:
                    for s in result['positive_tokens']+result['negative_tokens']:token_key(s)
                except Exception:result['accepted']=False;result['reason']='reward_parser_cannot_represent_certified_target'
            return {'sample_id':r['sample_id'],'original_id':r['original_id'],'split':r['split'],'review':review,'pruning':result,'request_id':response['response'].get('id'),'raw_review':ch['message'].get('content','')}
        reviewed=execute_batch('p3_fact_review',records,audit,4)
        dump(REPORT/'p3_summary.json',{'audited':len(reviewed),'accepted_by_split':dict(Counter(r['split'] for r in reviewed if r.get('pruning',{}).get('accepted'))),'task_statuses':dict(Counter(r.get('review',{}).get('task_status','transport_or_local_error') for r in reviewed)),'scope':'AI-certified fact subsets; not perfect truth or whole-image precision'})
    dump(REPORT/'frozen_protocol.json',{'cap_total_usd':8,'includes_before_spending':read(REPORT/'before_ledger.json')[-1]['task_id'],'global_concurrency':8,'p1_length':[150,1500],'p1_no_semantic_resampling':True,'p2_dev_tests_before_AI':True,'p3_training_split_only':'train; validation never training','p3_metric':'sparse certified positive/explicit negative precision and recall; unknown masked','code_hashes':{p:sha(ROOT/p) for p in ['scripts/run_topup_v6.py','src/synthesis/compact_v4.py','src/synthesis/data_review_contract.py','src/data/perception_v4.py','src/synthesis/topup_budget.py']}})
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(fn):name for fn,name in [(p1,'p1'),(p2,'p2'),(p3,'p3')]}
            for f in as_completed(futures):
                try:f.result();print('Pipeline complete',futures[f],client.summary(),flush=True)
                except Exception as e:dump(REPORT/(futures[f]+'_pipeline_error.json'),{'type':type(e).__name__,'message':str(e)});print('Pipeline error',futures[f],type(e).__name__,flush=True)
    finally:dump(REPORT/'budget_after.json',client.summary());client.close()
if __name__=='__main__':run(getpass.getpass('API key (hidden): '))
