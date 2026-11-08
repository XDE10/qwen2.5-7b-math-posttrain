"""Resume remaining work after the first owner's lock releases, same $8 ledger."""
import getpass,time,threading,base64
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.topup_common import *
from src.synthesis.topup_budget import TopupClient
from src.synthesis.shared_cost_client import BudgetExhausted
from src.synthesis.data_review_contract import payload as review_payload,parse as review_parse
from src.data.perception_v4 import SYSTEM,parse_review,prune

def run(key):
    while not (REPORT/'budget_after.json').exists():time.sleep(3)
    while not (DATA/'project2_dev_reserve/manifest.json').exists():time.sleep(3)
    while not (DATA/'project2_train_reserve/manifest.json').exists():time.sleep(3)
    while not (DATA/'project1_supplement/manifest.json').exists():time.sleep(3)
    while True:
        try:client=TopupClient(BUDGET,key);break
        except Exception as e:
            if type(e).__name__!='Timeout':raise
            time.sleep(2)
    slots=threading.BoundedSemaphore(8)
    def call(p,task):
        with slots:
            try:return client.call(p,task)
            except RuntimeError as e:
                if isinstance(e,BudgetExhausted) or 'URLError' not in str(e):raise
                return client.call(p,task+'/transport2')
    def batch(name,records,fn,workers=8):
        path=REPORT/(name+'.jsonl');previous=rows(path) if path.exists() else [];seen={r['id'] for r in previous};results=previous[:]
        with path.open('a',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=workers) as pool:
            fs={pool.submit(fn,r):r for r in records if r['id'] not in seen}
            for future in as_completed(fs):
                r=fs[future]
                try:x=future.result()
                except BudgetExhausted:x={'verdict':'not_reviewed_budget'}
                except Exception as e:x={'verdict':'uncertain_transport_or_local_error','error_type':type(e).__name__}
                x.update(id=r['id'],sample_id=r.get('sample_id',r['id']));results.append(x);f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush()
                if len(results)%25==0:print(name,len(results),'/',len(records),client.summary(),flush=True)
        return results
    def p1_repair():
        # Transport failures are audit records too; normalize identifier only.
        if (REPORT/'p1_summary.json').exists():return
        generated=rows(REPORT/'p1_generation.jsonl');assert len(generated)==len(rows(DATA/'project1/blind_requests.jsonl'))
        generated=[{**r,'sample_id':r.get('sample_id',r['id'])} for r in generated]
        import scripts.project1_batch500 as old
        old.DATA=DATA/'project1';old.REPORT=REPORT/'p1_local';old.REPORT.mkdir(exist_ok=True);old.prepare=lambda:None
        write(old.DATA/'verifier_references.jsonl',rows(old.DATA/'native_new1500.jsonl'));write(old.REPORT/'generation_results.jsonl',generated)
        if not (old.REPORT/'common_candidates.json').exists():old.verify_local()
        pairs=read(old.REPORT/'common_candidates.json')
        def audit(r):
            req={'profile':'same_problem_two_complete_solution_audit','problem':r['problem'],'reference_answer':r['reference_answer'],'raw_solution':r['raw_solution'],'compact_solution':r['compact_solution'],'additional_contract':'Check both full trajectories. Reject uncorrected false intermediate equalities, missing cases or unsupported deductions. Explicit corrected exploration is allowed. Final answer match does not rescue invalid reasoning.'}
            response=call(review_payload(req),'v6/p1/pair/'+r['id']);ch=response['response']['choices'][0]
            return {**review_parse(ch['message'].get('content',''),ch.get('finish_reason')),'request_id':response['response'].get('id'),'raw_review':ch['message'].get('content','')}
        result=batch('p1_pair_review',[{**r,'id':r['sample_id']} for r in pairs],audit)
        dump(REPORT/'p1_summary.json',{'new_questions':len(generated),'local_common':len(pairs),'pair_pass':sum(r.get('verdict')=='pass' for r in result),'identifier_only_repair':True})
    def p2_extra():
        requests=rows(DATA/'project2_dev_reserve/audit_requests.jsonl')+rows(DATA/'project2_train_reserve/audit_requests.jsonl');records=[{**r,'id':r['record_key']} for r in requests]
        # Frozen candidates, no rejected semantic verdict resampled.
        def audit(r):
            response=call(review_payload({k:v for k,v in r.items() if k!='id'}),'v6/p2/dev_reserve/'+r['id']);ch=response['response']['choices'][0]
            return {**review_parse(ch['message'].get('content',''),ch.get('finish_reason')),'request_id':response['response'].get('id'),'raw_review':ch['message'].get('content',''),'domain':'code','split':r['split']}
        result=batch('p2_dev_reserve_review',records,audit,8);dump(REPORT/'p2_dev_reserve_summary.json',{'reviewed':len(result),'pass':sum(r.get('verdict')=='pass' for r in result)})
    def p1_supplement():
        from src.synthesis.compact_v4 import payload as generation_payload
        out=DATA/'project1_supplement';records=[{**r,'id':r['sample_id']} for r in rows(out/'blind_requests.jsonl')]
        def generate(r):
            response=call(generation_payload(r['problem']),'v6/p1/supplement_generate/'+r['id']);ch=response['response']['choices'][0]
            return {'status':'complete','raw_content':ch['message'].get('content',''),'finish_reason':ch.get('finish_reason'),'request_id':response['response'].get('id')}
        generated=batch('p1_supplement_generation',records,generate)
        for r in generated:
            if r.get('status')!='complete':r['status']='uncertain_transport'
        import scripts.project1_batch500 as old
        old.DATA=out;old.REPORT=REPORT/'p1_supplement_local';old.REPORT.mkdir(exist_ok=True);old.prepare=lambda:None
        write(out/'verifier_references.jsonl',rows(out/'native_candidates.jsonl'));write(old.REPORT/'generation_results.jsonl',generated);old.verify_local()
        pairs=read(old.REPORT/'common_candidates.json')
        def audit(r):
            req={'profile':'same_problem_two_complete_solution_audit','problem':r['problem'],'reference_answer':r['reference_answer'],'raw_solution':r['raw_solution'],'compact_solution':r['compact_solution'],'additional_contract':'Check both complete trajectories for unsupported deductions, false intermediate equality or missing cases. Final answer match cannot rescue a false derivation; corrected exploration is allowed.'}
            response=call(review_payload(req),'v6/p1/supplement_pair/'+r['id']);ch=response['response']['choices'][0]
            return {**review_parse(ch['message'].get('content',''),ch.get('finish_reason')),'request_id':response['response'].get('id'),'raw_review':ch['message'].get('content','')}
        batch('p1_supplement_pair_review',[{**r,'id':r['sample_id']} for r in pairs],audit)
    def p3_extra():
        from scripts.reparse_topup_geometry import run as reparse
        reparse();summary=read(REPORT/'p3_reparsed_summary.json')
        if summary['accepted_by_split'].get('train',0)>=500:return
        records=[{**r,'id':r['sample_id']} for r in rows(DATA/'project3_reserve/cases.jsonl')]
        def audit(r):
            facts=r['canonical_candidates_with_audit_provenance'];record={'problem':r['problem'],'stored_answer_for_audit_only':r['reference_answer'],'candidate_facts':[{'index':i,'fact':f} for i,f in enumerate(facts)],'unparsed_source_entries_for_audit_only':r['uncertain']}
            p={'model':'deepseek-flash','thinking':{'type':'disabled'},'temperature':0,'max_tokens':min(3500,700+len(facts)*65),'response_format':{'type':'json_object'},'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'text','text':json.dumps(record,ensure_ascii=False)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path(r['image_path']).read_bytes()).decode(),'detail':'original'}}]}]}
            response=call(p,'v6/p3/reserve_fact_audit/'+r['id']);ch=response['response']['choices'][0];review=parse_review(ch['message'].get('content',''),len(facts),ch.get('finish_reason'))
            return {'original_id':r['original_id'],'split':r['split'],'review':review,'pruning':prune(r,review),'request_id':response['response'].get('id'),'raw_review':ch['message'].get('content','')}
        result=[]
        for offset in range(0,len(records),100):
            result=batch('p3_reserve_fact_review',records[offset:offset+100],audit)
            reparse();current=read(REPORT/'p3_reparsed_summary.json')
            dump(REPORT/'p3_reserve_summary.json',{'audited':len(result),'strict_approved_train':current['accepted_by_split'].get('train',0),'stop_rule':'100 fresh train images per round, final sidecar checks; stop once500 approved or total$8'})
            if current['accepted_by_split'].get('train',0)>=500:break
    try:
        # Finish exact code quotas before spending remaining cap on extra images.
        for f,name in [(p1_repair,'p1'),(p1_supplement,'p1_supplement'),(p2_extra,'p2_extra'),(p3_extra,'p3_extra')]:
            try:f();print('Reserve complete',name,flush=True)
            except Exception as e:dump(REPORT/(name+'_reserve_error.json'),{'type':type(e).__name__,'message':str(e)});print('Reserve error',name,type(e).__name__,flush=True)
    finally:dump(REPORT/'budget_after_reserves.json',client.summary());client.close()
if __name__=='__main__':run(getpass.getpass('API key (hidden): '))
