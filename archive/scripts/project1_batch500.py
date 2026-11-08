"""Versioned <=500 blind generations, shared $5 cap, separate local/AI audits."""
import argparse,getpass,json,os,re,subprocess,sys,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from scripts.compact_distillation_v2 import ROOT,read,write_jsonl,file_sha
from scripts.review_project2_deepseek import jsonl,BUDGET
from scripts.compact_distillation_v3 import tokenizer
from src.synthesis.compact_v4 import payload,serialize_response,admission
from src.synthesis.deepseek_pilot import dump,sha
from src.synthesis.shared_cost_client import SharedCostClient,BudgetExhausted
from src.synthesis.data_review_contract import payload as review_payload,parse as parse_review
from src.data.screening_io import ScreeningSimilarityIndex

DATA=ROOT/'data/project1_batch500_20261024';REPORT=ROOT/'reports/project1_batch500_20261024'
QUOTAS={'Algebra':175,'Geometry':100,'Number Theory':100,'Combinatorics':75,'Inequalities':33,'Logic and Puzzles':17}


def prepare():
    DATA.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    if (DATA/'manifest.json').exists():
        m=read(DATA/'manifest.json')
        for p,h in m['inputs'].items():assert file_sha(ROOT/p)==h,p
        for p,h in m['files'].items():assert file_sha(DATA/p)==h,p
        return
    previous=ScreeningSimilarityIndex(.85);ids=set()
    names=['data/deepseek_pilot_20261023/verifier_references.json','data/compact_distillation_v2_20261023/verifier_references.json',
        'data/compact_distillation_v3_20261023/verifier_references.json','data/compact_distillation_v4_20261024/verifier_references.json',
        'data/deepseek_pilot_20261023/anchors.json','data/deepseek_pilot_20261023/calibration_dev.json','data/deepseek_pilot_20261023/frozen_acceptance.json']
    for name in names:
        for r in read(ROOT/name):previous.add(r);ids.add(r['sample_id'])
    pool=jsonl(ROOT/'data/project1_native_v5/native_candidates.jsonl');selected=[]
    for kind,count in QUOTAS.items():
        chosen=[]
        for r in sorted([r for r in pool if r['problem_type']==kind],key=lambda r:sha('batch500-v5/'+r['sample_id'])):
            if r['sample_id'] in ids or previous.match(r['problem']):continue
            if re.search(r'[<>≤≥≠=]|\\(?:leq?|geq?|neq?|lt|gt)\b',str(r['reference_answer'])):continue
            assert not re.search(r'(?im)\bANSWER\s*:\s*\S',r['problem'])
            selected.append(r);chosen.append(r);previous.add(r)
            if len(chosen)==count:break
        if len(chosen)!=count:raise RuntimeError('No quota relaxation; insufficient '+kind+': '+str(len(chosen)))
    assert len(selected)==500
    write_jsonl(DATA/'blind_requests.jsonl',[{k:r[k] for k in ['sample_id','problem','split','source_revision']} for r in selected]);write_jsonl(DATA/'verifier_references.jsonl',selected)
    policy={'version':'batch500-v5-20261024','max_generation_questions':500,'max_attempts_per_question':1,'workers':16,
        'budget':'shared $5 hard cap including P2 audits, completed peak usage + in-flight byte-envelope reservations; stop before dispatch if exhausted',
        'gate_override':'User authorized a bounded engineering batch despite v4 gate failure; original v4 results/gate remain unchanged',
        'acceptance':'150-1500 Qwen tokens, source and generated answers pass conservative Math-Verify type guard, complete JSON/no control chars/no repeats/no empty fractions; then delegated AI process review',
        'not_a_success_guarantee':True,'no_training_or_bulk_auto_approval':True,'reference_never_sent_to_generation':True,
        'six_type_quotas':QUOTAS,'request_example':payload('PROBLEM_ONLY'),'training_approved':False}
    dump(DATA/'policy.json',policy)
    inputs=['data/project1_native_v5/manifest.json','data/project1_native_v5/native_candidates.jsonl','src/synthesis/compact_v4.py',
        'src/synthesis/compact_v3.py','src/synthesis/compact_format_review.py','src/synthesis/shared_cost_client.py','scripts/project1_batch500.py',
        'src/synthesis/data_review_contract.py','scripts/math_verify_v3_strict_worker.py']
    dump(DATA/'manifest.json',{'inputs':{p:file_sha(ROOT/p) for p in inputs},'files':{p.name:file_sha(p) for p in DATA.iterdir() if p.is_file() and p.name!='manifest.json'}})
    print('Frozen batch500, six strata',QUOTAS,flush=True)


def generate(key):
    prepare();client=SharedCostClient(BUDGET,key,budget=5,max_requests=6500);start=time.monotonic()
    reqs=jsonl(DATA/'blind_requests.jsonl');path=REPORT/'generation_results.jsonl'
    previous={r['sample_id']:r for r in jsonl(path)} if path.exists() else {};results=list(previous.values())
    def one(r):
        assert set(r)=={'sample_id','problem','split','source_revision'}
        try:
            c=client.call(payload(r['problem']),'p1/batch500/'+r['sample_id']);ch=c['response']['choices'][0]
            return {'sample_id':r['sample_id'],'raw_content':ch['message'].get('content',''),'finish_reason':ch.get('finish_reason'),
                'usage':c['response'].get('usage',{}),'request_id':c['response'].get('id'),'returned_model':c['response'].get('model'),'status':'complete'}
        except BudgetExhausted:return {'sample_id':r['sample_id'],'status':'not_dispatched_budget'}
        except Exception as e:return {'sample_id':r['sample_id'],'status':'uncertain_transport','error_type':type(e).__name__}
    with path.open('a',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=16) as pool:
        fs=[pool.submit(one,r) for r in reqs if r['sample_id'] not in previous]
        for future in as_completed(fs):
            r=future.result();results.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            if len(results)%50==0:print('P1 generation',len(results),'/500',client.summary(),flush=True)
    dump(REPORT/'generation_summary.json',{'target':500,'statuses':dict(Counter(r['status'] for r in results)),
        'wall_seconds':time.monotonic()-start,'shared_budget':client.summary(),'generation_saw_reference_fields':False,'training_approved':False})
    client.close()


def verify_local():
    prepare();refs={r['sample_id']:r for r in jsonl(DATA/'verifier_references.jsonl')};generated=jsonl(REPORT/'generation_results.jsonl')
    tok=tokenizer();parsed={r['sample_id']:serialize_response(r['raw_content']) for r in generated if r['status']=='complete'}
    jobs=[]
    for sid,p in parsed.items():
        jobs += [{'id':sid+'/compact','output':p['text'],'reference':refs[sid]['reference_answer']},{'id':sid+'/raw','output':refs[sid]['solution'],'reference':refs[sid]['reference_answer']}]
    batches=[jobs[i:i+20] for i in range(0,len(jobs),20)];verified={}
    def batch(rows):
        try:
            p=subprocess.run([sys.executable,'-m','scripts.project1_batch500','worker'],input=json.dumps(rows),text=True,encoding='utf-8',capture_output=True,timeout=90,cwd=ROOT)
            result=json.loads(p.stdout.strip().splitlines()[-1]);assert set(result)=={r['id'] for r in rows};return result
        except Exception as e:return {r['id']:{'correct':False,'status':'batch_verifier_error','error_type':type(e).__name__} for r in rows}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(batch,batches),1):
            verified.update(result)
            if i%5==0:print('P1 local verifier',i,'/',len(batches),flush=True)
    results=[];pairs=[]
    for r in generated:
        sid=r['sample_id']
        if sid not in parsed:results.append(r);continue
        p=parsed[sid];ref=refs[sid];math=verified[sid+'/compact'];rawmath=verified[sid+'/raw']
        a=admission(p,ref['problem'],r['finish_reason'],tok,math);a['rule_version']='batch500-v5'
        if re.search(r'\\frac\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*\{\s*\}',p['text']):
            a['errors'].append('empty_fraction_denominator');a['accepted_candidate']=False
        common=a['accepted_candidate'] and rawmath.get('correct',False)
        results.append({**r,'parsed':p,'admission':a,'raw_answer_verification':rawmath,'common_candidate':common,'training_approved':False})
        if common:pairs.append({'sample_id':sid,'problem_type':ref['problem_type'],'problem':ref['problem'],'reference_answer':ref['reference_answer'],
            'raw_solution':ref['solution'],'compact_solution':p['text'],'answer_tokens':a['answer_tokens'],'training_approved':False,'process_review':'pending'})
    dump(REPORT/'verification_results.json',results);dump(REPORT/'common_candidates.json',pairs)
    summary={'target_questions':500,'generation_completed':len(parsed),'schema_valid':sum(p['valid'] for p in parsed.values()),
        'answer_verified':sum(v.get('correct',False) for k,v in verified.items() if k.endswith('/compact')),
        'automatic_admitted':sum(r.get('admission',{}).get('accepted_candidate',False) for r in results),'common_pairs':len(pairs),
        'failure_signals':dict(Counter(e for r in results for e in r.get('admission',{}).get('errors',[]))),
        'verifier_errors':sum(v.get('status')=='batch_verifier_error' for v in verified.values()),'training_approved':False,'process_review':'pending'}
    dump(REPORT/'verification_summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


def review_pairs(key):
    prepare();pairs=read(REPORT/'common_candidates.json');client=SharedCostClient(BUDGET,key,budget=5,max_requests=6500);results=[]
    def one(r):
        request={'profile':'math_pair_full_process_audit','problem':r['problem'],'reference_answer':r['reference_answer'],
            'raw_solution':r['raw_solution'],'compact_solution':r['compact_solution'],
            'additional_conditions':'Audit BOTH full trajectories, including any uncorrected false intermediate equalities, missing cases or malformed fractions. Matching final answers cannot rescue an invalid derivation. A correct explicitly corrected exploration is not itself an error.'}
        try:
            c=client.call(review_payload(request),'p1/pair_review/'+r['sample_id']);ch=c['response']['choices'][0]
            return {'sample_id':r['sample_id'],**parse_review(ch['message'].get('content',''),ch.get('finish_reason')),
                'request_id':c['response'].get('id'),'raw_review':ch['message'].get('content',''),'training_approved':False,'review_method':'user_delegated_AI','human_review':'not_performed'}
        except BudgetExhausted:return {'sample_id':r['sample_id'],'verdict':'not_reviewed_budget','training_approved':False}
        except Exception as e:return {'sample_id':r['sample_id'],'verdict':'uncertain_transport','error_type':type(e).__name__,'training_approved':False}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i,r in enumerate(pool.map(one,pairs),1):
            results.append(r)
            if i%25==0:dump(REPORT/'pair_review_partial.json',results);print('P1 pair review',i,'/',len(pairs),flush=True)
    dump(REPORT/'pair_reviews.json',results)
    dump(REPORT/'pair_review_summary.json',{'pairs':len(pairs),'verdicts':dict(Counter(r['verdict'] for r in results)),'shared_budget':client.summary(),
        'all_human_reviewed':False,'no_whole_pool_approval':True})
    client.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','generate','verify','review','worker']);a=p.parse_args()
    if a.action=='worker':
        from scripts.math_verify_v3_strict_worker import check
        results={}
        for r in json.load(sys.stdin):
            try:results[r['id']]=check(r)
            except Exception as e:results[r['id']]={'correct':False,'status':str(e),'error_type':type(e).__name__}
        print(json.dumps(results))
    elif a.action=='prepare':prepare()
    elif a.action=='verify':verify_local()
    else:
        key=os.environ.get('DEEPSEEK_API_KEY') or getpass.getpass('API key (hidden): ')
        if not key:raise RuntimeError('No key; no dispatch')
        if a.action=='generate':generate(key)
        else:review_pairs(key)
