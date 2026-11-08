"""User-delegated itemwise AI audit of fixed P2 math/code train/dev. No training."""
import argparse
import ast
import getpass
import json
import os,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from src.synthesis.shared_cost_client import SharedCostClient,BudgetExhausted
from src.synthesis.data_review_contract import payload,parse,SYSTEM
from scripts.compact_distillation_v2 import read,write_jsonl,file_sha,ROOT
from src.synthesis.deepseek_pilot import dump,sha

OUT=ROOT/'data/project2_ai_review_20261024';REPORT=ROOT/'reports/project2_ai_review_20261024'
BUDGET=ROOT/'reports/data_release_preparation_20261024/shared_api_cache'


def jsonl(path):
    # JSON strings may contain literal U+2028; str.splitlines would corrupt them.
    with Path(path).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]


def tests_excerpt(r):
    tests=r.get('test_cases') or (r.get('verification_info') or {}).get('test_cases') or []
    values=[]
    for t in tests[:3]:
        s=json.dumps(t,ensure_ascii=False);values.append({'data':s[:1800],'truncated':len(s)>1800})
    return {'source_case_count':len(tests),'shown_cases':values,'all_cases_shown':len(tests)<=3 and all(not v['truncated'] for v in values),
        'execution_evidence':r.get('execution'),'scope':'source tests are not an independent semantic oracle'}


def prepare():
    OUT.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():
        m=read(OUT/'manifest.json')
        for p,h in m['inputs'].items():assert file_sha(ROOT/p)==h,p
        for p,h in m['files'].items():assert file_sha(OUT/p)==h,p
        return
    math={r['sample_id']:r for name in ['math_train','gsm8k_train','math_dev'] for r in jsonl(ROOT/f'data/math/{name}.jsonl')}
    train=jsonl(ROOT/'data/project2_code_v3_dev_verified/train_prompts.jsonl');dev=jsonl(ROOT/'data/project2_code_v3_dev_verified/dev_verification.jsonl')
    refs={r['sample_id']:r for r in jsonl(ROOT/'data/project2_code_v2/function_candidates.jsonl')+dev}
    missing={r['sample_id']:r for r in train if r['sample_id'] not in refs}
    source_paths=[]
    if missing:
        import pyarrow.parquet as pq
        from src.data.screen_code_interfaces import python_code
        metadata=read(ROOT/'data/screening_sources/code_source.json')
        hashes={r['path']:r['sha256'] for r in metadata['files']}
        for p in sorted((ROOT/'data/screening_sources/code/data').glob('*.parquet')):
            rel=str(p.relative_to(ROOT)).replace('\\','/');assert file_sha(p)==hashes[rel]
            source_paths.append(rel)
            for batch in pq.ParquetFile(p).iter_batches(batch_size=128,columns=['problem_id','problem_statement','gold_standard_solution','verification_info']):
                for r in batch.to_pylist():
                    sid='code/'+str(r['problem_id'])
                    if sid not in missing:continue
                    original=missing[sid];assert r['problem_statement']==original['problem']
                    code,tree=python_code(r['gold_standard_solution'])
                    assert sha_ast(tree)==original['reference_ast_sha256']
                    refs[sid]={'problem':r['problem_statement'],'reference_code':code,'verification_info':r['verification_info']}
            if set(missing)<=refs.keys():break
        assert set(missing)<=refs.keys()
    requests=[]
    for split,records in [('train',jsonl(ROOT/'data/project2_code/math_train_prompts.jsonl')),('dev',jsonl(ROOT/'data/math/math_dev.jsonl'))]:
        for r in records:
            ref=math[r['sample_id']];assert r['problem']==ref['problem']
            requests.append({'record_key':'math/'+split+'/'+r['sample_id'],'sample_id':r['sample_id'],'split':split,'domain':'math',
                'profile':'OPD_prompt_validity_and_reference_consistency' if split=='train' else 'fixed_dev_specification_and_reference',
                'problem':r['problem'],'reference_answer':ref.get('answer',ref.get('reference_answer')),
                'reference_solution':ref.get('solution'),'boundary':'references are provided solely for this explicitly authorized audit; never used in blind synthesis'})
    for split,records in [('train',train),('dev',dev)]:
        for r in records:
            ref=refs[r['sample_id']];assert r['problem']==ref['problem']
            requests.append({'record_key':'code/'+split+'/'+r['sample_id'],'sample_id':r['sample_id'],'split':split,'domain':'code',
                'profile':'OPD_prompt_validity_and_reference_consistency' if split=='train' else 'fixed_dev_specification_and_reference',
                'problem':r['prompt'],'interface':r['interface'],'entry_point':r.get('entry_point'),'function_signature':r.get('function_signature'),
                'reference_code':ref['reference_code'],'tests':tests_excerpt(ref),
                'boundary':'Full reference program is review-only. Execution is recorded only where supplied; do not infer train tests were executed.'})
    assert len(requests)==5456 and len({r['record_key'] for r in requests})==5456
    requests.sort(key=lambda r:(r['split']!='dev',sha('delegated-ai-review/'+r['record_key'])))
    write_jsonl(OUT/'requests.jsonl',requests)
    inputs=['data/project2_code_v3_dev_verified/manifest.json','data/project2_code_v3_dev_verified/train_prompts.jsonl','data/project2_code_v3_dev_verified/dev_verification.jsonl',
        'data/project2_code/math_train_prompts.jsonl','data/math/math_dev.jsonl','data/math/math_train.jsonl','data/math/gsm8k_train.jsonl',
        'src/synthesis/data_review_contract.py','src/synthesis/shared_cost_client.py','scripts/review_project2_deepseek.py']+source_paths
    dump(OUT/'policy.json',{'reviewer':'DeepSeek flash; user-delegated AI review, not human certification','system':SYSTEM,'system_sha256':sha(SYSTEM),
        'items':5456,'all_four_pools':True,'budget':'shared total $5 with P1 <=500 generation; completed usage + inflight reserve; stop at cap',
        'workers':8,'transport_policy':'At most one extra transport attempt for URLError only; retain original full cost reservation. No retry of a completed semantic or schema verdict; same exact audit payload.',
        'no_relabel_or_rewrite':True,'no_human_signoff_fabrication':True,'training_approved_before_audit':False})
    dump(OUT/'manifest.json',{'inputs':{p:file_sha(ROOT/p) for p in inputs},'files':{p.name:file_sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}})
    print('Frozen delegated review requests',len(requests),flush=True)


def sha_ast(tree):
    import hashlib
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def run(key):
    prepare();client=SharedCostClient(BUDGET,key,budget=5,max_requests=6500)
    requests=jsonl(OUT/'requests.jsonl');cache=REPORT/'results.jsonl'
    previous={r['record_key']:r for r in jsonl(cache)} if cache.exists() else {};results=list(previous.values())
    def one(r):
        content=json.dumps(r,ensure_ascii=False)
        if len(content)>60000:return {'record_key':r['record_key'],'sample_id':r['sample_id'],'domain':r['domain'],'split':r['split'],
            'verdict':'uncertain','issues':['full_reference_exceeds_declared_review_input_bound; no hidden truncation'],'checks':[],'api_called':False,'training_approved':False}
        try:
            attempts=1
            try:response=client.call(payload(r),'p2/review/'+r['record_key'])
            except RuntimeError as e:
                if isinstance(e,BudgetExhausted) or 'URLError' not in str(e):raise
                time.sleep(.3);attempts=2
                response=client.call(payload(r),'p2/review/'+r['record_key']+'/transport2')
            choice=response['response']['choices'][0]
            parsed=parse(choice['message'].get('content',''),choice.get('finish_reason'))
            return {'record_key':r['record_key'],'sample_id':r['sample_id'],'domain':r['domain'],'split':r['split'],**parsed,
                'api_called':True,'transport_attempts':attempts,'request_id':response['response'].get('id'),'returned_model':response['response'].get('model'),
                'raw_review':choice['message'].get('content',''),'review_method':'user_delegated_deepseek_AI','human_review':'not_performed',
                'engineering_candidate':parsed['verdict']=='pass','training_approved':False}
        except BudgetExhausted:return {'record_key':r['record_key'],'sample_id':r['sample_id'],'domain':r['domain'],'split':r['split'],
            'verdict':'not_reviewed_budget','api_called':False,'training_approved':False}
        except Exception as e:return {'record_key':r['record_key'],'sample_id':r['sample_id'],'domain':r['domain'],'split':r['split'],
            'verdict':'uncertain_transport','error_type':type(e).__name__,'api_called':False,'training_approved':False}
    pending=[r for r in requests if r['record_key'] not in previous]
    with cache.open('a',encoding='utf-8') as f, ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(one,r) for r in pending]
        for future in as_completed(fs):
            row=future.result();results.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            if len(results)%100==0:print('P2 delegated AI review',len(results),'/',len(requests),client.summary(),flush=True)
    summary={'items':5456,'results':len(results),'verdicts':dict(Counter(r['verdict'] for r in results)),
        'by_pool':{d+'/'+s:dict(Counter(r['verdict'] for r in results if r['domain']==d and r['split']==s)) for d in ['math','code'] for s in ['train','dev']},
        'shared_budget':client.summary(),'review_method':'user_delegated_AI; no human itemwise review claimed',
        'whole_dataset_approved':False,'training_approved':False,
        'release_rule':'Only itemwise pass can be proposed for delegated engineering release; isolate/uncertain/not_reviewed never auto-pass. No quota replacement or dev relabeling.'}
    dump(REPORT/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        key=os.environ.get('DEEPSEEK_API_KEY') or getpass.getpass('API key (hidden): ')
        if not key:raise RuntimeError('Missing key; no dispatch')
        run(key)
