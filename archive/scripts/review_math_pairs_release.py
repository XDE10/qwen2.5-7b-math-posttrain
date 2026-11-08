"""Delegated review of new batch pairs plus the prior ten AI-pass v4 pairs."""
import getpass,json,os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from scripts.compact_distillation_v2 import ROOT,read,file_sha
from scripts.review_project2_deepseek import BUDGET
from src.synthesis.shared_cost_client import SharedCostClient,BudgetExhausted
from src.synthesis.data_review_contract import payload,parse,SYSTEM
from src.synthesis.deepseek_pilot import dump,sha

REPORT=ROOT/'reports/math_pair_delegated_review_20261024'


def run(key):
    REPORT.mkdir(exist_ok=True)
    new=read(ROOT/'reports/project1_batch500_20261024/common_candidates.json')
    oldverdicts=read(ROOT/'reports/compact_distillation_v4_20261024/ai_process_review.json')['reviews']
    ids={r['sample_id'] for r in oldverdicts if r['pair_decision']=='pass'}
    old=[r for r in read(ROOT/'reports/compact_distillation_v4_20261024/common_candidates.json') if r['sample_id'] in ids]
    pairs=[{**r,'cohort':'new_batch500'} for r in new]+[{**r,'cohort':'prior_v4_10'} for r in old]
    assert len(old)==10 and len({r['sample_id'] for r in pairs})==len(pairs)
    inputs={p:file_sha(ROOT/p) for p in ['reports/project1_batch500_20261024/common_candidates.json','reports/compact_distillation_v4_20261024/ai_process_review.json',
        'reports/compact_distillation_v4_20261024/common_candidates.json','src/synthesis/data_review_contract.py','scripts/review_math_pairs_release.py']}
    policy={'inputs':inputs,'system':SYSTEM,'system_hash':sha(SYSTEM),'pairs':len(pairs),'old_human_review_claimed':False,
        'purpose':'User delegated AI process audit, both full trajectories; no automatic gold certification; shared $5 cap'}
    if (REPORT/'policy.json').exists():assert read(REPORT/'policy.json')==policy
    else:dump(REPORT/'policy.json',policy)
    client=SharedCostClient(BUDGET,key,budget=5,max_requests=6500)
    results=[]
    def one(r):
        record={'profile':'same_problem_two_complete_solution_audit','problem':r['problem'],'reference_answer':r['reference_answer'],
            'raw_solution':r['raw_solution'],'compact_solution':r['compact_solution'],
            'additional_contract':'Check both trajectories. An uncorrected false equality or invalid branch in either is material. Explicitly corrected exploration is not itself an error. Do not infer source correctness from matching answers. Distinguish source ambiguity from solution errors. Keep evidence concise.'}
        try:
            c=client.call(payload(record),'math_pair_delegated/'+r['sample_id']);ch=c['response']['choices'][0]
            return {'sample_id':r['sample_id'],'cohort':r['cohort'],**parse(ch['message'].get('content',''),ch.get('finish_reason')),
                'request_id':c['response'].get('id'),'raw_review':ch['message'].get('content',''),'human_review':'not_performed','review_method':'user_delegated_deepseek_AI'}
        except BudgetExhausted:return {'sample_id':r['sample_id'],'cohort':r['cohort'],'verdict':'not_reviewed_budget'}
        except Exception as e:return {'sample_id':r['sample_id'],'cohort':r['cohort'],'verdict':'uncertain_transport','error_type':type(e).__name__}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,r in enumerate(pool.map(one,pairs),1):
            results.append(r)
            if i%25==0:dump(REPORT/'partial.json',results);print('Delegated pair review',i,'/',len(pairs),flush=True)
    dump(REPORT/'reviews.json',results)
    summary={'pairs':len(pairs),'verdicts':dict(Counter(r['verdict'] for r in results)),
        'by_cohort':{k:dict(Counter(r['verdict'] for r in results if r['cohort']==k)) for k in ['new_batch500','prior_v4_10']},
        'shared_budget':client.summary(),'human_review_performed':False,'whole_native1500_approved':False}
    dump(REPORT/'summary.json',summary);client.close();print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    key=os.environ.get('DEEPSEEK_API_KEY') or getpass.getpass('API key (hidden): ')
    if not key:raise RuntimeError('No API key')
    run(key)
