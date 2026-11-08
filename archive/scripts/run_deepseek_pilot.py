"""Run only the frozen pilot; never automatically expand to 1500 questions."""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transformers import AutoTokenizer
from src.synthesis.deepseek_pilot import PilotClient, difficulty_payload, solve_payload, review_payload, dump, sha
from src.data.openr1_quality import quality_signals
from scripts.prepare_deepseek_pilot import ROOT, OUT, file_sha

REPORT=ROOT/'reports/deepseek_pilot_20261023'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def content(call):return call['response']['choices'][0]['message'].get('content') or ''
def parsed(call):
    text=content(call).strip()
    if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
    return json.loads(text)


def check_math(output,reference):
    try:
        p=subprocess.run([sys.executable,str(ROOT/'scripts/math_verify_pilot_worker.py')],
            input=json.dumps({'output':output,'reference':reference}),text=True,encoding='utf-8',
            capture_output=True,timeout=20,cwd=ROOT)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as exc:return {'correct':False,'status':'verifier_error','error_type':type(exc).__name__}


def difficulty_metrics(rows):
    valid=[r for r in rows if r.get('valid')]
    n=len(rows);low=[r for r in valid if r['gold_level']<=2];pred_high=[r for r in valid if r['estimated_level']>=3]
    confusion=[[0]*5 for _ in range(5)]
    for r in valid:confusion[r['gold_level']-1][r['estimated_level']-1]+=1
    return {'n':n,'valid':len(valid),'valid_schema_rate':len(valid)/n if n else 0,
        'exact_agreement':sum(r['estimated_level']==r['gold_level'] for r in valid)/n if n else 0,
        'within_one':sum(abs(r['estimated_level']-r['gold_level'])<=1 for r in valid)/n if n else 0,
        'mae_valid':sum(abs(r['estimated_level']-r['gold_level']) for r in valid)/len(valid) if valid else None,
        'low_gold_n':len(low),'low_to_high_count':sum(r['estimated_level']>=3 for r in low),
        'low_to_high_false_positive_rate':sum(r['estimated_level']>=3 for r in low)/len(low) if low else None,
        'predicted_high_n':len(pred_high),
        'predicted_high_precision':sum(r['gold_level']>=3 for r in pred_high)/len(pred_high) if pred_high else None,
        'confusion_gold_rows_pred_columns':confusion}


def parallel(items,fn,label,workers=4):
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,r):r for r in items}
        for f in as_completed(futures):
            r=futures[f]
            try:result=f.result()
            except Exception as exc:result={'sample_id':r.get('sample_id'),'error_type':type(exc).__name__,'status':'failed'}
            results.append(result)
            dump(REPORT/(label+'_partial.json'),results)
            print(f'{label}: {len(results)}/{len(items)}',flush=True)
    return sorted(results,key=lambda r:r['sample_id'])


def run(live=False):
    REPORT.mkdir(exist_ok=True)
    m=load(OUT/'manifest.json');policy=load(OUT/'policy.json')
    for name,h in m['files'].items():assert file_sha(OUT/name)==h,name
    for path,h in m['input_hashes'].items():assert file_sha(ROOT/path)==h,path
    anchors=load(OUT/'anchors.json');requests=load(OUT/'blind_requests.json')
    refs={r['sample_id']:r for r in load(OUT/'verifier_references.json')}
    assert len(requests)==50 and all(set(r)=={'sample_id','problem','source_revision','split'} for r in requests)
    # Materialize exact non-secret payloads for a privacy audit, including no gold label on targets.
    payloads={r['sample_id']:solve_payload(r['problem']) for r in requests}
    dump(REPORT/'blind_payload_audit.json',{'request_keys':['sample_id','problem','source_revision','split'],
        'payload_hashes':{k:sha(v) for k,v in payloads.items()},'reference_or_original_solution_sent':False,
        'prompt_sha256':sha(policy['prompt_examples']),'mode':policy['generation_mode']})
    if not live:
        print('Dry run passed: 50 blind payloads; no network');return
    client=PilotClient(REPORT/'api_cache',os.environ.get('DEEPSEEK_API_KEY',''),budget=policy['budget_usd_reserved'],max_requests=policy['max_requests'])
    before=REPORT/'balance_before.json'
    if not before.exists():dump(before,client.balance())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    for split in ['calibration_dev','frozen_acceptance']:
        def label(r):
            call=client.call(difficulty_payload(r['problem'],anchors),'difficulty/'+split+'/'+r['sample_id'])
            result={'sample_id':r['sample_id'],'gold_level':int(r['level'][-1]),'subject':r['problem_type'],'valid':False}
            try:
                d=parsed(call);v=d.get('estimated_level')
                valid=type(v) is int and 1<=v<=5 and type(d.get('uncertain')) is bool
                result.update(estimated_level=v,valid=valid,annotation=d)
            except Exception:pass
            return result
        output=parallel(load(OUT/(split+'.json')),label,split)
        dump(REPORT/(split+'_results.json'),output)
        dump(REPORT/(split+'_metrics.json'),difficulty_metrics(output))
    metrics=load(REPORT/'frozen_acceptance_metrics.json');gate=policy['difficulty_gate']
    gate_ok=(metrics['valid_schema_rate']>=gate['valid_schema_min'] and metrics['exact_agreement']>=gate['exact_agreement_min']
        and metrics['within_one']>=gate['within_one_min'] and metrics['low_to_high_false_positive_rate'] is not None
        and metrics['low_to_high_false_positive_rate']<=gate['low_to_high_false_positive_rate_max'])
    dump(REPORT/'difficulty_gate.json',{'passed':gate_ok,'policy':gate,'metrics':metrics,
        'bulk_labeling_enabled':False,'interpretation':'Pilot only; no full candidate labeling in this run'})
    if gate_ok:
        def label_candidate(r):
            call=client.call(difficulty_payload(r['problem'],anchors),'difficulty/openr1/'+r['sample_id'])
            return {'sample_id':r['sample_id'],'annotation':parsed(call),'official_level':None,'status':'estimated_only_unvalidated_on_openr1'}
        dump(REPORT/'openr1_estimated_levels.json',parallel(requests,label_candidate,'openr1_difficulty'))
    def generate(r):
        reference=refs[r['sample_id']]
        attempts=[];chosen=None
        for attempt in range(1,policy['max_generation_attempts']+1):
            call=client.call(solve_payload(r['problem'],attempt),'solve/'+r['sample_id']+'/'+str(attempt))
            text=content(call)
            answer_tokens=len(tokenizer.encode(text,add_special_tokens=False))
            total=len(tokenizer.apply_chat_template([
                {'role':'system','content':'Please reason step by step, and put your final answer within \\boxed{}.'},
                {'role':'user','content':r['problem']},{'role':'assistant','content':text}],tokenize=True,add_generation_prompt=False))
            finish=call['response']['choices'][0].get('finish_reason')
            structure=(text.count('<think>')==text.count('</think>')==1 and text.find('<think>')<text.find('</think>')
                and '\\boxed{' in text[text.find('</think>')+8:])
            verification=check_math(text,reference['reference_answer'])
            quality=quality_signals(text)
            accepted=(finish=='stop' and structure and 500<=answer_tokens<=1500 and total<=3800
                and verification['correct'] and quality['automatic_pass'])
            entry={'attempt':attempt,'answer_tokens':answer_tokens,'total_tokens':total,'finish_reason':finish,
                'structure_ok':structure,'math_verify':verification,'quality':quality,'auto_accepted':accepted,
                'solution':text,'request_id':call['response'].get('id')}
            attempts.append(entry)
            if accepted:chosen=entry;break
        final=chosen or attempts[-1]
        # An independent model critiques the visible solution; it gets no gold or source solution.
        review_call=client.call(review_payload(r['problem'],final['solution']),'review/'+r['sample_id']+'/'+sha(final['solution']))
        try:review=parsed(review_call)
        except Exception:review={'verdict':'uncertain','issues':['review_schema_error']}
        review_ok=review.get('verdict')=='pass' and review.get('missing_information') is False and isinstance(review.get('checked_steps'),list) and bool(review['checked_steps']) and review.get('issues')==[]
        raw_check=check_math(reference['raw_solution'],reference['reference_answer'])
        return {'sample_id':r['sample_id'],'problem_type':reference['selection']['problem_type'],
            'original_answer_tokens':reference['selection']['trajectory']['answer_tokens'],
            'original_math_verify':raw_check,'attempts':attempts,'selected_attempt':final['attempt'],
            'automatic_pass':chosen is not None,'process_review':review,'process_review_kind':'independent_model_not_human',
            'silver_pair_eligible':chosen is not None and review_ok and raw_check['correct'],
            'human_review':'not_performed','formal_training_approved':False}
    solutions=parallel(requests,generate,'solutions',workers=3)
    dump(REPORT/'solution_results.json',solutions)
    dump(REPORT/'balance_after.json',client.balance())
    pairs=[]
    for r in solutions:
        if not r.get('silver_pair_eligible'):continue
        entry=next(x for x in r['attempts'] if x['attempt']==r['selected_attempt'])
        ref=refs[r['sample_id']]
        pairs.append({'sample_id':r['sample_id'],'problem':ref['problem'],'reference_answer':ref['reference_answer'],
            'raw_solution':ref['raw_solution'],'compact_solution':entry['solution'],
            'status':'silver_candidate_pending_final_review','generation_mode':'blind_independent_compact_solution'})
    dump(REPORT/'paired_silver_candidates.json',pairs)
    for path,h in m['input_hashes'].items():assert file_sha(ROOT/path)==h,path
    summary={'difficulty_gate_passed':gate_ok,'solution_n':len(solutions),
        'automatic_pass':sum(r.get('automatic_pass',False) for r in solutions),'silver_pair_candidates':len(pairs),
        'api_requests_reserved':len(client.ledger),'reserved_usd':sum(r['reserved_usd'] for r in client.ledger),
        'usage_price_peak_usd':sum(r.get('usage_cost_peak_usd',0) for r in client.ledger),
        'formal_training_approved':False,'frozen_inputs_unchanged':True,'paid_api':True}
    dump(REPORT/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');args=parser.parse_args()
    run(args.live)
