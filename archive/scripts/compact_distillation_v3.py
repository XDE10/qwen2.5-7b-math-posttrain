"""Data-only v3 pool, frozen 30-question pilot; bulk generation is disabled."""
import argparse
import getpass
import hashlib
import json
import os
import time
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from transformers import AutoTokenizer
from src.synthesis.compact_v3 import VERSION,SYSTEM,screen_problem,payload,serialize_response,admission
from src.synthesis.deepseek_pilot import PilotClient,dump,sha,PRICES
from src.data.screening_io import ScreeningSimilarityIndex
from scripts.analyze_openr1_candidates import choose_for_budget
from scripts.compact_distillation_v2 import replay,file_sha,read,write_jsonl,ROOT,jsonl,batch_verify

DATA=ROOT/'data/project1_textonly_v3'
PILOT=ROOT/'data/compact_distillation_v3_20261023'
REPORT=ROOT/'reports/compact_distillation_v3_20261023'
QUOTAS={'Algebra':525,'Geometry':300,'Number Theory':300,'Combinatorics':225,'Inequalities':100,'Logic and Puzzles':50}
PILOT_QUOTAS={'Algebra':9,'Geometry':6,'Number Theory':6,'Combinatorics':5,'Inequalities':3,'Logic and Puzzles':1}


def tokenizer():return AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
def identity():
    paths=['src/synthesis/compact_v3.py','src/synthesis/compact_format_review.py',
        'src/synthesis/compact_v2.py','src/synthesis/deepseek_pilot.py','scripts/compact_distillation_v3.py',
        'scripts/math_verify_pilot_worker.py','data/math_manifest.json',
        'reports/openr1_full_audit_final_20261023/manifest.json',
        'reports/openr1_full_audit_final_20261023/questions.jsonl',
        'reports/project1_sft_v2_plan_20261023/coverage_proposal.json',
        'data/deepseek_pilot_20261023/manifest.json','data/compact_distillation_v2_20261023/manifest.json',
        'data/project1_sft/sft_train.jsonl']
    paths += [v['path'] for v in read(ROOT/'data/math_manifest.json')['files'].values()]
    paths += ['data/tokenizers/Qwen2.5-7B-Instruct/'+n for n in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','config.json')]
    return {p:file_sha(ROOT/p) for p in paths}


def validate_manifest(folder):
    m=read(folder/'manifest.json')
    for p,h in m['files'].items():assert file_sha(folder/p)==h,p
    return m


def build_pool():
    if (DATA/'manifest.json').exists():
        m=validate_manifest(DATA);assert m['input_identity']==identity(),'Use new version for changed inputs/code'
        print('Text-only v3 candidate pool sealed and verified');return
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    ident=identity();prior=read(ROOT/'reports/project1_sft_v2_plan_20261023/coverage_proposal.json')
    priorids={r['question_hash'] for r in prior};pools=defaultdict(list);rejects=[];stats=Counter()
    for q in jsonl(ROOT/'reports/openr1_full_audit_final_20261023/questions.jsonl'):
        t=choose_for_budget(q,3000)
        if not t:continue
        assert 'evaluation_overlap' in q and 'internal_near_duplicate' in q
        if q['evaluation_overlap'] or q['internal_near_duplicate']:continue
        stats['source_eligible_3000']+=1;c=screen_problem(q['problem'])
        if not c['screen_pass']:
            rejects.append({'sample_id':q['question_hash'],'problem_type':q['problem_type'],
                'was_in_old_1500':q['question_hash'] in priorids,'reasons':c['screen_reasons'],
                'problem':q['problem'],'semantics':'rule isolation, not proof of actual missing information'})
            for reason in c['screen_reasons']:stats['rejected_'+reason]+=1
            continue
        pools[q['problem_type']].append({'question_hash':q['question_hash'],'problem_type':q['problem_type'],
            'source':q['source'],'trajectory':t,'source_cleaning':c,'linked_train_level':q.get('linked_train_level'),
            'rl_train_overlap':q.get('rl_train_overlap'),'source_verification_ratio':q.get('verification_ratio')})
    evalindex=ScreeningSimilarityIndex(.85);seen=ScreeningSimilarityIndex(.85)
    mm=read(ROOT/'data/math_manifest.json')
    for key in ('math_dev','math500_test','gsm8k_test'):
        assert file_sha(ROOT/mm['files'][key]['path'])==mm['files'][key]['sha256']
        for r in jsonl(ROOT/mm['files'][key]['path']):evalindex.add(r)
    chosen=[];shortfalls={}
    for kind,count in QUOTAS.items():
        ordered=sorted(pools[kind],key=lambda r:(r['question_hash'] not in priorids,sha('textonly-v3/'+r['question_hash'])))
        picked=[]
        for r in ordered:
            text=r['source_cleaning']['problem'];hit=evalindex.match(text) or seen.match(text)
            if hit:continue
            picked.append(r);seen.add({'sample_id':r['question_hash'],'problem':text})
            if len(picked)==count+30:break
        chosen+=picked
    raw=replay(chosen);byid={r['sample_id']:r for r in raw};tok=tokenizer();records=[]
    final_seen=ScreeningSimilarityIndex(.85);raw_screen_rejects=[];final_counts=Counter()
    for selected in chosen:
        kind=selected['problem_type']
        if final_counts[kind]>=QUOTAS[kind]:continue
        r=byid[selected['question_hash']];original=r['problem']
        # Normalized duplicate rows can have different exact source wording.
        # Screen the ACTUAL selected trajectory's problem, not its representative.
        cleaned=screen_problem(original)
        if not cleaned['screen_pass'] or evalindex.match(cleaned['problem']) or final_seen.match(cleaned['problem']):
            raw_screen_rejects.append({'sample_id':r['sample_id'],'reasons':cleaned['screen_reasons'],
                'note':'actual selected source row wording screened again'});continue
        problem=cleaned['problem'];answer_tokens=len(tok.encode(r['raw_solution'],add_special_tokens=False))
        total=len(tok.apply_chat_template([
            {'role':'system','content':r'Please reason step by step, and put your final answer within \boxed{}.'},
            {'role':'user','content':problem},{'role':'assistant','content':r['raw_solution']}],tokenize=True,add_generation_prompt=False))
        assert 500<=answer_tokens<=3000 and total<=3800
        records.append({'sample_id':r['sample_id'],'problem':problem,'original_problem':original,
            'solution':r['raw_solution'],'reference_answer':r['reference_answer'],'problem_type':selected['problem_type'],
            'source_revision':r['source_revision'],'source_position':selected['trajectory'],
            'source_cleaning':cleaned,'answer_tokens':answer_tokens,'total_tokens':total,
            'source_verified':True,'local_math_verify_rechecked':False,
            'semantic_self_containedness':'pending','semantic_review':'pending','training_approved':False,
            'linked_train_level':selected['linked_train_level'],'split':'train'})
        final_seen.add({'sample_id':r['sample_id'],'problem':problem});final_counts[kind]+=1
    shortfalls={k:QUOTAS[k]-final_counts[k] for k in QUOTAS}
    write_jsonl(DATA/'native_candidates.jsonl',records);write_jsonl(DATA/'screened_out.jsonl',rejects)
    dump(DATA/'actual_source_row_rechecks.json',raw_screen_rejects)
    summary={'version':VERSION,'target':1500,'actual':len(records),'quota_shortfalls':shortfalls,
        'per_type':dict(Counter(r['problem_type'] for r in records)),
        'retained_from_old_1500':sum(r['sample_id'] in priorids for r in records),
        'replacement_count':sum(r['sample_id'] not in priorids for r in records),
        'old_1500_rule_isolated':sum(r['was_in_old_1500'] for r in rejects),
        'full_eligible_screening':dict(stats),'heldout_and_internal_match_after_cleaning':0,
        'training_approved':False,'claim':'passes declared text-only screening; not 100 percent semantic self-containedness'}
    dump(DATA/'summary.json',summary)
    dump(DATA/'manifest.json',{'input_identity':ident,'files':{p.name:file_sha(p) for p in DATA.iterdir() if p.is_file() and p.name!='manifest.json'},
        'training_approved':False,'quota_shortfalls':shortfalls})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def estimate_reserve(p):
    ip,_,op=PRICES[p['model']]
    return ((len(json.dumps(p['messages'],ensure_ascii=False).encode())+1024)*ip+p['max_tokens']*op)/1e6


def prepare():
    build_pool()
    if (PILOT/'manifest.json').exists():
        m=validate_manifest(PILOT);assert m['input_identity']==identity(),'Code/input changed'
        print('V3 independent 30-question policy verified');return
    PILOT.mkdir(parents=True,exist_ok=True)
    old=[]
    for p in ['data/deepseek_pilot_20261023/verifier_references.json','data/compact_distillation_v2_20261023/verifier_references.json',
              'data/deepseek_pilot_20261023/anchors.json','data/deepseek_pilot_20261023/calibration_dev.json','data/deepseek_pilot_20261023/frozen_acceptance.json']:
        old+=read(ROOT/p)
    previous=ScreeningSimilarityIndex(.85)
    for r in old:previous.add(r)
    pool=jsonl(DATA/'native_candidates.jsonl');selected=[]
    for kind,quota in PILOT_QUOTAS.items():
        candidates=sorted([r for r in pool if r['problem_type']==kind],key=lambda r:sha('independent-v3-pilot/'+r['sample_id']))
        chosen=[]
        for r in candidates:
            if previous.match(r['problem']) or r['source_cleaning']['edits']:continue
            chosen.append(r);previous.add({'sample_id':r['sample_id'],'problem':r['problem']})
            if len(chosen)==quota:break
        assert len(chosen)==quota,kind
        selected+=chosen
    selected.sort(key=lambda r:r['sample_id']);assert len(selected)==30
    reserve=sum(estimate_reserve(payload(r['problem'])) for r in selected)
    assert reserve<.095,f'Preflight worst-case reserve {reserve} exceeds pilot ceiling'
    requests=[{k:r[k] for k in ['sample_id','problem','source_revision','split']} for r in selected]
    dump(PILOT/'blind_requests.json',requests);dump(PILOT/'verifier_references.json',selected)
    policy={'version':VERSION,'model':'deepseek-flash','thinking':{'type':'disabled'},'prompt_sha256':sha(SYSTEM),
        'output_contract':'JSON finished solution and bare final_answer -> local think/boxed serialization',
        'max_generation_attempts':1,'max_requests':30,'workers':4,'reserved_budget_usd':.095,
        'preflight_worst_case_reserve_usd':reserve,'request_example':payload('PROBLEM_ONLY'),
        'admission':'correct Math-Verify answer; valid JSON; no long repetition; stop; 150-1500 Qwen tokens; full context<=3800',
        'engineering_gate':{'automatic_candidate_rate_min':.70,'json_format_rate_min':.90,
            'all_accepted_candidates_require_ai_process_review':True,'human_approval_required_before_bulk':True},
        'comparison':'prompt optimization only; no thinking-on arm; cannot infer on/off causal effect',
        'bulk_generation_enabled':False,'training_approved':False,'no_rule_tuning_after_dispatch':True,
        'independence':'30 new IDs/no Jaccard>=0.85 against 50+40 prior generation topics and 135 prior calibration topics',
        'max_tokens_note':'API 1400-token cap serves strict cost preflight; Qwen window is separately measured; truncations are failures'}
    dump(PILOT/'policy.json',policy)
    dump(PILOT/'manifest.json',{'input_identity':identity(),'native_pool_manifest_sha256':file_sha(DATA/'manifest.json'),
        'files':{p.name:file_sha(p) for p in PILOT.iterdir() if p.is_file() and p.name!='manifest.json'},
        'sample_counts':PILOT_QUOTAS,'old_question_matches':0,'training_approved':False})
    print(f'Frozen 30 new questions; worst-case reserve ${reserve:.6f} < $0.095',flush=True)


def run(live=False,key_prompt=False):
    prepare();REPORT.mkdir(parents=True,exist_ok=True)
    m=validate_manifest(PILOT);policy=read(PILOT/'policy.json');requests=read(PILOT/'blind_requests.json')
    assert all(set(r)=={'sample_id','problem','source_revision','split'} for r in requests)
    refs={r['sample_id']:r for r in read(PILOT/'verifier_references.json')}
    dump(REPORT/'blind_request_audit.json',{'hashes':{r['sample_id']:sha(payload(r['problem'])) for r in requests},
        'reference_or_original_solution_sent':False,'difficulty_and_review_requests':0})
    if not live:print('V3 dry-run: 30 payloads; no network');return
    key=os.environ.get('DEEPSEEK_API_KEY') or (getpass.getpass('API key (hidden): ') if key_prompt else '')
    if not key:raise RuntimeError('Missing API key; no dispatch')
    client=PilotClient(REPORT/'api_cache',key,budget=policy['reserved_budget_usd'],max_requests=30)
    start=time.monotonic();tok=tokenizer();results=[]
    def one(r):
        ref=refs[r['sample_id']]
        cached=client.call(payload(r['problem']),'v3/solve/'+r['sample_id']+'/1')
        choice=cached['response']['choices'][0];content=choice['message'].get('content') or ''
        parsed=serialize_response(content)
        from scripts.run_deepseek_pilot import check_math
        ver=check_math(parsed['text'],ref['reference_answer']) if parsed['text'] else {'correct':False,'status':'no_parseable_output'}
        rawver=check_math(ref['solution'],ref['reference_answer'])
        admitted=admission(parsed,r['problem'],choice.get('finish_reason'),tok,ver)
        return {'sample_id':r['sample_id'],'problem_type':ref['problem_type'],'raw_content':content,'parsed':parsed,
            'admission':admitted,'original_math_verify':rawver,'common_candidate':admitted['accepted_candidate'] and rawver.get('correct',False),
            'request_id':cached['response'].get('id'),'returned_model':cached['response'].get('model'),
            'finish_reason':choice.get('finish_reason'),'usage':cached['response'].get('usage',{}),
            'latency_seconds':cached['latency_seconds'],'ai_process_review':'pending','human_review':'pending','training_approved':False}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(one,r):r for r in requests}
        for f in as_completed(futures):
            r=futures[f]
            try:result=f.result()
            except Exception as e:result={'sample_id':r['sample_id'],'problem_type':refs[r['sample_id']]['problem_type'],
                'error_type':type(e).__name__,'status':'failed','common_candidate':False,'training_approved':False}
            results.append(result);dump(REPORT/'partial_results.json',results)
            print(f'V3 independent pilot {len(results)}/30',flush=True)
    results.sort(key=lambda r:r['sample_id']);dump(REPORT/'results.json',results)
    pairs=[]
    for r in results:
        if r['common_candidate']:
            ref=refs[r['sample_id']]
            pairs.append({'sample_id':r['sample_id'],'problem_type':r['problem_type'],'problem':ref['problem'],
                'reference_answer':ref['reference_answer'],'raw_solution':ref['solution'],'compact_solution':r['admission']['text'],
                'answer_tokens':r['admission']['answer_tokens'],'source_revision':ref['source_revision'],
                'ai_process_review':'pending','human_review':'pending','training_approved':False})
    dump(REPORT/'common_candidates.json',pairs)
    s={'questions':30,'completed':sum('admission' in r for r in results),'generation_requests':len(client.ledger),
        'answer_verified':sum(r.get('admission',{}).get('math_verify',{}).get('correct',False) for r in results),
        'json_format_pass':sum(r.get('parsed',{}).get('valid',False) for r in results),
        'automatic_candidates':sum(r.get('admission',{}).get('accepted_candidate',False) for r in results),
        'common_pairs':len(pairs),'wall_seconds':time.monotonic()-start,
        'reserved_usd':sum(r['reserved_usd'] for r in client.ledger),
        'usage_estimated_peak_usd':sum(r.get('usage_cost_peak_usd',0) for r in client.ledger),
        'request_states':dict(Counter(r['state'] for r in client.ledger)),
        'returned_models':dict(Counter(r.get('returned_model') for r in results if r.get('returned_model'))),
        'failure_signals':dict(Counter(e for r in results for e in r.get('admission',{}).get('errors',[]))),
        'per_type':{k:{'questions':n,'candidates':sum(r['problem_type']==k for r in pairs)} for k,n in PILOT_QUOTAS.items()},
        'bulk_generation_enabled':False,'human_approval':'pending','training_approved':False,
        'thinking_comparison':'not_run; this pilot tests an optimized final-written-solution JSON prompt'}
    s['engineering_gate_passed']=(s['automatic_candidates']/30>=.7 and s['json_format_pass']/30>=.9)
    dump(REPORT/'summary.json',s)
    dump(REPORT/'frozen_run_manifest.json',{'policy_manifest_sha256':file_sha(PILOT/'manifest.json'),
        'input_identity':identity(),'files':{str(p.relative_to(REPORT)):file_sha(p) for p in REPORT.rglob('*') if p.is_file() and p.name!='frozen_run_manifest.json'},
        'bulk_generation_enabled':False,'training_approved':False})
    print(json.dumps(s,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pool','prepare','run'])
    p.add_argument('--live',action='store_true');p.add_argument('--key-prompt',action='store_true');a=p.parse_args()
    if a.action=='pool':build_pool()
    elif a.action=='prepare':prepare()
    else:run(a.live,a.key_prompt)
