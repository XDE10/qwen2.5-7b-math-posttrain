"""Local v3 cache reaudit, explicit scope pool, frozen independent v4 pilot."""
import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from src.synthesis.compact_v4 import VERSION,SYSTEM,KNOWN_WRONG_OUTPUTS,starts,clean_source,media_class,source_triage,serialize_response,payload,admission
from src.synthesis.deepseek_pilot import PilotClient,dump,sha
from scripts.compact_distillation_v3 import tokenizer,estimate_reserve,PILOT_QUOTAS
from scripts.compact_distillation_v2 import file_sha,jsonl,read,write_jsonl,ROOT
from src.data.screening_io import ScreeningSimilarityIndex
from scripts.math_verify_v3_strict_worker import relation_answer

DATA=ROOT/'data/compact_distillation_v4_20261024';REPORT=ROOT/'reports/compact_distillation_v4_20261024'
POOL=ROOT/'data/project1_textonly_v4'


def typed_check(text,reference):
    try:
        p=subprocess.run([sys.executable,'-m','scripts.math_verify_v3_strict_worker'],input=json.dumps({'output':text,'reference':reference}),text=True,encoding='utf-8',capture_output=True,timeout=20,cwd=ROOT)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:return {'correct':False,'status':'verifier_error','error_type':type(e).__name__}


def identity():
    paths=['src/synthesis/compact_v4.py','scripts/compact_distillation_v4.py','src/synthesis/compact_v3.py',
        'src/synthesis/compact_v2.py','src/synthesis/compact_format_review.py','src/synthesis/deepseek_pilot.py',
        'scripts/math_verify_v3_strict_worker.py','scripts/math_verify_pilot_worker.py',
        'data/project1_textonly_v3/manifest.json','data/math_manifest.json',
        'reports/compact_distillation_v3_20261023/results.json','reports/data_human_review_pack/metadata.json']
    paths += [v['path'] for v in read(ROOT/'data/math_manifest.json')['files'].values()]
    paths += ['data/tokenizers/Qwen2.5-7B-Instruct/'+n for n in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')]
    return {p:file_sha(ROOT/p) for p in paths}


def build_pool():
    POOL.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    if (POOL/'manifest.json').exists():return
    candidates=[];audits=[]
    for r in jsonl(ROOT/'data/project1_textonly_v3/native_candidates.jsonl'):
        a=source_triage(r);audits.append(a)
        if a['automatic_pool_eligible']:
            candidates.append({**r,'problem':a['cleaning']['problem'],'v4_source_triage':a,'training_approved':False})
    write_jsonl(POOL/'native_candidates.jsonl',candidates);write_jsonl(POOL/'source_triage.jsonl',audits)
    media=[{'sample_id':r['sample_id'],'media_class':media_class(r['problem'])} for r in jsonl(ROOT/'data/math/math_train.jsonl')]
    write_jsonl(POOL/'math_train_media_audit.jsonl',media)
    summary={'input_pool':len(audits),'automatic_scope_candidates':len(candidates),
        'quality_isolated':sum(bool(a['quality_isolation_reasons']) for a in audits),
        'experiment_scope_excluded':sum(bool(a['experiment_scope_exclusions']) for a in audits),
        'manual_review':sum(bool(a['manual_review_flags']) for a in audits),
        'cleaned_questions':sum(bool(a['cleaning']['edits']) for a in audits),
        'quality_reasons':dict(Counter(x for a in audits for x in a['quality_isolation_reasons'])),
        'scope_reasons':dict(Counter(x for a in audits for x in a['experiment_scope_exclusions'])),
        'box_categories':dict(Counter(a['boxes']['kind'] for a in audits)),
        'math_train_media_classes':dict(Counter(r['media_class'] for r in media)),
        'length_is_not_difficulty':True,'human_review':'pending','training_approved':False}
    dump(POOL/'summary.json',summary)
    dump(POOL/'manifest.json',{'input_identity':identity(),'files':{p.name:file_sha(p) for p in POOL.glob('*.jsonl')},'training_approved':False})
    print('V4 source pool',json.dumps(summary,ensure_ascii=False),flush=True)


def reaudit():
    build_pool()
    if (REPORT/'cache_reaudit_summary.json').exists():return
    old=read(ROOT/'reports/compact_distillation_v3_20261023/results.json')
    refs={r['sample_id']:r for r in read(ROOT/'data/compact_distillation_v3_20261023/verifier_references.json')}
    ai={r['sample_id']:r for r in read(ROOT/'reports/compact_distillation_v3_20261023/ai_process_review.json')['candidate_reviews']}
    tok=tokenizer();result=[]
    def one(r):
        ref=refs[r['sample_id']];p=serialize_response(r['raw_content'])
        math=typed_check(p['text'],ref['reference_answer']) if p['text'] else {'correct':False,'status':'no_parseable_output'}
        a=admission(p,ref['problem'],r['finish_reason'],tok,math)
        wrong=starts(r['sample_id'],KNOWN_WRONG_OUTPUTS);triage=source_triage(ref)
        airow=ai.get(r['sample_id'],{});known_ai=airow.get('compact_process_decision',airow.get('decision','not_reviewed'))
        return {'sample_id':r['sample_id'],'raw_content_sha256':sha(r['raw_content']),
            'old_schema_valid':r['parsed']['valid'],'v4_parsed':p,'v4_admission':a,
            'parsing_recovered':not r['parsed']['valid'] and p['valid'],
            'source_triage':triage,'known_mathematical_error_isolated':wrong,'prior_ai_process_status':known_ai,
            'candidate_status':'mathematical_error_isolated' if wrong else 'source_or_scope_review' if not triage['automatic_pool_eligible'] else 'automatic_pending_process_and_human' if a['accepted_candidate'] else 'not_automatically_admitted',
            'request_problem_unchanged':True,'posthoc_not_independent_pilot':True,'training_approved':False}
    with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(one,old))
    dump(REPORT/'cache_reaudit_v3.json',result)
    summary={'v3_records':len(result),'parsing_recovered':sum(r['parsing_recovered'] for r in result),
        'parsing_recovered_ids':[r['sample_id'] for r in result if r['parsing_recovered']],
        'known_mathematical_error_isolated':sum(bool(r['known_mathematical_error_isolated']) for r in result),
        'mathematical_error_ids':[r['sample_id'] for r in result if r['known_mathematical_error_isolated']],
        'v4_schema_valid':sum(r['v4_parsed']['valid'] for r in result),'v4_automatic_output_candidates':sum(r['v4_admission']['accepted_candidate'] for r in result),
        'candidate_statuses':dict(Counter(r['candidate_status'] for r in result)),
        'api_calls':0,'interpretation':'post-hoc cache robustness check; frozen v3 14/30 and 12 common pairs unchanged; no API repair or reference-fed generation'}
    # Legacy text outputs did not request JSON: applying numeric JSON conversion to
    # them would be a fabricated improvement. Preserve their earlier format audit.
    legacy=read(ROOT/'reports/deepseek_pilot_20261023/solution_results.json')
    summary['legacy_text_records']=sum(len(r['attempts']) for r in legacy)
    summary['legacy_text_json_repair']='not_applicable; prior deterministic text-format audit preserved'
    dump(REPORT/'cache_reaudit_summary.json',summary);print('Cache reaudit',json.dumps(summary,ensure_ascii=False),flush=True)


def prepare():
    reaudit();DATA.mkdir(exist_ok=True)
    if (DATA/'manifest.json').exists():
        m=read(DATA/'manifest.json');assert m['input_identity']==identity(),'Frozen identity changed'
        for name,h in m['files'].items():assert file_sha(DATA/name)==h
        return
    previous=ScreeningSimilarityIndex(.85);oldids=set()
    for name in ['data/deepseek_pilot_20261023/verifier_references.json','data/compact_distillation_v2_20261023/verifier_references.json',
        'data/compact_distillation_v3_20261023/verifier_references.json','data/deepseek_pilot_20261023/anchors.json',
        'data/deepseek_pilot_20261023/calibration_dev.json','data/deepseek_pilot_20261023/frozen_acceptance.json']:
        for r in read(ROOT/name):previous.add(r);oldids.add(r['sample_id'])
    for r in read(ROOT/'reports/data_human_review_pack/metadata.json')['selected']:
        if r['category'].startswith('p1_'):oldids.add(r['sample_id'])
    oldids.update(r['sample_id'] for r in read(ROOT/'reports/compact_distillation_v3_20261023/source_review_flags.json')['rows'])
    # Protect heldout after cleaning, and intra-pilot lexical duplicates.
    heldout=ScreeningSimilarityIndex(.85)
    for k in ['math_dev','math500_test','gsm8k_test']:
        for r in jsonl(ROOT/read(ROOT/'data/math_manifest.json')['files'][k]['path']):heldout.add(r)
    selected=[];pool=jsonl(POOL/'native_candidates.jsonl')
    for kind,n in PILOT_QUOTAS.items():
        chosen=[]
        for r in sorted([r for r in pool if r['problem_type']==kind],key=lambda r:sha('independent-v4/'+r['sample_id'])):
            if r['sample_id'] in oldids or previous.match(r['problem']) or heldout.match(r['problem']):continue
            # Declared verifier scope: relation-valued answers require task-aware review.
            if relation_answer(str(r['reference_answer'])):continue
            selected.append(r);chosen.append(r);previous.add(r)
            if len(chosen)==n:break
        assert len(chosen)==n,kind
    assert len(selected)==30
    reserve=sum(estimate_reserve(payload(r['problem'])) for r in selected);assert reserve<.095
    dump(DATA/'blind_requests.json',[{k:r[k] for k in ['sample_id','problem','source_revision','split']} for r in selected])
    dump(DATA/'verifier_references.json',selected)
    policy={'version':VERSION,'teacher_alias':'deepseek-flash','documented_model_version':'DeepSeek-V4.1-Flash-1017',
        'price_verified_url':'https://api-docs.deepseek.com/quick_start/pricing/','price_verified_date':'2026-10-24',
        'peak_prices_per_million_tokens':{'input_miss':.3,'input_hit':.006,'output':1.2},
        'thinking':{'type':'disabled'},'request_example':payload('PROBLEM_ONLY'),'prompt_sha256':sha(SYSTEM),
        'budget_ceiling_usd':.095,'max_requests':30,'workers':4,'max_attempts_per_question':1,
        'preflight_worst_case_reserve_usd':reserve,'no_auto_bulk':True,'no_rule_tuning_after_dispatch':True,
        'admission_window_qwen_tokens':[150,1500],'context_max':3800,
        'answer_verifier':'Math-Verify 0.9 + conservative relation/multiple-distinct-box type guard',
        'scope':'open solution questions; isolate damaged/ambiguous sources; known elementary tasks excluded; not a calibrated medium/hard dataset',
        'newness':'exclude prior generation, anchors/calibration, v3, human pack IDs, 79 flagged IDs; Jaccard .85 prior/heldout/within pilot',
        'gate':{'automatic_candidate_rate_min':.7,'schema_rate_min':.9,'process_review_required':True,'human_approval_before_bulk':True},
        'training_approved':False}
    dump(DATA/'policy.json',policy)
    dump(DATA/'manifest.json',{'input_identity':identity(),'pool_manifest_sha256':file_sha(POOL/'manifest.json'),
        'files':{p.name:file_sha(p) for p in DATA.iterdir() if p.is_file() and p.name!='manifest.json'},'training_approved':False})
    print('V4 frozen independent30 reserve',reserve,flush=True)


def run(live=False,key_prompt=False):
    prepare()
    if (REPORT/'results.json').exists():raise FileExistsError('Pilot completed; do not redispatch')
    if not live:print('30 frozen payloads; no network');return
    key=os.environ.get('DEEPSEEK_API_KEY') or (getpass.getpass('API key (hidden): ') if key_prompt else '')
    if not key:raise RuntimeError('Missing API key; zero requests')
    policy=read(DATA/'policy.json');reqs=read(DATA/'blind_requests.json');refs={r['sample_id']:r for r in read(DATA/'verifier_references.json')}
    client=PilotClient(REPORT/'api_cache',key,budget=.095,max_requests=30)
    start=time.monotonic();tok=tokenizer();result=[]
    def one(r):
        assert set(r)=={'sample_id','problem','source_revision','split'}
        c=client.call(payload(r['problem']),'v4/solve/'+r['sample_id']+'/1');choice=c['response']['choices'][0]
        raw=choice['message'].get('content') or '';p=serialize_response(raw);ref=refs[r['sample_id']]
        verified=typed_check(p['text'],ref['reference_answer']) if p['text'] else {'correct':False,'status':'no_parseable_output'}
        rawver=typed_check(ref['solution'],ref['reference_answer']);a=admission(p,r['problem'],choice.get('finish_reason'),tok,verified)
        return {'sample_id':r['sample_id'],'problem_type':ref['problem_type'],'raw_content':raw,'parsed':p,'admission':a,
            'original_math_verify':rawver,'common_candidate':a['accepted_candidate'] and rawver.get('correct',False),
            'request_id':c['response'].get('id'),'returned_model':c['response'].get('model'),'finish_reason':choice.get('finish_reason'),
            'usage':c['response'].get('usage',{}),'latency_seconds':c['latency_seconds'],'process_review':'pending','human_review':'pending','training_approved':False}
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(one,r):r for r in reqs}
        for future in as_completed(fs):
            r=fs[future]
            try:row=future.result()
            except Exception as e:row={'sample_id':r['sample_id'],'error_type':type(e).__name__,'common_candidate':False,'training_approved':False}
            result.append(row);dump(REPORT/'partial_results.json',result);print(f'V4 independent pilot {len(result)}/30',flush=True)
    result.sort(key=lambda r:r['sample_id']);dump(REPORT/'results.json',result)
    pairs=[{'sample_id':r['sample_id'],'problem_type':r['problem_type'],'problem':refs[r['sample_id']]['problem'],
        'reference_answer':refs[r['sample_id']]['reference_answer'],'raw_solution':refs[r['sample_id']]['solution'],
        'compact_solution':r['admission']['text'],'answer_tokens':r['admission']['answer_tokens'],
        'process_review':'pending','human_review':'pending','training_approved':False} for r in result if r['common_candidate']]
    dump(REPORT/'common_candidates.json',pairs)
    s={'questions':30,'completed':sum('admission' in r for r in result),'generation_requests':len(client.ledger),
        'schema_valid':sum(r.get('parsed',{}).get('valid',False) for r in result),
        'answer_verified':sum(r.get('admission',{}).get('math_verify',{}).get('correct',False) for r in result),
        'automatic_candidates':sum(r.get('admission',{}).get('accepted_candidate',False) for r in result),'common_pairs':len(pairs),
        'numeric_normalizations':sum(bool(r.get('parsed',{}).get('normalization_operations')) for r in result),
        'failure_signals':dict(Counter(e for r in result for e in r.get('admission',{}).get('errors',[]))),
        'wall_seconds':time.monotonic()-start,'reserved_usd':sum(r['reserved_usd'] for r in client.ledger),
        'usage_estimated_peak_usd':sum(r.get('usage_cost_peak_usd',0) for r in client.ledger),
        'returned_models':dict(Counter(r.get('returned_model') for r in result)),
        'engineering_gate_passed':sum(r.get('admission',{}).get('accepted_candidate',False) for r in result)>=21 and sum(r.get('parsed',{}).get('valid',False) for r in result)>=27,
        'process_accuracy':None,'training_approved':False,'bulk_generation_enabled':False,'human_review':'pending',
        'comparison_limits':'different heldout development topics and explicit scope; v3->v4 rates are not causal parser or teacher improvements'}
    dump(REPORT/'summary.json',s)
    dump(REPORT/'frozen_run_manifest.json',{'input_identity':identity(),'policy_manifest_sha256':file_sha(DATA/'manifest.json'),
        'files':{str(p.relative_to(REPORT)):file_sha(p) for p in REPORT.rglob('*') if p.is_file() and p.name!='frozen_run_manifest.json'}})
    print(json.dumps(s,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pool','reaudit','prepare','run']);p.add_argument('--live',action='store_true');p.add_argument('--key-prompt',action='store_true');a=p.parse_args()
    if a.action=='pool':build_pool()
    elif a.action=='reaudit':reaudit()
    elif a.action=='prepare':prepare()
    else:run(a.live,a.key_prompt)
