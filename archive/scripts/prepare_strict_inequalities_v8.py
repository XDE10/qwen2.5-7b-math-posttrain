import json,re,sys,subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from scripts.strict_v8_common import *
from scripts.compact_distillation_v2 import replay
from scripts.refill_project1_native_v5 import source_filter
from scripts.reaudit_typed_inequalities_v8 import PROFILES,EXCLUSIONS
from src.data.screening_io import ScreeningSimilarityIndex
def infer_profile(r):
    prefix=r['sample_id'][:8]
    if prefix in EXCLUSIONS:return None
    if prefix in PROFILES:return PROFILES[prefix]
    ref=r['reference_answer'].strip();p=r['problem']
    if re.fullmatch(r'[A-Ea-e]',ref):return None
    relational=bool(re.search(r'[<>=≤≥]|\\(?:le|ge|in)\b|\\(?:leq|geq)',ref))
    setlike=ref.startswith(('[','(',r'\{')) or ',' in ref
    if not relational and not setlike:return {'kind':'scalar'}
    scrub=ref
    for op in [r'\leqslant',r'\geqslant',r'\leq',r'\geq',r'\le',r'\ge']:scrub=scrub.replace(op,' ')
    scrub=re.sub(r'\\[A-Za-z]+|_[{]?[0-9]+[}]?',' ',scrub)
    names=set(re.findall(r'(?<![A-Za-z])[a-z](?![A-Za-z])',scrub))
    if not names:
        blocks=re.findall(r'\$+([^$]+)\$+|\\\[([\s\S]*?)\\\]',p)
        chars=re.sub(r'\\[A-Za-z]+',' ',''.join(''.join(x) for x in blocks));names=set(re.findall(r'(?<![A-Za-z])[a-z](?![A-Za-z])',chars))
    if len(names)!=1:return None
    variable=next(iter(names));domain='real'
    if re.search(r'(?i)\b(?:integers?|natural numbers?)\b',p):
        # Integer restrictions on another variable must never narrow this answer.
        prose=re.sub(r'\$|\\[()\[\]]',' ',p);prose=re.sub(r'\s+',' ',prose)
        bound=re.search(r'(?i)\b(?:integers?|natural numbers?)\s+'+re.escape(variable)+r'\b|\binteger values? of\s+'+re.escape(variable)+r'\b',prose)
        if not bound:return None
        domain='integer'
    return {'kind':'one_variable_solution_set','domain':domain,'variable':variable,'enumerated_roots':bool(re.search(r'[a-z]_\{?1',ref))}
def worker(r):
    from src.synthesis.typed_answers_v8 import check_roles
    profile=infer_profile(r)
    if not profile:return {'passed':False,'reason':'unsupported_or_semantically_incomplete_profile'}
    if profile['kind']=='scalar':
        from scripts.math_verify_v3_strict_worker import check
        text=r['solution'];zone=text.split('</think>',1)[1] if text.count('</think>')==1 and text.count('<think>')==1 else text
        checked=check({'output':zone,'reference':r['reference_answer']})
    else:checked=check_roles(r['solution'],r['reference_answer'],profile)
    return {'passed':checked.get('correct',False),'profile':profile,'raw_answer_check':checked}
def run():
    out=DATA/'inequalities';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return
    used=set();held=ScreeningSimilarityIndex(.85)
    paths=['data/project1_batch500_20261024/verifier_references.jsonl','data/topup_v6_20261024/project1/native_new1500.jsonl','data/topup_v6_20261024/project1_supplement/native_candidates.jsonl']
    paths += [str(p.relative_to(ROOT)) for p in (ROOT/'data').glob('compact_distillation_v*/verifier_references.json')]
    paths += ['data/deepseek_pilot_20261023/'+n+'.json' for n in ['verifier_references','anchors','calibration_dev','frozen_acceptance']]
    for path in paths:
        rs=rows(ROOT/path) if path.endswith('.jsonl') else read(ROOT/path)
        for r in rs:used.add(r['sample_id']);held.add(r)
    for path in ['data/math/math_dev.jsonl','data/math/math500_test.jsonl','data/math/gsm8k_test.jsonl','data/topup_v6_20261024/release/project2_math_dev.jsonl','data/topup_v6_20261024/release/project2_code_dev.jsonl']:
        for r in rows(ROOT/path):used.add(r['sample_id']);held.add(r)
    for folder in ['project2','project2_dev_reserve']:
        for r in rows(ROOT/'data/topup_v6_20261024'/folder/'candidates.jsonl'):
            if r['split']=='dev':used.add(r['sample_id']);held.add(r)
    proposals=[]
    with (ROOT/'reports/openr1_full_audit_final_20261023/questions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            q=json.loads(line)
            if q['problem_type']!='Inequalities' or q['question_hash'] in used or q.get('flag_conflict') or q.get('evaluation_overlap') or q.get('internal_near_duplicate'):continue
            ts=[t for t in q['trajectories'] if t.get('verified') and t.get('complete') and t.get('answer_tokens') is not None]
            if not ts:continue
            n=min(t['answer_tokens'] for t in ts)
            if not 1200<=n<=3800:continue
            valid=[t for t in ts if t['answer_tokens']==n and t.get('quality',{}).get('automatic_pass') and t.get('total_tokens',99999)<=4096]
            if not valid or held.match(q['problem']):continue
            proposals.append({'question_hash':q['question_hash'],'problem_type':'Inequalities','source':q['source'],'trajectory':valid[0],'linked_train_level':q.get('linked_train_level')})
    print('V8 raw inequality proposals',len(proposals),flush=True);replayed={r['sample_id']:r for r in replay(proposals)};audit=[];selected=[];prior_native={r['sample_id'] for r in rows(ROOT/'data/project1_native_v5/native_candidates.jsonl')}
    def bounded_probe(row):
        try:
            proc=subprocess.run([sys.executable,'-m','scripts.prepare_strict_inequalities_v8','worker'],input=json.dumps(row),text=True,encoding='utf-8',capture_output=True,timeout=25,cwd=ROOT);return json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as e:return {'passed':False,'reason':'bounded_raw_probe_error','error_type':type(e).__name__}
    pool=ThreadPoolExecutor(max_workers=4);pending={}
    for sid,r in replayed.items():
        row={**r,'solution':r['raw_solution']};tri=source_filter(row)['v4_triage']
        if infer_profile(row) and not tri['quality_isolation_reasons'] and not tri['experiment_scope_exclusions']:pending[sid]=pool.submit(bounded_probe,row)
    for i,p in enumerate(proposals,1):
        r=replayed[p['question_hash']];row={'sample_id':r['sample_id'],'problem':r['problem'],'original_problem':r['problem'],'solution':r['raw_solution'],'reference_answer':r['reference_answer'],'source_revision':r['source_revision'],'source_position':p['trajectory'],'source':p['source'],'problem_type':'Inequalities','split':'train','training_approved':False,'source_answer_tokens':p['trajectory']['answer_tokens'],'source_total_tokens':p['trajectory']['total_tokens'],'prior_native_only_not_API':r['sample_id'] in prior_native}
        a=source_filter(row);tri=a['v4_triage'];blocking=tri['quality_isolation_reasons']+tri['experiment_scope_exclusions']+[x for x in tri['manual_review_flags'] if x not in ['malformed_boxes','distinct_boxes_components_collection_or_intermediate_requires_review']]
        blocking += [x for x in a['reasons'] if x!='v4_scope_or_source_or_structure_review']
        result={'sample_id':r['sample_id'],'blocking_source_flags':blocking}
        if not blocking and not held.match(a['problem']):
            row['problem']=a['problem']
            checked=pending[r['sample_id']].result() if r['sample_id'] in pending else {'passed':False,'reason':'no_supported_profile'}
            result.update(checked)
            if checked.get('passed'):selected.append({**row,'answer_profile':checked['profile'],'raw_answer_check':checked['raw_answer_check'],'source_scope_v8_waivers':tri['manual_review_flags']});held.add(row)
        audit.append(result)
        if i%10==0:write(out/'source_audit.jsonl',audit);write(out/'candidates.jsonl',selected);print('V8 source probes',i,'/',len(proposals),'passed',len(selected),flush=True)
    pool.shutdown(wait=True)
    selected.sort(key=lambda r:(r['sample_id'][:8] in PROFILES,stable('v8ineq/'+r['sample_id'])))
    write(out/'source_audit.jsonl',audit);write(out/'candidates.jsonl',selected)
    dump(out/'manifest.json',{'candidate_count':len(selected),'source_proposals':len(proposals),'type':'Inequalities','raw_tokens':[1200,3800],'full_sft_max':4096,'compact_tokens':[150,1500],'already_API_sampled_ids_excluded':True,'prior_native_only_not_API_included':sum(r['prior_native_only_not_API'] for r in selected),'development_cases':sum(r['sample_id'][:8] in PROFILES for r in selected),'files':{p.name:sha(p) for p in out.glob('*.jsonl')},'verifier_sha':sha(ROOT/'src/synthesis/typed_answers_v8.py')});print(read(out/'manifest.json'),flush=True)
if __name__=='__main__':
    if len(sys.argv)>1:print(json.dumps(worker(json.load(sys.stdin))))
    else:run()
