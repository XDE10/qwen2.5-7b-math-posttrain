"""Seal new dev BEFORE train; no rejected or historical train IDs recycled."""
import ast,re,hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from scripts.topup_common import *
from src.data.screening_io import ScreeningSimilarityIndex,source_files,parquet_rows
from src.data.code_v2_dedup import Protected
from src.data.code_triviality_v3 import analyze
from src.data.build_code_v2 import prompt_record
from src.data.screen_code_interfaces import python_code
from src.data.code_sandbox import run_jobs,function_job
from scripts.review_project2_deepseek import tests_excerpt

CONSTRUCTIVE=re.compile(r'(?is)(?:print|output|return)\s+any\b|multiple\s+(?:valid\s+)?(?:solutions|answers)|(?:any|one)\s+(?:valid\s+)?(?:reordering|permutation|arrangement)|not\s+unique')
def substantive(r):
    if r['interface']=='function':
        a=analyze(r['reference_code'],r['entry_point']);m=a['metrics'];return not a['isolate'] and sum(m.get(k,0) for k in ['control_flow','comprehensions','arithmetic'])>0
    _,tree=python_code(r['reference_code']);return any(isinstance(n,(ast.If,ast.For,ast.While,ast.BinOp,ast.ListComp,ast.DictComp,ast.SetComp,ast.GeneratorExp)) for n in ast.walk(tree))
def sandbox(r):
    if r['interface']=='function':
        if len({repr((t['args'],t['kwargs'])) for t in r['test_cases']})<3:return {'passed':False,'reason':'too_few_distinct_function_cases'}
        return run_jobs([function_job(r)])['results'][0]
    cases=r['verification_info']['test_cases']
    if len(cases)<3 or len({c['input'] for c in cases})<3 or len({tuple(c['output'].split()) for c in cases})<2:return {'passed':False,'reason':'insufficient_distinct_cases_or_outputs_for_mutant'}
    jobs=[{'kind':'stdio','sample_id':r['sample_id'],'reference_code':r['reference_code'],'cases':cases}]
    for i,value in enumerate([cases[0]['output'],'']):jobs.append({'kind':'stdio','sample_id':r['sample_id']+f'/constant{i}','reference_code':'import sys\nsys.stdout.write('+repr(value)+')','cases':cases})
    result=run_jobs(jobs)['results'];return {**result[0],'constant_mutant_passes':[x['passed'] for x in result[1:]]}
def audit_request(r,domain,split):
    return {'record_key':domain+'/'+split+'/'+r['sample_id'],'sample_id':r['sample_id'],'split':split,'domain':domain,'profile':'topup_OPD_prompt_and_reference_audit' if split=='train' else 'sealed_new_dev_unique_output_specification_and_reference',
        'problem':r.get('prompt',r['problem']),'reference_answer':r.get('reference_answer'),'reference_solution':r.get('solution'),'reference_code':r.get('reference_code'),'interface':r.get('interface'),'entry_point':r.get('entry_point'),'function_signature':r.get('function_signature'),'tests':tests_excerpt(r),
        'additional_conditions':'Dev code must have unambiguous unique expected output under the problem contract; isolate any constructive or multiple-valid-output task even if source tests pass. Evaluate full reference; never claim sample tests exhaustive. Train is prompt-only OPD, not supervised reference code.'}
def run():
    out=DATA/'project2';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():print('P2 already frozen');return
    old=ROOT/'data/delegated_ai_releases_20261024';dev=rows(old/'project2_fixed_dev_audit.jsonl')
    # Existing dev remains withheld forever; new version contains replacements.
    kept_math_dev=[r for r in dev if r['domain']=='math' and r['delegated_review_status']=='pass']
    kept_code_dev=[r for r in dev if r['domain']=='code' and r['delegated_review_status']=='pass' and not CONSTRUCTIVE.search(r['problem'])]
    protected=ScreeningSimilarityIndex(.85);priorids=set()
    for file in ['data/project2_code/math_train_prompts.jsonl','data/math/math_dev.jsonl','data/math/math500_test.jsonl','data/math/gsm8k_test.jsonl']:
        for r in rows(ROOT/file):protected.add(r);priorids.add(r['sample_id'])
    media={r['sample_id']:r['media_class'] for r in rows(ROOT/'data/project1_textonly_v4/math_train_media_audit.jsonl')}
    math=[]
    for r in sorted(rows(ROOT/'data/math/math_train.jsonl'),key=lambda r:(r.get('level') not in ['Level 3','Level 4','Level 5'],stable('p2topup/'+r['sample_id']))):
        if r['sample_id'] in priorids or media.get(r['sample_id'])!='prose_only_screen_pass_semantics_pending' or protected.match(r['problem']):continue
        split='dev' if len(math)<20 else 'train';math.append({**r,'split':split,'topup_origin':'previously unassigned MATH train; dev sealed before train','training_approved':False});protected.add(r)
        if len(math)==180:break
    print('Math topup frozen',len(math),flush=True)
    protection=Protected();idx=ScreeningSimilarityIndex(.85);ids=set();names=set();bodies=set();families=set()
    for file in ['data/project2_code/train_prompts.jsonl','data/project2_code/dev_verification.jsonl','data/project2_code_v2/train_prompts.jsonl','data/project2_code_v2/dev_verification.jsonl','data/project2_code_v3_dev_verified/train_prompts.jsonl','data/project2_code_v3_dev_verified/dev_verification.jsonl']:
        for r in rows(ROOT/file):
            ids.add(r['sample_id']);idx.add(r)
            if r.get('entry_point'):names.add(r['entry_point'])
            if r.get('reference_body_sha256'):bodies.add(r['reference_body_sha256'])
            if r.get('family_id'):families.add(r['family_id'])
    tok=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-1.5B-Instruct',local_files_only=True)
    def eligible(r):return r['sample_id'] not in ids and not protection.match(r) and not idx.match(r['problem']) and (not r.get('entry_point') or r['entry_point'] not in names) and (not r.get('reference_body_sha256') or r['reference_body_sha256'] not in bodies) and (not r.get('family_id') or r['family_id'] not in families)
    def claim(r):
        ids.add(r['sample_id']);idx.add(r)
        if r.get('entry_point'):names.add(r['entry_point'])
        if r.get('reference_body_sha256'):bodies.add(r['reference_body_sha256'])
        if r.get('family_id'):families.add(r['family_id'])
    functions=[]
    for r in sorted(rows(ROOT/'data/project2_code_v2/function_candidates.jsonl'),key=lambda r:stable('topupfunc/'+r['sample_id'])):
        if r['source_key']!='ace' or not eligible(r) or not substantive(r):continue
        p={**r,**prompt_record(r,'candidate')};n=len(tok.apply_chat_template([{'role':'user','content':p['prompt']}],tokenize=True,add_generation_prompt=True))
        if n>2560:continue
        split='dev' if len(functions)<80 else 'train';functions.append({**p,'split':split,'training_approved':False});claim(r)
        if len(functions)==680:break
    print('Function candidates',len(functions),flush=True)
    proposals=[]
    for r in sorted(rows(ROOT/'data/project2_code/structural_candidates.jsonl'),key=lambda r:(r['test_case_count']>8,stable('topupstdio/'+r['sample_id']))):
        if r['interface']!='stdin_stdout' or not eligible(r):continue
        if CONSTRUCTIVE.search(r['problem']):continue
        if re.search(r"(?i)you (?:do not|don't) need to read input",r['problem']):continue
        n=len(tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True))
        if n>2560:continue
        proposals.append(r);claim(r)
        if len(proposals)==1400:break
    lookup={r['sample_id']:r for r in proposals};stdio=[];source=source_files('code')
    for pos,raw in enumerate(parquet_rows(source)):
        sid='code/'+str(raw.get('problem_id',pos))
        if sid not in lookup:continue
        r=lookup.pop(sid);code,tree=python_code(raw['gold_standard_solution']);assert hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==r['reference_ast_sha256']
        r={**r,'reference_code':code,'verification_info':raw['verification_info']}
        if substantive(r):stdio.append(r)
        if not lookup:break
    stdio.sort(key=lambda r:(r['test_case_count']>8,stable('topupstdio/'+r['sample_id'])))
    stdio=[{**r,'split':'dev' if i<120 else 'train','training_approved':False} for i,r in enumerate(stdio[:1120])]
    # Retained old stdin dev also receives the new constant-mutant check.
    old_by_id={r['sample_id']:r for r in rows(ROOT/'data/project2_code_v3_dev_verified/dev_verification.jsonl')};dev_jobs=[r for r in functions+stdio if r['split']=='dev']
    for oldr in kept_code_dev:
        r=old_by_id[oldr['sample_id']]
        if r['interface']=='stdin_stdout':
            code,_=python_code(r.get('reference_code',r.get('gold_standard_solution')));dev_jobs.append({**r,'reference_code':code,'retained_old':True})
    executions=[];dev_pass=set();execution_lookup={}
    def execute(r):
        try:return {'sample_id':r['sample_id'],'retained_old':r.get('retained_old',False),'execution':sandbox(r)}
        except Exception as e:return {'sample_id':r['sample_id'],'execution':{'passed':False,'infrastructure_error':type(e).__name__}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,x in enumerate(pool.map(execute,dev_jobs),1):
            executions.append(x);execution_lookup[x['sample_id']]=x['execution'];e=x['execution']
            if e.get('passed') and not any(e.get('constant_mutant_passes',[True])):dev_pass.add(x['sample_id'])
            if i%10==0:write(out/'dev_execution.jsonl',executions);print('Restricted dev checks',i,'/',len(dev_jobs),'passed',len(dev_pass),flush=True)
    kept_code_dev=[{**r,'topup_constant_mutant_execution':execution_lookup.get(r['sample_id'])} for r in kept_code_dev if r['interface']=='function' or r['sample_id'] in dev_pass]
    candidates=math+[ {**r,'execution':execution_lookup.get(r['sample_id'])} for r in functions+stdio if r['split']=='train' or r['sample_id'] in dev_pass]
    write(out/'candidates.jsonl',candidates);write(out/'dev_execution.jsonl',executions);write(out/'retained_math_dev.jsonl',kept_math_dev);write(out/'retained_code_dev.jsonl',kept_code_dev)
    write(out/'audit_requests.jsonl',[audit_request(r,r['domain'],r['split']) for r in candidates])
    summary={'new_candidates':dict(Counter((r['domain']+'/'+r['split']+'/'+r.get('interface','math')) for r in candidates)),'retained_math_dev':len(kept_math_dev),'retained_code_dev':dict(Counter(r['interface'] for r in kept_code_dev)),'new_dev_sealed_before_train':True,'container_jobs':len(executions),'container_pass':len(dev_pass),'source_tests_not_exhaustive':True,'constructive_dev_excluded':True}
    dump(out/'summary.json',summary);dump(out/'manifest.json',{'summary':summary,'files':{p.name:sha(p) for p in out.glob('*.jsonl')},'source_revision':source['revision']});print(summary,flush=True)
if __name__=='__main__':run()
