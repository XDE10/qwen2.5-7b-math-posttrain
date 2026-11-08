"""Additional never-assigned stdin dev reserves; exact same test gates."""
import time,ast,hashlib
from concurrent.futures import ThreadPoolExecutor
from scripts.topup_common import *
from scripts.prepare_topup_p2 import CONSTRUCTIVE,substantive,sandbox,audit_request
from src.data.screening_io import ScreeningSimilarityIndex,source_files,parquet_rows
from src.data.code_v2_dedup import Protected
from src.data.screen_code_interfaces import python_code
from transformers import AutoTokenizer
def run():
    out=DATA/'project2_dev_reserve';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return
    while not (DATA/'project2/manifest.json').exists():time.sleep(3)
    protected=Protected();index=ScreeningSimilarityIndex(.85);ids=set();families=set()
    for name in ['data/project2_code/train_prompts.jsonl','data/project2_code/dev_verification.jsonl','data/project2_code_v2/train_prompts.jsonl','data/project2_code_v2/dev_verification.jsonl','data/project2_code_v3_dev_verified/train_prompts.jsonl','data/project2_code_v3_dev_verified/dev_verification.jsonl','data/topup_v6_20261024/project2/candidates.jsonl']:
        for r in rows(ROOT/name):
            ids.add(r['sample_id']);index.add(r)
            if r.get('family_id'):families.add(r['family_id'])
    tok=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-1.5B-Instruct',local_files_only=True);selected={}
    for r in sorted(rows(ROOT/'data/project2_code/structural_candidates.jsonl'),key=lambda r:(r['test_case_count']>8,stable('devreserve6/'+r['sample_id']))):
        if r['interface']!='stdin_stdout' or r['test_case_count']<3 or r['sample_id'] in ids or r.get('family_id') in families or CONSTRUCTIVE.search(r['problem']) or protected.match(r) or index.match(r['problem']):continue
        if len(tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True))>2560:continue
        selected[r['sample_id']]=r;index.add(r);families.add(r['family_id'])
        if len(selected)==650:break
    print('Fresh dev source proposals',len(selected),flush=True);candidates=[]
    for pos,raw in enumerate(parquet_rows(source_files('code'))):
        sid='code/'+str(raw.get('problem_id',pos))
        if sid not in selected:continue
        r=selected.pop(sid);code,tree=python_code(raw['gold_standard_solution']);assert hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==r['reference_ast_sha256'];cases=raw['verification_info']['test_cases']
        if len({c['input'] for c in cases})<3 or len({tuple(c['output'].split()) for c in cases})<2:continue
        r={**r,'reference_code':code,'verification_info':raw['verification_info'],'split':'dev','training_approved':False}
        if substantive(r):candidates.append(r)
        if not selected:break
    candidates.sort(key=lambda r:stable('devreserve6/'+r['sample_id']));print('Fresh dev distinct cases/output reserves',len(candidates),flush=True);tested=[];passing=[]
    def execute(r):
        try:return r,sandbox(r)
        except Exception as e:return r,{'passed':False,'infrastructure_error':type(e).__name__}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(r,e) in enumerate(pool.map(execute,candidates),1):
            tested.append({'sample_id':r['sample_id'],'execution':e})
            if e.get('passed') and not any(e.get('constant_mutant_passes',[True])):passing.append({**r,'execution':e})
            if i%10==0:write(out/'execution.jsonl',tested);write(out/'candidates.jsonl',passing);print('Extra dev tested',i,'/',len(candidates),'passed',len(passing),flush=True)
    write(out/'execution.jsonl',tested);write(out/'candidates.jsonl',passing);write(out/'audit_requests.jsonl',[audit_request(r,'code','dev') for r in passing]);dump(out/'manifest.json',{'candidate_count':len(candidates),'source_test_and_mutant_pass':len(passing),'files':{p.name:sha(p) for p in out.glob('*.jsonl')},'dev_sealed_from_previously_unassigned_source':True})
if __name__=='__main__':run()
