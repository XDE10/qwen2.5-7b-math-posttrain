"""Fresh stdin train reserve; no dev candidate recycled across splits."""
import ast,hashlib
from scripts.topup_common import *
from scripts.prepare_topup_p2 import substantive,audit_request,CONSTRUCTIVE
from src.data.screening_io import ScreeningSimilarityIndex,source_files,parquet_rows
from src.data.code_v2_dedup import Protected
from src.data.screen_code_interfaces import python_code
from transformers import AutoTokenizer
def run():
    out=DATA/'project2_train_reserve';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return
    idx=ScreeningSimilarityIndex(.85);ids=set();families=set();protected=Protected()
    for name in ['data/project2_code/train_prompts.jsonl','data/project2_code/dev_verification.jsonl','data/project2_code_v2/train_prompts.jsonl','data/project2_code_v2/dev_verification.jsonl','data/project2_code_v3_dev_verified/train_prompts.jsonl','data/project2_code_v3_dev_verified/dev_verification.jsonl','data/topup_v6_20261024/project2/candidates.jsonl','data/topup_v6_20261024/project2_dev_reserve/candidates.jsonl']:
        for r in rows(ROOT/name):
            ids.add(r['sample_id']);idx.add(r)
            if r.get('family_id'):families.add(r['family_id'])
    for p in [DATA/'project2/dev_execution.jsonl',DATA/'project2_dev_reserve/execution.jsonl']:
        for r in rows(p):ids.add(r['sample_id'])
    tok=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-1.5B-Instruct',local_files_only=True);wanted={}
    for r in sorted(rows(ROOT/'data/project2_code/structural_candidates.jsonl'),key=lambda r:stable('train-reserve6/'+r['sample_id'])):
        if r['interface']!='stdin_stdout' or r['test_case_count']<3 or r['sample_id'] in ids or r['family_id'] in families or CONSTRUCTIVE.search(r['problem']) or protected.match(r) or idx.match(r['problem']):continue
        if len(tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True))>2560:continue
        wanted[r['sample_id']]=r;idx.add(r);families.add(r['family_id'])
        if len(wanted)==180:break
    result=[]
    for pos,raw in enumerate(parquet_rows(source_files('code'))):
        sid='code/'+str(raw.get('problem_id',pos))
        if sid not in wanted:continue
        r=wanted.pop(sid);code,tree=python_code(raw['gold_standard_solution']);assert hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==r['reference_ast_sha256'];cases=raw['verification_info']['test_cases']
        # This also excludes dev-screened source rows that failed static diversity.
        if len({c['input'] for c in cases})<3 or len({tuple(c['output'].split()) for c in cases})<2:continue
        r={**r,'reference_code':code,'verification_info':raw['verification_info'],'split':'train','training_approved':False}
        if substantive(r):result.append(r)
        if not wanted:break
    write(out/'candidates.jsonl',result);write(out/'audit_requests.jsonl',[audit_request(r,'code','train') for r in result]);dump(out/'manifest.json',{'candidates':len(result),'files':{p.name:sha(p) for p in out.glob('*.jsonl')},'prior_dev_ids_excluded':True});print('Fresh stdin train reserve',len(result),flush=True)
if __name__=='__main__':run()
