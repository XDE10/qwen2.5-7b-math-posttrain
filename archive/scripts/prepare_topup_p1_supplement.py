"""Separate five-question supplement, not a rewrite of the frozen 1495 cohort."""
import re
from scripts.topup_common import *
from scripts.compact_distillation_v2 import replay
from scripts.refill_project1_native_v5 import source_filter
from scripts.compact_distillation_v3 import tokenizer
from src.data.screening_io import ScreeningSimilarityIndex
def run():
    out=DATA/'project1_supplement';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return
    index=ScreeningSimilarityIndex(.85);ids=set()
    for file in ['data/project1_native_v5/native_candidates.jsonl','data/topup_v6_20261024/project1/native_new1500.jsonl','data/math/math_dev.jsonl','data/math/math500_test.jsonl','data/math/gsm8k_test.jsonl','data/topup_v6_20261024/project2/candidates.jsonl']:
        for r in rows(ROOT/file):index.add(r);ids.add(r['sample_id'])
    for path in list((ROOT/'data').glob('compact_distillation_v*/verifier_references.json'))+list((ROOT/'data/deepseek_pilot_20261023').glob('*.json')):
        d=read(path)
        if isinstance(d,list):
            for r in d:
                if isinstance(r,dict) and 'sample_id' in r and 'problem' in r:index.add(r);ids.add(r['sample_id'])
    proposals=[]
    with (ROOT/'reports/openr1_full_audit_final_20261023/questions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            q=json.loads(line);ts=[t for t in q['trajectories'] if t.get('verified') and t.get('complete') and t.get('answer_tokens') is not None]
            if q['question_hash'] in ids or q['problem_type']!='Algebra' or not ts or q.get('flag_conflict') or q.get('evaluation_overlap') or q.get('internal_near_duplicate'):continue
            n=min(t['answer_tokens'] for t in ts);t=q.get('shortest_quality_pass')
            if not 1200<=n<=3200 or not t or t['answer_tokens']!=n or t['total_tokens']>3800 or index.match(q['problem']):continue
            proposals.append({'question_hash':q['question_hash'],'problem_type':'Algebra','source':q['source'],'trajectory':t,'linked_train_level':q.get('linked_train_level')})
            if len(proposals)==30:break
    result=[];tok=tokenizer();lookup={r['sample_id']:r for r in replay(proposals)}
    for p in proposals:
        r=lookup[p['question_hash']];row={'sample_id':r['sample_id'],'problem':r['problem'],'original_problem':r['problem'],'solution':r['raw_solution'],'reference_answer':r['reference_answer'],'source_revision':r['source_revision'],'source_position':p['trajectory'],'problem_type':'Algebra','split':'train','training_approved':False,'shortest_correct_source_tokens':p['trajectory']['answer_tokens'],'cohort':'separate_five_algebra_supplement'};a=source_filter(row)
        if not a['accepted'] or index.match(a['problem']) or re.search(r'[<>≤≥≠=]|\\(?:leq?|geq?|neq?|lt|gt)\b',str(row['reference_answer'])):continue
        row['problem']=a['problem'];assert len(tok.encode(row['solution'],add_special_tokens=False))==p['trajectory']['answer_tokens'];result.append(row);index.add(row)
        if len(result)==5:break
    write(out/'native_candidates.jsonl',result);write(out/'blind_requests.jsonl',[{'sample_id':r['sample_id'],'problem':r['problem']} for r in result]);dump(out/'manifest.json',{'count':len(result),'why_separate':'Only95 inequalities remain under all filters. Preserve original1495 cohort; supplement five equally filtered Algebra questions, report resulting530/300/300/225/95/50 honestly. No admission or difficulty-proxy threshold change.','files':{p.name:sha(p) for p in out.glob('*.jsonl')}});print('P1 supplementary candidates',len(result),flush=True)
if __name__=='__main__':run()
