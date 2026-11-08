"""Unused same-standard reserves; preparation itself grants no approval."""
import re,json
from collections import Counter,defaultdict
from scripts.strict_v8_common import *
from scripts.compact_distillation_v2 import replay
from scripts.refill_project1_native_v5 import source_filter
from src.synthesis.compact_v3 import screen_problem
from src.data.screening_io import ScreeningSimilarityIndex

def run():
    out=DATA/'other_type_reserves';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return
    held=ScreeningSimilarityIndex(.85);ids=set();paths=list(read(ROOT/'data/finalfill_v7_20261024/project1/manifest.json')['inputs'])
    paths+=['data/finalfill_v7_20261024/project1/candidates.jsonl','data/topup_v6_20261024/project2/candidates.jsonl','data/topup_v6_20261024/project2_dev_reserve/candidates.jsonl']
    for path in paths:
        rs=rows(ROOT/path) if path.endswith('.jsonl') else read(ROOT/path)
        for r in rs:
            if '/project2/' in path or '/project2_dev_reserve/' in path:
                if r.get('split')!='dev':continue
            ids.add(r['sample_id']);held.add(r)
    pools=defaultdict(list);targets={'Combinatorics':250,'Logic and Puzzles':100}
    with (ROOT/'reports/openr1_full_audit_final_20261023/questions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            q=json.loads(line);kind=q['problem_type']
            if kind not in targets or q['question_hash'] in ids or q.get('flag_conflict') or q.get('evaluation_overlap') or q.get('internal_near_duplicate'):continue
            ts=[t for t in q['trajectories'] if t.get('verified') and t.get('complete') and t.get('answer_tokens') is not None]
            if not ts:continue
            n=min(t['answer_tokens'] for t in ts)
            if not 1200<=n<=3200:continue
            valid=[t for t in ts if t['answer_tokens']==n and t.get('quality',{}).get('automatic_pass') and t.get('total_tokens',99999)<=3800]
            if not valid or not screen_problem(q['problem'])['screen_pass']:continue
            pools[kind].append({'question_hash':q['question_hash'],'problem_type':kind,'source':q['source'],'trajectory':valid[0],'priority':(not bool(re.search('olympiad|aime|amc|competition|contest',q['source'],re.I)),not (0<q.get('verification_ratio',1)<1),stable('strictreserve8/'+q['question_hash']))})
    for values in pools.values():values.sort(key=lambda p:p['priority'])
    selected=[];audit=[]
    for kind in targets:
        for i in range(0,len(pools[kind]),100):
            if sum(r['problem_type']==kind for r in selected)>=targets[kind]:break
            props=pools[kind][i:i+100];sources={r['sample_id']:r for r in replay(props)}
            for p in props:
                r=sources[p['question_hash']];r={**r,'solution':r['raw_solution'],'problem_type':kind,'source':p['source'],'source_position':p['trajectory'],'shortest_correct_source_tokens':p['trajectory']['answer_tokens'],'split':'train','training_approved':False}
                a=source_filter(r);ok=a['accepted'] and not held.match(a['problem']) and not re.search(r'[<>≤≥≠=]|\\(?:leq?|geq?|neq?|lt|gt)\b',str(r['reference_answer']))
                audit.append({'sample_id':r['sample_id'],'eligible':bool(ok),'source_filter':a})
                if ok:
                    selected.append({**r,'problem':a['problem']});held.add(r)
                    if sum(x['problem_type']==kind for x in selected)>=targets[kind]:break
            print('Strict type reserve',kind,dict(Counter(r['problem_type'] for r in selected)),flush=True)
    write(out/'candidates.jsonl',selected);write(out/'audit.jsonl',audit);dump(out/'manifest.json',{'counts':dict(Counter(r['problem_type'] for r in selected)),'source_tokens':[1200,3200],'full_max':3800,'same_gates':True,'training_approved':False,'api_calls':0,'files':{p.name:sha(p) for p in out.glob('*.jsonl')}})
if __name__=='__main__':run()
