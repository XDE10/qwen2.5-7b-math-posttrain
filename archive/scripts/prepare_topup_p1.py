"""Fresh 1500 questions with shortest complete correct source length 1200..3200."""
import re
from collections import Counter,defaultdict
from scripts.topup_common import *
from scripts.refill_project1_native_v5 import source_filter
from scripts.compact_distillation_v2 import replay
from scripts.compact_distillation_v3 import tokenizer,QUOTAS
from src.data.screening_io import ScreeningSimilarityIndex
from src.synthesis.compact_v3 import screen_problem

def run():
    out=DATA/'project1';out.mkdir(exist_ok=True);snapshot()
    if (out/'manifest.json').exists():print('P1 already frozen');return
    excluded=ScreeningSimilarityIndex(.85);ids=set()
    sources=['data/project1_native_v5/native_candidates.jsonl','data/project1_batch500_20261024/verifier_references.jsonl']
    for p in (ROOT/'data').glob('compact_distillation_v*/verifier_references.json'):
        sources.append(p.relative_to(ROOT).as_posix())
    sources+=['data/deepseek_pilot_20261023/'+n+'.json' for n in ['verifier_references','anchors','calibration_dev','frozen_acceptance']]
    for name in sources:
        rs=rows(ROOT/name) if name.endswith('jsonl') else read(ROOT/name)
        for r in rs:excluded.add(r);ids.add(r['sample_id'])
    for name in ['math_dev','math500_test','gsm8k_test']:
        for r in rows(ROOT/read(ROOT/'data/math_manifest.json')['files'][name]['path']):excluded.add(r)
    candidates=defaultdict(list);audit=Counter()
    with (ROOT/'reports/openr1_full_audit_final_20261023/questions.jsonl').open(encoding='utf-8') as f:
        for l in f:
            q=__import__('json').loads(l);ts=[t for t in q['trajectories'] if t.get('verified') and t.get('complete') and t.get('answer_tokens') is not None]
            if not ts:continue
            shortest=min(t['answer_tokens'] for t in ts)
            if not 1200<=shortest<=3200:continue
            valid=[t for t in ts if t['answer_tokens']==shortest and t.get('quality',{}).get('automatic_pass') and t.get('total_tokens',99999)<=3800]
            if not valid or q['question_hash'] in ids or q.get('flag_conflict') or q.get('evaluation_overlap') or q.get('internal_near_duplicate'):continue
            if not screen_problem(q['problem'])['screen_pass']:continue
            t=valid[0];priority=(not bool(re.search(r'olympiad|aime|amc|competition|contest',q['source'],re.I)),not (0<q.get('verification_ratio',1)<1),stable('topup6/'+q['question_hash']))
            candidates[q['problem_type']].append({'question_hash':q['question_hash'],'problem_type':q['problem_type'],'source':q['source'],'trajectory':t,'linked_train_level':q.get('linked_train_level'),'shortest_correct_source_tokens':shortest,'source_verification_ratio':q.get('verification_ratio'),'priority':priority})
    for k in candidates:candidates[k].sort(key=lambda r:r['priority'])
    selected=[];counts=Counter();offset=Counter();tok=tokenizer()
    for rnd in range(12):
        proposals=[]
        for k,n in QUOTAS.items():
            need=n-counts[k];take=max(0,min(len(candidates[k])-offset[k],2*need+20 if need else 0));proposals+=candidates[k][offset[k]:offset[k]+take];offset[k]+=take
        if not proposals:break
        print('P1 replay round',rnd+1,len(proposals),flush=True)
        rr={r['sample_id']:r for r in replay(proposals)}
        for p in proposals:
            k=p['problem_type']
            if counts[k]>=QUOTAS[k]:continue
            r=rr[p['question_hash']];row={'sample_id':r['sample_id'],'problem':r['problem'],'original_problem':r['problem'],'solution':r['raw_solution'],'reference_answer':r['reference_answer'],'source_revision':r['source_revision'],'source_position':p['trajectory'],'problem_type':k,'linked_train_level':p['linked_train_level'],'split':'train','shortest_correct_source_tokens':p['shortest_correct_source_tokens'],'source_verification_ratio':p['source_verification_ratio'],'difficulty_label_kind':'length/source/correctness proxies; no official level unless linked','training_approved':False}
            a=source_filter(row)
            if not a['accepted'] or excluded.match(a['problem']) or re.search(r'[<>≤≥≠=]|\\(?:leq?|geq?|neq?|lt|gt)\b',str(row['reference_answer'])):audit['source_or_overlap_or_answer_type_reject']+=1;continue
            row['problem']=a['problem'];n=len(tok.encode(row['solution'],add_special_tokens=False))
            if n!=p['shortest_correct_source_tokens']:raise ValueError('Actual tokenizer/source shortest length mismatch')
            row['answer_tokens']=n;selected.append(row);counts[k]+=1;excluded.add(row)
        print('P1 selected',len(selected),dict(counts),flush=True)
        if len(selected)==1500:break
    write(out/'native_new1500.jsonl',selected)
    write(out/'native_pool_union.jsonl',rows(ROOT/'data/project1_native_v5/native_candidates.jsonl')+selected)
    write(out/'blind_requests.jsonl',[{'sample_id':r['sample_id'],'problem':r['problem']} for r in selected])
    media={r['sample_id']:r['media_class'] for r in rows(ROOT/'data/project1_textonly_v4/math_train_media_audit.jsonl')}
    clean=[r for r in rows(ROOT/'data/math/math_train.jsonl') if media.get(r['sample_id'])=='prose_only_screen_pass_semantics_pending']
    priority=[{**r,'rl_priority':'official_Level_3_to_5','training_approved':False,'requires_7b_rollout_selection':True} for r in clean if r.get('level') in ['Level 3','Level 4','Level 5']]
    write(out/'rl_official_level3to5_priority.jsonl',priority)
    summary={'fresh_questions':len(selected),'type_counts':dict(counts),'target':1500,'shortest_correct_length':[1200,3200],'native_union':1500+len(selected),'clean_math_count':len(clean),'official_level_distribution':dict(Counter(r.get('level','unknown') for r in clean)),'rl_priority_count':len(priority),'rejections':dict(audit),'difficulty_proxy_not_official_level':True,'no_generation_references':True}
    dump(out/'summary.json',summary);dump(out/'manifest.json',{'summary':summary,'inputs':{n:sha(ROOT/n) for n in sources},'files':{p.name:sha(p) for p in out.glob('*.jsonl')}});print(summary,flush=True)
if __name__=='__main__':run()
