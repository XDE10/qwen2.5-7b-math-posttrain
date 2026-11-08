"""Inventory and independently freeze unused sources without weakening filters."""
import json,re
from collections import Counter,defaultdict
from scripts.finalfill_common import *
from scripts.compact_distillation_v2 import replay
from scripts.compact_distillation_v3 import tokenizer,QUOTAS
from scripts.refill_project1_native_v5 import source_filter
from src.synthesis.compact_v3 import screen_problem
from src.data.screening_io import ScreeningSimilarityIndex

def p1():
    out=DATA/'project1';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return
    excluded=ScreeningSimilarityIndex(.85);ids=set()
    paths=['data/project1_native_v5/native_candidates.jsonl','data/topup_v6_20261024/release/project1_native_fresh1500_candidates_not_blanket_approved.jsonl','data/topup_v6_20261024/release/project2_math_dev.jsonl','data/math/math_dev.jsonl','data/math/math500_test.jsonl','data/math/gsm8k_test.jsonl']
    paths += [str(p.relative_to(ROOT)) for p in (ROOT/'data').glob('compact_distillation_v*/verifier_references.json')]
    paths += ['data/deepseek_pilot_20261023/'+n+'.json' for n in ['verifier_references','anchors','calibration_dev','frozen_acceptance']]
    for path in paths:
        rs=rows(ROOT/path) if path.endswith('.jsonl') else read(ROOT/path)
        for r in rs:ids.add(r['sample_id']);excluded.add(r)
    for r in rows(ROOT/'data/topup_v6_20261024/project2/candidates.jsonl'):
        if r['split']=='dev':ids.add(r['sample_id']);excluded.add(r)
    pool=defaultdict(list);scanned=Counter()
    with (ROOT/'reports/openr1_full_audit_final_20261023/questions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            q=json.loads(line);ts=[t for t in q['trajectories'] if t.get('verified') and t.get('complete') and t.get('answer_tokens') is not None]
            if not ts or q['question_hash'] in ids or q.get('flag_conflict') or q.get('evaluation_overlap') or q.get('internal_near_duplicate'):continue
            minimum=min(t['answer_tokens'] for t in ts)
            if not 1200<=minimum<=3200:continue
            valid=[t for t in ts if t['answer_tokens']==minimum and t.get('quality',{}).get('automatic_pass') and t.get('total_tokens',99999)<=3800]
            if not valid or not screen_problem(q['problem'])['screen_pass']:continue
            priority=(not bool(re.search(r'olympiad|aime|amc|competition|contest',q['source'],re.I)),not (0<q.get('verification_ratio',1)<1),stable('finalfill7/'+q['question_hash']))
            pool[q['problem_type']].append({'question_hash':q['question_hash'],'problem_type':q['problem_type'],'source':q['source'],'trajectory':valid[0],'linked_train_level':q.get('linked_train_level'),'shortest_correct_source_tokens':minimum,'source_verification_ratio':q.get('verification_ratio'),'priority':priority})
    for k in pool:pool[k].sort(key=lambda x:x['priority'])
    dump(out/'inventory_before_source_replay.json',{k:len(v) for k,v in pool.items()});print('V7 pre-replay inventory',{k:len(v) for k,v in pool.items()},flush=True)
    # A finite new reservoir, not a promise of1500 accepts; at most needed chunks
    # will ever be submitted. Ratio deviations are recorded, never hidden.
    weights={k:v/1500 for k,v in QUOTAS.items()};offset=Counter();selected=[];counts=Counter();tok=tokenizer()
    for roundno in range(12):
        if len(selected)>=1800:break
        missing=1800-len(selected);proposals=[]
        available=[k for k in weights if offset[k]<len(pool[k])]
        if not available:break
        total=sum(weights[k] for k in available)
        for k in available:
            take=min(len(pool[k])-offset[k],max(20,int(2*missing*weights[k]/total)+10));proposals+=pool[k][offset[k]:offset[k]+take];offset[k]+=take
        print('V7 replay',roundno+1,len(proposals),flush=True);rr={r['sample_id']:r for r in replay(proposals)}
        queues={k:[p for p in proposals if p['problem_type']==k] for k in weights};positions=Counter()
        while len(selected)<1800:
            ks=[k for k in weights if positions[k]<len(queues[k])]
            if not ks:break
            k=min(ks,key=lambda k:(counts[k]+1)/weights[k]);p=queues[k][positions[k]];positions[k]+=1
            r=rr[p['question_hash']];row={'sample_id':r['sample_id'],'problem':r['problem'],'original_problem':r['problem'],'solution':r['raw_solution'],'reference_answer':r['reference_answer'],'source_revision':r['source_revision'],'source_position':p['trajectory'],'source':p['source'],'problem_type':p['problem_type'],'linked_train_level':p['linked_train_level'],'shortest_correct_source_tokens':p['shortest_correct_source_tokens'],'source_verification_ratio':p['source_verification_ratio'],'split':'train','training_approved':False}
            a=source_filter(row)
            if not a['accepted'] or excluded.match(a['problem']) or re.search(r'[<>≤≥≠=]|\\(?:leq?|geq?|neq?|lt|gt)\b',str(row['reference_answer'])):scanned['source_or_overlap_or_reference_type_reject']+=1;continue
            row['problem']=a['problem'];assert len(tok.encode(row['solution'],add_special_tokens=False))==p['shortest_correct_source_tokens']
            selected.append(row);counts[row['problem_type']]+=1;excluded.add(row)
    # Weighted scheduling, respecting actual finite stock. No relabeling.
    ordered=[];bytype={k:[r for r in selected if r['problem_type']==k] for k in weights};used=Counter()
    while len(ordered)<len(selected):
        ks=[k for k in weights if used[k]<len(bytype[k])]
        k=min(ks,key=lambda k:(used[k]+1)/weights[k]);ordered.append(bytype[k][used[k]]);used[k]+=1
    write(out/'candidates.jsonl',ordered);dump(out/'manifest.json',{'count':len(ordered),'type_counts':dict(counts),'requested_weight_counts':QUOTAS,'same_hard_filters':True,'stock_shortfall_must_be_reported':True,'first_plan':1100,'reserve_only_max':1800,'files':{'candidates.jsonl':sha(out/'candidates.jsonl')},'inputs':{p:sha(ROOT/p) for p in paths}});print('V7 P1 reservoir',len(ordered),dict(counts),flush=True)

def p3():
    out=DATA/'project3';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return
    old=rows(ROOT/'reports/topup_v6_20261024/p3_fact_review_reparsed.jsonl');attempted={r.get('sample_id',r.get('id')) for r in old}
    full=rows(ROOT/'data/topup_v6_20261024/project3_reserve/cases.jsonl');remaining=[r for r in full if r['sample_id'] not in attempted]
    assert len(remaining)==896 and all(r['split']=='train' for r in remaining)
    write(out/'remaining896.jsonl',remaining)
    pending=[r for r in old if r.get('review',{}).get('task_status')=='uncertain'];positive=[r for r in pending if r.get('pruning',{}).get('positive_tokens')]
    report={'uncertain_total':len(pending),'uncertain_by_split':dict(Counter(r['split'] for r in pending)),'uncertain_with_certified_positive_by_split':dict(Counter(r['split'] for r in positive)),'same_pruning_already_applied':True,'automatic_recoveries_without_new_answer_evidence':0,'reason':'Fact-level issues are already masked. Remaining task_status uncertain concerns question/answer validity; do not convert to pass solely because some visual facts are certified.'}
    dump(out/'uncertain_zero_cost_audit.json',report);write(out/'uncertain_with_positives_queue.jsonl',positive)
    dump(out/'manifest.json',{'remaining_train':896,'already_audited_train':1200,'expected_unique_train_coverage':2096,'no_validation_or_test_seeds':True,'files':{p.name:sha(p) for p in out.glob('*.jsonl')}});print('V7 P3 remaining',len(remaining),report,flush=True)
if __name__=='__main__':snapshot();p3();p1()
