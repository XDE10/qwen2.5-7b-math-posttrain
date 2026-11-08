"""Refill fixed six-stratum native proposal after v4 posthoc exclusions. No API."""
import json,re
from pathlib import Path
from collections import Counter,defaultdict
from scripts.compact_distillation_v2 import ROOT,read,replay,file_sha,write_jsonl
from scripts.compact_distillation_v3 import QUOTAS,tokenizer
from scripts.analyze_openr1_candidates import choose_for_budget
from src.synthesis.compact_v4 import source_triage
from src.synthesis.compact_v3 import screen_problem
from src.synthesis.deepseek_pilot import dump,sha
from src.data.screening_io import ScreeningSimilarityIndex

DATA=ROOT/'data/project1_native_v5';REPORT=ROOT/'reports/project1_native_v5'


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for l in f:
            if l.strip():yield json.loads(l)


def source_filter(r):
    a=source_triage(r);p=a['cleaning']['problem'];reasons=[]
    if not a['automatic_pool_eligible']:reasons.append('v4_scope_or_source_or_structure_review')
    if re.search(r'(?im)\bANSWER\s*:\s*\S|^\s*(?:#{1,6}\s*)?SOLUTION\s*:\s*\S',p):reasons.append('explicit_answer_or_solution_metadata')
    if re.search(r'(?i)translated into|translation result|ignore previous',p):reasons.append('remaining_meta_instruction')
    if re.search(r'\\(?:mathrm|text)\{\s*(?:~|\\,|\\;|\\!|\\quad|\\ )*\s*\}',str(r['reference_answer'])):reasons.append('reference_empty_unit_markup_review')
    s=screen_problem(p)
    if not s['screen_pass']:reasons+=s['screen_reasons']
    return {'accepted':not reasons,'problem':p,'reasons':sorted(set(reasons)),'v4_triage':a}


def run():
    DATA.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    if (DATA/'manifest.json').exists():
        m=read(DATA/'manifest.json')
        for name,h in m['files'].items():assert file_sha(DATA/name)==h
        print('Native v5 already prepared');return
    old=list(rows(ROOT/'data/project1_textonly_v3/native_candidates.jsonl'));oldids={r['sample_id'] for r in old}
    reviews=read(ROOT/'reports/compact_distillation_v4_20261024/ai_process_review.json')['reviews']
    isolated={r['sample_id'] for r in reviews if r['pair_decision']!='pass'}
    selected=[];audit=[];counts=Counter();index=ScreeningSimilarityIndex(.85);heldout=ScreeningSimilarityIndex(.85)
    for name in ['math_dev','math500_test','gsm8k_test']:
        for r in rows(ROOT/read(ROOT/'data/math_manifest.json')['files'][name]['path']):heldout.add(r)
    def accept(r,origin):
        a=source_filter(r)
        if r['sample_id'] in isolated:a['accepted']=False;a['reasons'].append('v4_process_review_nonpass_source_or_trajectory')
        hit=heldout.match(a['problem']) or index.match(a['problem'])
        if hit:a['accepted']=False;a['reasons'].append('heldout_or_internal_lexical_overlap')
        audit.append({'sample_id':r['sample_id'],'origin':origin,**a})
        if not a['accepted'] or counts[r['problem_type']]>=QUOTAS[r['problem_type']]:return False
        r={**r,'problem':a['problem'],'v5_source_filter':a,'training_approved':False,'semantic_review':'pending'}
        selected.append(r);index.add(r);counts[r['problem_type']]+=1;return True
    for r in old:accept(r,'v3_original_pool')
    pools=defaultdict(list)
    for q in rows(ROOT/'reports/openr1_full_audit_final_20261023/questions.jsonl'):
        if q['question_hash'] in oldids:continue
        t=choose_for_budget(q,3000)
        if not t:continue
        assert 'evaluation_overlap' in q and 'internal_near_duplicate' in q
        if q['evaluation_overlap'] or q['internal_near_duplicate'] or not screen_problem(q['problem'])['screen_pass']:continue
        pools[q['problem_type']].append({'question_hash':q['question_hash'],'problem_type':q['problem_type'],'source':q['source'],
            'trajectory':t,'linked_train_level':q.get('linked_train_level')})
    ordered={k:sorted(v,key=lambda r:sha('native-v5-refill/'+r['question_hash'])) for k,v in pools.items()};offsets=Counter();tok=tokenizer()
    for roundno in range(8):
        missing={k:n-counts[k] for k,n in QUOTAS.items() if counts[k]<n}
        if not missing:break
        proposals=[]
        for kind,n in missing.items():
            start=offsets[kind];take=min(len(ordered[kind])-start,2*n+20)
            proposals+=ordered[kind][start:start+take];offsets[kind]+=take
        if not proposals:break
        print('Replaying actual source rows, round',roundno+1,'candidates',len(proposals),flush=True)
        replayed={r['sample_id']:r for r in replay(proposals)}
        for p in proposals:
            rr=replayed[p['question_hash']]
            row={'sample_id':rr['sample_id'],'problem':rr['problem'],'original_problem':rr['problem'],
                'solution':rr['raw_solution'],'reference_answer':rr['reference_answer'],'source_revision':rr['source_revision'],
                'source_position':p['trajectory'],'source_verified':True,'local_math_verify_rechecked':False,
                'problem_type':p['problem_type'],'linked_train_level':p['linked_train_level'],'split':'train'}
            accept(row,'fresh_full_audit_refill')
    if len(selected)!=1500:raise RuntimeError('Cannot fill fixed quotas without relaxing scope: '+str(counts))
    for r in selected:
        r['answer_tokens']=len(tok.encode(r['solution'],add_special_tokens=False))
        r['total_tokens']=len(tok.apply_chat_template([{'role':'system','content':r'Please reason step by step, and put your final answer within \boxed{}.'},{'role':'user','content':r['problem']},{'role':'assistant','content':r['solution']}],tokenize=True,add_generation_prompt=False))
        assert 500<=r['answer_tokens']<=3000 and r['total_tokens']<=3800
        assert source_filter(r)['accepted']
    write_jsonl(DATA/'native_candidates.jsonl',selected);write_jsonl(DATA/'screening_audit.jsonl',audit)
    retained=sum(r['sample_id'] in oldids for r in selected)
    summary={'version':'native-v5-refill-20261024','count':1500,'retained_from_original_1500':retained,'refilled':1500-retained,
        'per_type':dict(counts),'originals_rejected':1500-retained,'heldout_and_internal_jaccard_threshold':.85,
        'claim':'fixed-rule native candidate proposal; not 1500 semantically gold or globally Math-Verify-rechecked solutions',
        'training_approved':False,'human_review':'not_performed','api_calls':0}
    dump(DATA/'summary.json',summary)
    dump(DATA/'manifest.json',{'summary':summary,'inputs':{p:file_sha(ROOT/p) for p in ['data/project1_textonly_v3/manifest.json','reports/compact_distillation_v4_20261024/ai_process_review.json','reports/openr1_full_audit_final_20261023/manifest.json','data/math_manifest.json','scripts/refill_project1_native_v5.py']},
        'files':{p.name:file_sha(p) for p in DATA.iterdir() if p.is_file() and p.name!='manifest.json'}})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
