"""Versioned cache audit + independent 40-question pilot, no automatic scaling.

python -m scripts.compact_distillation_v2 reaudit|prepare|run [--live --key-prompt]
Old frozen data, policies and reports are read only. No LLM level/review requests.
"""
import argparse
import getpass
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transformers import AutoTokenizer
from src.synthesis.compact_v2 import VERSION, SYSTEM, admit, clean_problem, payload, digest_text
from src.synthesis.deepseek_pilot import PilotClient, dump, sha
from src.data.screening_io import ScreeningSimilarityIndex
from scripts.run_deepseek_pilot import check_math

ROOT = Path(__file__).resolve().parents[1]
OLD_DATA = ROOT/'data/deepseek_pilot_20261023'
OLD_REPORT = ROOT/'reports/deepseek_pilot_20261023'
DATA = ROOT/'data/compact_distillation_v2_20261023'
REPORT = ROOT/'reports/compact_distillation_v2_20261023'
QUOTAS = {'Algebra':12, 'Geometry':8, 'Number Theory':8,
          'Combinatorics':6, 'Inequalities':4, 'Logic and Puzzles':2}


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def file_sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()
def tok():
    return AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct', local_files_only=True)
def jsonl(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def write_jsonl(p, rs):
    Path(p).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rs), encoding='utf-8')


def frozen_identity():
    paths = [OLD_DATA/'manifest.json', OLD_REPORT/'delivery_manifest.json',
             OLD_REPORT/'solution_results.json', ROOT/'data/project1_sft/sft_train.jsonl',
             ROOT/'data/math_manifest.json']
    paths += [ROOT/v['path'] for v in read(ROOT/'data/math_manifest.json')['files'].values()]
    return {str(p.relative_to(ROOT)):file_sha(p) for p in paths}


def batch_verify(jobs):
    """Bounded separate Math-Verify processes, memoized by exact output/reference."""
    unique = {sha([text, ref]):(text, ref) for text, ref in jobs}
    result = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        future = {pool.submit(check_math, text, ref):key for key,(text,ref) in unique.items()}
        for n,f in enumerate(as_completed(future),1):
            result[future[f]] = f.result()
            if n%25==0: print(f'Math-Verify {n}/{len(unique)}', flush=True)
    return result


def reaudit():
    OUT = REPORT/'cache_reaudit'
    if (OUT/'manifest.json').exists():
        m=read(OUT/'manifest.json')
        assert m['input_identity']==frozen_identity(), 'Inputs changed'
        for name,h in m['files'].items(): assert file_sha(OUT/name)==h
        print('Cache audit already sealed; hashes verified'); return
    before=frozen_identity(); OUT.mkdir(parents=True,exist_ok=True)
    old=read(OLD_REPORT/'solution_results.json')
    refs={r['sample_id']:r for r in read(OLD_DATA/'verifier_references.json')}
    jobs=[(a['solution'],refs[r['sample_id']]['reference_answer']) for r in old for a in r['attempts']]
    jobs += [(r['raw_solution'],r['reference_answer']) for r in refs.values()]
    checked=batch_verify(jobs); tokenizer=tok(); attempts=[]; pairs=[]; strict_pairs=[]; all_questions=[]
    for r in old:
        ref=refs[r['sample_id']]; cleaned=clean_problem(ref['problem'])
        raw_check=checked[sha([ref['raw_solution'],ref['reference_answer']])]
        selected=None; strict=None
        for a in r['attempts']:
            ver=checked[sha([a['solution'],ref['reference_answer']])]
            # Audit against ORIGINAL problem. Cleaning cannot retroactively change the request.
            s=admit(a['solution'],ref['problem'],a['finish_reason'],tokenizer,ver,False)
            n=admit(a['solution'],ref['problem'],a['finish_reason'],tokenizer,ver,True)
            row={'sample_id':r['sample_id'],'attempt':a['attempt'],'request_id':a['request_id'],
                 'strict':s,'normalized':n,'source_cleaning':cleaned,
                 'old_verification_agrees':ver.get('correct')==a['math_verify'].get('correct')}
            attempts.append(row)
            if s['accepted_candidate'] and strict is None: strict=row
            if n['accepted_candidate'] and selected is None: selected=row
        common={'sample_id':r['sample_id'],'problem_type':r['problem_type'],
                'problem':ref['problem'],'reference_answer':ref['reference_answer'],
                'raw_solution':ref['raw_solution'],'original_math_verify':raw_check,
                'training_approved':False,'human_review':'pending',
                'source_cleaning':cleaned,'selection_scope':'retrospective_development_cache_only'}
        if strict and raw_check.get('correct') and not cleaned['review_flags']:
            strict_pairs.append({**common,'compact_solution':strict['strict']['text'],
                                 'attempt':strict['attempt']})
        if selected and raw_check.get('correct') and not cleaned['review_flags']:
            pairs.append({**common,'compact_solution':selected['normalized']['text'],
                          'attempt':selected['attempt'],'normalization':selected['normalized']['operations'],
                          'compact_tokens':selected['normalized']['answer_tokens']})
        all_questions.append({'sample_id':r['sample_id'],'strict_candidate':strict is not None,
            'normalized_candidate':selected is not None,'original_verified':raw_check.get('correct',False),
            'source_review_flags':cleaned['review_flags']})
    summary={'version':VERSION,'questions':len(old),'cached_attempts':len(attempts),
        'strict_candidate_questions':sum(r['strict_candidate'] for r in all_questions),
        'normalized_candidate_questions':sum(r['normalized_candidate'] for r in all_questions),
        'strict_common_pairs':len(strict_pairs),'normalized_common_pairs':len(pairs),
        'math_verify_disagreements':sum(not r['old_verification_agrees'] for r in attempts),
        'normalization_operations':dict(Counter(x for r in pairs for x in r['normalization'])),
        'difficulty_labeling':'archived_negative_result_no_new_calls',
        'llm_process_judge_calls':0,'api_calls':0,'training_approved':False,
        'process_correctness':'not_certified_by_deterministic_rules','human_review_completed':False,
        'limitation':'Retrospective rules tuned on this cache; not independent new-pilot evidence.'}
    dump(OUT/'attempts.json',attempts);dump(OUT/'questions.json',all_questions)
    dump(OUT/'normalized_pairs.json',pairs);dump(OUT/'strict_pairs.json',strict_pairs)
    dump(OUT/'summary.json',summary)
    assert before==frozen_identity()
    dump(OUT/'manifest.json',{'input_identity':before,'files':{p.name:file_sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'},'old_data_unchanged':True})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def replay(selections):
    import pyarrow.parquet as pq
    groups=defaultdict(dict)
    meta=read(ROOT/'data/screening_sources/openr1_source.json')
    digests={r['path']:r['sha256'] for r in meta['files']}
    for r in selections: groups[r['trajectory']['source_path']][r['trajectory']['source_row']]=r
    found=[]
    for path,positions in groups.items():
        assert file_sha(ROOT/path)==digests[path]
        offset=0
        for batch in pq.ParquetFile(ROOT/path).iter_batches(batch_size=512,
                columns=['problem','answer','generations','is_reasoning_complete','correctness_math_verify']):
            for j,raw in enumerate(batch.to_pylist()):
                if offset+j not in positions: continue
                s=positions[offset+j]; i=s['trajectory']['trajectory_index']
                assert raw['is_reasoning_complete'][i] is True and raw['correctness_math_verify'][i] is True
                assert digest_text(raw['generations'][i])==s['trajectory']['text_sha256']
                found.append({'sample_id':s['question_hash'],'problem':raw['problem'],
                    'reference_answer':raw['answer'],'raw_solution':raw['generations'][i],
                    'selection':s,'source_revision':meta['revision'],'split':'train'})
            offset+=batch.num_rows
    assert len(found)==len(selections)
    return found


def prepare():
    if (DATA/'manifest.json').exists():
        m=read(DATA/'manifest.json')
        assert m['protected_identity']==frozen_identity()
        assert m['code_identity']==code_identity(), 'Code changed: use a new version'
        for name,h in m['files'].items():assert file_sha(DATA/name)==h
        print('Independent pilot already frozen; verified');return
    DATA.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    protected=frozen_identity()
    prior=read(OLD_DATA/'verifier_references.json')
    excluded=prior+sum([read(OLD_DATA/(name+'.json')) for name in ('anchors','calibration_dev','frozen_acceptance')],[])
    index=ScreeningSimilarityIndex(.85)
    for r in excluded:index.add(r)
    for name in ['math_dev','math500_test','gsm8k_test']:
        for r in jsonl(ROOT/read(ROOT/'data/math_manifest.json')['files'][name]['path']):index.add(r)
    proposed=read(ROOT/'reports/project1_sft_v2_plan_20261023/coverage_proposal.json')
    oldids={r['sample_id'] for r in prior}
    # Read-only replay, stable candidate order; no outcome-based sampling.
    short=[]
    for kind,quota in QUOTAS.items():
        pool=sorted([r for r in proposed if r['problem_type']==kind and r['question_hash'] not in oldids],
                    key=lambda r:sha('independent-compact-v2/'+r['question_hash']))
        short+=pool[:quota+6]
    raw=replay(short); buckets=defaultdict(list)
    order={r['question_hash']:n for n,r in enumerate(short)}
    for r in sorted(raw,key=lambda r:order[r['sample_id']]):buckets[r['selection']['problem_type']].append(r)
    selected=[];rejected=[]
    for kind,quota in QUOTAS.items():
        for r in buckets[kind]:
            c=clean_problem(r['problem']);hit=index.match(c['problem'])
            if c['review_flags'] or hit:
                rejected.append({'sample_id':r['sample_id'],'flags':c['review_flags'],'overlap':hit});continue
            r['original_problem']=r['problem'];r['problem']=c['problem'];r['source_cleaning']=c
            selected.append(r);index.add({'sample_id':r['sample_id'],'problem':r['problem']})
            if sum(x['selection']['problem_type']==kind for x in selected)==quota:break
        assert sum(x['selection']['problem_type']==kind for x in selected)==quota,kind
    assert len(selected)==40
    selected.sort(key=lambda r:r['sample_id'])
    dump(DATA/'verifier_references.json',selected)
    requests=[{k:r[k] for k in ['sample_id','problem','source_revision','split']} for r in selected]
    dump(DATA/'blind_requests.json',requests);dump(DATA/'selection_exclusions.json',rejected)
    policy={'version':VERSION,'generation_mode':'blind_independent_compact_solution',
        'model':'deepseek-flash','thinking':{'type':'disabled'},'temperature':.4,'max_tokens':2400,
        'fallback_teacher':'disabled; future pro rescue requires a separate experiment version',
        'budget_usd_reserved':2.0,'max_requests':135,'max_generation_attempts':3,'workers':4,
        'answer_tokens_min':150,'answer_tokens_max':1500,'total_tokens_max':3800,
        'selection':'first accepted deterministic candidate after documented wrapping normalization',
        'system':SYSTEM,'system_sha256':digest_text(SYSTEM),'math_verify_version':'0.9.0',
        'difficulty_calls':False,'llm_judge_calls':False,'human_review':'pending',
        'engineering_gate':{'candidate_rate_min':.6,'first_strict_structure_rate_min':.9},
        'training_approved':False,'automatic_bulk_expansion':False,
        'format_semantics':'think tags wrap visible written solution, not certified private reasoning',
        'independence':'new question IDs and no Jaccard>=.85 against previous 50 + 135 calibration questions and protected dev/test',
        'limits':'same source pool, not IID random sample; no difficulty-distribution guarantee'}
    dump(DATA/'policy.json',policy)
    dump(DATA/'manifest.json',{'protected_identity':protected,'code_identity':code_identity(),
        'files':{p.name:file_sha(p) for p in DATA.iterdir() if p.is_file() and p.name!='manifest.json'},
        'counts':dict(Counter(r['selection']['problem_type'] for r in selected)),
        'prior_pilot_overlap':0,'generation_input_keys':['sample_id','problem','source_revision','split']})
    assert protected==frozen_identity()
    print('Independent 40-question pilot frozen; no API request',flush=True)


def code_identity():
    return {p:file_sha(ROOT/p) for p in ['src/synthesis/compact_v2.py',
        'scripts/compact_distillation_v2.py','scripts/math_verify_pilot_worker.py',
        'src/synthesis/deepseek_pilot.py','src/data/openr1_quality.py']}


def run(live=False,key_prompt=False):
    prepare(); m=read(DATA/'manifest.json'); policy=read(DATA/'policy.json')
    requests=read(DATA/'blind_requests.json');refs={r['sample_id']:r for r in read(DATA/'verifier_references.json')}
    allowed={'sample_id','problem','source_revision','split'}
    assert all(set(r)==allowed for r in requests)
    dump(REPORT/'blind_payload_audit.json',{'payload_hashes':{r['sample_id']:sha(payload(r['problem'])) for r in requests},
        'reference_or_original_sent':False,'difficulty_or_process_judge_calls':0,'generation_model':policy['model']})
    if not live: print('Dry run: 40 blind payloads, 0 network requests');return
    key=os.environ.get('DEEPSEEK_API_KEY') or (getpass.getpass('API key (hidden): ') if key_prompt else '')
    if not key: raise RuntimeError('DEEPSEEK_API_KEY missing; no paid request dispatched')
    client=PilotClient(REPORT/'api_cache',key,budget=policy['budget_usd_reserved'],max_requests=policy['max_requests'])
    before=REPORT/'balance_before.json'
    if not before.exists():dump(before,client.balance())
    tokenizer=tok();start=time.monotonic()
    def one(r):
        ref=refs[r['sample_id']];attempts=[];chosen=None
        rawcheck=check_math(ref['raw_solution'],ref['reference_answer'])
        for n in range(1,policy['max_generation_attempts']+1):
            response=client.call(payload(r['problem']),'solve-v2/'+r['sample_id']+'/'+str(n))
            choice=response['response']['choices'][0];text=choice['message'].get('content') or ''
            ver=check_math(text,ref['reference_answer'])
            strict=admit(text,r['problem'],choice.get('finish_reason'),tokenizer,ver,False)
            normalized=admit(text,r['problem'],choice.get('finish_reason'),tokenizer,ver,True)
            entry={'attempt':n,'raw_text':text,'strict':strict,'normalized':normalized,
                'request_id':response['response'].get('id'),'returned_model':response['response'].get('model'),
                'latency_seconds':response['latency_seconds'],'usage':response['response'].get('usage',{})}
            attempts.append(entry)
            if normalized['accepted_candidate']:
                chosen=entry;break
        return {'sample_id':r['sample_id'],'problem_type':ref['selection']['problem_type'],
            'attempts':attempts,'selected_attempt':chosen['attempt'] if chosen else None,
            'candidate':bool(chosen) and rawcheck.get('correct',False),'original_math_verify':rawcheck,
            'training_approved':False,'human_review':'pending','process_correctness':'not_certified'}
    results=[]
    with ThreadPoolExecutor(max_workers=policy['workers']) as pool:
        futures={pool.submit(one,r):r for r in requests}
        for f in as_completed(futures):
            r=futures[f]
            try:result=f.result()
            except Exception as e:result={'sample_id':r['sample_id'],'status':'failed','error_type':type(e).__name__,'training_approved':False}
            results.append(result);dump(REPORT/'partial_results.json',results)
            print(f'Independent pilot {len(results)}/40',flush=True)
    results.sort(key=lambda r:r['sample_id']);dump(REPORT/'results.json',results)
    dump(REPORT/'balance_after.json',client.balance())
    aa=[a for r in results for a in r.get('attempts',[])]
    first=[r['attempts'][0] for r in results if r.get('attempts')]
    selected=[(r,next(a for a in r['attempts'] if a['attempt']==r['selected_attempt'])) for r in results if r.get('candidate')]
    raw=[];compact=[]
    for r,a in selected:
        ref=refs[r['sample_id']]
        base={'sample_id':r['sample_id'],'problem':ref['problem'],'source_revision':ref['source_revision'],
              'source_cleaning':ref['source_cleaning'],'teacher_model':a['returned_model'],
              'training_approved':False,'human_review':'pending','selection':'first_deterministic_acceptance'}
        raw.append({**base,'solution':ref['raw_solution'],'solution_kind':'openr1_original'})
        compact.append({**base,'solution':a['normalized']['text'],'solution_kind':'api_blind_compact',
                        'normalization':a['normalized']['operations'],'request_id':a['request_id']})
    write_jsonl(REPORT/'paired_raw_candidates.jsonl',raw);write_jsonl(REPORT/'paired_compact_candidates.jsonl',compact)
    cost=sum(x.get('usage_cost_peak_usd',0) for x in client.ledger if x['state']=='complete')
    summary={'questions':40,'completed_question_jobs':len(first),'failed_jobs':len(results)-len(first),
        'generation_attempts':len(aa),'first_answer_verified':sum(a['strict']['math_verify'].get('correct',False) for a in first),
        'any_answer_verified':sum(any(a['strict']['math_verify'].get('correct') for a in r.get('attempts',[])) for r in results),
        'first_strict_structure_pass':sum(a['strict']['structure_pass'] for a in first),
        'first_strict_admitted':sum(a['strict']['accepted_candidate'] for a in first),
        'first_normalized_admitted':sum(a['normalized']['accepted_candidate'] for a in first),
        'strict_candidate_questions':sum(any(a['strict']['accepted_candidate'] for a in r.get('attempts',[])) for r in results),
        'normalized_common_pairs':len(compact),'returned_models':dict(Counter(a['returned_model'] for a in aa)),
        'per_type':{k:{'n':n,'candidates':sum(r.get('candidate',False) for r in results if r.get('problem_type')==k)} for k,n in QUOTAS.items()},
        'usage_estimated_peak_usd':cost,'usage_estimated_offpeak_usd':cost/2,
        'reserved_usd':sum(x['reserved_usd'] for x in client.ledger),'wall_seconds':time.monotonic()-start,
        'difficulty_calls':0,'llm_judge_calls':0,'training_approved':False,'automatic_bulk_expansion':False,
        'process_audit':'pending; deterministic structure is not mathematical proof'}
    summary['engineering_gate_passed']=(len(compact)/40>=policy['engineering_gate']['candidate_rate_min'] and
        summary['first_strict_structure_pass']/40>=policy['engineering_gate']['first_strict_structure_rate_min'])
    dump(REPORT/'summary.json',summary)
    assert m['protected_identity']==frozen_identity()
    dump(REPORT/'delivery_manifest.json',{'protected_identity':m['protected_identity'],'code_identity':code_identity(),
        'files':{str(p.relative_to(REPORT)):file_sha(p) for p in REPORT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'},
        'training_approved':False})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['reaudit','prepare','run']);p.add_argument('--live',action='store_true');p.add_argument('--key-prompt',action='store_true')
    a=p.parse_args()
    if a.action=='reaudit':reaudit()
    elif a.action=='prepare':prepare()
    else:run(a.live,a.key_prompt)
