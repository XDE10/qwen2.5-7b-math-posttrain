"""Bounded blind Flash/Pro teacher pilot: 12 train-pool questions, one call each.

No GPU, no automatic expansion, no training approval. Existing R1 outputs are
offline references and are never included in the generation payload.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/qfft_teacher_pilot_20261028_v60'
MODELS = ('deepseek-flash', 'deepseek-v4-pro')
SYSTEM = r'Please reason step by step, and put your final answer within \boxed{}.'
MAX_OUTPUT = 16384
S1_SELECTION = {'Algebra': [690, 702, 826], 'Geometry': [53, 225, 830],
                'Number Theory': [165, 248, 673], 'Combinatorics': [374]}
RAW_SELECTION = [413, 592]


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def normalized(text):
    return ''.join(text.split())


def payload(model, question):
    if model not in MODELS:
        raise ValueError('Only the two authorized teacher models')
    return {'model':model,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':question}],
            'thinking':{'type':'enabled'},'reasoning_effort':'high','max_tokens':MAX_OUTPUT,
            'top_p':1.0,'stream':False}


def prepare():
    import pandas as pd
    frame=pd.read_parquet(ROOT/'artifacts/research_s1_7b/s1K-1.1.parquet')
    raw=rows(ROOT/'data/portable_release/project1_raw_paired.jsonl')
    blocked=set()
    for name in ['math_dev.jsonl','math500_test.jsonl','gsm8k_test.jsonl']:
        blocked.update(normalized(r['problem']) for r in rows(ROOT/'data/math'/name))
    selected=[]
    for category,indices in S1_SELECTION.items():
        for index in indices:
            r=frame.iloc[index]
            meta=ast.literal_eval(r['metadata']) if isinstance(r['metadata'],str) else r['metadata']
            assert meta['unique_id'].startswith('train/') and r['deepseek_grade']=='Yes'
            selected.append({'sample_id':f's1k11/row/{index:04d}','category':category,
                'problem':str(r['question']),'reference_answer':str(meta['answer']),
                'source_original_id':meta['unique_id'],'reference_teacher':'DeepSeek-R1 dataset response',
                'r1_reasoning':str(r['deepseek_thinking_trajectory']),'r1_final':str(r['deepseek_attempt']),
                'reference_selection_note':'R1 answer-labelled Yes before this pilot; not a fresh randomized R1 generation.'})
    for index in RAW_SELECTION:
        r=raw[index]
        assert r['training_approved'] and r['split']=='train' and r['arm']=='raw'
        cut=r['solution'].index('</think>')
        selected.append({'sample_id':'openr1/'+r['sample_id'],'category':'Combinatorics',
            'problem':r['problem'],'reference_answer':str(r['reference_answer']),
            'source_original_id':r['source_position']['source_uuid'],
            'reference_teacher':'Existing approved OpenR1 raw response',
            'r1_reasoning':r['solution'][len('<think>'):cut],'r1_final':r['solution'][cut+len('</think>'):],
            'reference_selection_note':'Previously approved raw trajectory; not a fresh randomized R1 generation.'})
    assert len(selected)==12 and len({r['sample_id'] for r in selected})==12
    assert Counter(r['category'] for r in selected)=={'Algebra':3,'Geometry':3,'Number Theory':3,'Combinatorics':3}
    assert all(normalized(r['problem']) not in blocked for r in selected)
    protocol={'scope':'12 frozen training-pool questions x 2 teachers x 1 completion; no expansion',
        'models':list(MODELS),'thinking':'enabled','reasoning_effort':'high','max_tokens':MAX_OUTPUT,
        'top_p':1.0,'temperature':'omitted; ineffective in thinking mode','tools':False,
        'source_rows_sha256':digest(selected),'question_ids':[r['sample_id'] for r in selected],
        'selection':'Purposive diagnostic: 3 each algebra/geometry/number theory/combinatorics; not representative benchmark accuracy.',
        'teacher_payload':'Original question plus common system instruction only; never reference answers or R1 trajectories.',
        'training_approved':False,'automatic_expansion':False,
        'reference_note':'Existing answer-verified R1 reference selection is biased toward correct R1 answers; cannot rank general model ability from this pilot.'}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists() and read(OUT/'protocol.json')!=protocol:
        raise RuntimeError('Pilot protocol differs; preserve existing paid outputs')
    save(OUT/'protocol.json',protocol)
    save(OUT/'source_questions_and_r1.json',selected)
    return selected


def generate(model, row, key):
    body=payload(model,row['problem'])
    tag=model+'--'+digest(row['sample_id'])[:20]
    request_path=OUT/'requests'/(tag+'.json')
    response_path=OUT/'api_responses'/(tag+'.json')
    if request_path.exists() and read(request_path)!=body:
        raise RuntimeError('Saved request changed')
    save(request_path,body)
    if response_path.exists():
        return read(response_path)
    request=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    start=time.perf_counter()
    try:
        with urllib.request.urlopen(request,timeout=600) as response:
            result=json.load(response)
    except urllib.error.HTTPError as error:
        save(OUT/'errors'/(tag+'.json'),{'sample_id':row['sample_id'],'model':model,'http_status':error.code,
              'note':'No automatic retry; paid usage, if any, not returned.'})
        raise RuntimeError('Teacher API HTTP '+str(error.code)) from None
    except Exception as error:
        save(OUT/'errors'/(tag+'.json'),{'sample_id':row['sample_id'],'model':model,'error_type':type(error).__name__,
              'note':'No automatic retry; server-side usage may be unknown.'})
        raise RuntimeError('Teacher request failed; no credential logging') from None
    record={'sample_id':row['sample_id'],'model_requested':model,'request_sha256':digest(body),
            'latency_seconds':time.perf_counter()-start,'completed_utc':datetime.now(timezone.utc).isoformat(),
            'response':result,'training_approved':False}
    # Save the complete paid response, before validating or scoring its content.
    save(response_path,record)
    return record


def score(selected):
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct/tokenizer.json'))
    spec=importlib.util.spec_from_file_location('pilot_judge60',ROOT/'tools/evaluate_sft_ds_v29.py')
    j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
    scored=[]
    for row in selected:
        rtext=row['r1_reasoning']+'<|im_start|>answer\n'+row['r1_final']
        scored.append({'sample_id':row['sample_id'],'category':row['category'],'model':'R1_existing_reference',
            'native_grade':j.grade_bounded(row['r1_final'],row['reference_answer'],ROOT),
            'student_qwen_tokens':len(tok.encode(rtext,add_special_tokens=False).ids),
            'reference_only':True,'training_approved':False})
        for model in MODELS:
            tag=model+'--'+digest(row['sample_id'])[:20]
            path=OUT/'api_responses'/(tag+'.json')
            if not path.exists():continue
            record=read(path);response=record['response'];choice=response['choices'][0];message=choice['message']
            reasoning=message.get('reasoning_content') or '';final=message.get('content') or ''
            text=reasoning+'<|im_start|>answer\n'+final
            count=len(tok.encode(text,add_special_tokens=False).ids)
            grade=j.grade_bounded(final,row['reference_answer'],ROOT)
            complete=choice['finish_reason']=='stop' and bool(reasoning.strip()) and bool(final.strip())
            scored.append({'sample_id':row['sample_id'],'category':row['category'],'model':model,
                'model_returned':response.get('model'),'system_fingerprint':response.get('system_fingerprint'),
                'native_grade':grade,'finish_reason':choice['finish_reason'],'complete_thinking_and_final':complete,
                'student_qwen_tokens':count,'fits_student_16k':count<=16384,
                'latency_seconds':record['latency_seconds'],'usage':response.get('usage',{}),
                'answer_correct_complete_fits16k':grade['correct'] and complete and count<=16384,
                'training_approved':False})
    save(OUT/'native_scores.json',scored)
    summary={}
    for model in ('R1_existing_reference',)+MODELS:
        rr=[r for r in scored if r['model']==model]
        usage=Counter()
        for r in rr:
            for key,value in r.get('usage',{}).items():
                if isinstance(value,int):usage[key]+=value
        summary[model]={'count':len(rr),'native_answer_correct':sum(r['native_grade']['correct'] for r in rr),
            'mean_student_qwen_tokens':sum(r['student_qwen_tokens'] for r in rr)/max(1,len(rr)),
            'answer_complete_fits16k':sum(r.get('answer_correct_complete_fits16k',False) for r in rr),
            'usage':dict(usage)}
        if model == 'R1_existing_reference':
            summary[model].pop('answer_complete_fits16k')
            summary[model]['reference_only'] = True
        if model in MODELS:
            # Official public rates: cache hit/miss/output per 1M, USD.
            rates=(.003,.15,.6) if model=='deepseek-flash' else (.022,.66,1.98)
            low=sum(usage[k]*v for k,v in zip(('prompt_cache_hit_tokens','prompt_cache_miss_tokens','completion_tokens'),rates))/1e6
            summary[model]['estimated_cost_usd_range_offpeak_to_peak']=[low,2*low]
    save(OUT/'summary.json',{'models':summary,'note':'Native answer check only; derivations still need inspection.12-question diagnostic, not a benchmark. R1 references preselected; no fresh R1 API run. Costs are public-rate estimates, not bill readback.','training_approved':False})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--score-only',action='store_true')
    args=parser.parse_args()
    selected=prepare()
    print('FROZEN_PILOT: 12 train-only questions; 24 maximum single-attempt generations; no GPU',flush=True)
    if args.run:
        key=read(Path('C:/Users/Mr.Z/.codex/private/project1/deepseek.json'))['DEEPSEEK_API_KEY']
        errors=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks={pool.submit(generate,model,row,key):(model,row['sample_id']) for row in selected for model in MODELS}
            for n,future in enumerate(as_completed(tasks),1):
                model,sid=tasks[future]
                try:
                    record=future.result();choice=record['response']['choices'][0]
                    print(f'TEACHER {n}/24 saved {model} {sid} finish={choice["finish_reason"]}',flush=True)
                except Exception as error:
                    errors.append({'model':model,'sample_id':sid,'error_type':type(error).__name__})
                    print(f'TEACHER {n}/24 failed {model} {sid}',flush=True)
        if errors:save(OUT/'request_errors.json',errors)
        score(selected)
        if errors:raise RuntimeError('Some requests failed; successes saved. No automatic paid retries.')
    elif args.score_only:score(selected)


if __name__=='__main__':main()
