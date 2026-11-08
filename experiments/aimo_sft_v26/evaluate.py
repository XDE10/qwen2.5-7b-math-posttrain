"""Resumable pass@1 estimation, 32768 output tokens, AIME 64 samples/question."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.aimo_sft_v26.common import config,prompt_ids,read,rows,save,sha


def append(path,value):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')


def summary(records,expected,samples):
    return {'responses':len(records),'expected_responses':expected,'complete':len(records)==expected,
        'samples_per_question':samples,'pass_at_1':sum(r['correct'] for r in records)/len(records) if records else None,
        'grade_categories':dict(Counter(r['grade_category'] for r in records)),
        'finish_reasons':dict(Counter(r['finish_reason'] for r in records)),
        'mean_output_tokens':sum(r['output_tokens'] for r in records)/len(records) if records else None,
        'metric':'Mean correctness of individual sampled answers; NOT pass@64 or majority-vote accuracy'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--variant',choices=['base','sft'],required=True)
    p.add_argument('--benchmark',nargs='+',choices=['dev','aime2024','aime2025','math500'],required=True)
    p.add_argument('--time-budget-minutes',type=float,default=90)
    a=p.parse_args();c=config();protocol=c['evaluation'];out=Path(a.output_root);out.mkdir(parents=True,exist_ok=True)
    sources=read(ROOT/'reports/aimo_sft_20261026_v26/evaluation_sources.json')
    from src.training.frozen_verifier_v4 import Worker,grader_identity
    from src.training.math_reward_v2 import category
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True)
    for name,expected in c['tokenizer_sha256'].items():
        if sha(Path(a.model_path)/name)!=expected:raise ValueError('Wrong Qwen Instruct tokenizer')
    model_identity=read(Path(a.model_path)/('projection_model_identity.json' if a.variant=='base' else 'sft_identity.json'))
    llm=None;worker=Worker();started=time.monotonic()
    try:
        for benchmark in a.benchmark:
            directory=out/benchmark;directory.mkdir(exist_ok=True)
            path=ROOT/sources[benchmark]['file']
            if sha(path)!=sources[benchmark]['sha256']:raise ValueError('Evaluation inputs changed')
            selected=rows(path);by_id={r['sample_id']:r for r in selected}
            samples=1 if benchmark in ['dev','math500'] else protocol['aime_samples_per_question']
            identity={'model':model_identity,'variant':a.variant,'benchmark':sources[benchmark],
                'protocol':protocol,'samples':samples,'seed':c['seed'],'grader':grader_identity(),'code':sha(__file__)}
            if (directory/'identity.json').exists() and read(directory/'identity.json')!=identity:
                raise ValueError('Evaluation protocol changed; use another directory')
            save(directory/'identity.json',identity)
            records=rows(directory/'results.jsonl') if (directory/'results.jsonl').exists() else []
            done={(r['sample_id'],r['draw']) for r in records}
            if len(done)!=len(records):raise ValueError('Duplicate evaluation records')
            def grade(raw):
                result=dict(raw);result['grade']=worker.check(raw['text'],by_id[raw['sample_id']]['answer'])
                result['grade_category']=category(result['grade'])
                if result['grade_category'] in ['timeout','verifier_error']:
                    raise RuntimeError('Saved raw output; rerun to retry grading without regeneration')
                result['correct']=result['grade_category']=='equivalent'
                append(directory/'results.jsonl',result);records.append(result)
                done.add((result['sample_id'],result['draw']))
            raw_saved=rows(directory/'raw.jsonl') if (directory/'raw.jsonl').exists() else []
            for raw in raw_saved:
                if (raw['sample_id'],raw['draw']) not in done:grade(raw)
            jobs=[(r,draw) for r in selected for draw in range(samples) if (r['sample_id'],draw) not in done]
            for start in range(0,len(jobs),4):
                if time.monotonic()-started>=a.time_budget_minutes*60:
                    save(directory/'summary.json',summary(records,len(selected)*samples,samples))
                    print('Evaluation budget reached; repeat identical command to resume.',flush=True);return
                if llm is None:
                    from vllm import LLM,SamplingParams
                    llm=LLM(model=a.model_path,tokenizer=a.model_path,dtype='bfloat16',
                        max_model_len=protocol['max_model_len'],max_num_seqs=4,max_num_batched_tokens=2048,
                        enable_chunked_prefill=True,enable_prefix_caching=True,gpu_memory_utilization=.88,
                        hf_overrides={'rope_scaling':protocol['rope_scaling']},
                        generation_config='vllm',seed=c['seed'])
                batch=jobs[start:start+4];prompts=[];params=[]
                for row,draw in batch:
                    ids=prompt_ids(tok,row['problem'])
                    if len(ids)+protocol['max_new_tokens']>protocol['max_model_len']:
                        raise ValueError('Evaluation prompt+generation exceeds context; no truncation')
                    seed=(c['seed']+int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8],16)+draw)%2**31
                    prompts.append({'prompt_token_ids':ids})
                    params.append(SamplingParams(max_tokens=protocol['max_new_tokens'],temperature=0. if benchmark=='dev' else protocol['temperature'],top_p=1. if benchmark=='dev' else protocol['top_p'],repetition_penalty=1.,seed=seed))
                outputs=llm.generate(prompts,params,use_tqdm=False)
                if len(outputs)!=len(batch):raise RuntimeError('Engine result count mismatch')
                raw_batch=[]
                for (row,draw),output in zip(batch,outputs):
                    answer=output.outputs[0]
                    raw={'sample_id':row['sample_id'],'draw':draw,'text':answer.text,
                         'output_tokens':len(answer.token_ids),'finish_reason':answer.finish_reason}
                    append(directory/'raw.jsonl',raw)
                    raw_batch.append(raw)
                for raw in raw_batch:
                    grade(raw)
                save(directory/'summary.json',summary(records,len(selected)*samples,samples))
                print(f'{a.variant} {benchmark}: {len(records)}/{len(selected)*samples}, sampled pass@1={sum(r["correct"] for r in records)/len(records):.3%}',flush=True)
            save(directory/'summary.json',summary(records,len(selected)*samples,samples))
            if len(records)!=len(selected)*samples:raise RuntimeError('Evaluation incomplete')
            save(directory/'COMPLETE.json',{'records':len(records),'results_sha256':sha(directory/'results.jsonl')})
    finally:worker.close()


if __name__=='__main__':main()
