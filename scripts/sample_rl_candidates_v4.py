"""Single-GPU vLLM screening, durable sharded raw/graded journals, no Drive gate.

Only runs when explicitly invoked with local verified 7B weights. Never trains.
"""
import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from src.training.frozen_verifier_v4 import Worker,sha,digest
from src.training.math_reward_v2 import category
from src.training.rollout_pipeline_v4 import read,jsonl,indexed,append,write_new,trial_seed,selection,load_prepared,validate_model,Metrics


def grade_one(raw,row,worker):
    result=worker.check(raw['text'],row['reference_answer'])
    return {'sample_id':raw['sample_id'],'trial':raw['trial'],'seed':raw['seed'],'raw_sha256':digest(raw),
            'finish_reason':raw['finish_reason'],'category':category(result),'verification':result}


def complete(root,rows,protocol):
    raw=[];graded=[];files={}
    for shard in range(protocol['num_shards']):
        for kind,target in [('raw',raw),('graded',graded)]:
            path=root/kind/f'shard-{shard:05d}.jsonl'
            if path.exists():target.extend(jsonl(path));files[str(path.relative_to(root))]=sha(path)
    r=indexed(raw);g=indexed(graded)
    expected={(row['sample_id'],trial) for row in rows for trial in range(protocol['num_samples'])}
    if set(r)!=expected or set(g)!=expected:return False
    for key,value in g.items():
        if value['raw_sha256']!=digest(r[key]):raise ValueError('Graded/raw linkage changed')
    summary={'schema':'project1-complete-sampling-v4','identity_sha256':sha(root/'sampling_identity.json'),
             'candidate_count':len(rows),'trial_count':len(expected),'files':files,'complete':True}
    path=root/'sampling_complete.json'
    if path.exists():
        if read(path)!=summary:raise ValueError('Completed run mutated')
    else:write_new(path,summary)
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidates',required=True);p.add_argument('--preparation-seal',required=True)
    p.add_argument('--model-path',required=True);p.add_argument('--model-identity',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--num-samples',type=int,choices=[4,8],default=8)
    p.add_argument('--num-shards',type=int,default=16);p.add_argument('--shard-id',default='all')
    p.add_argument('--batch-prompts',type=int,default=8)
    p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--temperature',type=float,default=.8);p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--max-num-seqs',type=int,default=32);p.add_argument('--gpu-memory-utilization',type=float,default=.85)
    p.add_argument('--swanlab-mode',choices=['offline','online','disabled'],default='offline')
    p.add_argument('--time-budget-seconds',type=float,default=0,help='Pause after a durable batch; incomplete runs cannot be approved')
    a=p.parse_args()
    from filelock import FileLock
    lock_root=Path(a.output_root).resolve();lock_root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(lock_root/'.sampling.lock'),timeout=0):
        execute(a)


def execute(a):
    if a.num_shards<1 or a.batch_prompts<1 or not 1<=a.max_new_tokens<=2048:raise ValueError('Invalid shard/batch/token budget')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Sampler requires exactly one visible CUDA GPU; CPU uses tests/mocks only')
    rows,seal=load_prepared(a.candidates,a.preparation_seal)
    model_identity=validate_model(a.model_path,a.model_identity)
    token_path=Path(a.preparation_seal).parent/'prompt_tokens.jsonl'
    if sha(token_path)!=seal['prompt_tokens_sha256']:raise ValueError('Pretokenized prompts changed')
    tokens={r['sample_id']:r['prompt_token_ids'] for r in jsonl(token_path)}
    if set(tokens)!={r['sample_id'] for r in rows}:raise ValueError('Prompt coverage mismatch')
    from importlib.metadata import version
    if version('vllm')!='0.11.2' or version('transformers')!='4.57.6':raise ValueError('Use the pinned separate sampler environment')
    protocol={'schema':'project1-sampling-v4','candidate_sha256':sha(a.candidates),'preparation_sha256':sha(a.preparation_seal),
              'grader_identity_sha256':seal['grader_identity_sha256'],'model_identity':model_identity,
              'num_samples':a.num_samples,'num_shards':a.num_shards,'seed':a.seed,'temperature':a.temperature,'top_p':a.top_p,'top_k':-1,
              'max_new_tokens':a.max_new_tokens,'max_prompt_length':1752,'max_model_len':1752+a.max_new_tokens,
              'max_num_seqs':a.max_num_seqs,'gpu_memory_utilization':a.gpu_memory_utilization,'batch_prompts':a.batch_prompts,
              'sampler_source_sha256':sha(__file__),'versions':{n:version(n) for n in ['vllm','torch','transformers','tokenizers','sympy']},
              'pipeline_source_sha256':sha(Path(__file__).resolve().parents[1]/'src/training/rollout_pipeline_v4.py'),
              'truncation_policy':'any non-stop finish makes the candidate ineligible; no resampling to repair wrong answers',
              'selection_cap_policy':'seeded SHA256 ordering; maximum3000; no relaxation for shortfall'}
    root=Path(a.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    ip=root/'sampling_identity.json'
    if ip.exists() and read(ip)!=protocol:raise ValueError('Resume identity changed; use a separate output-root (4 and 8 cannot mix)')
    if not ip.exists():write_new(ip,protocol)
    append(root/'runtime_environments.jsonl',{'gpu':torch.cuda.get_device_name(0),'torch':version('torch'),'cuda_runtime':torch.version.cuda})
    # Store portable, self-contained approval inputs; preserve bytes and identity.
    copies={'candidate_pool.jsonl':Path(a.candidates),'preparation_seal.json':Path(a.preparation_seal),
            'prompt_tokens.jsonl':token_path,'protected_dev_ids_never_train.json':Path('data/portable_release/reserved_dev_ids_never_train.json')}
    for name,source in copies.items():
        dst=root/name
        if dst.exists() and sha(dst)!=sha(source):raise ValueError('Sampling copy changed')
        if not dst.exists():dst.write_bytes(source.read_bytes())
    ids=list(range(a.num_shards)) if a.shard_id=='all' else [int(a.shard_id)]
    if any(i<0 or i>=a.num_shards for i in ids):raise ValueError('shard-id outside range')
    from vllm import LLM,SamplingParams
    engine=None
    metrics=Metrics(root,a.swanlab_mode,'E0-rl-screening',{'num_samples':a.num_samples,'candidate_count':len(rows),'max_new_tokens':a.max_new_tokens})
    workers=[Worker() for _ in range(4)]
    elapsed_generation=0.;generated_tokens=0;step=0
    started=time.monotonic();new_trials=0
    initial_trials=sum(len(jsonl(root/'raw'/f'shard-{s:05d}.jsonl')) for s in range(a.num_shards))
    try:
        for shard in ids:
            subset=[row for i,row in enumerate(rows) if i%a.num_shards==shard]
            raw_path=root/'raw'/f'shard-{shard:05d}.jsonl';grade_path=root/'graded'/f'shard-{shard:05d}.jsonl'
            raw=indexed(jsonl(raw_path));graded=indexed(jsonl(grade_path))
            allowed={(r['sample_id'],t) for r in subset for t in range(a.num_samples)}
            if not set(raw)<=allowed or not set(graded)<=set(raw):raise ValueError('Shard has foreign/orphan trial records')
            for start in range(0,len(subset),a.batch_prompts):
                batch=subset[start:start+a.batch_prompts]
                pending=[(r,t) for r in batch for t in range(a.num_samples) if (r['sample_id'],t) not in raw]
                if pending:
                    if engine is None:
                        engine=LLM(model=str(Path(a.model_path).resolve()),tokenizer='data/tokenizers/Qwen2.5-7B-Instruct',
                                   dtype='bfloat16',tensor_parallel_size=1,max_model_len=protocol['max_model_len'],
                                   max_num_seqs=a.max_num_seqs,gpu_memory_utilization=a.gpu_memory_utilization,
                                   enable_prefix_caching=True,enable_chunked_prefill=True,max_num_batched_tokens=8192,
                                   seed=a.seed,generation_config='vllm')
                    params=[SamplingParams(n=1,temperature=a.temperature,top_p=a.top_p,top_k=-1,max_tokens=a.max_new_tokens,
                                           seed=trial_seed(a.seed,r['sample_id'],t)) for r,t in pending]
                    begin=time.monotonic()
                    outputs=engine.generate([{'prompt_token_ids':tokens[r['sample_id']]} for r,t in pending],params,use_tqdm=False)
                    elapsed_generation+=time.monotonic()-begin
                    if len(outputs)!=len(pending):raise ValueError('Backend output alignment mismatch')
                    for (row,t),response in zip(pending,outputs):
                        if len(response.outputs)!=1:raise ValueError('Unexpected backend generation count')
                        o=response.outputs[0]
                        record={'sample_id':row['sample_id'],'trial':t,'seed':trial_seed(a.seed,row['sample_id'],t),
                                'text':o.text,'token_ids':list(o.token_ids),'finish_reason':o.finish_reason,'stop_reason':o.stop_reason,
                                'prompt_tokens':len(tokens[row['sample_id']])}
                        if len(record['token_ids'])>a.max_new_tokens:raise ValueError('Backend completion budget exceeded')
                        append(raw_path,record);raw[(row['sample_id'],t)]=record;generated_tokens+=len(record['token_ids']);new_trials+=1
                # Raw predictions are durable before grading; a crash never regenerates a persisted answer.
                todo=[(r,t) for r in batch for t in range(a.num_samples) if (r['sample_id'],t) not in graded]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for offset in range(0,len(todo),4):
                        work=todo[offset:offset+4]
                        futures=[pool.submit(grade_one,raw[(r['sample_id'],t)],r,workers[j]) for j,(r,t) in enumerate(work)]
                        for future in futures:
                            rec=future.result();append(grade_path,rec);graded[(rec['sample_id'],rec['trial'])]=rec
                step+=1;rate=generated_tokens/max(elapsed_generation,1e-9)
                remaining=sum((r['sample_id'],t) not in raw for r in subset for t in range(a.num_samples))
                remaining_global=len(rows)*a.num_samples-initial_trials-new_trials
                eta=remaining_global*(time.monotonic()-started)/new_trials if new_trials else None
                metrics.log({'generated_tokens':generated_tokens,'generation_seconds':elapsed_generation,'output_tokens_per_second':rate,
                             'shard_remaining_trials':remaining,'num_samples':a.num_samples,'remaining_seconds_estimate':eta},step)
                print(json.dumps({'shard':shard,'persisted_trials':len(raw),'output_tokens_per_second':rate,
                                  'remaining_seconds_estimate':eta,
                                  'eta_note':'estimate after measured lengths; 1-2h is not guaranteed'}),flush=True)
                if a.time_budget_seconds>0 and time.monotonic()-started>=a.time_budget_seconds:
                    print(json.dumps({'paused_for_time_budget':True,'approval':'incomplete run is not approvable'}));return
        done=complete(root,rows,protocol)
        print(json.dumps({'all_shards_complete':done,'approval':'run approve_rl_pool_v4.py only after all shards complete'}))
    finally:
        for worker in workers:worker.close()
        metrics.finish()


if __name__=='__main__':main()
