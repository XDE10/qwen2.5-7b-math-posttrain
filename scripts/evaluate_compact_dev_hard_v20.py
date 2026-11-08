"""v20: completed compact v18 on unchanged dev/hard protocols, no SwanLab."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil

from src.training.eval_identity_v20 import validate_completed_v18
from scripts.evaluate_project1_dev_v9 import DEV_SHA, messages, summary, save_rows
from src.training.frozen_verifier_v4 import sha, digest, grader_identity
from src.training.rollout_pipeline_v4 import read, jsonl, append, write_new, trial_seed, REV
from src.training.eval_helpers_v13 import hard_summary


def cache_merged(run, cache_root, arm):
    run=Path(run); source=run/'merged'
    state=read(run/'training_completion.json'); origin=read(source/'merged_sft_identity.json')
    validate_completed_v18(state,origin,arm)
    if origin!=read(run/'run_identity.json'):
        raise ValueError('Merged/run origin mismatch')
    checks=read(source/'weight_manifest.json')
    if not any(n.endswith('.safetensors') for n in checks):raise ValueError('No merged weights')
    target=Path(cache_root)/f'E1_{arm}_{sha(source/"weight_manifest.json")[:12]}'
    target.mkdir(parents=True,exist_ok=True)
    total=0; missing=0
    for name in checks:
        if Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name:
            raise ValueError('Weight filename escapes model')
        size=(source/name).stat().st_size; total+=size
        local=target/name
        if not local.exists():missing+=size
        elif sha(local)!=checks[name]:raise ValueError('Local model cache changed; preserve it and use a different cache-root')
    if shutil.disk_usage(target).free<missing+1_000_000_000:raise ValueError('Insufficient local model-cache disk space')
    from tqdm import tqdm
    import hashlib
    with tqdm(total=total,unit='B',unit_scale=True,desc=f'Cache {arm} merged model') as bar:
        for name,expected in checks.items():
            src=source/name; dst=target/name
            if dst.exists():bar.update(dst.stat().st_size);continue
            dst.parent.mkdir(parents=True,exist_ok=True); temporary=dst.with_name(dst.name+'.copying')
            h=hashlib.sha256()
            with src.open('rb') as reader,temporary.open('wb') as writer:
                for block in iter(lambda:reader.read(16*1024*1024),b''):
                    writer.write(block);h.update(block);bar.update(len(block))
            if h.hexdigest()!=expected:raise ValueError('Merged file hash differs from training export')
            temporary.replace(dst)
    return target,origin,sha(source/'weight_manifest.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['raw','compact'],required=True)
    p.add_argument('--model-run',required=True);p.add_argument('--baseline-root',required=True)
    p.add_argument('--sampling-root',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--cache-root',default='/content/models')
    p.add_argument('--max-new-tokens',type=int,default=4096)
    a=p.parse_args();root=Path(a.output_root); baseline=Path(a.baseline_root); source=Path(a.sampling_root)
    if root.exists():raise ValueError('Use a new evaluation output-root')
    protocol=read(source/'sampling_identity.json'); frozen=grader_identity()
    if protocol['num_samples']!=4 or protocol['grader_identity_sha256']!=digest(frozen):raise ValueError('Original four-sample grader changed')
    if sha('data/math/math_dev.jsonl')!=DEV_SHA:raise ValueError('Dev content changed')
    base_identity=read(baseline/'evaluation_identity.json'); base_summary=read(baseline/'summary.json')
    if base_identity['dev_sha256']!=DEV_SHA or digest(base_identity['grader_identity'])!=digest(frozen):raise ValueError('E0 dev protocol changed')
    probe=read(baseline/'hard_probe_identity.json'); hard_file=baseline/'hard_candidates_e0_0of4.jsonl'
    if sha(hard_file)!=probe['output_sha256'] or probe['source_sampling_identity_sha256']!=sha(source/'sampling_identity.json'):
        raise ValueError('Hard probe source changed')
    hard=jsonl(hard_file); dev=jsonl('data/math/math_dev.jsonl')
    if len(hard)!=637 or {r['sample_id'] for r in hard}!=set(probe['sample_ids']):raise ValueError('Require fixed 637 hard candidates')
    if any(r['training_approved'] is not False for r in hard):raise ValueError('Hard candidates must remain unapproved')
    seal=read(source/'preparation_seal.json'); token_file=source/'prompt_tokens.jsonl'
    if sha(token_file)!=seal['prompt_tokens_sha256']:raise ValueError('Original hard prompt tokens changed')
    tokens={r['sample_id']:r['prompt_token_ids'] for r in jsonl(token_file)}
    model_path,origin,weight_sha=cache_merged(a.model_run,a.cache_root,a.arm)
    from transformers import AutoTokenizer
    from src.training.sft_encoding_v6 import token_ids
    tok_path=Path('data/tokenizers/Qwen2.5-7B-Instruct')
    if {p.name:sha(p) for p in tok_path.iterdir() if p.is_file()}!=base_identity['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from E0 dev')
    tokenizer=AutoTokenizer.from_pretrained(tok_path,local_files_only=True)
    dev_tokens=[token_ids(tokenizer.apply_chat_template(messages(r),tokenize=True,add_generation_prompt=True,
                return_dict=False,return_tensors=None,padding=False,truncation=False)) for r in dev]
    max_model_len=max(max(map(len,dev_tokens)),max(len(tokens[r['sample_id']]) for r in hard))+a.max_new_tokens
    root.mkdir(parents=True)
    identity={'experiment':f'E1_{a.arm}_v18_dev_hard_v20','arm':a.arm,'model_run':str(a.model_run),
              'merged_manifest_sha256':weight_sha,'sft_origin':origin,'dev_sha256':DEV_SHA,
              'e0_baseline_root':str(baseline),'e0_baseline_summary':base_summary,
              'hard_probe_identity':probe,'original_hard_sampling_protocol':protocol,
              'hard_generation':{**protocol,'max_new_tokens':a.max_new_tokens},
              'comparison_scope':'Historical E0/old SFT used 2048; new budget differs; no matched-budget improvement claim',
              'dev_generation':{'temperature':0,'top_p':1,'top_k':-1,'max_new_tokens':a.max_new_tokens,'seed':42},
              'max_model_len':max_model_len,'max_num_seqs':128,'gpu_memory_utilization':0.90,
              'grader_identity':frozen,'versions':{n:version(n) for n in ['vllm','torch','transformers','tokenizers']},
              'source_sha256':sha(__file__),'helper_sha256':sha(Path('src/training/eval_helpers_v13.py')),
              'model_guard_sha256':sha(Path('src/training/eval_identity_v20.py'))}
    write_new(root/'evaluation_identity.json',identity)
    from vllm import LLM,SamplingParams
    from src.training.runner_support_v4 import PooledVerifier
    checker=PooledVerifier(4)
    try:
        engine=LLM(model=str(model_path),tokenizer=str(tok_path),dtype='bfloat16',tensor_parallel_size=1,
                   max_model_len=max_model_len,max_num_seqs=128,gpu_memory_utilization=0.90,
                   enable_prefix_caching=True,enable_chunked_prefill=True,max_num_batched_tokens=8192,seed=42,generation_config='vllm')
        def generate(prompts,params,work,filename):
            outputs=engine.generate([{'prompt_token_ids':t} for t in prompts],params,use_tqdm=True)
            if len(outputs)!=len(work):raise ValueError('Generation coverage differs')
            raw=[]
            for (row,trial),response in zip(work,outputs):
                if len(response.outputs)!=1:raise ValueError('Expected one output per trial')
                o=response.outputs[0]
                record={'sample_id':row['sample_id'],'trial':trial,'text':o.text,'token_ids':list(o.token_ids),
                        'generated_tokens':len(o.token_ids),'finish_reason':o.finish_reason,
                        'seed':42 if filename.startswith('dev') else trial_seed(protocol['seed'],row['sample_id'],trial)}
                append(root/filename,record);raw.append(record)
            def grade(pair):
                record,(row,_) = pair
                result=checker.check(record['text'],row['reference_answer'])
                from src.training.math_reward_v2 import category
                return {**record,'verification':result,'category':category(result)}
            with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(grade,zip(raw,work)))
        dev_results=generate(dev_tokens,SamplingParams(temperature=0,top_p=1,top_k=-1,max_tokens=a.max_new_tokens,seed=42),
                             [(r,0) for r in dev],'dev_predictions.jsonl')
        save_rows(root/'dev_graded.jsonl',dev_results)
        dev_report=summary(dev_results)
        dev_report['historical_e0_2048_correct_over_all']=base_summary['correct_over_all']
        dev_report['max_new_tokens']=a.max_new_tokens
        dev_report['comparison_scope']='Budget differs from historical 2048; do not infer matched-budget gains'
        write_new(root/'dev_summary.json',dev_report)
        print('DEV '+json.dumps(dev_report,ensure_ascii=False),flush=True)
        all_hard=[]
        for batch,start in enumerate(range(0,len(hard),64),start=1):
            work=[(r,t) for r in hard[start:start+64] for t in range(4)]
            parameters=[SamplingParams(n=1,temperature=protocol['temperature'],top_p=protocol['top_p'],top_k=-1,
                                      max_tokens=a.max_new_tokens,seed=trial_seed(protocol['seed'],r['sample_id'],t)) for r,t in work]
            graded=generate([tokens[r['sample_id']] for r,t in work],parameters,work,'hard_predictions.jsonl')
            for rec in graded:append(root/'hard_graded.jsonl',rec)
            all_hard.extend(graded)
            print(f'HARD {len(all_hard)}/2548 trials persisted',flush=True)
        hard_report=hard_summary(all_hard,probe['sample_ids'])
        hard_report['max_new_tokens']=a.max_new_tokens
        write_new(root/'hard_summary.json',hard_report)
        print('HARD SUMMARY '+json.dumps(hard_report,ensure_ascii=False),flush=True)
    finally:
        checker.close()


if __name__=='__main__':main()
