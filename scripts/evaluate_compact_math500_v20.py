"""v20: user-authorized MATH-500 snapshot for compact v18, frozen v15 protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
import os
from pathlib import Path

from scripts.evaluate_compact_dev_hard_v20 import cache_merged
from scripts.evaluate_project1_dev_v9 import messages, summary, save_rows
from src.training.frozen_verifier_v4 import sha, digest, grader_identity
from src.training.rollout_pipeline_v4 import read, jsonl, append, write_new, validate_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-key',choices=['compact_v18'],required=True)
    p.add_argument('--runs-root',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--cache-root',default='/content/models')
    p.add_argument('--model-run',required=True)
    a=p.parse_args(); root=Path(a.output_root); runs=Path(a.runs_root)
    protocol=read('configs/project1_math500_protocol_v20.json'); data=Path(protocol['data'])
    if sha(data)!=protocol['data_sha256']:raise ValueError('Fixed test bytes changed')
    rows=jsonl(data)
    if len(rows)!=500 or len({r['sample_id'] for r in rows})!=500 or any(r['split']!='test' for r in rows):
        raise ValueError('Require all original 500 unique test rows')
    frozen=grader_identity()
    if digest(frozen)!=protocol['grader_identity_sha256']:raise ValueError('Frozen scorer changed; do not tune it using test results')
    if root.exists():raise ValueError('Use a new output-root; preserve prior test results')
    model_path,origin,h=cache_merged(a.model_run,a.cache_root,'compact')
    weight_identity={'sft_origin':origin,'merged_manifest_sha256':h}
    from transformers import AutoTokenizer
    from src.training.sft_encoding_v6 import token_ids
    token_path=Path('data/tokenizers/Qwen2.5-7B-Instruct')
    tok=AutoTokenizer.from_pretrained(token_path,local_files_only=True)
    prompts=[token_ids(tok.apply_chat_template(messages(r),tokenize=True,add_generation_prompt=True,return_dict=False,
              return_tensors=None,padding=False,truncation=False)) for r in rows]
    root.mkdir(parents=True)
    identity={'model_key':a.model_key,'protocol':protocol,'protocol_sha256':sha('configs/project1_math500_protocol_v20.json'),
              'max_model_len':max(map(len,prompts))+protocol['max_new_tokens'],'grader_identity':frozen,**weight_identity,
              'tokenizer_sha256':{p.name:sha(p) for p in token_path.iterdir() if p.is_file()},
              'source_sha256':sha(__file__),'versions':{n:version(n) for n in ['vllm','torch','transformers','tokenizers']}}
    write_new(root/'evaluation_identity.json',identity)
    from src.training.runner_support_v4 import PooledVerifier
    checker=PooledVerifier(4)
    try:
        from vllm import LLM,SamplingParams
        engine=LLM(model=str(model_path),tokenizer=str(token_path),dtype='bfloat16',tensor_parallel_size=1,
                   max_model_len=identity['max_model_len'],max_num_seqs=128,gpu_memory_utilization=0.90,
                   enable_prefix_caching=True,enable_chunked_prefill=True,max_num_batched_tokens=8192,seed=42,generation_config='vllm')
        outputs=engine.generate([{'prompt_token_ids':t} for t in prompts],
                                SamplingParams(temperature=0,top_p=1,top_k=-1,max_tokens=protocol['max_new_tokens'],seed=42),use_tqdm=True)
        if len(outputs)!=500:raise ValueError('Test generation coverage differs')
        raw=[]
        for row,response in zip(rows,outputs):
            if len(response.outputs)!=1:raise ValueError('One greedy answer required')
            o=response.outputs[0]
            record={'sample_id':row['sample_id'],'text':o.text,'token_ids':list(o.token_ids),
                    'generated_tokens':len(o.token_ids),'finish_reason':o.finish_reason}
            append(root/'predictions.jsonl',record);raw.append(record)
        def grade(pair):
            record,row=pair
            result=checker.check(record['text'],row['reference_answer'])
            from src.training.math_reward_v2 import category
            return {**record,'verification':result,'category':category(result)}
        with ThreadPoolExecutor(max_workers=4) as pool:graded=list(pool.map(grade,zip(raw,rows)))
        save_rows(root/'graded_predictions.jsonl',graded)
        report=summary(graded)
        report.update(model_key=a.model_key,scope='Fixed MATH-500 test snapshot; no test-driven tuning; frozen conservative scorer, not human verification',
                      test_data_sha256=protocol['data_sha256'],protocol_sha256=identity['protocol_sha256'],
                      predictions_sha256=sha(root/'predictions.jsonl'),grader_identity_sha256=digest(frozen),
                      tokenizer_sha256=identity['tokenizer_sha256'])
        report['max_new_tokens']=protocol['max_new_tokens']
        report['comparison_scope']='4096 budget differs from historical E0/raw/compact 2048; not a matched-budget comparison'
        report['length_finish_rate']=report['finish_reason_counts'].get('length',0)/500
        report['unscorable_rate']=(500-report['scorable_count'])/500
        write_new(root/'summary.json',report)
        print('MATH500 SUMMARY '+json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    finally:
        checker.close()


if __name__=='__main__':main()
