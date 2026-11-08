"""Seal CPU evidence and pre-render problem-only prompts for the sampler."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from src.training.frozen_verifier_v4 import sha,grader_identity,digest
from src.training.rollout_pipeline_v4 import jsonl,write_new,SYSTEM


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);a=p.parse_args();out=Path(a.output_root).resolve()
    support=json.loads((out/'support_summary.json').read_text(encoding='utf-8'))
    embedding=json.loads((out/'embedding_screening.json').read_text(encoding='utf-8'))
    pool=out/'rl_candidates_verifier_supported_semantic_screened_v3.jsonl'
    if sha(pool)!=embedding['final_candidate_sha256']:raise ValueError('Embedding final candidate changed')
    if digest(grader_identity())!=support['grader_identity_sha256']:raise ValueError('Scorer changed since selfcheck')
    rows=jsonl(pool);checks={r['sample_id']:r for r in jsonl(out/'support_checks.jsonl')}
    tok=AutoTokenizer.from_pretrained('data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    rendered=[]
    for row in rows:
        if not checks[row['sample_id']]['supported_and_consistent']:raise ValueError('Unsupported candidate in final pool')
        ids=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':row['problem']}],tokenize=True,add_generation_prompt=True)
        if len(ids)>1752:raise ValueError('Prompt exceeds registered 1752-token budget')
        rendered.append({'sample_id':row['sample_id'],'prompt_token_ids':ids})
    (out/'prompt_tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rendered),encoding='utf-8')
    seal={'schema':'project1-cpu-preparation-v4','final_candidate_sha256':sha(pool),'candidate_count':len(rows),
          'grader_identity_sha256':support['grader_identity_sha256'],'grader':grader_identity(),
          'support_evidence_sha256':sha(out/'support_checks.jsonl'),'support_summary_sha256':sha(out/'support_summary.json'),
          'embedding_evidence_sha256':sha(out/'embedding_screening.json'),'lexical_evidence_sha256':sha('reports/handover_20261024_v3/freeze_record.json'),
          'prompt_tokens_sha256':sha(out/'prompt_tokens.jsonl'),'prompt_max':max(len(r['prompt_token_ids']) for r in rendered),
          'tokenizer_files':{p.name:sha(p) for p in Path('data/tokenizers/Qwen2.5-7B-Instruct').iterdir() if p.is_file()},
          'protected_ids_sha256':sha('data/portable_release/reserved_dev_ids_never_train.json'),
          'sft_semantic_suspect_ids':embedding['sft_suspect_ids'],'training_approved':False,
          'scope':'fixed lexical+embedding screening and supplied-answer/solution support checks; not universal semantic cleanliness'}
    write_new(out/'preparation_seal.json',seal)
    print(json.dumps({'sealed_candidates':len(rows),'prompt_max':seal['prompt_max'],'sft_semantic_suspects':len(seal['sft_semantic_suspect_ids'])}))


if __name__=='__main__':main()
