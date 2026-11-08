"""Final OPD input-budget gate and deterministic training-prompt refill."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from src.data.screening_io import rows,sha,json_write,ScreeningSimilarityIndex
from src.data.prepare_math_data import write_jsonl
from src.data.build_code_v2 import order,prompt_record
from src.data.code_v2_dedup import Protected


def run():
    base=Path('data/project2_code_v2');path=base/'manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if 'input_budget' in manifest:
        assert manifest['input_budget']['script_sha256']==sha(__file__)
        print('Input-budget stage already sealed');return manifest
    tokenizer=AutoTokenizer.from_pretrained('data/tokenizers/Qwen2.5-1.5B-Instruct',local_files_only=True)
    def length(r):return len(tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True))
    train=list(rows(base/'train_prompts.jsonl'));dev=list(rows(base/'dev_metadata.jsonl'))
    verification=list(rows(base/'dev_verification.jsonl'));audit=[]
    for r in dev+verification:
        r['prompt_token_count']=length(r)
        if r['prompt_token_count']>2560:raise ValueError('Dev needs budget-aware reselection: '+r['sample_id'])
    accepted=[]
    for r in train:
        r['prompt_token_count']=length(r)
        if r['prompt_token_count']>2560:audit.append({'sample_id':r['sample_id'],'reason':'prompt_exceeds_2560','prompt_tokens':r['prompt_token_count'],'interface':r['interface']})
        else:accepted.append(r)
    deficits={key:sum(r['interface']==key for r in audit) for key in ['function','stdin_stdout']}
    if deficits['function']:raise ValueError('Function pool needs budget-aware refill')
    index=ScreeningSimilarityIndex(.85);protected=Protected()
    for r in accepted+dev:index.add(r)
    for raw in sorted(rows('data/project2_code/train_prompts.jsonl'),key=order):
        if not deficits['stdin_stdout']:break
        if raw['interface']!='stdin_stdout' or protected.match(raw) or index.match(raw['problem']):continue
        candidate=prompt_record({**raw,'source_key':'openr1','family_id':'legacy/'+raw['source_family']},'train')
        candidate['prompt_token_count']=length(candidate)
        if candidate['prompt_token_count']>2560:continue
        accepted.append(candidate);index.add(candidate);deficits['stdin_stdout']-=1
    if any(deficits.values()):raise ValueError('Unable to refill input-budget exclusions')
    json_write('reports/backups/code_v2_manifest_before_input_budget.json',manifest)
    write_jsonl(base/'train_prompts.jsonl',accepted);write_jsonl(base/'dev_metadata.jsonl',dev)
    write_jsonl(base/'dev_verification.jsonl',verification);write_jsonl(base/'input_budget_audit.jsonl',audit)
    tokenizer_path=Path('data/tokenizers/Qwen2.5-1.5B-Instruct')
    tokenizer_audit=json.loads(Path('reports/tokenizer_alignment_report.json').read_text(encoding='utf-8'))
    tokenizer_source=next(m for m in tokenizer_audit['models'] if m['model']=='Qwen/Qwen2.5-1.5B-Instruct')
    manifest['input_budget']={'script_sha256':sha(__file__),'max_total_context':4096,'generation_reserve':1536,
        'max_prompt_tokens':2560,'rejected_and_refilled':len(audit),
        'actual_train_max':max(r['prompt_token_count'] for r in accepted),
        'actual_dev_max':max(r['prompt_token_count'] for r in dev),
        'tokenizer_model':tokenizer_source['model'],'tokenizer_revision':tokenizer_source['revision'],
        'template':'Qwen2.5-1.5B-Instruct user-only apply_chat_template, add_generation_prompt=True; uses tokenizer default system. Training must use this exact template or re-audit.',
        'tokenizer_files':{p.name:sha(p) for p in tokenizer_path.iterdir() if p.is_file()}}
    manifest['files']={p.name:{'path':p.as_posix(),'sha256':sha(p)} for p in base.glob('*.jsonl')}
    json_write(path,manifest)
    print('Input budget checked: refilled',len(audit),'train max',manifest['input_budget']['actual_train_max'],
          'dev max',manifest['input_budget']['actual_dev_max'])
    return manifest


if __name__=='__main__':run()
