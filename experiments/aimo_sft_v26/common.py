"""Shared fixed identities and loss-preserving R1 formatting for direct SFT."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SUFFIX='\n\nPlease reason step by step, and put your final answer within \\boxed{}.'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def prompt_ids(tok,question):
    ids=tok.apply_chat_template([{'role':'user','content':question+SUFFIX}],
        tokenize=True,add_generation_prompt=True,return_dict=False)
    return list(ids['input_ids'] if isinstance(ids,dict) else ids)


def encode(tok,question,solution,max_length):
    """Qwen Instruct prefix plus the entire original response; assistant-only loss."""
    prefix=prompt_ids(tok,question)
    rendered=tok.decode(prefix,skip_special_tokens=False)
    if not rendered.endswith('<|im_start|>assistant\n'):
        raise ValueError('Expected Qwen2.5-Instruct assistant generation prefix')
    text=solution.lstrip()
    if not text:
        raise ValueError('Empty original response')
    if ('<think>' in text) != ('</think>' in text):
        raise ValueError('Unbalanced original think tags')
    # Unlike R1, Qwen's generation prefix contains no think tag. Supervise it too.
    target=text
    answer=tok.encode(target,add_special_tokens=False)
    if not answer or answer[-1]!=tok.eos_token_id:
        answer.append(tok.eos_token_id)
    ids=prefix+answer
    if len(ids)>max_length:
        raise OverflowError(len(ids))
    return {'input_ids':ids,'attention_mask':[1]*len(ids),
        'labels':[-100]*len(prefix)+answer,'length':len(ids),
        'prompt_tokens':len(prefix),'supervised_tokens':len(answer)}


def config():
    return read(ROOT/'configs/project1_aimo_sft_v26.json')


def verify_data(c):
    manifest=read(ROOT/c['manifest'])
    path=ROOT/c['train_file']
    if manifest['approved_training_files']!=[c['train_file']] or sha(path)!=manifest['train_sha256']:
        raise ValueError('Training data/approval manifest mismatch')
    data=rows(path)
    if len(data)!=manifest['train_count'] or len({r['sample_id'] for r in data})!=len(data):
        raise ValueError('Training rows/count mismatch')
    if any(r.get('training_approved') is not True for r in data):
        raise ValueError('Unapproved training row')
    return data,manifest
