"""Fixed greedy math evaluation; sealed JSONL and model identity in every run."""
import argparse,json,hashlib
from pathlib import Path
from src.training.approved_hf import check_math

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--revision',required=True);p.add_argument('--adapter');p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--max-new-tokens',type=int,default=2048);p.add_argument('--problem-field',default='problem');p.add_argument('--answer-field',default='reference_answer');a=p.parse_args()
    if a.revision=='main':raise ValueError('Pin evaluation revision')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if not torch.cuda.is_available():raise RuntimeError('Model evaluation requires CUDA')
    tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision);model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,a.adapter).eval()
    data=Path(a.data);seen=set();results=[]
    with data.open(encoding='utf-8') as f,Path(a.output).open('x',encoding='utf-8') as out:
        for line in f:
            r=json.loads(line);sid=r['sample_id']
            if sid in seen:raise ValueError('Duplicate evaluation ID')
            seen.add(sid);msgs=[{'role':'system','content':r'Please reason step by step, and put your final answer within \boxed{}.'},{'role':'user','content':r[a.problem_field]}]
            x=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda')
            with torch.no_grad():ids=model.generate(x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=a.max_new_tokens,pad_token_id=tok.eos_token_id)
            response=ids[0,x.shape[1]:];text=tok.decode(response,skip_special_tokens=True);v=check_math(text,r[a.answer_field])
            rec={'sample_id':sid,'text':text,'generated_tokens':len(response),'budget_hit_without_eos':len(response)==a.max_new_tokens and int(response[-1])!=tok.eos_token_id,'verification':v}
            results.append(rec);out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush()
    meta={'config':vars(a),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'count':len(results),'conservative_correct':sum(r['verification'].get('correct',False) for r in results),'unsupported_or_error':sum(r['verification'].get('status') not in ['equivalent','mismatch'] for r in results),'budget_hits':sum(r['budget_hit_without_eos'] for r in results),'grader_scope':'Conservative expression equivalence; relational/set ambiguities separate from mismatches'}
    Path(a.output+'.summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
