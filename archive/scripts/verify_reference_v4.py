"""v4 run-identity aware reference validation; arbitrary local output-root."""
import argparse
import json
from pathlib import Path
from scripts.verify_sft_reference_v2 import verify


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,GPT2Config,GPT2LMHeadModel
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--tiny',action='store_true');g.add_argument('--config')
    p.add_argument('--rl-adapter');p.add_argument('--device',default='cpu');p.add_argument('--output-root',required=True)
    a=p.parse_args();torch.manual_seed(42)
    if a.tiny:
        if a.device!='cpu':raise ValueError('Tiny selfcheck is CPU-only')
        model=GPT2LMHeadModel(GPT2Config(vocab_size=24,n_positions=32,n_embd=16,n_layer=1,n_head=2,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))
        result=verify(model,torch.tensor([[1,2,3,4,5,6]]),['c_attn'])
    else:
        if not a.rl_adapter or not Path(a.rl_adapter).is_dir():raise ValueError('Provide the existing local already-updated RL adapter; no implicit download or real-model training')
        from src.training.guards_v4 import approved_config
        config,_,identity=approved_config(a.config)
        if config['stage']!='rl':raise ValueError('Reference check requires RL config')
        adapter=Path(a.rl_adapter);ip=adapter/'run_identity.json'
        if not ip.exists():ip=adapter.parent/'run_identity.json'
        if json.loads(ip.read_text(encoding='utf-8'))!=identity:raise ValueError('Adapter/base/run identity mismatch')
        model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,torch_dtype=torch.bfloat16).to(a.device)
        tok=AutoTokenizer.from_pretrained(config['tokenizer_path'],local_files_only=True)
        ids=tok.apply_chat_template([{'role':'user','content':'Compute 2 + 3.'}],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(a.device)
        result=verify(model,ids,['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],atol=.005,rtol=.005,
                      rank=config.get('lora_r',16),alpha=config.get('lora_alpha',32),updated_adapter=str(adapter))
    out=Path(a.output_root).resolve();out.mkdir(parents=True,exist_ok=True)
    result['real_7b_gpu']='未验证' if a.tiny else 'see this run only'
    with (out/'reference_check.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
