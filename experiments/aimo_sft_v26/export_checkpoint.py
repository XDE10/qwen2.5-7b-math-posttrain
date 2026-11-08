"""Merge one saved epoch adapter on CPU; child exits before vLLM starts."""
import argparse
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.aimo_sft_v26.common import config,read,save,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['base','checkpoint','output']:p.add_argument('--'+arg,required=True)
    a=p.parse_args();c=config();base=Path(a.base);cp=Path(a.checkpoint);out=Path(a.output)
    identity=read(cp/'run_identity.json')
    effective=dict(c)
    for key in ['batch_size','gradient_accumulation_steps']:effective[key]=identity['config'][key]
    if identity['config']!=effective:raise ValueError('Checkpoint configuration differs')
    identity={**identity,'selected_checkpoint':cp.name,'adapter_sha256':sha(cp/'adapter_model.safetensors')}
    if (out/'sft_identity.json').exists() and read(out/'sft_identity.json')==identity:return
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    model=AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16,device_map='cpu')
    model=PeftModel.from_pretrained(model,cp).merge_and_unload(safe_merge=True)
    model.save_pretrained(out,safe_serialization=True,max_shard_size='4GB')
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    for name in c['tokenizer_sha256']:shutil.copyfile(base/name,out/name)
    save(out/'sft_identity.json',identity)

if __name__=='__main__':main()
