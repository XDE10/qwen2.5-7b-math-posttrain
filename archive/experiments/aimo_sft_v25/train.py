"""Single-A100 BF16 LoRA SFT, assistant-only loss with full R1 thoughts."""
import argparse
import inspect
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.aimo_sft_v25.common import config,encode,read,save,sha,verify_data
from experiments.aimo_sft_v25.telemetry import start_online,device_metrics


class Collator:
    def __init__(self,pad):self.pad=pad
    def __call__(self,records):
        import torch
        width=max(len(r['input_ids']) for r in records)
        return {'input_ids':torch.tensor([r['input_ids']+[self.pad]*(width-len(r['input_ids'])) for r in records]),
                'attention_mask':torch.tensor([r['attention_mask']+[0]*(width-len(r['input_ids'])) for r in records]),
                'labels':torch.tensor([r['labels']+[-100]*(width-len(r['input_ids'])) for r in records])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True)
    p.add_argument('--output-root',default='/content/drive/MyDrive/runs/project1/aimo_sft_v25')
    a=p.parse_args();c=config();data,manifest=verify_data(c)
    out=Path(a.output_root);out.mkdir(parents=True,exist_ok=True)
    model_path=Path(a.model_path)
    identity=read(model_path/'projection_model_identity.json')
    if identity['repo']!=c['model'] or identity['revision']!=c['model_revision']:
        raise ValueError('Wrong model identity')
    for name,expected in c['tokenizer_sha256'].items():
        if sha(model_path/name)!=expected:raise ValueError('Tokenizer differs: '+name)
    run_identity={'config':c,'manifest':manifest,'model_identity':identity,'train_code_sha256':sha(__file__),
                  'encoding_code_sha256':sha(ROOT/'experiments/aimo_sft_v25/common.py'),
                  'telemetry_code_sha256':sha(ROOT/'experiments/aimo_sft_v25/telemetry.py')}
    if (out/'run_identity.json').exists() and read(out/'run_identity.json')!=run_identity:
        raise ValueError('Different experiment; choose a new output-root')
    save(out/'run_identity.json',run_identity)
    if (out/'TRAIN_COMPLETE.json').exists():
        print('Training and export already complete.',flush=True);return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,TrainerCallback,set_seed
    from peft import LoraConfig,get_peft_model
    from datasets import Dataset
    from transformers.trainer_utils import get_last_checkpoint
    if not torch.cuda.is_available():raise RuntimeError('Connect A100 before training')
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='right'
    encoded=[encode(tok,r['question'],r['solution'],c['max_length']) for r in data]
    if any(len(x['input_ids'])!=r['total_tokens'] for x,r in zip(encoded,data)):
        raise ValueError('Remote tokenization differs from frozen CPU preparation')
    save(out/'tokenization.json',{'examples':len(encoded),'supervised_tokens':sum(x['supervised_tokens'] for x in encoded),
         'max_total_tokens':max(x['length'] for x in encoded),'loss':'assistant thought body + closing think + final answer + EOS only'})
    set_seed(c['seed'])
    model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=c['lora_r'],lora_alpha=c['lora_alpha'],
        lora_dropout=c['lora_dropout'],target_modules=c['target_modules'],task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    fields=['input_ids','attention_mask','labels','length']
    dataset=Dataset.from_list([{k:r[k] for k in fields} for r in encoded])
    kwargs=dict(output_dir=str(out),num_train_epochs=1,max_steps=-1,learning_rate=c['learning_rate'],
        per_device_train_batch_size=c['batch_size'],gradient_accumulation_steps=c['gradient_accumulation_steps'],
        bf16=True,fp16=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        optim='adamw_torch',lr_scheduler_type='cosine',warmup_ratio=.05,weight_decay=0.,max_grad_norm=1.,
        logging_steps=1,save_strategy='steps',save_steps=20,save_total_limit=2,
        report_to='none',remove_unused_columns=False,dataloader_num_workers=0,seed=c['seed'],data_seed=c['seed'])
    if 'group_by_length' in inspect.signature(TrainingArguments).parameters:
        kwargs.update(group_by_length=True,length_column_name='length')
    args=TrainingArguments(**kwargs)
    from importlib.metadata import version
    environment={'versions':{n:version(n) for n in ['torch','transformers','peft','accelerate','datasets','swanlab']},
        'gpu':torch.cuda.get_device_name(0),'gpu_total_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
        'cuda_runtime':torch.version.cuda,'training_arguments':args.to_dict()}
    save(out/'environment.json',environment)
    trainable,total=model.get_nb_trainable_parameters()
    payload={'recipe':c,'data':manifest,'environment':environment,
        'model_identity':identity,'source_identity':run_identity,
        'parameter_counts':{'trainable':trainable,'total':total,'trainable_fraction':trainable/total},
        'logging_step_semantics':'SwanLab step=monotone event index; train/global_step=optimizer step; replay after checkpoint resume retained'}
    swan=start_online(out,payload)
    class Log(TrainerCallback):
        def __init__(self):self.started=None;self.step_started=None;self.last_step_seconds=0.
        def on_train_begin(self,args,state,control,**kw):
            self.started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            swan.log({'lifecycle/train_started':1,'train/target_examples':len(data),
                'train/target_supervised_tokens':manifest['tokens']['supervised_sum']},state.global_step)
        def on_step_begin(self,args,state,control,**kw):self.step_started=time.monotonic()
        def on_step_end(self,args,state,control,**kw):
            if self.step_started is not None:self.last_step_seconds=time.monotonic()-self.step_started
        def on_log(self,args,state,control,logs=None,**kw):
            values={'train/'+key:value for key,value in (logs or {}).items()}
            values.update(device_metrics(torch))
            if self.started is not None:values['time/train_session_seconds']=time.monotonic()-self.started
            if self.last_step_seconds>0:
                values['time/optimizer_step_seconds']=self.last_step_seconds
                values['throughput/optimizer_steps_per_second']=1/self.last_step_seconds
            swan.log(values,state.global_step)
        def on_save(self,args,state,control,**kw):
            save(Path(args.output_dir)/f'checkpoint-{state.global_step}'/'run_identity.json',run_identity)
            swan.log({'lifecycle/checkpoint_saved':1},state.global_step)
    trainer=Trainer(model=model,args=args,train_dataset=dataset,data_collator=Collator(tok.pad_token_id),callbacks=[Log()])
    checkpoint=get_last_checkpoint(str(out))
    if checkpoint and read(Path(checkpoint)/'run_identity.json')!=run_identity:
        raise ValueError('Checkpoint experiment identity differs')
    started=time.monotonic()
    print(f'TRAIN {len(data)} original traces, one epoch, effective batch={c["batch_size"]*c["gradient_accumulation_steps"]}, expected updates={math.ceil(len(data)/(c["batch_size"]*c["gradient_accumulation_steps"]))}',flush=True)
    try:
        result=trainer.train(resume_from_checkpoint=checkpoint)
        swan.log({'final/'+key:value for key,value in result.metrics.items()},trainer.state.global_step)
        trainer.save_model(str(out/'adapter'));tok.save_pretrained(out/'adapter')
        for name in c['tokenizer_sha256']:
            shutil.copyfile(model_path/name,out/'adapter'/name)
        save(out/'adapter'/'base_identity.json',identity)
        trainer.save_state()
        save(out/'TRAIN_ADAPTER_COMPLETE.json',{'metrics':result.metrics,'steps':trainer.state.global_step})
        print('Saving merged BF16 model for vLLM evaluation...',flush=True)
        swan.log({'lifecycle/merge_started':1},trainer.state.global_step)
        merged=model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(out/'merged',safe_serialization=True,max_shard_size='4GB')
        tok.save_pretrained(out/'merged')
        for name in c['tokenizer_sha256']:
            shutil.copyfile(model_path/name,out/'merged'/name)
        save(out/'merged'/'sft_identity.json',run_identity)
        save(out/'TRAIN_COMPLETE.json',{'examples':len(data),'steps':trainer.state.global_step,
            'train_and_export_seconds':time.monotonic()-started,'adapter':str(out/'adapter'),'merged':str(out/'merged')})
        print('训练完成：'+str(out),flush=True)
        swan.log({'lifecycle/export_complete':1,'time/train_and_export_seconds':time.monotonic()-started},trainer.state.global_step)
    except BaseException:
        swan.log({'lifecycle/failed_or_interrupted':1},trainer.state.global_step)
        raise
    finally:
        swan.finish()


if __name__=='__main__':main()
