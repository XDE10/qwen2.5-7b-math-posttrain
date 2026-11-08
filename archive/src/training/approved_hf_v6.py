"""Version6 SFT-only compatibility derivative of v4; GPU acceptance unverified."""
import argparse,hashlib,json,subprocess,sys
from importlib.metadata import version
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.data.approved_training_inputs_v2 import load_approved,model_view
from src.training.guards_v6 import approved_config
from src.training.sft_encoding_v6 import encode_sft
from src.training.guards_v2 import validate_checkpoint,rl_lengths
from src.training.runner_support_v4 import PooledVerifier,swanlab_callback
from src.training.math_reward_v2 import AuditedMathReward

ROOT=Path(__file__).resolve().parents[2]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class CompletionCollator:
    def __init__(self,pad):self.pad=pad
    def __call__(self,rows):
        import torch
        n=max(len(r['input_ids']) for r in rows)
        return {'input_ids':torch.tensor([r['input_ids']+[self.pad]*(n-len(r['input_ids'])) for r in rows]),
            'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(n-len(r['input_ids'])) for r in rows]),
            'labels':torch.tensor([r['labels']+[-100]*(n-len(r['input_ids'])) for r in rows])}


class VisualCompletionCollator:
    def __init__(self,processor,max_length):self.processor=processor;self.max_length=max_length
    def __call__(self,rows):
        import torch
        from PIL import Image
        images=[];texts=[];prefixes=[]
        for row in rows:
            v=model_view(row,'perception_sft');image=Image.open(v['input']['image_path']).convert('RGB');images.append(image)
            problem=v['input']['problem'].removeprefix('<image>').strip()
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':problem}]}]
            prefixes.append(self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
            texts.append(self.processor.apply_chat_template(messages+[{'role':'assistant','content':v['target']}],tokenize=False,add_generation_prompt=False))
        batch=self.processor(text=texts,images=images,padding=True,return_tensors='pt')
        labels=batch['input_ids'].clone();labels[batch['attention_mask']==0]=-100
        for i,(image,prefix) in enumerate(zip(images,prefixes)):
            p=self.processor(text=[prefix],images=[image],return_tensors='pt')['input_ids'][0]
            n=int(batch['attention_mask'][i].sum())
            if n>self.max_length or not torch.equal(batch['input_ids'][i,:len(p)],p):raise ValueError('VL prefix/budget violation; do not truncate image/question')
            labels[i,:len(p)]=-100
        batch['labels']=labels;return batch


def run(config_path,swanlab_mode='offline'):
    c,rows,identity=approved_config(config_path)
    if c['stage']!='sft' or c.get('vision'):raise ValueError('v6 is text SFT only; RL/vision compatibility is unverified')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Formal run requires CUDA; use CPU small-tensor smoke separately')
    if torch.cuda.device_count()!=1:raise RuntimeError('Initial configuration supports exactly one visible GPU')
    from datasets import Dataset
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,set_seed
    from peft import LoraConfig,get_peft_model
    set_seed(c['seed']);out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'run_identity.json').exists() and read(out/'run_identity.json')!=identity:raise ValueError('Run/data/model identity changed; choose new output')
    (out/'run_identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    class IdentityCallback(TrainerCallback):
        def on_save(self,args,state,control,**kwargs):
            p=Path(args.output_dir)/f'checkpoint-{state.global_step}'/'run_identity.json';p.write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    resume=c.get('resume_from_checkpoint')
    if not resume and list(out.glob('checkpoint-*')):raise ValueError('Existing checkpoints require explicit resume or a new output directory')
    if resume:validate_checkpoint(resume,identity)
    swan_callback,swan_metrics=swanlab_callback(out,swanlab_mode,c)
    import atexit
    atexit.register(swan_metrics.finish)
    vision=c.get('vision',False)
    loader=AutoModelForCausalLM
    model=loader.from_pretrained(c['model'],revision=c['model_revision'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model.config.use_cache=False
    processor=AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True)
    tok=processor.tokenizer if vision else processor;tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left' if c['stage']=='rl' else 'right'
    lora=LoraConfig(r=c.get('lora_r',16),lora_alpha=c.get('lora_alpha',32),lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM')
    common=dict(output_dir=str(out),max_steps=c['max_steps'],learning_rate=c['learning_rate'],per_device_train_batch_size=c.get('batch_size',1),
        gradient_accumulation_steps=c.get('gradient_accumulation_steps',8),bf16=True,fp16=False,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},report_to='none',logging_steps=1,save_steps=c.get('save_steps',20),
        save_only_model=False,seed=c['seed'],data_seed=c['seed'],dataloader_num_workers=0,remove_unused_columns=False)
    if c['stage']=='sft':
        model=get_peft_model(model,lora)
        if vision:
            fields=['problem','image_path','model_perception','image_sha256','training_approved','model_input_whitelist','model_target_whitelist']
            data=Dataset.from_list([{k:r[k] for k in fields} for r in rows]);collator=VisualCompletionCollator(processor,c['max_length'])
        else:data=Dataset.from_list(encode_sft(rows,tok,c['max_length']));collator=CompletionCollator(tok.pad_token_id)
        trainer=Trainer(model=model,args=TrainingArguments(**common),train_dataset=data,data_collator=collator,processing_class=processor,callbacks=[IdentityCallback(),swan_callback])
    trainer.train(resume_from_checkpoint=resume);trainer.save_model(out/'final');processor.save_pretrained(out/'final')
    if c['stage']=='sft' and c.get('export_merged_base',False):
        merged=trainer.model.merge_and_unload();merged.save_pretrained(out/'merged');processor.save_pretrained(out/'merged')
        (out/'merged'/'merged_sft_identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
        checks={str(p.relative_to(out/'merged')):sha(p) for p in (out/'merged').rglob('*') if p.is_file()}
        (out/'merged'/'weight_manifest.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    swan_metrics.finish()
    return trainer.state


def cli(expected_stage=None,expected_vision=None):
    import os
    from src.training.guards_v2 import expand_config
    from src.training.rollout_pipeline_v4 import digest,validate_model
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--e0-model-path');p.add_argument('--e0-model-identity');p.add_argument('--sft-raw-base');p.add_argument('--sft-compact-base')
    p.add_argument('--max-steps',type=int);p.add_argument('--resume-from-checkpoint');p.add_argument('--swanlab-mode',choices=['offline','online','disabled'],default='offline')
    a=p.parse_args();source=Path(a.config).resolve();release=source.parent.parent
    os.environ['APPROVED_RELEASE_ROOT']=str(release);os.environ['OUTPUT_ROOT']=str(Path(a.output_root).resolve())
    for value,name in [(a.e0_model_path,'E0_MODEL_PATH'),(a.sft_raw_base,'SFT_RAW_MERGED'),(a.sft_compact_base,'SFT_COMPACT_MERGED')]:
        if value:os.environ[name]=str(Path(value).resolve())
    raw_config=read(source)
    raw_config['output_dir']=str(Path(a.output_root).resolve()/raw_config.get('experiment',raw_config['stage']+'_'+raw_config.get('arm','shared')))
    c=expand_config(raw_config)
    if c['stage']!='sft' or c.get('vision'):raise ValueError('v6 is text SFT only')
    if expected_stage is not None and c['stage']!=expected_stage:raise ValueError('Wrong stage')
    if c.get('vision'):raise ValueError('Text-only project')
    if c['stage']=='sft':
        if a.e0_model_path:c['model']=str(Path(a.e0_model_path).resolve())
        if a.e0_model_identity:c['e0_model_identity_path']=str(Path(a.e0_model_identity).resolve())
    if a.max_steps is not None:
        if a.max_steps<1:raise ValueError('max-steps must be positive')
        c['max_steps']=a.max_steps
    if a.resume_from_checkpoint:c['resume_from_checkpoint']=a.resume_from_checkpoint
    root=Path(a.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    resolved=root/('resolved_'+c.get('experiment',c['stage'])+'_'+digest(c)[:12]+'.json')
    if not resolved.exists():resolved.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
    run(resolved,a.swanlab_mode)
