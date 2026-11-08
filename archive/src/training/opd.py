"""Single-GPU, sequential-domain teacher OPD with exact student-token prefixes."""
import argparse,gc,json,random,time
from pathlib import Path
from src.data.approved_training_inputs import load_approved,model_view
from src.teachers.opd_contract import render_student_prompt,teacher_batch_from_student
from src.losses.reference import aligned_kl
from src.training.approved_hf import read,sha


def response_batch(ids,prompt,eos):
    import torch
    n=len(prompt);end=ids.shape[1]
    positions=(ids[0,n:]==eos).nonzero().flatten()
    if len(positions):end=n+int(positions[0])+1
    ids=ids[:,:end];attention=torch.ones_like(ids)
    return ids,attention,n


def update_loss(student,teacher,ids,attention,prefix,support):
    batch,mask=teacher_batch_from_student(ids,attention,[len(prefix)],[prefix],support)
    student_logits=student(**batch).logits
    import torch
    teacher.eval()
    with torch.no_grad():teacher_logits=teacher(**batch).logits
    loss=aligned_kl(student_logits,teacher_logits,mask,range(support),range(support),'reverse')
    with torch.no_grad():
        response=mask[:,1:]
        def removed(logits):
            rows=logits[:,:-1][response].float()
            return float((1-(rows[:,:support].logsumexp(-1)-rows.logsumexp(-1)).exp()).mean())
        mass={'student_removed_mass':removed(student_logits),'teacher_removed_mass':removed(teacher_logits)}
    return loss,mass


def save_checkpoint(path,model,optimizer,scheduler,state,identity):
    import torch
    path=Path(path);path.mkdir(parents=True,exist_ok=False);model.save_pretrained(path/'student')
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'state':state,
        'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},path/'engine.pt')
    (path/'run_identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    files={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}
    (path/'checkpoint_manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')


def run(config_path):
    c=read(config_path);manifest=c['data_manifest'];pools={}
    for domain,name in c['data_files'].items():pools[domain]=load_approved(manifest,name)
    audit=read(c['tokenizer_audit']);ids=audit['common_support']['ids'];support=len(ids)
    if ids!=list(range(support)) or audit['mapping_difference_count']!=0:raise ValueError('This version requires audited identical contiguous effective token IDs')
    expected={r['model']:r['revision'] for r in audit['models']}
    for spec in [c['student'],*c['teachers'].values()]:
        if expected.get(spec['model'])!=spec['revision']:raise ValueError('Model revision differs from token alignment audit')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Formal OPD requires exactly one visible CUDA GPU')
    from transformers import AutoTokenizer,AutoModelForCausalLM,LogitsProcessor,LogitsProcessorList,set_seed
    from peft import LoraConfig,get_peft_model,PeftModel
    from importlib.metadata import version
    identity={'config':{k:v for k,v in c.items() if k!='resume_from_checkpoint'},'manifest_sha':sha(manifest),'audit_sha':sha(c['tokenizer_audit']),
        'versions':{k:version(k) for k in ['torch','transformers','peft']}}
    out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'run_identity.json').exists() and read(out/'run_identity.json')!=identity:raise ValueError('OPD run identity changed')
    (out/'run_identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    set_seed(c['seed']);tok=AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
    spec=c['student'];student=AutoModelForCausalLM.from_pretrained(spec['model'],revision=spec['revision'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    resume=c.get('resume_from_checkpoint');saved=None
    if not resume and list(out.glob('checkpoint-*')):raise ValueError('Existing checkpoints require explicit resume or a new output directory')
    if resume:
        path=Path(resume);checks=read(path/'checkpoint_manifest.json')
        if read(path/'run_identity.json')!=identity or any(sha(path/p)!=h for p,h in checks.items()):raise ValueError('OPD checkpoint identity or digest mismatch')
        student=PeftModel.from_pretrained(student,path/'student',is_trainable=True)
        saved=torch.load(path/'engine.pt',map_location='cpu',weights_only=False)
    else:student=get_peft_model(student,LoraConfig(r=c.get('lora_r',16),lora_alpha=32,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});student.enable_input_require_grads();student.config.use_cache=False
    optimizer=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=c['learning_rate'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:max(0.,1-s/c['max_steps']))
    state={'step':0,'positions':{k:0 for k in pools}}
    if saved:
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);state=saved['state']
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    class SupportMask(LogitsProcessor):
        def __call__(self,input_ids,scores):scores[:,support:]=-torch.inf;return scores
    teacher=None;teacher_domain=None;domains=list(c['domain_order']);accum=c.get('gradient_accumulation_steps',4)
    with (out/'metrics.jsonl').open('a',encoding='utf-8') as log:
        for step in range(state['step'],c['max_steps']):
            domain=domains[(step//c.get('domain_block_steps',10))%len(domains)]
            if teacher_domain!=domain:
                if teacher is not None:del teacher;gc.collect();torch.cuda.empty_cache()
                spec=c['teachers'][domain]
                # Loading a frozen teacher must not perturb the student rollout RNG.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    teacher=AutoModelForCausalLM.from_pretrained(spec['model'],revision=spec['revision'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
                teacher_domain=domain
            optimizer.zero_grad();losses=[];start=time.monotonic()
            for _ in range(accum):
                i=state['positions'][domain];row=pools[domain][i%len(pools[domain])];state['positions'][domain]+=1
                view=model_view(row,'opd');prefix=render_student_prompt(tok,view['input']['prompt'],domain)
                x=torch.tensor([prefix],dtype=torch.long,device='cuda');student.eval()
                with torch.no_grad():sample=student.generate(x,attention_mask=torch.ones_like(x),max_new_tokens=c['max_new_tokens'][domain],do_sample=True,
                    temperature=1.,top_p=1.,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,logits_processor=LogitsProcessorList([SupportMask()]))
                full,attention,n=response_batch(sample,prefix,tok.eos_token_id);student.train()
                loss,mass=update_loss(student,teacher,full,attention,prefix,support);(loss/accum).backward();losses.append(float(loss.detach()))
                del loss,sample,full,attention,x
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.);optimizer.step();scheduler.step();state['step']=step+1
            record={'step':state['step'],'domain':domain,'loss':sum(losses)/len(losses),'seconds':time.monotonic()-start,**mass,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30};log.write(json.dumps(record)+'\n');log.flush();print(record,flush=True)
            if state['step']%c.get('save_steps',20)==0 or state['step']==c['max_steps']:save_checkpoint(out/f"checkpoint-{state['step']}",student,optimizer,scheduler,state,identity)
    return state


def cli():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
