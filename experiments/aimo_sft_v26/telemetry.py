"""Required cloud SwanLab telemetry. Credentials never enter config or metrics."""
import json
import inspect
import math
import os
from pathlib import Path
import subprocess
import time

from experiments.aimo_sft_v26.common import read,save


def device_metrics(torch):
    values={'gpu/allocated_gib':torch.cuda.memory_allocated()/2**30,
        'gpu/reserved_gib':torch.cuda.memory_reserved()/2**30,
        'gpu/peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'gpu/peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True,timeout=2)
        if result.returncode==0:
            for name,value in zip(['gpu/utilization_pct','gpu/device_used_mib','gpu/power_w'],result.stdout.strip().split(',')):
                try:values[name]=float(value.strip())
                except ValueError:pass
    except (OSError,subprocess.TimeoutExpired):
        pass
    return values


class OnlineLog:
    def __init__(self,module,out,run_id,url=None):
        self.module,self.out,self.run_id=module,Path(out),run_id
        self.path=self.out/'swanlab_run.json'
        previous=read(self.path) if self.path.exists() else {}
        self.url=url or previous.get('url')
        self.sequence=previous.get('last_log_sequence',-1)
        self.closed=False
        self.persist()

    def persist(self):
        save(self.path,{'run_id':self.run_id,'url':self.url,'project':'project1-7b-math','mode':'online',
                        'last_log_sequence':self.sequence})

    def log(self,values,optimizer_step):
        numeric={}
        for key,value in values.items():
            try:number=float(value)
            except (TypeError,ValueError):continue
            if math.isfinite(number):numeric[key]=number
        numeric['train/global_step']=int(optimizer_step)
        self.sequence+=1
        # A monotone event index preserves replayed optimizer steps after resume.
        self.module.log(numeric,step=self.sequence)
        with (self.out/'metrics.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'event_index':self.sequence,**numeric},allow_nan=False)+'\n')
        self.persist()

    def finish(self):
        if not self.closed:
            self.module.finish()
            self.closed=True


def start_online(out,payload):
    import swanlab
    key=os.environ.get('SWANLAB_API_KEY')
    if not key:
        try:
            from google.colab import userdata
            key=userdata.get('SWANLAB_API_KEY')
        except Exception:
            # Non-notebook terminal may not have access to the Colab frontend.
            pass
    if key:
        authenticated=swanlab.login(api_key=key,save=False)
    else:
        # SDK reuses its existing login; first-time login prompts only if absent.
        authenticated=swanlab.login()
    if authenticated is False:
        raise RuntimeError('SwanLab online login failed; no offline fallback')
    previous=read(Path(out)/'swanlab_run.json') if (Path(out)/'swanlab_run.json').exists() else {}
    kwargs={'project':'project1-7b-math',
        'config':payload,
        'tags':['Qwen2.5-7B-Instruct','AIMO2-original','LoRA','v26'],'reinit':True}
    # SwanLab 0.10.2 uses online/name/log_dir; older SDKs use cloud/experiment_name/logdir.
    if 'name' in inspect.signature(swanlab.init).parameters:
        kwargs.update(name='Qwen25_7B_Instruct_aimo_original_v26',mode='online',log_dir=str(Path(out)/'swanlog'))
    else:
        kwargs.update(experiment_name='Qwen25_7B_Instruct_aimo_original_v26',mode='cloud',logdir=str(Path(out)/'swanlog'))
    if previous.get('run_id'):
        kwargs.update(id=previous['run_id'],resume='allow')
    run=swanlab.init(**kwargs)
    run_id=getattr(run,'id',None)
    if not run_id:
        raise RuntimeError('SwanLab cloud run did not return an experiment ID; training not started')
    print('SwanLab cloud logging initialized; run ID: '+str(run_id),flush=True)
    url=getattr(run,'url',None)
    if isinstance(url,str):print('SwanLab URL: '+url,flush=True)
    return OnlineLog(swanlab,out,str(run_id),url if isinstance(url,str) else None)
