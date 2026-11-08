"""Training-only live curves, full configuration and actual CUDA memory peaks."""
import os
from pathlib import Path
import time

from src.training.rollout_pipeline_v4 import Metrics
from src.training.swanlab_online_v11 import safe_config


def cuda_metrics(torch):
    if not torch.cuda.is_available():return {}
    scale=1024**3
    free,total=torch.cuda.mem_get_info()
    return {'gpu/allocated_gib':torch.cuda.memory_allocated()/scale,
            'gpu/reserved_gib':torch.cuda.memory_reserved()/scale,
            'gpu/peak_allocated_gib':torch.cuda.max_memory_allocated()/scale,
            'gpu/peak_reserved_gib':torch.cuda.max_memory_reserved()/scale,
            'gpu/device_used_gib':(total-free)/scale,
            'gpu/total_gib':total/scale}


def numeric_logs(logs):
    result={}
    for key,value in (logs or {}).items():
        try:result[key]=float(value)
        except (TypeError,ValueError):continue
    return result


def swanlab_callback(output,mode,config):
    from transformers import TrainerCallback
    import torch
    metrics=Metrics(output,'disabled');swan=None
    if mode!='disabled':
        if mode=='online' and not os.environ.get('SWANLAB_API_KEY'):raise ValueError('Automatic SwanLab credential is missing')
        import swanlab as swan
        options={'project':os.environ.get('SWANLAB_PROJ_NAME','project1-7b-math'),
                 'experiment_name':'math7b-'+config['experiment'],'mode':mode,
                 'logdir':str(Path(output)/'swanlog'),'config':safe_config(config),'reinit':True}
        if os.environ.get('SWANLAB_WORKSPACE'):options['workspace']=os.environ['SWANLAB_WORKSPACE']
        swan.init(**options);metrics.swan=swan
    class Callback(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            self.started=time.monotonic();self.initial_step=state.global_step
            model=kwargs.get('model');details={'actual_trainer_arguments':safe_config(args.to_dict())}
            if model is not None:
                details['model_parameter_counts']={'total':sum(p.numel() for p in model.parameters()),
                                                   'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
            if torch.cuda.is_available():
                details['hardware']={'gpu':torch.cuda.get_device_name(0),
                                     'total_memory_gib':torch.cuda.get_device_properties(0).total_memory/(1024**3),
                                     'torch_cuda_runtime':torch.version.cuda}
            if swan:swan.config.update(details)
        def on_log(self,args,state,control,logs=None,**kwargs):
            values=numeric_logs(logs);values.update(cuda_metrics(torch))
            if hasattr(self,'started'):
                elapsed=time.monotonic()-self.started
                values['train/elapsed_seconds']=elapsed
                values['train/updates_per_second']=(state.global_step-self.initial_step)/max(elapsed,1e-9)
            metrics.log(values,state.global_step)
    return Callback(),metrics
