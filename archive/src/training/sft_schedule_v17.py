"""Explicit epoch, cosine and optimizer parameters; all are part of run identity."""
import math


def expected_updates(config, num_examples=1500):
    values=[config[k] for k in ['batch_size','gradient_accumulation_steps','num_train_epochs']]
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in values):
        raise ValueError('Batch, accumulation and epochs must be positive integers')
    batch,accumulation,epochs=values
    return math.ceil(math.ceil(num_examples/batch)/accumulation)*epochs


def training_schedule(config, num_examples=1500):
    updates=expected_updates(config,num_examples)
    if config['expected_optimizer_updates']!=updates:
        raise ValueError('Update expected_optimizer_updates with the new batch/epoch protocol')
    if config['lr_scheduler_type']!='cosine' or config['lr_scheduler_kwargs']!={'num_cycles':0.5}:
        raise ValueError('v17 requires one cosine decay without restarts')
    ratio=config['warmup_ratio']
    if not 0<=ratio<1:raise ValueError('warmup_ratio must be in [0,1)')
    return {'max_steps':-1,'num_train_epochs':float(config['num_train_epochs']),
            'lr_scheduler_type':'cosine','lr_scheduler_kwargs':dict(config['lr_scheduler_kwargs']),
            'warmup_steps':math.ceil(updates*ratio),
            **{k:config[k] for k in ['optim','adam_beta1','adam_beta2','adam_epsilon','weight_decay','max_grad_norm']}}
