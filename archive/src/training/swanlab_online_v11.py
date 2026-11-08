"""Explicit cloud logging, complete hyperparameters, no credential persistence."""
import os
from pathlib import Path

from src.training.rollout_pipeline_v4 import Metrics


def safe_config(value):
    if isinstance(value, dict):
        return {k: ('[REDACTED]' if any(marker in k.lower() for marker in
                    ('api_key', 'password', 'secret', 'access_token', 'hub_token', 'authorization'))
                    else safe_config(v)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_config(v) for v in value]
    return value


def swanlab_callback(output, mode, config):
    from transformers import TrainerCallback
    metrics = Metrics(output, 'disabled')
    swan = None
    if mode != 'disabled':
        if mode == 'online' and not os.environ.get('SWANLAB_API_KEY'):
            raise ValueError('Set SWANLAB_API_KEY in the terminal; it is never saved in configuration')
        import swanlab as swan
        options = dict(project=os.environ.get('SWANLAB_PROJ_NAME', 'project1-7b-math'),
                       experiment_name='math7b-' + config.get('experiment', config['stage']),
                       mode=mode, logdir=str(Path(output) / 'swanlog'),
                       config=safe_config(config), reinit=True)
        if os.environ.get('SWANLAB_WORKSPACE'):
            options['workspace'] = os.environ['SWANLAB_WORKSPACE']
        swan.init(**options)
        metrics.swan = swan
    class Callback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if swan:
                swan.config.update({'actual_trainer_arguments': safe_config(args.to_dict())})
        def on_log(self, args, state, control, logs=None, **kwargs):
            metrics.log(logs or {}, state.global_step)
    return Callback(), metrics
