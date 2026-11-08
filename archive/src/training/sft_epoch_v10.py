"""Explicit epoch stopping for the equal-question-exposure SFT protocol."""


def training_schedule(config):
    if config.get('num_train_epochs') != 1 or config.get('max_steps') != 94:
        raise ValueError('v10 formal protocol requires one epoch and 94 expected updates')
    if config.get('batch_size') != 1 or config.get('gradient_accumulation_steps') != 16:
        raise ValueError('Keep the paired formal effective batch at 16')
    return {'max_steps': -1, 'num_train_epochs': 1.0}
