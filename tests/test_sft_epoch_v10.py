import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_sft_epoch_v10 import build
from src.training.sft_epoch_v10 import training_schedule


def test_epoch_schedule_overrides_step_cap_and_preserves_last_batch():
    from transformers import Trainer
    config = dict(num_train_epochs=1, max_steps=94, batch_size=1, gradient_accumulation_steps=16)
    args = SimpleNamespace(**training_schedule(config), gradient_accumulation_steps=16)
    trainer = SimpleNamespace(num_examples=lambda dataloader: len(dataloader))
    epochs, updates, examples, samples, epoch_based, length, max_steps = Trainer.set_initial_training_values(trainer, args, list(range(1500)), 16)
    assert (epochs, updates, examples, samples, epoch_based, length, max_steps) == (1,94,1500,1500,True,1500,94)
    assert 1500 % 16 == 12


def test_wrong_budget_rejected():
    with pytest.raises(ValueError):
        training_schedule(dict(num_train_epochs=2,max_steps=94,batch_size=1,gradient_accumulation_steps=16))


def test_new_configs_and_runner_keep_paired_protocol():
    root = Path(__file__).resolve().parents[1]
    output = build(root)
    assert output == build(root)
    for arm in ('raw','compact'):
        config = json.loads(output[f'configs/project1_{arm}_sft_equal_exposure_v10.json'])
        assert config['draft'] is False
        assert config['arm'] == arm
        assert config['num_train_epochs'] == 1
        assert config['max_length'] == 4096
        assert config['data_file'] == f'project1_{arm}_paired.jsonl'
        assert config['learning_rate'] == 1e-5
    assert '**training_schedule(c)' in output['src/training/approved_hf_v10.py']
    assert 'save_total_limit=2' in output['src/training/approved_hf_v10.py']
