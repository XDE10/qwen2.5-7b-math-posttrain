import json
import math
from pathlib import Path

import pytest
from src.training.sft_schedule_v17 import training_schedule, expected_updates
from scripts.build_compact_cosine_v17 import build


def config():return json.loads(Path('configs/project1_compact_cosine_v17.json').read_text(encoding='utf-8'))


def test_real_cosine_scheduler_has_five_warmup_steps_and_decays_to_zero():
    import torch
    from transformers import get_scheduler,TrainingArguments,Trainer
    from types import SimpleNamespace
    c=config();schedule=training_schedule(c)
    assert expected_updates(c)==141 and schedule['warmup_steps']==5
    args=TrainingArguments(output_dir='unused',per_device_train_batch_size=2,gradient_accumulation_steps=16,
                           learning_rate=1e-4,report_to='none',**schedule)
    data_loader=list(range(750))
    values=Trainer.set_initial_training_values(SimpleNamespace(num_examples=lambda _:1500),args,data_loader,32)
    assert values[0]==3 and values[1]==47 and values[-1]==141
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=c['learning_rate'])
    scheduler=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=5,num_training_steps=141,
                            scheduler_specific_kwargs=c['lr_scheduler_kwargs'])
    rates=[optimizer.param_groups[0]['lr']]
    for _ in range(141):optimizer.step();scheduler.step();rates.append(optimizer.param_groups[0]['lr'])
    assert rates[0]==0 and rates[5]==pytest.approx(1e-4)
    assert all(rates[i+1]<=rates[i]+1e-12 for i in range(5,141))
    assert rates[-1]==pytest.approx(0,abs=1e-12)
    assert rates[73]==pytest.approx(0.5e-4)


def test_generated_runner_passes_schedule_and_respects_epoch_budget():
    outputs=build(Path.cwd());runner=outputs['src/training/approved_hf_v17.py']
    assert 'from src.training.guards_v17 import approved_config' in runner
    assert 'from src.training.sft_schedule_v17 import training_schedule' in runner
    assert '**training_schedule(c)' in runner
    assert "trainer.state.epoch-c['num_train_epochs']" in runner
    assert "lora_dropout=c['lora_dropout']" in runner
    c=config();assert c['arm']=='compact' and c['lora_r']==32 and c['lora_alpha']==64
    assert c['batch_size']*c['gradient_accumulation_steps']==32
    c['expected_optimizer_updates']=94
    with pytest.raises(ValueError):training_schedule(c)


def test_real_approved_compact_guard_without_loading_weights(tmp_path,monkeypatch):
    from src.training.guards_v17 import approved_config
    from src.training import guards_v4
    calls=[];monkeypatch.setattr(guards_v4,'validate_model',lambda m,i:calls.append((m,i)))
    c=config();c['e0_model_identity_path']='cpu_fixture_no_weights'
    path=tmp_path/'resolved.json';path.write_text(json.dumps(c),encoding='utf-8')
    actual,rows,identity=approved_config(path)
    assert len(rows)==1500 and actual['arm']=='compact'
    assert identity['actual_trainer_schedule']['lr_scheduler_type']=='cosine'
    assert len(calls)==1
    c['arm']='raw';path.write_text(json.dumps(c),encoding='utf-8')
    with pytest.raises(ValueError,match='compact-only'):approved_config(path)
