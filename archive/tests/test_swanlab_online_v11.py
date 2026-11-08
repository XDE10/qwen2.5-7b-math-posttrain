from pathlib import Path
from types import SimpleNamespace
import sys

from src.training.swanlab_online_v11 import safe_config, swanlab_callback
from scripts.build_swanlab_online_v11 import build


def test_online_callback_records_config_args_and_metrics(tmp_path, monkeypatch):
    from transformers import TrainerCallback  # Import dependency discovery before installing the SDK mock.
    init = []; updates = []; logs = []
    swan = SimpleNamespace(init=lambda **kw: init.append(kw), config=SimpleNamespace(update=updates.append),
                           log=lambda values, step: logs.append((values, step)), finish=lambda: None)
    monkeypatch.setitem(sys.modules, 'swanlab', swan)
    monkeypatch.setenv('SWANLAB_API_KEY', 'unit-test-placeholder')
    monkeypatch.setenv('SWANLAB_WORKSPACE', 'example-team')
    monkeypatch.delenv('SWANLAB_PROJ_NAME', raising=False)
    callback, metrics = swanlab_callback(tmp_path, 'online', {'stage':'sft','batch_size':1,'lora_r':16,'api_key':'fixture'})
    assert init[0]['mode'] == 'online'
    assert init[0]['workspace'] == 'example-team'
    assert init[0]['project'] == 'project1-7b-math'
    assert init[0]['config']['lora_r'] == 16
    assert init[0]['config']['api_key'] == '[REDACTED]'
    callback.on_train_begin(SimpleNamespace(to_dict=lambda:{'optim':'fixture','hub_token':'fixture'}),None,None)
    assert updates[0]['actual_trainer_arguments']['hub_token'] == '[REDACTED]'
    callback.on_log(None,SimpleNamespace(global_step=1),None,logs={'loss':0.5})
    assert logs == [({'loss':0.5},1)]
    metrics.finish()


def test_nested_credentials_redacted_without_losing_token_budget():
    assert safe_config({'max_new_tokens':2048,'nested':{'api_key':'fixture'}}) == {'max_new_tokens':2048,'nested':{'api_key':'[REDACTED]'}}


def test_builder_keeps_training_budget():
    import json
    root = Path(__file__).resolve().parents[1]
    output = build(root)
    for arm in ['raw','compact']:
        old=json.loads((root/f'configs/project1_{arm}_sft_equal_exposure_v10.json').read_text(encoding='utf-8'))
        new=json.loads(output[f'configs/project1_{arm}_sft_equal_exposure_v11.json'])
        for key in old.keys()-{'experiment','output_dir'}:
            assert old[key] == new[key]
    assert "default='online')" in output['src/training/approved_hf_v11.py']
