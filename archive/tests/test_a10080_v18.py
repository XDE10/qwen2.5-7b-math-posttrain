import json
from pathlib import Path
from types import SimpleNamespace
from scripts.build_compact_a10080_v18 import build
from src.training.sft_schedule_v17 import training_schedule
from src.training.swanlab_training_v18 import cuda_metrics,numeric_logs


def test_larger_microbatch_preserves_recipe_budget():
    root=Path.cwd();build(root)
    a=json.loads((root/'configs/project1_compact_cosine_v17.json').read_text(encoding='utf-8'))
    b=json.loads((root/'configs/project1_compact_cosine_A10080_v18.json').read_text(encoding='utf-8'))
    assert a['batch_size']*a['gradient_accumulation_steps']==b['batch_size']*b['gradient_accumulation_steps']==32
    assert training_schedule(a)==training_schedule(b)
    assert b['batch_size']==16 and b['gradient_accumulation_steps']==2
    assert b['expected_optimizer_updates']==141


def test_memory_metrics_have_explicit_units_and_cpu_does_not_fake_zero():
    assert numeric_logs({'loss':'0.5','learning_rate':'1e-4','note':'text'})=={'loss':0.5,'learning_rate':1e-4}
    scale=1024**3
    gpu=SimpleNamespace(is_available=lambda:True,mem_get_info=lambda:(10*scale,80*scale),
                        memory_allocated=lambda:50*scale,memory_reserved=lambda:60*scale,
                        max_memory_allocated=lambda:55*scale,max_memory_reserved=lambda:65*scale)
    actual=cuda_metrics(SimpleNamespace(cuda=gpu))
    assert actual['gpu/peak_allocated_gib']==55
    assert actual['gpu/device_used_gib']==70
    assert cuda_metrics(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))=={}


def test_runner_binds_logger_and_records_real_token_counts():
    output=build(Path.cwd());runner=output['src/training/approved_hf_v18.py']
    assert 'from src.training.guards_v18 import approved_config' in runner
    assert 'from src.training.swanlab_training_v18 import swanlab_callback' in runner
    assert 'supervised_tokens_per_epoch' in runner
    assert 'minimum_gpu_memory_gib' in runner
