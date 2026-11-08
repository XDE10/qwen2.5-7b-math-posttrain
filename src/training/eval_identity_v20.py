"""Accept only the completed user-requested compact v18 recipe."""
from src.training.rollout_pipeline_v4 import REV

EXPECTED = dict(stage='sft', arm='compact', experiment='E1_compact_r32_lr1e4_e3_b32_cosine_A10080_v18',
    model_revision=REV, data_file='project1_compact_paired.jsonl', num_train_epochs=3,
    expected_optimizer_updates=141, batch_size=16, gradient_accumulation_steps=2,
    lora_r=32, lora_alpha=64, learning_rate=0.0001, lr_scheduler_type='cosine', max_length=3072)


def validate_completed_v18(state, origin, arm='compact'):
    config=origin['config']
    if arm!='compact' or any(config.get(k)!=v for k,v in EXPECTED.items()):
        raise ValueError('Require the declared compact v18 recipe; preserve other runs')
    if state.get('global_step')!=141 or abs(state.get('epoch',0)-3)>1e-6 or state.get('num_examples')!=1500:
        raise ValueError('Require completed 3 epochs / 141 updates / 1500 rows')
