import json

import pytest

from experiments.qfft_v93.common import CONFIG_DIR, load_config
from experiments.qfft_v93.grpo import grpo_config


def test_vllm_mode_reaches_trl_from_an_inherited_json_config(tmp_path, monkeypatch):
    trl = pytest.importorskip('trl')
    monkeypatch.setattr('experiments.qfft_v93.grpo.gpu_profile', lambda: {'kind': 'g4'})
    monkeypatch.setattr('transformers.training_args.is_torch_bf16_gpu_available', lambda: True)
    pilot = CONFIG_DIR / 'grpo/grpo_qfft_r1_16k_pilot.json'
    override = tmp_path / 'token_mask.json'
    override.write_text(json.dumps({
        'base': str(pilot),
        'vllm_importance_sampling_mode': 'token_mask',
    }))

    # Build the same config as the GPU entry point. This does not start a trainer or vLLM.
    baseline = grpo_config(load_config(pilot), tmp_path, smoke=False, max_steps=None)
    candidate = grpo_config(load_config(override), tmp_path, smoke=False, max_steps=None)
    assert isinstance(candidate, trl.GRPOConfig)
    assert baseline.use_vllm and baseline.vllm_mode == 'colocate'
    assert baseline.vllm_importance_sampling_correction
    assert baseline.vllm_importance_sampling_clip_min is None
    assert baseline.vllm_importance_sampling_clip_max == 3.0
    assert baseline.vllm_importance_sampling_mode == 'sequence_mask'
    assert candidate.vllm_importance_sampling_mode == 'token_mask'
    baseline_fields = baseline.to_dict()
    assert {key for key, value in candidate.to_dict().items() if value != baseline_fields[key]} == {
        'vllm_importance_sampling_mode',
    }


def test_vllm_mode_rejects_typos(tmp_path, monkeypatch):
    pytest.importorskip('trl')
    monkeypatch.setattr('experiments.qfft_v93.grpo.gpu_profile', lambda: {'kind': 'g4'})
    cfg = load_config('grpo/grpo_qfft_r1_16k_pilot.json')
    cfg['vllm_importance_sampling_mode'] = 'token-mask'
    with pytest.raises(ValueError, match='Invalid vllm_importance_sampling_mode'):
        grpo_config(cfg, tmp_path, smoke=False, max_steps=None)


def test_capture_preserves_trl_alignment_and_compares_same_tokens():
    torch = pytest.importorskip('torch')
    from experiments.qfft_v93.grpo_is_diagnostic import capture_rows, compare_weights

    result = {
        'prompt_ids': torch.tensor([[0, 1], [0, 1]]),
        'prompt_mask': torch.tensor([[0, 1], [0, 1]]),
        'completion_ids': torch.tensor([[3, 4, 2], [4, 5, 0]]),
        'completion_mask': torch.tensor([[1, 1, 1], [0, 0, 0]]),
        'sampling_per_token_logps': torch.tensor([[-2, -2, float('nan')], [-2, -2, 0]]),
        'old_per_token_logps': torch.tensor([[-1.4, -1.4, -1], [-1, -1, 0]]),
        'advantages': torch.tensor([0.5, -0.5]),
        'importance_sampling_ratio': torch.tensor([[0.0], [1.0]]),
    }
    diagnostics = [
        {'sample_id': 'a', 'tokens': 3, 'trl_clipped': False, 'text': 'answer',
         'finish_reason': 'stop', 'stop_reason': None, 'reward_status': 'correct'},
        {'sample_id': 'a', 'tokens': 2, 'trl_clipped': True, 'text': 'unfinished',
         'finish_reason': 'length', 'stop_reason': None, 'reward_status': 'wrong'},
    ]
    rows = list(capture_rows(result, diagnostics, ['a'], 2))
    assert rows[0]['sampling_logps'] == [-2.0, -2.0, None]
    assert rows[1]['completion_ids'] == [4, 5]
    assert rows[0]['prompt_ids'] == [1]
    summary = compare_weights(rows)
    assert summary['missing_sampling_logps_on_loss_tokens'] == 1
    sequence = summary['modes']['sequence_mask']['all']
    token = summary['modes']['token_mask']['all']
    assert sequence['loss_tokens'] == token['loss_tokens'] == 3
    assert sequence['zero_weight_completions'] == 1
    assert sequence['weighted_nonzero_advantage_tokens'] == 0
    assert token['weighted_nonzero_advantage_tokens'] == 3
    assert token['absolute_advantage_weight_sum'] == pytest.approx(2.3221188)
    with pytest.raises(ValueError, match='disagrees with TRL'):
        compare_weights([{**rows[0], 'trl_sequence_weight': 1.0}])
    with pytest.raises(RuntimeError, match='Unexpected prompt groups'):
        list(capture_rows(result, diagnostics, ['b'], 2))


def test_diagnostic_colab_stage_uses_rollout_capture(monkeypatch):
    from types import SimpleNamespace

    from experiments.qfft_v93.colab import stages

    calls = []
    monkeypatch.setattr(stages, 'gpu_banner', lambda: None)
    context = SimpleNamespace(py_module=lambda *args: args, run=lambda *args: calls.append(args))
    stages._dispatch(context, 'r1-is-diagnostic')
    assert calls == [(('rl', 'experiments.qfft_v93.grpo_is_diagnostic', 'capture'), 'r1_is_diagnostic')]
