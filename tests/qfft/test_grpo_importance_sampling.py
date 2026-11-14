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
