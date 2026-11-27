"""Unit tests for MMR-DR-GRPO algorithm components and RL config wiring."""
from __future__ import annotations

import math
import numpy as np
import pytest

from math_posttrain.common import load_config


def test_mmr_reweight_group_penalizes_duplicates():
    pytest.importorskip('trl')
    from math_posttrain.mmr_grpo import reweight_group

    rewards = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    # Completions 0 and 1 are identical; completion 2 is orthogonal.
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    order: list[int] = []
    adjusted = reweight_group(rewards, embeddings, order=order)

    assert order[0] == 0
    assert order[1] == 2  # orthogonal completion preferred over identical duplicate
    assert adjusted[2] > adjusted[1]


def test_cosine_reward_bounds():
    pytest.importorskip('trl')
    from math_posttrain.mmr_grpo import cosine_reward

    assert math.isclose(cosine_reward(True, 0, 16384), 1.0, rel_tol=1e-6)
    assert math.isclose(cosine_reward(True, 16384, 16384), 0.5, rel_tol=1e-6)
    assert math.isclose(cosine_reward(False, 0, 16384), 0.0, abs_tol=1e-6)
    assert math.isclose(cosine_reward(False, 16384, 16384), -0.5, rel_tol=1e-6)


def test_grpo_configs_build_cleanly(tmp_path):
    pytest.importorskip('trl')
    from math_posttrain.grpo import build_grpo_config

    for cfg_name in ('grpo/mmr_dr50.json', 'grpo/dapo_16k.json'):
        cfg = load_config(cfg_name)
        args = build_grpo_config(cfg, tmp_path / cfg['run_name'], smoke=True, max_steps=2)
        assert args.max_steps == 2
        assert args.use_vllm is False
        assert args.loss_type == cfg['loss_type']


def test_mmr_reward_groups_and_format_scoring():
    pytest.importorskip('trl')
    pytest.importorskip('math_verify')
    from math_posttrain.mmr_grpo import MmrReward

    class _Tok:
        eos_token_id = 151645
        pad_token_id = 151643

    cfg = load_config('grpo/mmr_dr50.json')
    cfg['num_generations'] = 2
    embed = lambda texts: np.eye(len(texts), dtype=np.float32)  # noqa: E731
    reward = MmrReward(_Tok(), cfg, embed)

    metrics = {}
    scores = reward(
        completions=[
            [{'content': 'work\n</think>\n<answer>\\boxed{42}</answer>'}],
            [{'content': 'work\n</think>\n<answer>\\boxed{99}</answer>'}],
        ],
        completion_ids=[[1, 2, 151645], [1, 2, 151645]],
        reference_answer=['42', '42'],
        sample_id=['p/0', 'p/0'],
        log_metric=lambda k, v: metrics.__setitem__(k, v),
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]
    assert metrics['reward/correct_frac'] == 0.5
    assert metrics['reward/format_frac'] == 1.0
