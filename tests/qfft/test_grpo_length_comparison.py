from types import SimpleNamespace

import pytest

from experiments.qfft_v93.common import load_config
from experiments.qfft_v93.grpo import approved_pool_sources, build_pool


def test_length_comparison_configs_differ_only_in_budget_and_name(monkeypatch, tmp_path):
    configs = [load_config(f'grpo/compare_grpo_qfft_r1_{length}k.json') for length in (8, 12, 16)]
    ignored = {'max_completion_length', 'run_name', 'note'}
    shared = {k: v for k, v in configs[0].items() if k not in ignored}
    for cfg, budget in zip(configs, (8192, 12288, 16384), strict=True):
        assert cfg['max_completion_length'] == budget
        assert cfg['max_steps'] == 2
        assert cfg['init_model'] == 'run:qfft_r1_16k_b8_restart_v93'
        assert not cfg['shuffle_dataset']
        assert cfg['last_box_fallback'] and cfg['mask_truncated_completions']
        assert {k: v for k, v in cfg.items() if k not in ignored} == shared
        # An approved run proceeds to opening the first source.
        with pytest.raises(FileNotFoundError, match='project1_rl_rollout_approved_v1.jsonl'):
            list(approved_pool_sources(cfg, tmp_path))
    ids = configs[0]['pool']['comparison_sample_ids']
    assert len(ids) == len(set(ids)) == 16
    medium = [{'sample_id': sid, 'question': f'Medium question {i}', 'answer': '1'}
              for i, sid in enumerate(reversed(ids[::2]))]
    hard = [{'sample_id': sid, 'question': f'Hard question {i}', 'answer': '2'}
            for i, sid in enumerate(reversed(ids[1::2]))]
    monkeypatch.setattr('experiments.qfft_v93.grpo.approved_pool_sources',
                        lambda cfg, root: [(root / 'medium.jsonl', medium), (root / 'hard.jsonl', hard)])
    monkeypatch.setattr('experiments.qfft_v93.grpo.decontam.load_benchmarks', lambda path: {
        name: [(name, 'Benchmark question')] for name in ('aime24', 'aime25', 'amc23', 'math500', 'gsm8k')
    })
    for cfg in configs:
        pool, report = build_pool(cfg, SimpleNamespace(root=tmp_path), tmp_path, set())
        assert [row['sample_id'] for row in pool] == ids
        assert report['comparison_source_counts'] == {'medium.jsonl': 8, 'hard.jsonl': 8}


def test_whole_gpu_measurement_and_phase_accounting(monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from experiments.qfft_v93 import grpo_telemetry

    def fake_smi(argv, **kwargs):
        assert 'memory.used,memory.total,utilization.gpu' in argv[1]
        assert kwargs['timeout'] == 3
        return SimpleNamespace(stdout='66560, 98304, 75\n')

    monkeypatch.setattr(grpo_telemetry.subprocess, 'run', fake_smi)
    monkeypatch.setattr(grpo_telemetry.torch.cuda, 'synchronize', lambda: None)
    ticks = iter([0, 2, 3, 7, 10, 15])
    monkeypatch.setattr(grpo_telemetry.time, 'monotonic', lambda: next(ticks))
    telemetry = grpo_telemetry.ComparisonTelemetry()
    state = SimpleNamespace(global_step=0)
    llm = SimpleNamespace(generate=lambda: 'completions')
    trainer = SimpleNamespace(state=state, _generate_and_score_completions=lambda: llm.generate(),
                              vllm_generation=SimpleNamespace(llm=llm))
    telemetry.attach(trainer)
    telemetry.on_step_begin(None, state, None)
    assert trainer._generate_and_score_completions() == 'completions'
    telemetry.samples['generation'].append(grpo_telemetry.gpu_sample())
    telemetry.samples['update'].append((70, 96, 55))
    state.global_step = 1
    telemetry.on_step_end(None, state, None)
    assert telemetry.summary['comparison/generation_seconds'] == 4
    assert telemetry.summary['comparison/rollout_other_seconds'] == 4
    assert telemetry.summary['comparison/update_seconds'] == 7
    assert telemetry.summary['comparison/gpu_step_peak_used_gib'] == 70
    assert telemetry.summary['comparison/gpu_step_mean_utilization_pct'] == 65
    assert telemetry.summary['comparison/gpu_rollout_other_samples'] == 0


def test_pinned_trl_sampler_preserves_prompt_order_and_rollout_groups():
    pytest.importorskip('trl')
    from trl.trainer.utils import RepeatSampler
    from experiments.qfft_v93.grpo import grpo_config

    cfg = load_config('grpo/compare_grpo_qfft_r1_16k.json')
    args = grpo_config(cfg, 'unused', smoke=True, max_steps=None)
    ids = cfg['pool']['comparison_sample_ids']
    sampler = RepeatSampler(
        ids, mini_repeat_count=args.num_generations,
        batch_size=args.generation_batch_size // args.num_generations,
        repeat_count=args.num_iterations, shuffle=args.shuffle_dataset, seed=args.seed,
    )
    assert args.generation_batch_size == 64
    assert args.steps_per_generation == args.gradient_accumulation_steps == 32
    assert list(sampler) == [index for index in range(16) for _ in range(8)]
