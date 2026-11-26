from types import SimpleNamespace

import pytest

from math_posttrain.common import load_config, write_json
from math_posttrain.validation import scheduled_steps, select_best, result_drive


def test_every_validation_step_is_saved_and_retained_including_odd_steps():
    from math_posttrain.mmr_grpo import MmrRecoveryCheckpoints
    cfg = load_config('grpo/grpo_mmr_dr500.json')
    steps = scheduled_steps(load_config('grpo/mmr_validation.json'))
    assert steps == list(range(150, 501, 25)) == cfg['retained_checkpoint_steps']
    assert cfg['save_steps'] == 25
    callback = MmrRecoveryCheckpoints(steps)
    for step in [174, 175, 176, 199, 200, 201, 225, 500]:
        control = SimpleNamespace(should_save=False)
        callback.on_step_end(None, SimpleNamespace(global_step=step, max_steps=500), control)
        assert control.should_save == (step in steps or step % 2 == 0 and step < 500)


def test_upload_failure_does_not_prune_validation_checkpoint(tmp_path):
    from math_posttrain import checkpointing as ck
    mirror = ck.CheckpointMirror(tmp_path / 'local', tmp_path / 'drive', keep=2,
                                mirror_optimizer_state=True, retain_steps=[175], min_free_gib=0)
    for step in [174, 175, 176]:
        write_json(mirror.local_dir / f'checkpoint-{step}' / ck.SAVED_MARK, {'step': step})
        write_json(mirror.local_dir / f'checkpoint-{step}' / 'trainer_state.json', {'global_step': step})
    mirror.error = 'simulated Drive quota error'
    mirror._prune_local()
    assert {p.name for p in mirror.local_dir.iterdir()} == {'checkpoint-175', 'checkpoint-176'}


def test_best_uses_only_complete_matching_dev_and_ties_choose_earliest():
    def row(step, score):
        return {'step': step, 'protocol_sha256': 'p', 'paired_problems': 249, 'mean_at_n': score}
    values = [row(200, .84), row(175, .84), row(150, .82)]
    assert select_best(values, 'p')['step'] == 175
    with pytest.raises(ValueError, match='mixed protocols'):
        select_best(values, 'other')
    with pytest.raises(ValueError, match='duplicate'):
        select_best(values + [row(175, .9)], 'p')
    values[0]['paired_problems'] = 248
    with pytest.raises(ValueError, match='incomplete'):
        select_best(values, 'p')


def test_separate_writable_result_drive_restores_input_configuration(monkeypatch):
    import os
    monkeypatch.setenv('P1_DRIVE_ROOT', '/input')
    monkeypatch.setenv('P1_RCLONE_REMOTE', 'input:')
    monkeypatch.setenv('P1_VALIDATION_DRIVE_ROOT', '/results')
    monkeypatch.setenv('P1_VALIDATION_RCLONE_REMOTE', 'results:')
    with pytest.raises(RuntimeError):
        with result_drive():
            assert os.environ['P1_DRIVE_ROOT'] == '/results'
            assert os.environ['P1_RCLONE_REMOTE'] == 'results:'
            raise RuntimeError('interrupted write')
    assert os.environ['P1_DRIVE_ROOT'] == '/input'
    assert os.environ['P1_RCLONE_REMOTE'] == 'input:'
