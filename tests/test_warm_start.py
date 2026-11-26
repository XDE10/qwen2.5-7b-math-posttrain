import copy
import sys

import pytest

from math_posttrain import drive_io
from math_posttrain.common import read_json, write_json, write_jsonl
from math_posttrain.warm_start import plan_warm_start


def publish_mirror(run, step, total, *, complete=True, artifact_type=None):
    mirror = run / 'ckpt_mirror' / f'checkpoint-{step}'
    write_json(mirror / 'config.json', {})
    write_json(mirror / 'trainer_state.json', {'global_step': step, 'max_steps': total})
    (mirror / 'model.safetensors').write_bytes(b'weights')
    if complete:
        write_json(mirror / 'MIRROR_COMPLETE.json', {'artifact_type': artifact_type} if artifact_type else {})
    return mirror


def test_selects_latest_usable_mirror_and_checks_identity(tmp_path):
    source = tmp_path / 'source'
    recipe = {'lr_scheduler_type': 'cosine', 'warmup_fraction': 0.1, 'learning_rate': 1e-5}
    identity = {'run': 'source', 'recipe': recipe, 'data_sha256': 'abc'}
    write_json(source / 'run_identity.json', {'identity': identity})
    publish_mirror(source, 125, 750)  # old light marker without artifact_type
    publish_mirror(source, 250, 750, artifact_type='full_state')
    publish_mirror(source, 375, 750, complete=False)
    broken = publish_mirror(source, 500, 750, artifact_type='weights_only')
    (broken / 'model.safetensors').unlink()
    plan = plan_warm_start(tmp_path, 'source', {**identity, 'run': 'destination'}, 750)
    assert plan['original_start_step'] == 125
    assert plan['remaining_steps'] == 625
    assert plan['learning_rate']['first_update_lr'] == pytest.approx(9.865224352899119e-6)
    assert plan['source_weight_files'] == {'config.json': 3, 'model.safetensors': 7}
    with pytest.raises(ValueError, match='same data'):
        plan_warm_start(tmp_path, 'source', {**identity, 'run': 'other', 'data_sha256': 'different'}, 750)

    chained = tmp_path / 'destination'
    write_json(chained / 'run_identity.json', {'identity': {**identity, 'run': 'destination', 'warm_start': plan}})
    publish_mirror(chained, 125, 625, artifact_type='weights_only')
    second = plan_warm_start(tmp_path, 'destination', {**identity, 'run': 'next'}, 750)
    assert (second['source_checkpoint_step'], second['original_start_step'], second['remaining_steps']) == (125, 250, 500)


def test_download_only_weights_validates_sizes(tmp_path, monkeypatch):
    monkeypatch.delenv('P1_RCLONE_REMOTE', raising=False)
    mirror = publish_mirror(tmp_path, 125, 750, artifact_type='weights_only')
    (mirror / 'optimizer.pt').write_bytes(b'optimizer')
    target = tmp_path / 'weights'
    files = {'model.safetensors': 7, 'config.json': 3}
    drive_io.download_tree(mirror, target, files=files)
    assert set(p.name for p in target.iterdir()) == files.keys()
    with pytest.raises(RuntimeError, match='Incomplete download'):
        drive_io.download_tree(mirror, target, files={**files, 'model.safetensors': 8})
    assert (target / 'model.safetensors').read_bytes() == b'weights'


def test_colab_stage_passes_source_and_destination(tmp_path, monkeypatch):
    from math_posttrain.colab import stages

    leases, commands = [], []

    class Lease:
        def __init__(self, stage, sha, force):
            leases.append(stage)

        def close(self, **values):
            assert values['state'] == 'complete'

    class Context:
        sha = 'test'
        pkg = stages.Path(stages.__file__).resolve().parents[1]

        def python(self, env):
            assert env == 'train'
            return 'python3'

        def run(self, args, log_name):
            commands.append((args, log_name))

    monkeypatch.setattr(stages, 'StageLease', Lease)
    monkeypatch.setattr(stages, 'gpu_banner', lambda: None)
    monkeypatch.setattr(stages, 'check_host_ram', lambda: None)
    monkeypatch.setattr(stages, 'PROJECT', tmp_path)
    (tmp_path / 'data/qfft1000_v93/r1').mkdir(parents=True)
    (tmp_path / 'data/qfft1000_v93/r1/train.jsonl').touch()
    stages.run_stage(Context(), 'warm-start:qfft_r1_16k_b8:old:new')
    assert leases == ['warm-start:new']
    args, log_name = commands[0]
    assert args[-6:] == ['--config', 'train/qfft_r1_16k_b8.json', '--warm-start-from', 'old', '--run-name', 'new']
    assert log_name == 'warm_start_new'


def test_cpu_trainer_loads_light_weights_and_runs_only_remaining_updates(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('accelerate')
    tokenizers = pytest.importorskip('tokenizers')
    from transformers import get_cosine_schedule_with_warmup

    from math_posttrain import models, train_sft
    from math_posttrain.common import drive_layout, load_config

    monkeypatch.setenv('P1_DRIVE_ROOT', str(tmp_path / 'drive'))
    monkeypatch.setenv('P1_LOCAL_ROOT', str(tmp_path / 'ssd'))
    monkeypatch.delenv('P1_RCLONE_REMOTE', raising=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    base = tmp_path / 'base'
    vocab = {'[PAD]': 0, '[UNK]': 1, 'one': 2, 'plus': 3, 'two': 4}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', unk_token='[UNK]').save_pretrained(base)
    transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(vocab), n_positions=16, n_embd=8, n_layer=1, n_head=1, pad_token_id=0,
        bos_token_id=1, eos_token_id=1,
    )).save_pretrained(base)
    monkeypatch.setattr(models, 'ensure_tokenizer', lambda **kwargs: base)
    data = tmp_path / 'data.jsonl'
    write_jsonl(data, [
        {'index': i, 'sample_id': str(i), 'question': 'one plus one', 'response': 'one plus one two'} for i in range(4)
    ])
    cfg = load_config('train/qfft_r1_16k_b8.json')
    cfg.update(run_name='source', gradient_accumulation_steps=1, num_train_epochs=2, save_steps=4,
               warmup_fraction=0.25, mirror_optimizer_state=False)
    monkeypatch.setattr(train_sft, 'load_config', lambda _: copy.deepcopy(cfg))
    common = ['train_sft', '--config', 'cpu', '--data', str(data), '--smoke', '--no-export']
    monkeypatch.setattr(sys, 'argv', [*common, '--model-path', str(base)])
    train_sft.main()
    runs = drive_layout().runs
    mirror = runs / 'source/ckpt_mirror/checkpoint-4'
    assert read_json(mirror / 'MIRROR_COMPLETE.json')['artifact_type'] == 'weights_only'
    source_weights = (mirror / 'model.safetensors').read_bytes()
    assert source_weights != (base / 'model.safetensors').read_bytes()

    original_scheduler = train_sft.warm_start_scheduler
    actual_lrs = []

    def record_optimizer(optimizer, plan):
        assert not optimizer.state
        unwrapped = getattr(optimizer, 'optimizer', optimizer)
        unwrapped.register_step_pre_hook(lambda opt, args, kwargs: actual_lrs.append(opt.param_groups[0]['lr']))
        return original_scheduler(optimizer, plan)

    original_loader = transformers.AutoModelForCausalLM.from_pretrained

    def record_model(path, **kwargs):
        assert (path / 'model.safetensors').read_bytes() == source_weights
        assert set(p.name for p in path.iterdir()) == {'config.json', 'generation_config.json', 'model.safetensors'}
        return original_loader(path, **kwargs)

    monkeypatch.setattr(train_sft, 'warm_start_scheduler', record_optimizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', record_model)
    monkeypatch.setattr(sys, 'argv', [*common, '--warm-start-from', 'source', '--run-name', 'destination'])

    def interrupted_download(*args, **kwargs):
        raise RuntimeError('interrupted download')

    with monkeypatch.context() as interrupted:
        interrupted.setattr(drive_io, 'download_tree', interrupted_download)
        with pytest.raises(RuntimeError, match='interrupted download'):
            train_sft.main()
    newer = publish_mirror(runs / 'source', 6, 8, artifact_type='weights_only')
    (newer / 'model.safetensors').write_bytes(source_weights)
    train_sft.main()

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=cfg['learning_rate'])
    reference = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=8)
    expected_lrs = []
    for step in range(8):
        if step >= 4:
            expected_lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        reference.step()
    assert actual_lrs == pytest.approx(expected_lrs)
    status = read_json(runs / 'destination/status.json')
    assert status['steps'] == 4
    assert status['warm_start']['optimizer'] == 'rebuilt_from_config'
    assert status['warm_start']['learning_rate']['first_update_lr'] == pytest.approx(expected_lrs[0])
    assert (mirror / 'model.safetensors').read_bytes() == source_weights
