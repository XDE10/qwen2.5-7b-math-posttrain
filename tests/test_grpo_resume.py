import random
import sys
from pathlib import Path

import pytest

from math_posttrain.common import CONFIG_DIR, load_config, read_json, read_jsonl, sha256_file, write_json, write_jsonl


def test_r1_grpo_mirrors_full_training_state_and_enables_safety_flags():
    formal = load_config('grpo/grpo_dapo_16k.json')
    assert formal['mirror_optimizer_state'] is True
    assert formal['max_steps'] == 150
    assert formal['last_box_fallback'] is True
    assert formal['mask_truncated_completions'] is True
    assert formal['vllm_importance_sampling_mode'] == 'token_mask'
    assert formal['max_completion_length'] == 16384


def test_grpo_cpu_resumes_full_mirror_on_an_empty_vm(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    trl = pytest.importorskip('trl')
    pytest.importorskip('datasets')
    pytest.importorskip('peft')
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from transformers.trainer_pt_utils import safe_globals

    from math_posttrain import grpo
    trainer_cls = trl.GRPOTrainer

    base, drive, vm1, vm2 = (tmp_path / name for name in ('base', 'drive', 'vm1', 'vm2'))
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'question': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tok.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tok.save_pretrained(base)
    GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=128, n_embd=8, n_layer=1, n_head=1,
                               pad_token_id=1, eos_token_id=2, bos_token_id=2)).save_pretrained(base)
    pool = tmp_path / 'pool.jsonl'
    write_jsonl(pool, [{'sample_id': str(i), 'question': f'question {i}', 'answer': '42'} for i in range(8)])
    cfg = load_config('grpo/grpo_dapo_16k.json')
    cfg.update(run_name='cpu_resume', max_steps=3, warmup_steps=0, save_steps=1,
               gradient_accumulation_steps=1, num_generations=2, max_completion_length=4, reward_workers=1)
    cfg['lora'] = {**cfg['lora'], 'r': 2, 'alpha': 2}
    monkeypatch.setattr(grpo, 'load_config', lambda _: cfg)
    monkeypatch.setenv('P1_DRIVE_ROOT', str(drive))
    monkeypatch.setenv('P1_LOCAL_ROOT', str(vm1))
    monkeypatch.delenv('P1_RCLONE_REMOTE', raising=False)
    command = ['grpo', '--config', 'cpu', '--smoke', '--no-merge', '--model-path', str(base), '--pool', str(pool)]
    monkeypatch.setattr(sys, 'argv', [*command, '--require-resume'])
    with pytest.raises(RuntimeError, match='No complete GRPO checkpoint'):
        grpo.main()

    def test_grade(self, items, strict=False):
        # Give the two real CPU rollouts different rewards, so AdamW updates its moments.
        assert strict
        return [{'status': 'correct' if i % 2 == 0 else 'wrong', 'how': 'last_box'}
                for i, _ in enumerate(items)]

    monkeypatch.setattr(grpo.Grader, 'grade_many', test_grade)
    monkeypatch.setattr(sys, 'argv', [*command, '--pause-after', '1'])
    grpo.main()
    grpo.main()  # Re-running phase one leaves the published checkpoint in place.

    run = drive / 'runs/project1/v93/runs/cpu_resume'
    checkpoint = run / 'ckpt_mirror/checkpoint-1'
    assert read_json(checkpoint / 'MIRROR_COMPLETE.json')['artifact_type'] == 'full_state'
    assert read_json(checkpoint / 'trainer_state.json')['global_step'] == 1
    assert (checkpoint / 'adapter_model.safetensors').is_file()
    assert not (run / 'GRPO_COMPLETE.json').exists()
    optimizer = torch.load(checkpoint / 'optimizer.pt', weights_only=True)
    scheduler = torch.load(checkpoint / 'scheduler.pt', weights_only=True)
    adapter = load_file(checkpoint / 'adapter_model.safetensors')
    with safe_globals():
        rng = torch.load(checkpoint / 'rng_state.pth', weights_only=True)
    assert optimizer['state'] and scheduler['last_epoch'] == 1

    class VerifyRestore(trl.GRPOTrainer):
        def _load_optimizer_and_scheduler(self, path):
            super()._load_optimizer_and_scheduler(path)
            if path:
                assert Path(path).is_relative_to(vm2)
                actual = self.optimizer.state_dict()
                assert actual['param_groups'] == optimizer['param_groups']
                for key, state in optimizer['state'].items():
                    for name, tensor in state.items():
                        torch.testing.assert_close(actual['state'][key][name], tensor, rtol=0, atol=0)
                assert self.lr_scheduler.state_dict() == scheduler
                actual_adapter = get_peft_model_state_dict(self.model)
                assert actual_adapter.keys() == adapter.keys()
                for key, tensor in adapter.items():
                    torch.testing.assert_close(actual_adapter[key], tensor, rtol=0, atol=0)

        def _load_rng_state(self, path):
            super()._load_rng_state(path)
            assert random.getstate() == rng['python']
            assert np.random.get_state()[0] == rng['numpy'][0]
            np.testing.assert_array_equal(np.random.get_state()[1], rng['numpy'][1])
            assert np.random.get_state()[2:] == rng['numpy'][2:]
            assert torch.equal(torch.random.get_rng_state(), rng['cpu'])
            restored_rng.append(path)

    restored_rng = []
    assert not vm2.exists()
    monkeypatch.setenv('P1_LOCAL_ROOT', str(vm2))
    monkeypatch.setattr(trl, 'GRPOTrainer', VerifyRestore)
    monkeypatch.setattr(sys, 'argv', [*command, '--require-resume'])
    with monkeypatch.context() as modified:
        modified.setenv('P1_CODE_SHA', 'different-code')
        with pytest.raises(RuntimeError, match='different experiment'):
            grpo.main()
    grpo.main()
    complete = read_json(run / 'GRPO_COMPLETE.json')
    assert complete['steps'] == 3
    restored = vm2 / 'runs/cpu_resume/checkpoint-1'
    assert complete['resumed_from'] == str(restored)
    assert restored_rng == [str(restored)]
    assert [r['step'] for r in read_jsonl(run / 'reward_status.jsonl')] == [0, 1, 2]

    extended = {
        **cfg, 'run_name': 'cpu_extend', 'max_steps': 5,
        'extend_from': {'config': 'source', 'code_bundle': None,
                        'approval_sha256': sha256_file(CONFIG_DIR / cfg['pool']['approval_manifest'])},
    }
    monkeypatch.setattr(grpo, 'load_config', lambda name: cfg if name == 'source' else extended)
    vm3 = tmp_path / 'vm3'
    monkeypatch.setenv('P1_LOCAL_ROOT', str(vm3))
    monkeypatch.setenv('P1_CODE_SHA', 'extension-code')
    final = run / 'ckpt_mirror/checkpoint-3'
    expected_optimizer = torch.load(final / 'optimizer.pt', weights_only=True)
    expected_scheduler = torch.load(final / 'scheduler.pt', weights_only=True)
    verified = []

    class VerifyExtension(trainer_cls):
        def _load_optimizer_and_scheduler(self, path):
            super()._load_optimizer_and_scheduler(path)
            assert Path(path) == vm3 / 'runs/cpu_extend/checkpoint-3'
            assert self.optimizer.state_dict()['param_groups'] == expected_optimizer['param_groups']
            for key, moments in expected_optimizer['state'].items():
                for name, tensor in moments.items():
                    torch.testing.assert_close(self.optimizer.state_dict()['state'][key][name], tensor, rtol=0, atol=0)
            assert self.lr_scheduler.state_dict() == expected_scheduler
            verified.append(path)

    monkeypatch.setattr(trl, 'GRPOTrainer', VerifyExtension)
    monkeypatch.setattr(sys, 'argv', command)
    grpo.main()
    extension_run = run.with_name('cpu_extend')
    complete = read_json(extension_run / 'GRPO_COMPLETE.json')
    assert complete['steps'] == 5 and complete['extension']['source_step'] == 3
    assert len(verified) == 1
    assert [r['step'] for r in read_jsonl(extension_run / 'reward_status.jsonl')] == [3, 4]
    assert [r['learning_rate'] for r in read_jsonl(extension_run / 'metrics.jsonl') if 'loss' in r] == [1e-5, 1e-5]
    assert read_json(run / 'GRPO_COMPLETE.json')['steps'] == 3


def test_corrupt_full_checkpoint_falls_back_and_low_disk_preserves_local(tmp_path, monkeypatch):
    from math_posttrain import checkpointing as ckpt

    local, drive = tmp_path / 'vm1', tmp_path / 'drive'
    monkeypatch.delenv('P1_RCLONE_REMOTE', raising=False)
    mirror = ckpt.CheckpointMirror(local, drive, mirror_optimizer_state=True, min_free_gib=0)
    for step in (1, 2):
        path = local / f'checkpoint-{step}'
        path.mkdir(parents=True)
        write_json(path / ckpt.SAVED_MARK, {'step': step})
        write_json(path / 'trainer_state.json', {'global_step': step})
        for name in ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            (path / name).write_bytes(b'original')
        mirror._mirror(path)
        assert mirror.has_full_mirror(step)
    newest = drive / 'ckpt_mirror/checkpoint-2'
    (newest / 'optimizer.pt').write_bytes(b'corrupt!')
    assert Path(ckpt.resolve_resume(tmp_path / 'vm2', drive)).name == 'checkpoint-1'
    (newest / ckpt.MIRROR_MARK).unlink()
    monkeypatch.setattr(ckpt, 'free_bytes', lambda path: 0)
    with pytest.raises(RuntimeError, match='retaining checkpoint-2'):
        mirror._before_save(3)
    assert (local / 'checkpoint-2/optimizer.pt').exists()


def test_holdout_scoring_rejects_missing_or_duplicate_draws(tmp_path):
    from math_posttrain.score import score_one

    rows = {'a': {'answer': '1'}, 'b': {'answer': '2'}}
    write_json(tmp_path / 'identity.json', {'protocol': {'n': 1}})
    write_jsonl(tmp_path / 'raw.jsonl', [{'sample_id': 'a', 'draw': 0}] * 2)
    with pytest.raises(RuntimeError, match='1 missing, 0 unexpected, 1 duplicates'):
        score_one(None, tmp_path, rows, rescore=False)


def test_known_scoring_recovery_is_bound_to_the_saved_generation(tmp_path):
    from math_posttrain.common import read_jsonl, sha256_text
    from math_posttrain.score import score_one

    class Grader:
        def grade_many(self, items):
            return [{'status': 'wrong', 'pred': '216112050', 'how': 'last_box'}
                    if text == r'\boxed{216112050}' else
                    {'status': 'timeout', 'how': 'comparison_timeout'} for text, _ in items]

    original = r'\boxed{2 \cdot (11!!)^2}'
    raw = {'sample_id': 'aime25/25', 'draw': 0, 'text': original, 'tokens': 12, 'finish_reason': 'stop'}
    raw_path = tmp_path / 'raw.jsonl'
    write_jsonl(raw_path, [raw])
    initial_bytes = raw_path.read_bytes()
    protocol = {'n': 1, 'model': 'run:r1', 'benchmark': 'aime25'}
    write_json(tmp_path / 'identity.json', {'protocol': protocol})
    rows = {'aime25/25': {'answer': '113'}}
    recovery = {**protocol, 'sample_id': 'aime25/25', 'draw': 0,
                'original_text_sha256': sha256_text(original), 'reference_answer': '113',
                'normalized_answer': '216112050'}
    summary = score_one(Grader(), tmp_path, rows, rescore=False, require_gradable=True, recovery=recovery)
    assert summary['exact_numeric_recoveries'] == [{'sample_id': 'aime25/25', 'draw': 0, 'status': 'wrong'}]
    scored, = read_jsonl(tmp_path / 'scored.jsonl')
    assert scored['pred'] == '216112050' and scored['exact_numeric_recovery']
    assert raw_path.read_bytes() == initial_bytes

    write_jsonl(raw_path, [{**raw, 'text': original + ' changed'}])
    with pytest.raises(RuntimeError, match='1 unscorable generations'):
        score_one(Grader(), tmp_path, rows, rescore=False, require_gradable=True, recovery=recovery)


def test_remote_full_upload_stops_before_publication_on_verification_failure(tmp_path, monkeypatch):
    from math_posttrain import drive_io

    src = tmp_path / 'source'
    src.mkdir()
    (src / 'optimizer.pt').write_bytes(b'original')
    monkeypatch.setenv('P1_DRIVE_ROOT', str(tmp_path))
    monkeypatch.setenv('P1_RCLONE_REMOTE', 'mydrive:')
    commands = []

    def remote(*args, **kwargs):
        commands.append(args)
        if args[0] == 'check':
            raise RuntimeError('checksum mismatch')

    monkeypatch.setattr(drive_io, '_rclone', remote)
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        drive_io.upload_tree(src, tmp_path / 'checkpoint-5', verify_full=True)
    assert [args[0] for args in commands] == ['copy', 'check']
    staging = commands[0][2]
    assert staging.startswith('mydrive:checkpoint-5.') and staging.endswith('.partial')
    assert commands[1][2] == staging
