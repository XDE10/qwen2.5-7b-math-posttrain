import random
import sys
from pathlib import Path

import pytest

from experiments.qfft_v93.common import load_config, read_json, read_jsonl, write_jsonl


def test_r1_grpo_mirrors_full_training_state_without_changing_smoke_reward():
    formal = load_config('grpo/grpo_qfft_r1_16k_b8.json')
    smoke = load_config('grpo/smoke_grpo_qfft_r1_fallback_8k.json')
    assert formal['mirror_optimizer_state'] is True
    assert smoke['mirror_optimizer_state'] is False
    assert (formal['max_steps'], smoke['max_steps']) == (150, 2)
    for key in ('init_model', 'last_box_fallback', 'mask_truncated_completions', 'generation_kwargs'):
        assert formal[key] == smoke[key]
    assert formal['max_completion_length'] == smoke['max_completion_length'] == 8192


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

    from experiments.qfft_v93 import grpo

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
    cfg = load_config('grpo/grpo_qfft_r1_16k_b8.json')
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
    grpo.main()
    complete = read_json(run / 'GRPO_COMPLETE.json')
    assert complete['steps'] == 3
    restored = vm2 / 'runs/cpu_resume/checkpoint-1'
    assert complete['resumed_from'] == str(restored)
    assert restored_rng == [str(restored)]
    assert [r['step'] for r in read_jsonl(run / 'reward_status.jsonl')] == [0, 1, 2]
