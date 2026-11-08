"""Compatibility is about token IDs, not loosening masks or length checks."""
import copy
import json
from pathlib import Path

import pytest
from transformers import BatchEncoding

from src.data.approved_training_inputs_v2 import load_approved
from src.training.approved_hf import CompletionCollator, encode_sft as old_encode
from src.training.sft_encoding_v6 import encode_sft, token_ids


@pytest.fixture
def row():
    return load_approved('data/portable_release/manifest.json',
                         'project1_compact_paired.jsonl')[0]


class Template:
    def __init__(self, mode='list', mismatch=False, no_completion=False):
        self.mode, self.mismatch, self.no_completion = mode, mismatch, no_completion
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), dict(kwargs)))
        prefix = kwargs['add_generation_prompt']
        ids = [1, 2] if prefix or self.no_completion else [1, 2, 3, 4]
        if self.mismatch and not prefix:
            ids[0] = 9
        if self.mode == 'mapping' or (self.mode == 'v5' and kwargs.get('return_dict', True)):
            return BatchEncoding({'input_ids': ids, 'attention_mask': [1] * len(ids)})
        return ids


def test_original_reproduces_mapping_failure(row):
    with pytest.raises(ValueError, match='prefix/length'):
        old_encode([row], Template('v5'), 4096)


@pytest.mark.parametrize('mode', ['list', 'v5', 'mapping'])
def test_compatibility_retains_assistant_only_mask(row, mode):
    tokenizer = Template(mode)
    actual = encode_sft([row], tokenizer, 4)
    assert actual == [{'input_ids': [1, 2, 3, 4], 'labels': [-100, -100, 3, 4]}]
    for messages, kw in tokenizer.calls:
        assert kw['return_dict'] is False
        assert kw['return_tensors'] is None
        assert kw['truncation'] is False and kw['padding'] is False
        assert messages[0]['content'] == r'Please reason step by step, and put your final answer within \boxed{}.'
        assert messages[1]['content'] == row['problem']
    assert tokenizer.calls[1][0][-1]['content'] == row['solution']


def test_mismatch_still_rejected(row):
    with pytest.raises(ValueError, match='prefix/length'):
        encode_sft([row], Template(mismatch=True), 4096)


def test_length_limit_not_silently_truncated(row):
    with pytest.raises(ValueError, match='prefix/length'):
        encode_sft([row], Template(), 3)


def test_no_completion_rejected(row):
    with pytest.raises(ValueError, match='prefix/length'):
        encode_sft([row], Template(no_completion=True), 4096)


def test_padding_and_eos_still_correct(row):
    encoded = encode_sft([row], Template(), 4096)
    batch = CompletionCollator(0)(encoded + [{'input_ids': [1, 2, 4], 'labels': [-100, -100, 4]}])
    assert batch['input_ids'].shape == (2, 4)
    assert batch['labels'].tolist() == [[-100, -100, 3, 4], [-100, -100, 4, -100]]
    assert batch['attention_mask'].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    # Last completion token (mock EOS=4) remains supervised.
    assert batch['labels'][0, -1].item() == 4


@pytest.mark.parametrize('value', [[], '123', [[1, 2]], {'attention_mask': [1]},
                                  [1, True], [1, -1], [1, 1.5]])
def test_malformed_template_output_rejected(value):
    with pytest.raises(ValueError):
        token_ids(value)


@pytest.mark.parametrize('length', [True, 0, -1, 1.5])
def test_invalid_budget_rejected(row, length):
    with pytest.raises(ValueError, match='positive integer'):
        encode_sft([row], Template(), length)


def test_real_tokenizer_old_and_new_ids_labels_identical():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'data/tokenizers/Qwen2.5-7B-Instruct', local_files_only=True)
    # Run full verification only on the locally validated 4.x contract.
    from importlib.metadata import version
    if not version('transformers').startswith('4.'):
        pytest.skip('Old encoder intentionally incompatible with 5.x; run v6 full checker instead')
    for arm in ('raw', 'compact'):
        rows = load_approved('data/portable_release/manifest.json',
                             f'project1_{arm}_paired.jsonl')
        assert encode_sft(rows, tokenizer, 4096) == old_encode(rows, tokenizer, 4096)


def test_v6_sft_entry_uses_fixed_encoder_without_loading_a_model(monkeypatch):
    from src.training import approved_hf_v6 as runner
    import scripts.train_project1_sft_v6 as entry
    import torch
    assert runner.encode_sft is encode_sft
    assert entry.cli is runner.cli
    monkeypatch.setattr(runner, 'approved_config', lambda _: ({'stage': 'sft'}, [], {}))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='requires CUDA'):
        runner.run('unused.json', 'disabled')


@pytest.mark.parametrize('config', [{'stage': 'rl'}, {'stage': 'sft', 'vision': True}])
def test_v6_guard_rejects_unverified_rl_vision(tmp_path, config):
    from src.training.guards_v6 import approved_config
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='text SFT only'):
        approved_config(path)


def test_v6_resume_identity_is_checked_after_adding_encoder_hashes(tmp_path, monkeypatch):
    from src.training import guards_v6 as guard
    path = tmp_path / 'config.json'
    def base(p):
        supplied = json.loads(Path(p).read_text(encoding='utf-8'))
        assert 'resume_from_checkpoint' not in supplied
        return supplied, [], {'versions': {'transformers': 'fixture'}}
    monkeypatch.setattr(guard, 'base_config', base)
    path.write_text('{"stage":"sft"}', encoding='utf-8')
    _, _, expected = guard.approved_config(path)
    assert expected['runner_sha256'] == guard.sha(Path(guard.__file__).with_name('approved_hf_v6.py'))
    assert expected['sft_encoder_v6_sha256'] == guard.sha(Path(guard.__file__).with_name('sft_encoding_v6.py'))
    checkpoint = tmp_path / 'checkpoint-1'
    checkpoint.mkdir()
    for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
        (checkpoint / name).write_bytes(b'fixture')
    (checkpoint / 'trainer_state.json').write_text('{"global_step":1}', encoding='utf-8')
    identity_path = checkpoint / 'run_identity.json'
    identity_path.write_text(json.dumps(expected), encoding='utf-8')
    path.write_text(json.dumps({'stage': 'sft', 'resume_from_checkpoint': str(checkpoint)}), encoding='utf-8')
    _, _, actual = guard.approved_config(path)
    assert actual == expected
    old = dict(expected)
    old.pop('sft_encoder_v6_sha256')
    identity_path.write_text(json.dumps(old), encoding='utf-8')
    with pytest.raises(ValueError, match='identity mismatch'):
        guard.approved_config(path)
