"""Test targeted fixes for VULN-02, VULN-05, VULN-07, and VULN-08."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.qfft_v93 import formats, grading
from experiments.qfft_v93.common import load_config


def test_formats_and_grading_final_segment_parity():
    samples = [
        'pure reasoning without any tag \\boxed{7}',
        'step 1\n<|im_start|>answer\n\n\\boxed{42}',
        '<think>draft \\boxed{1}</think>\nFinal: \\boxed{2}',
    ]
    for s in samples:
        for strict in (False, True):
            assert formats.final_segment(s, strict=strict) == grading.final_segment(s, strict=strict)


def test_grading_strict_anti_cheating():
    # Escaped set braces \{1, 2\} inside \boxed{} must extract cleanly
    text = 'thinking\n<|im_start|>answer\nThe solution set is \\boxed{\\{1, 2\\}}.'
    assert grading.boxed_contents(text) == ['\\{1, 2\\}']

    # In strict mode, no \boxed{} must return no_answer with missing_box (no tail guessing)
    res = grading.grade_text('<|im_start|>answer\nTherefore the value is 42.', '42', strict=True)
    assert res['status'] == 'no_answer'
    assert res['how'] == 'missing_box'

    # In strict mode, discarded candidate boxes must not rescue a wrong final box
    trick = '<|im_start|>answer\nCandidate \\boxed{1} and candidate \\boxed{2}, so final is \\boxed{99}.'
    res_trick = grading.grade_text(trick, '1, 2', strict=True)
    assert res_trick['status'] == 'wrong'


def test_grpo_defaults_config():
    cfg_path = Path(__file__).resolve().parents[2] / 'experiments' / 'qfft_v93' / 'configs' / 'grpo' / '_defaults.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    assert cfg['mask_truncated_completions'] is True
    assert cfg['learning_rate'] == 1e-05


def test_r1_reference_truncates_only_when_enabled():
    reference = load_config('train/qfft_r1_16k_b8.json')
    main = load_config('train/qfft_r1.json')
    assert reference['data'] == main['data']
    assert reference['run_name'] != main['run_name']
    assert reference['max_length'] == 16384
    assert reference['gradient_accumulation_steps'] == 8
    assert reference['save_steps'] == 250
    assert reference['mirror_optimizer_state'] is True

    def tokenizer(text, *, add_special_tokens):
        assert text == 'response'
        assert not add_special_tokens
        return {'input_ids': [1, 2, 3, 4]}

    with pytest.raises(OverflowError):
        formats.encode(tokenizer, 'qfft', 'question', 'response', 3)
    with pytest.raises(ValueError, match='only supported for QFFT'):
        formats.encode(tokenizer, 'sft', 'question', 'response', 3, truncate=True)
    encoded = formats.encode(tokenizer, 'qfft', 'question', 'response', 3, truncate=reference['truncate'])
    assert encoded['input_ids'] == encoded['labels'] == [1, 2, 3]
    assert encoded['truncated_tokens'] == 1
