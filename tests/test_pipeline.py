"""Core pipeline unit & zero-GPU end-to-end tests."""
from __future__ import annotations

import json
import pytest

from math_posttrain import cli, decontam, formats, grading, metrics
from math_posttrain.common import get_config_dir, load_config, read_json, write_jsonl


class _FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {'<|im_end|>': 151645}

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self.vocab[tok]

    def convert_ids_to_tokens(self, idx: int) -> str:
        return '<|im_end|>' if idx == 151645 else f't{idx}'

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_dict=False):
        return [ord(c) for c in formats.render_prompt(messages[1]['content'])]

    def decode(self, ids, skip_special_tokens=False) -> str:
        return ''.join(chr(i) for i in ids)

    def __call__(self, text: str, add_special_tokens=False):
        return {'input_ids': [ord(c) for c in text]}


def test_all_shipped_configs_load_without_hardcoded_paths():
    cfg_dir = get_config_dir()
    cfg_files = sorted(cfg_dir.rglob('*.json'))
    assert len(cfg_files) >= 6
    for path in cfg_files:
        raw = path.read_text(encoding='utf-8')
        for forbidden in ('/content/', 'mydrive', 'project1', 'rclone', 'P1_'):
            assert forbidden not in raw, f'Found legacy hardcoded token {forbidden!r} in {path}'
        data = load_config(path.relative_to(cfg_dir).as_posix())
        assert isinstance(data, dict)


def test_qfft_and_sft_encoding_invariants():
    tok = _FakeTokenizer()
    response = formats.canonical_response('Wait, check 2+2=4.', r'Therefore \boxed{4}.')
    qfft = formats.encode(tok, 'qfft', 'What is 2+2?', response, max_length=4096)
    sft = formats.encode(tok, 'sft', 'What is 2+2?', response, max_length=4096)

    assert qfft['prompt_tokens'] == 0
    assert qfft['input_ids'] == qfft['labels']
    assert qfft['input_ids'][-1] != 151645

    assert sft['labels'][: sft['prompt_tokens']] == [-100] * sft['prompt_tokens']
    assert sft['labels'][sft['prompt_tokens'] : -1] == qfft['labels']
    assert sft['labels'][-1] == 151645
    assert sft['supervised_tokens'] == qfft['supervised_tokens'] + 1


def test_qfft_truncation_behavior():
    tok = _FakeTokenizer()
    response = formats.canonical_response('a' * 50, 'b' * 10)
    with pytest.raises(OverflowError):
        formats.encode(tok, 'qfft', 'q', response, max_length=20, truncate=False)
    trunc = formats.encode(tok, 'qfft', 'q', response, max_length=20, truncate=True)
    assert len(trunc['input_ids']) == 20
    assert trunc['truncated_tokens'] == len(response) - 20


def test_final_segment_and_box_extraction():
    text = r'Try \boxed{999} first.' + formats.ANSWER_SEP + r'Correct is \boxed{\frac{1}{2}}.'
    assert formats.final_segment(text) == r'Correct is \boxed{\frac{1}{2}}.'
    assert grading.boxed_contents(formats.final_segment(text)) == [r'\frac{1}{2}']
    assert formats.final_segment('No separator here', strict=True) == ''
    assert grading.last_box_without_final_tag(r'Draft \boxed{1} final \boxed{2}') == '<|im_start|>answer\n\\boxed{2}'


def test_math_verify_grading_edge_cases():
    pytest.importorskip('math_verify')
    ok = grading.grade_text(r'Wrong \boxed{1}' + formats.ANSWER_SEP + r'Final \boxed{\frac{2}{4}}', '0.5')
    assert ok['status'] == 'correct'

    wrong = grading.grade_text(formats.ANSWER_SEP + r'\boxed{42}', '43')
    assert wrong['status'] == 'wrong'

    strict_missing = grading.grade_text(r'No answer tag \boxed{42}', '42', strict=True)
    assert strict_missing['status'] == 'no_answer' and strict_missing['how'] == 'missing_answer_tag'


def test_metrics_pass_at_k_and_paired_bootstrap():
    assert metrics.pass_at_k(16, 0, 1) == 0.0
    assert metrics.pass_at_k(16, 16, 1) == 1.0
    assert abs(metrics.pass_at_k(4, 2, 1) - 0.5) < 1e-12
    assert abs(metrics.pass_at_k(4, 2, 2) - (1 - 1 / 6)) < 1e-12

    paired = metrics.paired_bootstrap([1.0, 1.0, 0.5, 0.5] * 5, [0.5, 0.5, 0.5, 0.5] * 5, iters=500, seed=0)
    assert paired['diff'] > 0 and paired['ci95'][0] > 0


def test_decontamination_detects_exact_and_near_duplicates():
    q = 'Find the sum of all positive integers n such that n squared plus 12n minus 2007 is a perfect square.'
    flags = decontam.find_flags([('train/1', q)], [('math500/1', q + ' ')])
    assert len(flags) == 1 and flags[0]['train_id'] == 'train/1'


def test_cli_doctor_runs_cleanly(capsys):
    rc = cli.main(['doctor'])
    out = capsys.readouterr().out
    assert rc == 0
    assert '[ok] eval.json' in out
    assert '[ok] train/qfft_r1.json' in out
    assert '[ok] grpo/mmr_dr50.json' in out


def test_end_to_end_fake_eval_and_score(tmp_path, monkeypatch):
    pytest.importorskip('math_verify')
    data_dir = tmp_path / 'data'
    out_dir = tmp_path / 'outputs'
    monkeypatch.setenv('MATH_POSTTRAIN_DATA_DIR', str(data_dir))
    monkeypatch.setenv('MATH_POSTTRAIN_OUTPUT_DIR', str(out_dir))

    bench_dir = data_dir / 'benchmarks'
    write_jsonl(bench_dir / 'aime24.jsonl', [
        {'sample_id': 'aime24/00', 'problem': 'What is 6 * 7?', 'answer': '42'},
        {'sample_id': 'aime24/01', 'problem': 'What is 10 + 5?', 'answer': '15'},
    ])
    write_jsonl(bench_dir / 'math500.jsonl', [
        {'sample_id': 'math500/00', 'problem': 'Simplify 2/4.', 'answer': '0.5', 'level': 1},
    ])

    assert cli.main(['generate', '--model', 'base', '--tag', 'demo', '--suite', 'smoke', '--engine', 'fake']) == 0
    assert cli.main(['score', '--tags', 'demo', '--workers', '1']) == 0

    results = read_json(out_dir / 'evals' / 'results.json')
    assert results['results']['demo']['math500']['mean_at_n'] == 1.0
    assert results['results']['demo']['aime24']['mean_at_n'] == 0.5
    assert (out_dir / 'evals' / 'results.md').is_file()
