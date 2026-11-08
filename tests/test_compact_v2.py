import pytest
from src.synthesis.compact_v2 import clean_problem, inspect_structure, payload


def test_cleaner_narrow_suffix_only():
    raw = 'Find x such that x+1=2.\nPlease translate the above problem into English.'
    r = clean_problem(raw)
    assert r['problem'] == 'Find x such that x+1=2.' and len(r['edits']) == 1
    math = 'Translate the triangle two units to the right. Find its coordinates.'
    assert clean_problem(math)['problem'] == math
    assert clean_problem('Translate this into English: x+1=2')['review_flags']
    observed = "Find the integer solutions.\n\nTranslate the above text into English, please retain the original text's line breaks and format, and output the translation result directly."
    assert clean_problem(observed)['problem'] == 'Find the integer solutions.'


def test_normalization_does_not_create_solution():
    text = r'Subtract one from both sides of x+1=2. This gives x=1 and satisfies the equation. \boxed{1}'
    assert not inspect_structure(text)['structure_pass']
    r = inspect_structure(text, allow_wrap=True)
    assert r['structure_pass'] and r['operations'] == ['wrap_existing_written_derivation']
    assert r['process_correctness'] == 'not_certified'
    assert not inspect_structure(r'\boxed{1}', allow_wrap=True)['structure_pass']


@pytest.mark.parametrize('text', [r'<think>x=1 \boxed{1}',
    r'<think>Long mathematical reasoning with x=1 and conditions.</think>\boxed{1} contradicts this',
    r'<think>Long mathematical reasoning with x=1 and conditions.</think>\boxed{1'])
def test_malformed_and_conflicting_suffix_rejected(text):
    assert not inspect_structure(text, allow_wrap=True)['structure_pass']


def test_blind_payload_has_single_teacher_and_no_reference():
    with pytest.raises(ValueError): payload({'problem':'x','answer':'1'})
    p = payload('x+1=2')
    assert p['model'] == 'deepseek-flash' and p['thinking'] == {'type':'disabled'}
    assert p['messages'][1]['content'] == 'x+1=2'


def test_display_delimiters_and_existing_derivation_are_preserved():
    from src.synthesis.compact_format_review import normalize_written_solution
    s = r'<think>Subtract one from x+1=2 and obtain x=1.</think> Check by substitution: 1+1=2. \[\boxed{1}\]'
    n=normalize_written_solution(s)
    assert n['repairable'] and 'Check by substitution: 1+1=2.' in n['text']
    assert n['text'].endswith(r'\boxed{1}')
    assert not n['mathematical_content_added']


def test_repair_rejects_malformed_tags_and_trailing_prose():
    from src.synthesis.compact_format_review import normalize_written_solution
    assert not normalize_written_solution(r'<think><think>x=1</think>\boxed{1}')['repairable']
    assert not normalize_written_solution(r'Sufficient written proof with x=1 established. \boxed{1} But actually x=2.')['repairable']
