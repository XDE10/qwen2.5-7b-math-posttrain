import json
import pytest
from src.synthesis.compact_v4 import serialize_response,clean_source,source_triage,boxes_kind,media_class


def obj(v):return json.dumps({'status':'solved','solution':'Let v be the walking speed. Equating the two encounter distances gives 10(u-v)=5(u+3v), hence u=5v and x=8.','final_answer':v})


def test_finite_numeric_only_and_no_blind_repair():
    r=serialize_response(obj(8));assert r['valid'] and r['fields']['final_answer']=='8'
    assert r['normalization_operations']==['finite_integer_final_answer_to_string']
    assert not serialize_response(obj(True))['valid']
    assert not serialize_response(obj(float('inf')))['valid']
    assert not serialize_response(r'{"status":"solved","solution":"x=1","final_answer":"\log 2"}')['valid']
    r=serialize_response(obj(0.125));assert r['fields']['final_answer']=='0.125'
    assert not r['global_backslash_repair']


def test_source_tail_and_math_translate_preserved():
    s='Find a and b.\n\n## SOLUTION:\n\nThe above text has been translated into English, preserving the original text.'
    assert clean_source(s)['problem']=='Find a and b.'
    assert clean_source('Translate the triangle by vector (1,2).')['problem'].startswith('Translate')
    r=source_triage({'sample_id':'x','problem':'Pick (A) 1 (B) 2','reference_answer':'A','solution':r'\boxed{A}'})
    assert r['experiment_scope_exclusions'] and not r['quality_isolation_reasons']


def test_multi_boxes_and_media_have_separate_classes():
    assert boxes_kind(r'\boxed{1} and again \boxed{1}')['kind']=='repeated_identical_answer'
    assert boxes_kind(r'\boxed{-3}p+\boxed{2}q+\boxed{5}r')['kind'].startswith('distinct_boxes')
    assert media_class('As shown. [asy]draw(A--B);[/asy]')=='requires_drawing_code_review'
    assert media_class('Compute 1+1. [asy]draw(A--B);[/asy]')=='prose_sufficiency_pending_review'


def test_real_bounded_verifier_subprocess_entrypoint():
    from scripts.compact_distillation_v4 import typed_check
    assert typed_check(r'\boxed{8}','8')['correct']
    assert not typed_check(r'\boxed{x>1}','x>2')['correct']
