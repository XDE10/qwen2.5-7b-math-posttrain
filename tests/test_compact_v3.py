import json
from src.synthesis.compact_v3 import screen_problem,payload,serialize_response,repeated_derivation


def test_missing_media_and_diagram_triaged():
    assert not screen_problem('In the diagram shown below find angle ABC.')['screen_pass']
    assert not screen_problem('Find x given https://example.invalid/diagram.png and AB=2.')['screen_pass']
    assert screen_problem('A rectangle has length 6 and width 3. Determine its area.')['screen_pass']
    # Known false positives are explicit triage rather than a semantic claim.
    r=screen_problem('The graph is y=x+1. Find its intersection with the x-axis.')
    assert not r['screen_pass'] and r['semantic_self_containedness'].startswith('pending')


def test_json_render_never_uses_reference_and_accepts_display_math():
    content=json.dumps({'status':'solved','solution':r'Subtract 1 from both sides of \[x+1=2\]. Thus x=1, which satisfies the equation.','final_answer':'1'})
    r=serialize_response(content)
    assert r['valid'] and r['text'].endswith(r'\boxed{1}')
    assert not r['mathematics_added_by_serializer']
    assert payload('x+1=2')['thinking']=={'type':'disabled'}


def test_rejects_nested_drafts_unknown_fields_and_unclosed_answer():
    for d in [dict(status='solved',solution='<think>draft</think>',final_answer='1'),
              dict(status='solved',solution='Write a full derivation with x=1 and verify it.',final_answer=r'\frac{1}{2'),
              dict(status='solved',solution='x=1',final_answer='1',reference_answer='1')]:
        assert not serialize_response(json.dumps(d))['valid']


def test_duplicate_blocks_not_short_formula_reuse():
    p='Subtract the same constant from both sides and use positivity to conclude the unique real root of this equation is one.'
    assert repeated_derivation(p+'\n\n'+p)['signals']
    assert not repeated_derivation('x=1. Checking gives x+1=2. Therefore the solution is x=1.')['signals']
