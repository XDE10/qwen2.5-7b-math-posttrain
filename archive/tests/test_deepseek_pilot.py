import json
import pytest
from src.synthesis.deepseek_pilot import solve_payload,difficulty_payload,review_payload,PilotClient
from scripts.run_deepseek_pilot import difficulty_metrics,check_math


def test_generator_rejects_whole_record_and_sends_only_problem():
    with pytest.raises(ValueError):solve_payload({'problem':'1+1','reference_answer':'2'})
    p=solve_payload('Compute 1+1.')
    assert p['messages'][1]['content']=='Compute 1+1.'
    assert 'reference_answer' not in json.dumps(p)
    assert p['thinking']['type']=='disabled'


def test_target_level_not_in_difficulty_payload():
    anchors=[{'problem':'anchor','problem_type':'Algebra','level':'Level 3'}]
    p=difficulty_payload('target',anchors)
    d=json.loads(p['messages'][1]['content'])
    assert set(d)=={'anchors','target_problem'} and d['anchors'][0]['level']==3


def test_reserved_budget_prevents_dispatch(tmp_path,monkeypatch):
    def no_network(*args,**kwargs):raise AssertionError('Network must not run')
    monkeypatch.setattr('urllib.request.urlopen',no_network)
    c=PilotClient(tmp_path,'test-placeholder',budget=0)
    with pytest.raises(RuntimeError,match='budget'):c.call(solve_payload('test'),'one')


def test_unknown_previous_dispatch_not_repeated(tmp_path):
    from src.synthesis.deepseek_pilot import sha,dump
    p=solve_payload('test');key=sha({'task_id':'one','payload':p})
    dump(tmp_path/'ledger.json',[{'cache_key':key,'state':'reserved','reserved_usd':.1}])
    c=PilotClient(tmp_path,'placeholder')
    with pytest.raises(RuntimeError,match='Uncertain'):c.call(p,'one')


def test_low_to_high_error_denominator():
    rows=[{'valid':True,'gold_level':1,'estimated_level':3},{'valid':True,'gold_level':2,'estimated_level':2},
          {'valid':True,'gold_level':5,'estimated_level':5}]
    m=difficulty_metrics(rows)
    assert m['low_to_high_false_positive_rate']==.5 and m['predicted_high_precision']==.5


def test_math_verify_equivalence_conflict_and_missing():
    assert check_math(r'\boxed{\frac{1}{2}}','0.5')['correct']
    assert not check_math(r'\boxed{2} then \boxed{3}','2')['correct']
    assert not check_math('no final answer','2')['correct']


def test_windows_atomic_replace_retries(tmp_path,monkeypatch):
    from pathlib import Path
    from src.synthesis.deepseek_pilot import dump
    original=Path.replace
    calls=[]
    def locked_once(self,target):
        calls.append(1)
        if len(calls)==1:raise PermissionError('transient reader lock')
        return original(self,target)
    monkeypatch.setattr(Path,'replace',locked_once)
    dump(tmp_path/'out.json',{'ok':True})
    assert len(calls)==2 and json.loads((tmp_path/'out.json').read_text())=={'ok':True}
