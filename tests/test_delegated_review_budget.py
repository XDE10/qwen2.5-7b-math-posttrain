import io,json,urllib.error
import pytest
from src.synthesis.shared_cost_client import SharedCostClient,BudgetExhausted
from src.synthesis.data_review_contract import parse,payload


def test_review_schema_fails_closed():
    d={'verdict':'pass','issues':[],'checks':['2+2=4'],'requires_visual':False,'answer_leak':False}
    assert parse(json.dumps(d),'stop')['verdict']=='pass'
    d['answer_leak']=True;assert parse(json.dumps(d),'stop')['verdict']=='uncertain'
    assert parse('{','stop')['verdict']=='uncertain'
    assert parse(json.dumps(d),'length')['verdict']=='uncertain'


def test_completed_usage_releases_reserve_but_unknown_dispatch_does_not(tmp_path,monkeypatch):
    p=payload({'problem':'2+2'});p['messages']=[{'role':'user','content':'x'}]
    body={'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':20}}
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:io.BytesIO(json.dumps(body).encode()))
    c=SharedCostClient(tmp_path/'ok','TEST_ONLY',budget=.001)
    c.call(p,'1');c.call(p,'2')
    assert len(c.rows)==2 and c.committed()<.0001
    c.close()
    def fail(*a,**k):raise urllib.error.URLError('offline')
    monkeypatch.setattr('urllib.request.urlopen',fail)
    c=SharedCostClient(tmp_path/'bad','TEST_ONLY',budget=.001)
    with pytest.raises(RuntimeError):c.call(p,'1')
    with pytest.raises(BudgetExhausted):c.call(p,'2')
    assert len(c.rows)==1
    c.close()
