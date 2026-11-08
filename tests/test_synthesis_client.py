import asyncio
import json
import pytest
from src.synthesis.async_client import AsyncSynthesisClient, BlindRequest, MockTransport, TransientError, BudgetExceeded


def req(): return BlindRequest('train/1','Compute 1+1.','fixed-revision')
def valid(text): return {'accepted':text=='#### 2','reason':'independent exact check'}
def run(awaitable): return asyncio.run(awaitable)


def test_dry_run_never_dispatches_and_rejects_reference_fields(tmp_path):
    transport = MockTransport()
    c = AsyncSynthesisClient(tmp_path/'cache.db',transport)
    result = run(c.generate(req(),valid))
    assert result['state']=='dry_run' and transport.calls==[]
    with pytest.raises(ValueError): BlindRequest.from_record({'sample_id':'x','problem':'x','source_revision':'r','answer':'2'})
    with pytest.raises(ValueError): BlindRequest('x','x','r',split='test')
    with pytest.raises(ValueError): AsyncSynthesisClient(tmp_path/'bad.db',object(),dry_run=False)


def test_blind_resampling_caps_and_persistent_cache(tmp_path):
    t = MockTransport(['wrong','wrong','#### 2'])
    c = AsyncSynthesisClient(tmp_path/'c.db',t,dry_run=False,requests_per_second=10000)
    result = run(c.generate(req(),valid))
    assert result['state']=='accepted' and len(result['attempts'])==3
    assert [p['messages'] for p in t.calls] == [t.calls[0]['messages']]*3
    assert all('reason' not in json.dumps(p) and 'reference' not in json.dumps(p) for p in t.calls)
    c.cache.close()
    t2 = MockTransport()
    c2 = AsyncSynthesisClient(tmp_path/'c.db',t2,dry_run=False)
    assert run(c2.generate(req(),valid))==result and not t2.calls


def test_wrong_generations_stop_at_three_and_do_not_send_verifier_feedback(tmp_path):
    t = MockTransport(['bad']*10)
    c = AsyncSynthesisClient(tmp_path/'c.db',t,dry_run=False,requests_per_second=10000)
    assert run(c.generate(req(),lambda _: {'accepted':False,'gold':'TOP_SECRET'}))['state']=='rejected'
    assert len(t.calls)==3 and 'TOP_SECRET' not in json.dumps(t.calls)


def test_retry_backoff_rate_limit_and_transport_count(tmp_path):
    class Clock:
        now = 10.
        delays = []
        def time(self): return self.now
        async def sleep(self,d): self.delays.append(d); self.now+=d
    clock = Clock()
    t = MockTransport([TransientError('network'),TransientError('busy'),'#### 2'])
    c = AsyncSynthesisClient(tmp_path/'c.db',t,dry_run=False,requests_per_second=2,backoff_base=.1,sleep=clock.sleep,clock=clock.time)
    result = run(c.generate(req(),valid))
    assert result['state']=='accepted' and len(result['attempts'])==1 and len(t.calls)==3
    assert .1 in clock.delays and .2 in clock.delays
    assert sum(clock.delays)>=1.
    assert c.cache.db.execute('SELECT count(*) FROM calls').fetchone()[0]==3


@pytest.mark.parametrize('max_requests,max_cost',[(1,100),(100,10)])
def test_persistent_budget_ceiling(tmp_path,max_requests,max_cost):
    path = tmp_path/'c.db'
    t = MockTransport(['bad'])
    c = AsyncSynthesisClient(path,t,dry_run=False,max_requests=max_requests,max_cost_units=max_cost,requests_per_second=10000)
    with pytest.raises(BudgetExceeded): run(c.generate(req(),valid))
    assert len(t.calls)==1
    c.cache.close()
    c2 = AsyncSynthesisClient(path,t,dry_run=False,max_requests=max_requests,max_cost_units=max_cost)
    with pytest.raises(BudgetExceeded): run(c2.generate(req(),valid))
    assert len(t.calls)==1


def test_jsonl_restores_cache_and_budget(tmp_path):
    path = tmp_path/'c.db'
    c = AsyncSynthesisClient(path,MockTransport(),dry_run=False)
    result = run(c.generate(req(),valid))
    c.cache.close()
    path.unlink()
    t = MockTransport()
    c2 = AsyncSynthesisClient(path,t,dry_run=False)
    assert run(c2.generate(req(),valid))==result and not t.calls
    assert c2.cache.db.execute('SELECT COUNT(*) FROM calls').fetchone()[0]==1


def test_concurrent_duplicate_jobs_only_dispatch_once(tmp_path):
    t = MockTransport()
    c = AsyncSynthesisClient(tmp_path/'c.db',t,dry_run=False)
    async def both(): return await asyncio.gather(c.generate(req(),valid),c.generate(req(),valid))
    a,b = run(both())
    assert a==b and len(t.calls)==1


def test_cache_identity_changes_with_model_sampling_verifier(tmp_path):
    a = AsyncSynthesisClient(tmp_path/'c.db')
    b = AsyncSynthesisClient(tmp_path/'c.db',model_id='another')
    c = AsyncSynthesisClient(tmp_path/'c.db',verification_version='v2')
    assert len({x.key(req()) for x in [a,b,c]})==3


def test_verifier_failure_restarts_from_saved_output_without_regeneration(tmp_path):
    path = tmp_path/'c.db'
    t = MockTransport()
    c = AsyncSynthesisClient(path,t,dry_run=False)
    def crashed(_): raise RuntimeError('verification process interrupted')
    with pytest.raises(RuntimeError): run(c.generate(req(),crashed))
    assert len(t.calls)==1
    c.cache.close()
    resumed_transport = MockTransport()
    resumed = AsyncSynthesisClient(path,resumed_transport,dry_run=False)
    result = run(resumed.generate(req(),valid))
    assert result['state']=='accepted' and resumed_transport.calls==[]
    assert len(result['attempts'])==1
