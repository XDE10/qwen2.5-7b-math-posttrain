"""CPU checks for actual Qwen ChatML, score alignment, batched transitions and stats."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time
import pytest
from experiments.projection_pilot_v24.pilot import (
    ROOT, Backend, generation_user, proposal_ids, scoring_prefix, trajectory,
    run_jobs, summarize, CostLedger,
)


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)


ROW = {'sample_id':'a'*64,'question':'Solve x+1=2.', 'solution':'TEACHER_SECRET: x=1.', 'length_bin':'1-4096', 'solution_tokens':2000}


def test_empty_and_nonempty_real_chat_template_have_one_continuation(tok):
    prefix = tok.encode('UNIQUE_ASSISTANT_PREFIX: Let x be the unknown.',add_special_tokens=False)
    empty = proposal_ids(tok,ROW,[])
    full = proposal_ids(tok,ROW,prefix)
    assert full == empty+prefix
    text = tok.decode(full,skip_special_tokens=False)
    assert text.count('UNIQUE_ASSISTANT_PREFIX') == 1
    assert text.count('TEACHER_SECRET') == 1
    assert tok.decode(empty,skip_special_tokens=False).endswith('<|im_start|>assistant\n')
    user = generation_user(ROW)
    assert user.startswith('<QUERY>\n')
    assert user.index('<TRAJECTORY>') < user.index('Write a standalone')


def test_expert_user_prefix_is_identical_across_cuts(tok):
    base = proposal_ids(tok,ROW,[])
    for text in ['first prefix','second prefix with different length','']:
        prefix = tok.encode(text,add_special_tokens=False)
        assert proposal_ids(tok,ROW,prefix)[:len(base)] == base
    assert 'TEACHER_SECRET' not in tok.decode(scoring_prefix(tok,ROW['question']),skip_special_tokens=False)


def test_observed_score_uses_exact_assistant_offset_in_mixed_batch(tok):
    b = Backend.__new__(Backend)
    b.config = {'max_model_len':32768,'temperature':0.6,'repetition_penalty':1.1}
    b.tok, b.eos, b.fixed = tok, tok.eos_token_id, {}
    b.generated=b.scored=b.input_tokens=b.calls=0
    b.SamplingParams = lambda **kw:kw
    ids = tok.encode('x=1',add_special_tokens=False)
    n = len(scoring_prefix(tok,ROW['question']))
    captured = []
    def generate(prompts,params,**kw):
        captured.extend(prompts)
        assert prompts[0]['prompt_token_ids'] == scoring_prefix(tok,ROW['question'])+ids
        assert prompts[1]['prompt_token_ids'] == proposal_ids(tok,ROW,ids[:1])
        assert params[0]['prompt_logprobs'] == 0
        assert params[1]['seed'] == 123
        return [SimpleNamespace(prompt_logprobs=[None]*n+[{t:SimpleNamespace(logprob=-2.)} for t in ids],
                    outputs=[SimpleNamespace(token_ids=[7],finish_reason='length')]),
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=[8],finish_reason='stop')])]
    b.llm = SimpleNamespace(generate=generate)
    result = b.execute([('score',ROW['question'],ids),('propose',ROW,ids[:1],16,123)])
    assert result == [-2.,{'ids':ids[:1]+[8], 'stopped':True}]
    assert b.scored == len(ids)
    assert 'TEACHER_SECRET' not in tok.decode(captured[0]['prompt_token_ids'],skip_special_tokens=False)


def drive(gen, stop_checkpoint=None):
    try:
        req = next(gen)
        while True:
            if req[0]=='propose':
                _,row,prefix,limit,seed = req
                value = {'ids':prefix+[1+seed%7]*limit, 'stopped':False}
            elif req[0]=='score':
                value = -sum(req[2])/len(req[2])
            else:
                if stop_checkpoint is not None:
                    stop_checkpoint.append(deepcopy(req[1]))
                    return None
                value = None
            req = gen.send(value)
    except StopIteration as stop:
        return stop.value


def test_resume_preserves_rng_acceptance_and_cumulative_gain():
    c = {'seed':42,'max_new_tokens':12,'block_tokens':4,'block_num':3,'eos':999}
    full = drive(trajectory(ROW,c,4))
    saved = []
    drive(trajectory(ROW,c,4),saved)
    resumed = drive(trajectory(ROW,c,4,saved[0]))
    assert resumed == full
    assert full['blocks_used'] == 3
    assert len(full['trace']) == 12
    assert full['within_run_mcmc_gain'] == pytest.approx(sum(
        b['after_updates_mean_logprob']-b['before_updates_mean_logprob'] for b in full['blocks']))


def test_eos_still_gets_updates_then_stops_after_first_block():
    c = {'seed':42,'max_new_tokens':12,'block_tokens':4,'block_num':3,'eos':999}
    gen = trajectory(ROW,c,0)
    assert next(gen)[0] == 'propose'
    assert gen.send({'ids':[5], 'stopped':True})[0] == 'score'
    assert gen.send(-2.)[0] == 'checkpoint'
    with pytest.raises(StopIteration) as stop:
        gen.send(None)
    assert stop.value.value['blocks_used']==1
    assert stop.value.value['finish_reason']=='stop'


def test_paired_summary_uses_matched_ids_and_separates_local_gain():
    records=[]
    for name,u,score,gain,blocks in [('a',0,-2.,0.,1),('b',0,-4.,0.,2),
                                    ('a',4,-1.,.2,2),('b',4,-5.,.6,4),('c',4,-1.,.4,3)]:
        records.append({'sample_id':name,'method':f'projection_{u}','trace':[],
            'grade_category':'equivalent','finish_reason':'stop','token_ids':[1,2],
            'blocks_used':blocks,'within_run_mcmc_gain':gain,'final_mean_logprob':score,
            'original_mean_logprob':-3., 'length_bin':'1-4096', 'source_solution_tokens':2000})
    s=summarize(records)
    assert s['paired_count']==2
    assert s['mean_logprob_diff_proj4_minus_proj0']==0.
    assert s['proj4_gt_proj0_win_rate']==.5
    assert s['methods']['projection_4']['mean_within_run_mcmc_gain']==pytest.approx(.4)
    assert s['methods']['projection_4']['mean_blocks_used']==3


@pytest.mark.parametrize('concurrency',[1,4])
def test_scheduler_matches_individual_trajectories_and_persists_blocks(tmp_path,concurrency):
    selected=[dict(ROW,sample_id=f'{i+1:064x}') for i in range(5)]
    c={'seed':42,'max_new_tokens':8,'block_tokens':4,'block_num':2,'eos':999,'concurrency':concurrency}
    class Fake:
        def execute(self,requests):
            return [{'ids':r[2]+[1+r[4]%7]*r[3], 'stopped':False} if r[0]=='propose' else -sum(r[2])/len(r[2]) for r in requests]
    actual={}
    assert run_jobs(selected,c,4,Fake(),tmp_path,lambda r:actual.update({r['sample_id']:r}),time.monotonic()+10)
    for row in selected:
        assert actual[row['sample_id']]==drive(trajectory(row,c,4))
        assert (tmp_path/f"state_{row['sample_id']}_4.json").exists()


def test_accepted_short_candidate_does_not_shrink_next_candidate_ceiling():
    c={'seed':42,'max_new_tokens':512,'block_tokens':256,'block_num':2,'eos':999}
    gen=trajectory(ROW,c,2)
    assert next(gen)[3]==256
    gen.send({'ids':[1]*180,'stopped':True})
    first=gen.send(-4.)
    assert len(first[2])+first[3]==256
    gen.send({'ids':[2]*100,'stopped':True})
    second=gen.send(-3.)
    assert len(second[2])+second[3]==256  # not the accepted length of 100
    gen.send({'ids':[3]*240,'stopped':True})
    checkpoint=gen.send(-2.)
    assert len(checkpoint[1]['ids'])==240  # regrown beyond initial 180
    assert checkpoint[1]['blocks'][0]['block_ceiling']==256
    with pytest.raises(StopIteration) as done:
        gen.send(None)
    assert done.value.value['blocks_used']==1


def test_fixed_ceiling_clamps_at_total_budget_on_later_block():
    c={'seed':42,'max_new_tokens':6,'block_tokens':4,'block_num':2,'eos':999}
    gen=trajectory(ROW,c,0)
    assert next(gen)[3]==4
    gen.send({'ids':[1]*4,'stopped':False})
    gen.send(-1.)
    assert gen.send(None)[3]==2
    gen.send({'ids':[1]*6,'stopped':False})
    checkpoint=gen.send(-1.)
    assert checkpoint[1]['blocks'][-1]['block_ceiling']==6


def test_cost_flush_is_idempotent_and_resume_accumulates_without_original_leak(tmp_path):
    backend=SimpleNamespace(generated=0,input_tokens=0,scored=0,calls=0)
    clock=[0.]
    path=tmp_path/'costs.json'
    ledger=CostLedger(path,backend,clock=lambda:clock[0])
    ledger.begin('original_scoring')
    backend.scored=100; backend.input_tokens=120; clock[0]=2.
    ledger.end()
    ledger.begin('projection_0')
    backend.generated=10; backend.input_tokens=320; backend.scored=150; clock[0]=5.
    ledger.flush(); ledger.flush()
    assert ledger.totals['projection_0']['generation_tokens_spent']==10
    assert ledger.totals['projection_0']['scoring_tokens_spent']==50
    assert ledger.totals['projection_0']['wall_clock_seconds']==3.
    ledger.end()
    fresh=SimpleNamespace(generated=0,input_tokens=0,scored=0,calls=0)
    resumed=CostLedger(path,fresh,clock=lambda:clock[0])
    resumed.begin('projection_0')
    fresh.generated=5; fresh.input_tokens=80; clock[0]=7.
    resumed.end()
    assert resumed.totals['projection_0']['generation_tokens_spent']==15
    assert resumed.totals['projection_0']['wall_clock_seconds']==5.
    assert resumed.totals['original_scoring']['scoring_tokens_spent']==100


def test_length_buckets_pair_by_id_and_show_correct_cost_ratio():
    def record(key,u,bucket,length,score):
        return {'sample_id':key,'method':f'projection_{u}', 'trace':[],
            'grade_category':'equivalent','finish_reason':'stop','token_ids':[1]*length,
            'blocks_used':2,'within_run_mcmc_gain':0.,'final_mean_logprob':score,
            'original_mean_logprob':-3.,'length_bin':bucket}
    records=[record('short',0,'1-4096',10,-2.),record('short',4,'1-4096',20,-1.),
             record('long',0,'12289-16384',40,-1.),record('long',4,'12289-16384',30,-2.)]
    summary=summarize(records,{'projection_0':{'wall_clock_seconds':10.,'generation_tokens_spent':100},
                               'projection_4':{'wall_clock_seconds':25.,'generation_tokens_spent':400}})
    assert summary['cost_ratio_proj4_over_proj0']['wall_clock_seconds']==2.5
    assert summary['cost_ratio_proj4_over_proj0']['generation_tokens_spent']==4.
    buckets=summary['by_original_length_bucket']
    assert buckets['1-4096']['mean_logprob_diff_proj4_minus_proj0']==1.
    assert buckets['12289-16384']['mean_logprob_diff_proj4_minus_proj0']==-1.
    assert buckets['12289-16384']['methods']['projection_4']['mean_output_tokens']==30
