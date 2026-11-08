"""CPU checks for actual Qwen ChatML, score alignment, batched transitions and stats."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time
import pytest
from experiments.projection_pilot_v23.pilot import (
    ROOT, Backend, generation_user, proposal_ids, scoring_prefix, trajectory,
    run_jobs, summarize,
)


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)


ROW = {'sample_id':'a'*64,'question':'Solve x+1=2.', 'solution':'TEACHER_SECRET: x=1.'}


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
    assert result == [-2.,ids[:1]+[8,tok.eos_token_id]]
    assert b.scored == len(ids)
    assert 'TEACHER_SECRET' not in tok.decode(captured[0]['prompt_token_ids'],skip_special_tokens=False)


def drive(gen, stop_checkpoint=None):
    try:
        req = next(gen)
        while True:
            if req[0]=='propose':
                _,row,prefix,limit,seed = req
                value = prefix+[1+seed%7]*limit
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
    assert gen.send([5,999])[0] == 'score'
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
            'original_mean_logprob':-3.})
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
            return [r[2]+[1+r[4]%7]*r[3] if r[0]=='propose' else -sum(r[2])/len(r[2]) for r in requests]
    actual={}
    assert run_jobs(selected,c,4,Fake(),tmp_path,lambda r:actual.update({r['sample_id']:r}),time.monotonic()+10)
    for row in selected:
        assert actual[row['sample_id']]==drive(trajectory(row,c,4))
        assert (tmp_path/f"state_{row['sample_id']}_4.json").exists()
