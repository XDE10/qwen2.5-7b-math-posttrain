"""Meaningful CPU checks for frozen scoring, EOS, and sampler recovery."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from experiments.projection_pilot_v21.pilot import Backend, sample


class Fake:
    eos=999
    def __init__(self):
        self.generated=self.scored=self.input_tokens=self.calls=0
    def score(self,question,ids):
        self.scored+=len(ids);self.calls+=1
        return sum(ids)/len(ids)
    def propose(self,row,prefix,limit,seed):
        self.generated+=limit;self.calls+=1
        return prefix+[1+seed%3]*limit
    def decode(self,ids):return str(ids)


ROW={'sample_id':'01234567'+'a'*56,'question':'secret question','solution':'TEACHER_SECRET'}
C={'seed':42,'max_new_tokens':12,'block_num':3}


def test_accepted_moves_improve_the_actual_scored_mean():
    result=sample(ROW,Fake(),C,2,None)
    assert result['attempts']==6
    assert len(result['token_ids'])==12
    for e in result['trace']:
        assert e['accepted']==(e['candidate_mean_logprob']>e['old_mean_logprob'])
    assert result['training_approved'] is False


def test_recover_after_a_completed_block_matches_uninterrupted_tokens():
    class Interrupted(Exception):pass
    states=[]
    def checkpoint(s):
        states.append(deepcopy(s))
        raise Interrupted()
    with pytest.raises(Interrupted):sample(ROW,Fake(),C,2,None,checkpoint=checkpoint)
    recovered=sample(ROW,Fake(),C,2,None,state=states[-1])
    full=sample(ROW,Fake(),C,2,None)
    assert recovered['token_ids']==full['token_ids']
    assert recovered['trace']==full['trace']
    assert recovered['generation_tokens_spent']==full['generation_tokens_spent']


def test_eos_stops_further_block_generation():
    class Finished(Fake):
        def propose(self,row,prefix,limit,seed):return prefix+[2,self.eos]
    result=sample(ROW,Finished(),C,0,None)
    assert result['finish_reason']=='stop'
    assert len(result['token_ids'])==2


def test_question_only_likelihood_never_receives_expert_solution():
    backend=Backend.__new__(Backend)
    backend.config={'max_model_len':100}
    backend.generated=backend.scored=backend.input_tokens=backend.calls=0
    backend.prompt=lambda text: [10,11] if 'TEACHER_SECRET' not in text else (_ for _ in ()).throw(AssertionError('expert leaked'))
    backend.SamplingParams=lambda **kw:kw
    captured=[]
    class Engine:
        def generate(self,requests,params,use_tqdm=False):
            captured.extend(requests)
            return [SimpleNamespace(prompt_logprobs=[None,{}, {5:SimpleNamespace(logprob=-1.)}, {6:SimpleNamespace(logprob=-3.)}],
                                    outputs=[SimpleNamespace(token_ids=[])])]
    backend.llm=Engine()
    assert backend.score('plain question',[5,6])==-2.
    assert captured==[{'prompt_token_ids':[10,11,5,6]}]


def test_missing_logprobs_abort_instead_of_optimistic_partial_score():
    b=Backend.__new__(Backend);b.config={'max_model_len':100};b.prompt=lambda q:[10]
    b.SamplingParams=lambda **kw:kw
    b.llm=SimpleNamespace(generate=lambda *a,**k:[SimpleNamespace(prompt_logprobs=[None,{}])])
    with pytest.raises(RuntimeError,match='Missing observed-token'):b.score('q',[5])
