"""Scheduler parity against the actual serial sampler, without a GPU."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from experiments.projection_pilot_v21.pilot import sample
from experiments.projection_pilot_v21.parallel_bench import BatchBackend, choose_rows, schedule


class Fake:
    eos = 999
    def __init__(self):
        self.generated = self.scored = self.input_tokens = self.calls = 0
    def propose(self, row, prefix, limit, seed):
        # Different questions finish at different stages, exercising slot refill.
        if row['early']:
            return prefix + [seed % 5 + 1, self.eos][:limit]
        return prefix + [seed % 5 + 1] * limit
    def score(self, question, ids):
        return sum(ids)/len(ids)
    def decode(self, ids):
        return str(ids)


@pytest.mark.parametrize('concurrency', [1, 2, 4])
def test_batch_scheduler_matches_serial_transitions(concurrency):
    selected = [{'sample_id': f'{i+1:064x}', 'question': str(i), 'early': i % 3 == 0}
                for i in range(8)]
    c = {'seed': 42, 'max_new_tokens': 12, 'block_num': 3, 'eos': 999}
    expected = {r['sample_id']: sample(r, Fake(), c, 2, None) for r in selected}
    backend, actual, batch_sizes = Fake(), {}, []
    def execute(requests):
        batch_sizes.append(len(requests))
        return [backend.propose(*r[1:]) if r[0] == 'propose' else backend.score(*r[1:])
                for r in requests]
    schedule(selected, c, concurrency, execute, lambda r: actual.update({r['sample_id']: r}))
    assert max(batch_sizes) == concurrency
    assert actual.keys() == expected.keys()
    for key, result in actual.items():
        for field in ['token_ids', 'trace', 'finish_reason', 'final_mean_logprob', 'training_approved']:
            assert result[field] == expected[key][field]


def test_selection_is_fixed_and_balanced_without_mutation():
    selected = [{'sample_id': str(i), 'length_bin': i//8} for i in range(32)]
    original = deepcopy(selected)
    assert [r['sample_id'] for r in choose_rows(selected)] == ['0','8','16','24','1','9','17','25']
    assert selected == original


def test_wrong_engine_result_count_cannot_silently_drop_questions():
    with pytest.raises(RuntimeError, match='count mismatch'):
        schedule([{'sample_id': '0'*64}], {'seed':42,'max_new_tokens':12,'block_num':3},
                 1, lambda requests: [], lambda result: None)


def test_mixed_batch_keeps_teacher_out_of_score_and_maps_results():
    backend = BatchBackend.__new__(BatchBackend)
    backend.config = {'max_model_len': 100}
    backend.eos = 999
    backend.generated = backend.scored = backend.input_tokens = backend.calls = 0
    backend.SamplingParams = lambda **kw: kw
    backend.tok = SimpleNamespace(decode=lambda ids, **kw: 'prefix')
    texts = []
    def prompt(text):
        texts.append(text)
        return [10, 11]
    backend.prompt = prompt
    def generate(prompts, params, **kw):
        assert prompts[0]['prompt_token_ids'] == [10, 11, 3, 4]
        assert 'prompt_logprobs' in params[0]
        assert params[1]['seed'] == 123
        return [SimpleNamespace(prompt_logprobs=[None,None,{3:SimpleNamespace(logprob=-2.)},
                  {4:SimpleNamespace(logprob=-4.)}],
                  outputs=[SimpleNamespace(token_ids=[7],finish_reason='length')]),
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=[8],finish_reason='stop')])]
    backend.llm = SimpleNamespace(generate=generate)
    values = backend.execute([('score','question',[3,4]),
        ('propose',{'question':'question','solution':'TEACHER_SECRET'},[3],5,123)])
    assert values == [-3., [3,8,999]]
    assert 'TEACHER_SECRET' not in texts[0]
    assert 'TEACHER_SECRET' in texts[1]
    assert backend.scored == 2
