from types import SimpleNamespace

from experiments.qfft_v93.common import JsonlAppender, read_jsonl
from experiments.qfft_v93.grading import grade_text
from experiments.qfft_v93.grpo import FinalAnswerReward, RolloutDiagnostics
from experiments.qfft_v93.rollout_diagnostics import compare


class LocalGrader:
    def grade_many(self, rows, strict):
        return [grade_text(text, reference, strict=strict) for text, reference in rows]


def test_capture_stop_reasons_and_compare_without_changing_strict_reward(tmp_path):
    texts = [r'The answer is \boxed{42}<|im_end|>', r'The answer is \boxed{41}',
             '<|im_start|>answer\n' + r'\boxed{42}', '<|im_start|>answer\n42']
    ids = [[1, 9], [2, 3], [4, 5], [6, 9]]
    outputs = [
        SimpleNamespace(token_ids=seq, text=text, finish_reason=finish, stop_reason=stop)
        for seq, text, finish, stop in zip(
            ids, texts, ['stop', 'stop', 'length', 'stop'], [None, '\n\nProblem:', None, None], strict=True
        )
    ]
    llm = SimpleNamespace(generate=lambda: [SimpleNamespace(outputs=outputs)])
    trainer = SimpleNamespace(accelerator=SimpleNamespace(num_processes=1), eos_token_ids=[9],
                              _tokenizer=SimpleNamespace(pad_token_id=0), vllm_generation=SimpleNamespace(llm=llm))
    tokenizer = SimpleNamespace(batch_decode=lambda *args, **kwargs: texts)
    path = tmp_path / 'diagnostics.jsonl'
    with JsonlAppender(tmp_path / 'reward.jsonl') as journal, JsonlAppender(path) as diagnostics:
        reward = FinalAnswerReward(tokenizer, LocalGrader(), journal)
        reward.diagnostics = RolloutDiagnostics(trainer, diagnostics)
        assert llm.generate()[0].outputs is outputs
        assert reward([], [], completion_ids=ids, sample_id=['q1'] * 4, reference_answer=['42'] * 4) == [0., 0., 1., 0.]
    rows = read_jsonl(path)
    assert rows[0]['text'] == texts[0] and rows[0]['reward_text'] == r'The answer is \boxed{42}'
    assert rows[1]['stop_reason'] == '\n\nProblem:' and rows[1]['trl_clipped'] and not rows[1]['truncated']
    assert rows[2]['truncated']
    counts = compare(rows, LocalGrader())
    assert counts['last_box_fallback / non_length / correct'] == 1
    assert counts['last_box_fallback / non_length / wrong'] == 1
    assert counts['strict / non_length / no_answer / missing_box'] == 1


def test_r1_reward_fallback_only_without_final_segment_tag(tmp_path):
    texts = [
        r'\boxed{41} then \boxed{42} then \boxed{43',
        r'\boxed{42} then \boxed{41}',
        r'\boxed{42}' + '\n<|im_start|>answer\n42',
        r'\boxed{42}' + '\n</think>',
        'Final answer: 42',
        '<|im_start|>answer\n' + r'\boxed{42}',
    ]
    tokenizer = SimpleNamespace(batch_decode=lambda *args, **kwargs: texts)
    ids = [[i, 9] for i in range(len(texts))]
    path = tmp_path / 'reward.jsonl'
    with JsonlAppender(path) as journal:
        reward = FinalAnswerReward(tokenizer, LocalGrader(), journal, last_box_fallback=True)
        assert reward([], [], completion_ids=ids, reference_answer=['42'] * len(texts)) == [1., 0., 0., 0., 0., 1.]
    assert read_jsonl(path)[0]['counts'] == {'correct': 2, 'wrong': 1, 'no_answer': 3}
