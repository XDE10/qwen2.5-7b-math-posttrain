from scripts.evaluate_project1_dev_v9 import hard_ids, messages, summary


def test_hard_probe_requires_four_valid_wrong_outputs():
    records = [{'sample_id': 'hard', 'trial': t, 'category': 'mismatch', 'finish_reason': 'stop'} for t in range(4)]
    assert hard_ids(records) == ['hard']
    assert hard_ids(records[:3]) == []
    records[0]['category'] = 'parse_failure'
    assert hard_ids(records) == []
    records[0]['category'] = 'mismatch'
    records[0]['finish_reason'] = 'length'
    assert hard_ids(records) == []


def test_prompt_has_no_private_answer_or_solution():
    assert [r['content'] for r in messages({'problem': 'question', 'reference_answer': 'private', 'solution': 'secret'})][1:] == ['question']


def test_metrics_keep_unscorable_separate():
    rows = [{'category': c, 'generated_tokens': 10, 'finish_reason': 'stop'}
            for c in ['equivalent', 'mismatch', 'unsupported_relation', 'parse_failure']]
    actual = summary(rows)
    assert actual['correct_over_all'] == 0.25
    assert actual['correct_over_scorable'] == 0.5
    assert actual['scorable_count'] == 2
    assert actual['status_counts']['parse_failure'] == 1
