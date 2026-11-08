from src.data.screen_openr1_sft import trajectory_candidates, answer_is_complete, sealed_math


class Tokenizer:
    def encode(self,text,**kwargs): return list(range(int(text.split('COUNT=')[1].split()[0])))
    def apply_chat_template(self,messages,**kwargs): return [0]*1000


def row(counts=(500,1600),correct=(True,True),complete=(True,True)):
    return {'problem':'Question','generations':[f'<think>COUNT={n} work</think>\\boxed{{2}}' for n in counts],
            'is_reasoning_complete':list(complete),'correctness_math_verify':list(correct),'finish_reasons':None}


def test_per_trajectory_not_whole_row_correctness():
    r=row(correct=(False,True));r['correctness_count']=1
    accepted,logs=trajectory_candidates(r,Tokenizer())
    assert [x['trajectory_index'] for x in accepted]==[1]
    assert logs[0]['reason']=='not_math_verify_true'


def test_length_boundaries_and_full_content_not_truncated():
    accepted,_=trajectory_candidates(row((499,500,1600,1601),(True,)*4,(True,)*4),Tokenizer())
    assert [r['response_token_count'] for r in accepted]==[500,1600]
    assert all('\\boxed{2}' in r['solution'] for r in accepted)


def test_malformed_flags_finish_and_unclosed_structure():
    r=row();r['is_reasoning_complete']=[True]
    assert trajectory_candidates(r,Tokenizer())[1][0]['reason']=='unaligned_or_missing_flags'
    r=row();r['finish_reasons']=['stop']
    assert trajectory_candidates(r,Tokenizer())[1][0]['reason']=='unaligned_finish_reasons'
    r=row();r['finish_reasons']=['length','stop']
    assert [v['trajectory_index'] for v in trajectory_candidates(r,Tokenizer())[0]]==[1]
    assert not answer_is_complete('<think>unfinished \\boxed{2}')
    assert not answer_is_complete('<think>x</think>\\boxed{2')


def test_sft_total_budget_and_shared_dev_identity():
    class LongPrompt(Tokenizer):
        def apply_chat_template(self,messages,**kwargs): return [0]*4097
    assert not trajectory_candidates(row(),LongPrompt())[0]
    m,protected=sealed_math()
    assert len(protected['math_dev'])==256 and len(protected['math_train'])==7184
