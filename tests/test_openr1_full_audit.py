import json
from pathlib import Path
import pytest
from scripts.analyze_openr1_candidates import quantiles, ratio_group, collect_questions, choose_for_budget, inspect_row
from src.data.openr1_quality import quality_signals
from src.data.prepare_math_data import write_jsonl


def test_quantiles_and_ratio_do_not_invent_missing_difficulty():
    assert quantiles([100,200,300,400]) == {'n':4,'p50':250.0,'p75':325.0,'p90':370.0}
    assert quantiles([])['p50'] is None
    assert [ratio_group(k,4) for k in range(5)] == ['0','(0,0.5]','(0,0.5]','(0.5,1)','1']
    assert ratio_group(0,0) == 'unknown'


def test_noise_filter_rejects_loops_but_not_bilingual_or_normal_latex():
    assert quality_signals('Let x=2. 因此答案是 '+r'\boxed{2}.')['automatic_pass']
    assert quality_signals('hello\ufffd')['reject']
    repeated = 'let us try this same calculation once again '
    assert 'terminal_periodic_word_loop' in quality_signals(repeated*8)['reject']
    assert 'symbol_run_128' in quality_signals('! '*3+'@'*128)['reject']
    assert not quality_signals(('Wait, let me reconsider. '*11)+r'\boxed{2}')['automatic_pass']


def test_duplicate_rows_do_not_inflate_verification_ratio_and_flags_conflict_quarantined(tmp_path):
    base = {'question_hash':'q','problem':'p','uuid':'a','source':'s','problem_type':'Geometry',
        'aligned':True,'source_path':'f','source_row':0,'source_correctness_count_disagrees':False,
        'trajectories':[{'text_sha256':'h','trajectory_index':0,'verified':True,'complete':True,
            'answer_tokens':1000,'total_tokens':1100,'supervised_tokens':1002,
            'quality':{'automatic_pass':True}}]}
    f = tmp_path/'cache.jsonl'
    write_jsonl(f,[base,base])
    questions, counts = collect_questions([f])
    assert len(questions)==1 and questions[0]['verification_ratio']==1
    assert questions[0]['unique_trajectory_count']==1 and counts['raw_trajectories']==2
    changed = json.loads(json.dumps(base)); changed['trajectories'][0]['verified']=False
    write_jsonl(f,[base,changed])
    q = collect_questions([f])[0][0]
    assert q['flag_conflict'] and choose_for_budget(q,1600) is None


def test_budget_applies_to_whole_template_and_can_skip_too_short_trace():
    q = {'audit_usable':True,'trajectories':[
        {'verified':True,'complete':True,'answer_tokens':400,'total_tokens':500,'text_sha256':'a','quality':{'automatic_pass':True}},
        {'verified':True,'complete':True,'answer_tokens':1500,'total_tokens':3900,'text_sha256':'b','quality':{'automatic_pass':True}},
        {'verified':True,'complete':True,'answer_tokens':1550,'total_tokens':1700,'text_sha256':'c','quality':{'automatic_pass':True}}]}
    assert choose_for_budget(q,1600)['text_sha256']=='c'
    assert choose_for_budget(q,1500) is None


def test_real_tokenizer_uses_indexed_flags_not_whole_question_count():
    from transformers import AutoTokenizer
    from src.data.screen_openr1_sft import PROMPT
    tok=AutoTokenizer.from_pretrained('data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    row={'uuid':'u','problem':'What is 1+1?','source':'test','problem_type':'Algebra',
         'generations':[r'Incorrect \boxed{7}',r'One plus one is two. \boxed{2}'],
         'is_reasoning_complete':[True,True],'correctness_math_verify':[False,True],
         'correctness_count':0,'finish_reasons':['stop','stop']}
    result=inspect_row(row,'p',0,tok)
    assert result['source_correctness_count_disagrees']
    assert 'answer_tokens' not in result['trajectories'][0]
    t=result['trajectories'][1]
    assert t['answer_tokens']==len(tok.encode(row['generations'][1],add_special_tokens=False))
    messages=[{'role':'system','content':PROMPT},{'role':'user','content':row['problem']}]
    full=tok.apply_chat_template(messages+[{'role':'assistant','content':row['generations'][1]}],tokenize=True)
    prefix=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    assert t['total_tokens']==len(full) and t['supervised_tokens']==len(full)-len(prefix)
    row['is_reasoning_complete']=[True]
    assert not inspect_row(row,'p',0,tok)['aligned']
