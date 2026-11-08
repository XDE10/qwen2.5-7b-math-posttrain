"""Focused CPU checks: do not silently drop R1 reasoning or supervise prompts."""
from pathlib import Path
import pytest
from experiments.aimo_sft_v25.common import ROOT,config,encode,prompt_ids,verify_data
from experiments.aimo_sft_v25.train import Collator
from experiments.aimo_sft_v25.evaluate import summary
from experiments.aimo_sft_v25.telemetry import OnlineLog


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/DeepSeek-R1-Distill-Qwen-7B',local_files_only=True)


def test_actual_r1_template_drops_thought_but_training_encoder_preserves_it(tok):
    solution='<think>UNIQUE_REASONING: subtract one on both sides.</think>\nAnswer: \\boxed{1}.'
    bad=tok.apply_chat_template([{'role':'user','content':'x+1=2'},
        {'role':'assistant','content':solution}],tokenize=False,add_generation_prompt=False)
    assert 'UNIQUE_REASONING' not in bad
    good=encode(tok,'x+1=2',solution,8192)
    assert tok.decode(good['input_ids'],skip_special_tokens=False).count('<think>')==1
    supervised=tok.decode([t for t in good['labels'] if t!=-100],skip_special_tokens=False)
    assert 'UNIQUE_REASONING' in supervised and '</think>' in supervised and '\\boxed{1}' in supervised
    assert all(t==-100 for t in good['labels'][:good['prompt_tokens']])
    assert good['input_ids'][:good['prompt_tokens']]==prompt_ids(tok,'x+1=2')


def test_overlength_fails_instead_of_truncation_and_padding_has_no_loss(tok):
    short=encode(tok,'q','<think>A.</think>\\boxed{1}',8192)
    long=encode(tok,'a longer question','<think>A longer derivation.</think>\\boxed{1}',8192)
    with pytest.raises(OverflowError):encode(tok,'q','<think>A.</think>\\boxed{1}',3)
    batch=Collator(tok.eos_token_id)([short,long])
    assert (batch['labels'][0,len(short['input_ids']):]==-100).all()
    assert batch['labels'].shape==batch['input_ids'].shape


def test_frozen_training_selection_can_be_encoded_with_matching_lengths(tok):
    data,manifest=verify_data(config())
    assert len(data)==1000
    for row in [data[0],data[-1],max(data,key=lambda r:r['total_tokens'])]:
        encoded=encode(tok,row['question'],row['solution'],8192)
        assert encoded['length']==row['total_tokens']
    assert manifest['tokens']['max']<=8192


def test_pass_at_1_counts_each_draw_not_any_success():
    records=[{'correct':True,'grade_category':'equivalent','finish_reason':'stop','output_tokens':20},
             {'correct':False,'grade_category':'mismatch','finish_reason':'stop','output_tokens':10}]
    result=summary(records,64,64)
    assert result['pass_at_1']==.5
    assert result['complete'] is False


def test_online_logs_resume_monotone_and_keep_optimizer_steps(tmp_path):
    from types import SimpleNamespace
    captured=[]
    module=SimpleNamespace(log=lambda values,step:captured.append((values,step)),finish=lambda:None)
    logger=OnlineLog(module,tmp_path,'test-run')
    logger.log({'train/loss':1.2},20)
    logger.log({'train/loss':1.1},21)
    logger.finish()
    resumed=OnlineLog(module,tmp_path,'test-run')
    resumed.log({'train/loss':1.15},20)
    assert [item[1] for item in captured]==[0,1,2]
    assert captured[-1][0]['train/global_step']==20


def test_online_init_requires_cloud_and_does_not_log_credentials(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from experiments.aimo_sft_v25.telemetry import start_online
    calls={}
    def login(**kwargs):calls['login']=kwargs
    def init(**kwargs):calls['init']=kwargs;return SimpleNamespace(id='cloud-test-id')
    module=SimpleNamespace(login=login,init=init,log=lambda *a,**kw:None,finish=lambda:None)
    monkeypatch.setitem(sys.modules,'swanlab',module)
    monkeypatch.setenv('SWANLAB_API_KEY','FAKE_TEST_CREDENTIAL')
    start_online(tmp_path,{'recipe':{'learning_rate':2e-5}})
    assert calls['init']['mode']=='cloud'
    assert 'FAKE_TEST_CREDENTIAL' not in str(calls['init'])
    assert 'FAKE_TEST_CREDENTIAL' not in (tmp_path/'swanlab_run.json').read_text()


def test_current_swanlab_api_uses_online_name_and_log_dir(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from experiments.aimo_sft_v25.telemetry import start_online
    calls={}
    def init(*,name=None,log_dir=None,mode=None,**kwargs):
        calls.update(name=name,log_dir=log_dir,mode=mode,**kwargs)
        return SimpleNamespace(id='new-api-id',url='https://example.invalid/test-run')
    module=SimpleNamespace(login=lambda **kwargs:True,init=init,log=lambda *a,**kw:None,finish=lambda:None)
    monkeypatch.setitem(sys.modules,'swanlab',module)
    monkeypatch.setenv('SWANLAB_API_KEY','FAKE_TEST_CREDENTIAL')
    start_online(tmp_path,{'test':True})
    assert calls['mode']=='online'
    assert calls['name']=='R1_7B_aimo_original_v25'
    assert 'experiment_name' not in calls


def test_failed_login_never_silently_starts_offline(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from experiments.aimo_sft_v25.telemetry import start_online
    module=SimpleNamespace(login=lambda **kwargs:False,init=lambda **kwargs:pytest.fail('must not initialize'))
    monkeypatch.setitem(sys.modules,'swanlab',module)
    monkeypatch.setenv('SWANLAB_API_KEY','FAKE_TEST_CREDENTIAL')
    with pytest.raises(RuntimeError,match='online login failed'):
        start_online(tmp_path,{})
