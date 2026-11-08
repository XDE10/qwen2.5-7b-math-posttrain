import copy
import hashlib
import json
from pathlib import Path
import pytest
from src.data.approved_training_inputs_v2 import load_approved, assert_pairs
from src.training.guards_v2 import approved_config, validate_checkpoint, validate_merged, rl_lengths, expand_config
from src.training.math_reward_v2 import AuditedMathReward, category
from src.data.decontamination_v2 import Index, grams, jaccard


def release(tmp_path):
    files = {}
    row = dict(sample_id='a', pair_id='p', problem='Find 2+3', reference_answer='5', split='train', problem_type='Algebra', training_approved=True, solution='5')
    for arm in ('raw', 'compact'):
        path = tmp_path / f'project1_{arm}_paired.jsonl'
        path.write_text(json.dumps(dict(row, arm=arm))+'\n', encoding='utf-8')
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = tmp_path / 'protected.json'
    path.write_text(json.dumps({'sample_ids':['dev']}), encoding='utf-8')
    files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files':files, 'approved_training_files':list(files)[:2], 'protected_dev_ids_file':path.name}), encoding='utf-8')
    return manifest, row


def test_v2_actual_release_and_pairing():
    for arm in ('raw','compact'):
        assert len(load_approved('data/portable_release/manifest.json',f'project1_{arm}_paired.jsonl')) == 1500
    with pytest.raises(ValueError):
        load_approved('data/portable_release/manifest.json','project1_rl_priority_not_rollout_selected.jsonl')


@pytest.mark.parametrize('field,value', [('split','dev'),('sample_id','dev'),('pair_id','wrong'),('problem','changed')])
def test_training_guard_rejects_even_with_fixture_digest(tmp_path, field, value):
    # Synthetic fixtures only; no real release hashes are rewritten.
    manifest, _ = release(tmp_path)
    m = json.loads(manifest.read_text())
    path = tmp_path / 'project1_raw_paired.jsonl'
    row = json.loads(path.read_text());row[field]=value
    path.write_text(json.dumps(row)+'\n')
    m['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError):load_approved(manifest,path.name)


def test_pairs_duplicate_rejected():
    with pytest.raises(ValueError, match='Duplicate pair_id'):
        assert_pairs([{'pair_id':'p'},{'pair_id':'p'}],[])


def test_protected_file_tamper(tmp_path):
    manifest, _ = release(tmp_path)
    (tmp_path/'protected.json').write_text('{"sample_ids":[]}')
    with pytest.raises(ValueError,match='digest'):load_approved(manifest,'project1_raw_paired.jsonl')


@pytest.mark.parametrize('missing',['optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','run_identity.json'])
def test_checkpoint_missing_state(tmp_path,missing):
    for name in ['optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','run_identity.json']:
        (tmp_path/name).write_text(json.dumps({'global_step':1}) if name=='trainer_state.json' else '{}')
    (tmp_path/missing).unlink()
    with pytest.raises(ValueError,match='Incomplete'):validate_checkpoint(tmp_path,{})


def test_checkpoint_identity_changed(tmp_path):
    for name in ['optimizer.pt','scheduler.pt','rng_state.pth']:(tmp_path/name).write_bytes(b'state')
    (tmp_path/'trainer_state.json').write_text('{"global_step":1}')
    (tmp_path/'run_identity.json').write_text('{"versions":{"trl":"old"}}')
    with pytest.raises(ValueError,match='identity mismatch'):validate_checkpoint(tmp_path,{})


def test_rl_total_budget_and_draft_guard(tmp_path):
    assert rl_lengths({'max_prompt_length':1752,'max_new_tokens':2048,'max_length':3800})==(1752,2048)
    with pytest.raises(ValueError):rl_lengths({'max_prompt_length':512,'max_new_tokens':2048,'max_length':3800})
    path=tmp_path/'draft.json';path.write_text('{"draft":true}')
    with pytest.raises(ValueError,match='Draft'):approved_config(path)


@pytest.mark.parametrize('status,expected',[
    ('equivalent','equivalent'),('mismatch','mismatch'),('relational_answer_requires_task_aware_manual_review','unsupported_relation'),
    ('multiple_distinct_boxes_require_manual_collection_review','multiple_boxes'),('unclosed_box','unclosed_box'),
    ('unparsed_reference','parse_failure'),('missing_box','parse_failure'),('timeout','timeout'),('unexpected','verifier_error')])
def test_status_categories(status,expected):assert category({'status':status})==expected


@pytest.mark.parametrize('status',['timeout','verifier_error'])
def test_verifier_error_logs_then_aborts(tmp_path,status):
    reward=AuditedMathReward(tmp_path,checker=lambda *args:{'status':status})
    with pytest.raises(RuntimeError):reward(['x'],['1'])
    record=json.loads((tmp_path/'reward_status.jsonl').read_text())
    assert record['counts'][status]==1 and record['counts']['mismatch']==0


def test_reward_values_and_log(tmp_path):
    reward=AuditedMathReward(tmp_path,checker=lambda x,y:{'status':'equivalent' if x==y else 'mismatch'})
    assert reward(['1','2'],['1','1'])==[1.,0.]


@pytest.mark.parametrize('variant',[
    'Find the sum of the first 100 positive integers.',
    'Solve this mathematical problem. Find the sum of the first 100 positive integers. Explain your answer.',
    'Find the sum of the first 101 positive integers.'])
def test_decontamination_original_rewrite_numeric_family(variant):
    rows=[{'problem':'Find the sum of the first 100 positive integers.'}]
    assert Index(rows).query(variant,.4)


def test_prefix_index_equals_exhaustive():
    rows=[{'problem':x} for x in ['Compute 12+13.','Compute 12+14.','What is the area of a circle?','Compute twelve plus thirteen.','x']]
    for threshold in [.3,.5,.8,1.]:
        for row in rows:
            expected=[(i,jaccard(grams(row['problem']),grams(other['problem']))) for i,other in enumerate(rows) if jaccard(grams(row['problem']),grams(other['problem']))>=threshold]
            assert Index(rows).query(row['problem'],threshold)==expected


def test_merged_wrong_arm_and_weight_change(tmp_path):
    origin={'schema':'project1-run-v2','config':{'stage':'sft','arm':'raw','output_dir':'runs/sft'},'data_sha256':'data'}
    (tmp_path/'merged_sft_identity.json').write_text(json.dumps(origin))
    weight=tmp_path/'model.safetensors';weight.write_bytes(b'fixture weights')
    (tmp_path/'weight_manifest.json').write_text(json.dumps({'model.safetensors':hashlib.sha256(weight.read_bytes()).hexdigest()}))
    config={'model':str(tmp_path),'arm':'compact','expected_sft':{'arm':'compact','output_dir':'runs/sft','data_sha256':'data','identity_sha256':hashlib.sha256((tmp_path/'merged_sft_identity.json').read_bytes()).hexdigest(),'weight_manifest_sha256':hashlib.sha256((tmp_path/'weight_manifest.json').read_bytes()).hexdigest()}}
    with pytest.raises(ValueError,match='arm'):validate_merged(config)
    config['expected_sft']['arm']='raw'
    config['arm']='raw'
    assert validate_merged(config)
    weight.write_bytes(b'changed')
    with pytest.raises(ValueError,match='digest'):validate_merged(config)
    (tmp_path/'weight_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest'):validate_merged(config)


def test_reference_tiny():
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    from scripts.verify_sft_reference_v2 import verify
    torch.manual_seed(42)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=24,n_positions=32,n_embd=16,n_layer=1,n_head=2))
    assert verify(model,torch.tensor([[1,2,3,4,5,6]]),['c_attn'])['frozen_base_unchanged']


def test_drive_config_variables(monkeypatch):
    monkeypatch.setenv('PROJECT_DRIVE_ROOT','/content/drive/MyDrive/math7b')
    assert expand_config({'output':['${PROJECT_DRIVE_ROOT}/runs']})['output'][0]=='/content/drive/MyDrive/math7b/runs'
    monkeypatch.delenv('PROJECT_DRIVE_ROOT')
    with pytest.raises(ValueError):expand_config('${PROJECT_DRIVE_ROOT}/runs')


def test_bridge_budget_fields():
    from src.training.trl_bridge_v2 import ProjectGRPOConfig
    c=ProjectGRPOConfig(use_cpu=True,bf16=False,max_prompt_length=1752,max_length=3800,max_completion_length=2048)
    assert c.max_prompt_length+c.max_completion_length==c.max_length
    with pytest.raises(ValueError):ProjectGRPOConfig(use_cpu=True,bf16=False,max_prompt_length=1,max_length=3800,max_completion_length=2048)


def test_identity_detects_template_and_grader_changes(tmp_path,monkeypatch):
    # All modified files here are isolated synthetic fixtures.
    import src.training.guards_v2 as g
    manifest,_=release(tmp_path)
    tok=tmp_path/'tokenizer';tok.mkdir()
    tc=tok/'tokenizer_config.json';tc.write_text('{"chat_template":"template-one"}')
    cfg={'stage':'sft','vision':False,'arm':'raw','data_file':'project1_raw_paired.jsonl','data_manifest':str(manifest),'model_revision':'fixed','tokenizer_path':str(tok)}
    cp=tmp_path/'config.json';cp.write_text(json.dumps(cfg))
    _,_,one=g.approved_config(cp)
    tc.write_text('{"chat_template":"template-two"}')
    _,_,two=g.approved_config(cp)
    assert one['template_sha256']!=two['template_sha256'] and one['tokenizer_sha256']!=two['tokenizer_sha256']
    assert one['grader_sha256'] and one['bundled_grader_sha256']


@pytest.mark.parametrize('arm,limit,maximum',[('raw',4096,3876),('compact',3800,1443)])
def test_original_encode_full_release(arm,limit,maximum):
    from transformers import AutoTokenizer
    from src.training.approved_hf import encode_sft
    tok=AutoTokenizer.from_pretrained('data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    encoded=encode_sft(load_approved('data/portable_release/manifest.json',f'project1_{arm}_paired.jsonl'),tok,limit)
    assert len(encoded)==1500 and max(len(x['input_ids']) for x in encoded)==maximum
    assert all(tok.eos_token_id in x['labels'] for x in encoded)


def test_formal_drafts_and_budget_arithmetic():
    folder=Path('reports/handover_20261024_v2/formal_v1')
    for path in folder.rglob('*.draft.json'):
        value=json.loads(path.read_text(encoding='utf-8'))
        entries=value if isinstance(value,list) else [value]
        assert all(v['draft'] is True for v in entries)
        assert 'D:/' not in path.read_text(encoding='utf-8') and 'D:\\' not in path.read_text(encoding='utf-8')
    totals=json.loads((folder/'budget_totals.json').read_text(encoding='utf-8'))
    token_plan=totals['planning']['equal_supervised_tokens']
    for arm in ('raw','compact'):
        plan=token_plan['plans'][arm]
        assert plan['supervised_tokens']-plan['overshoot']==token_plan['target_supervised_tokens']
        assert plan['exposures']/1500==plan['effective_epochs']


def test_check_math_process_failures(monkeypatch):
    import subprocess
    import src.training.math_reward_v2 as reward
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('worker',20)
    monkeypatch.setattr(reward.subprocess,'run',timeout)
    assert reward.check_math('x','1')['status']=='timeout'
    from types import SimpleNamespace
    monkeypatch.setattr(reward.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stderr='dependency missing'))
    assert reward.check_math('x','1')['status']=='verifier_error'


def test_bridge_refuses_all_masked(monkeypatch):
    from types import SimpleNamespace
    import torch
    from trl import GRPOTrainer
    from src.training.trl_bridge_v2 import ProjectGRPOTrainer
    trainer=object.__new__(ProjectGRPOTrainer)
    trainer.args=SimpleNamespace(max_prompt_length=4)
    trainer._tokenizer=SimpleNamespace(encode=lambda *a,**k:[3])
    trainer.max_completion_length=4
    fake={'prompt_ids':torch.tensor([[3]]),'prompt_mask':torch.ones(1,1),'completion_ids':torch.tensor([[3,3,3,3]]),'completion_mask':torch.zeros(1,4)}
    monkeypatch.setattr(GRPOTrainer,'_generate_and_score_completions',lambda *a,**k:fake)
    with pytest.raises(ValueError,match='All completions'):trainer._generate_and_score_completions([{'prompt':'question'}])


def test_reference_existing_tiny_adapter(tmp_path):
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    from peft import LoraConfig,get_peft_model
    from scripts.verify_sft_reference_v2 import verify
    torch.manual_seed(42)
    base=GPT2LMHeadModel(GPT2Config(vocab_size=24,n_positions=32,n_embd=16,n_layer=1,n_head=2,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))
    original=copy.deepcopy(base)
    ids=torch.tensor([[1,2,3,4,5,6]])
    policy=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['c_attn'],task_type='CAUSAL_LM'))
    optimizer=torch.optim.SGD([p for p in policy.parameters() if p.requires_grad],lr=.1)
    policy(input_ids=ids,labels=ids).loss.backward();optimizer.step()
    policy.save_pretrained(tmp_path)
    result=verify(original,ids,['c_attn'],updated_adapter=str(tmp_path))
    assert result['updated_disabled_max_abs']==0 and result['active_adapter_change_max_abs']>0


def test_materialize_pins_existing_export_and_refuses_overwrite(tmp_path,monkeypatch):
    import sys
    from scripts.materialize_colab_config_v2 import main
    merged=tmp_path/'merged';merged.mkdir()
    (merged/'merged_sft_identity.json').write_text('{"fixture":"sft"}')
    (merged/'weight_manifest.json').write_text('{"fixture":"weights"}')
    template=tmp_path/'template.json';output=tmp_path/'resolved.json'
    template.write_text(json.dumps({'stage':'rl','initial_state':'sft','model':str(merged),'expected_sft':{'arm':'raw'}}))
    monkeypatch.setattr(sys,'argv',['materialize','--template',str(template),'--output',str(output)])
    main();config=json.loads(output.read_text())
    for source,key in [('merged_sft_identity.json','identity_sha256'),('weight_manifest.json','weight_manifest_sha256')]:
        assert config['expected_sft'][key]==hashlib.sha256((merged/source).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):main()


def test_materialize_refuses_formal_draft(tmp_path,monkeypatch):
    import sys
    from scripts.materialize_colab_config_v2 import main
    template=tmp_path/'draft.json';output=tmp_path/'resolved.json'
    template.write_text('{"draft":true}')
    monkeypatch.setattr(sys,'argv',['materialize','--template',str(template),'--output',str(output)])
    with pytest.raises(ValueError,match='formal drafts'):main()
    assert not output.exists()


def test_v2_runner_cpu_gate_precedes_model_loading(tmp_path,monkeypatch):
    import torch
    from src.training.approved_hf_v2 import run
    config=json.loads(Path('configs/project1_raw_sft_smoke.json').read_text())
    config.update(arm='raw',output_dir=str(tmp_path/'run'))
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA'):run(path)
    assert not (tmp_path/'run').exists()
