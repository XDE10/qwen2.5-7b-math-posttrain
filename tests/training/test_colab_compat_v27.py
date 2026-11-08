import inspect
from unittest.mock import Mock
import pytest
from experiments.aimo_sft_v26.common import config,read,save
from experiments.aimo_sft_v26.train import training_kwargs,reconcile_pretrain_identity
from experiments.aimo_sft_v26 import install_missing


@pytest.mark.parametrize('batch',[1,2,4])
def test_warmup_steps_is_22_and_supported_by_real_arguments(tmp_path,batch):
    from transformers import TrainingArguments
    c=config();c.update(batch_size=batch,gradient_accumulation_steps=16//batch)
    kwargs=training_kwargs(c,2275,tmp_path,TrainingArguments)
    assert kwargs['warmup_steps']==22 and 'warmup_ratio' not in kwargs
    kwargs['bf16']=False
    args=TrainingArguments(**kwargs)
    assert args.get_warmup_steps(429)==22


def test_modern_sampler_signature_uses_renamed_parameter(tmp_path):
    from transformers import TrainingArguments
    params=[p for p in inspect.signature(TrainingArguments).parameters.values()
            if p.name not in ['warmup_ratio','group_by_length','train_sampling_strategy']]
    params.append(inspect.Parameter('train_sampling_strategy',inspect.Parameter.KEYWORD_ONLY,default='random'))
    modern=Mock();modern.__signature__=inspect.Signature(params)
    kwargs=training_kwargs(config(),2275,tmp_path,modern)
    assert kwargs['train_sampling_strategy']=='group_by_length'
    assert 'group_by_length' not in kwargs and 'warmup_ratio' not in kwargs


def test_failed_initialization_identity_archived_without_changing_recipe(tmp_path):
    known='240d985b0130f18c70f89ee89ad39753bbf29b01382c8969d055369dc142f5d9'
    old={'train_code_sha256':known,'config':config(),'data_identity':'unchanged'}
    save(tmp_path/'run_identity.json',old)
    new={**old,'train_code_sha256':'new-source'}
    reconcile_pretrain_identity(tmp_path,new)
    assert read(tmp_path/'run_identity_before_compat_v27.json')==old
    assert read(tmp_path/'run_identity.json')==old  # caller writes new identity, original retained here
    (tmp_path/'checkpoint-1').mkdir()
    with pytest.raises(ValueError,match='Existing training state'):
        reconcile_pretrain_identity(tmp_path,new)


def test_torchao_known_fix_does_not_upgrade_torch(monkeypatch):
    monkeypatch.setattr(install_missing,'version',lambda name:'0.10.0')
    run=Mock();monkeypatch.setattr(install_missing.subprocess,'run',run)
    install_missing.repair_known_torchao()
    command=run.call_args.args[0]
    assert '--no-deps' in command and command[-1]=='torchao==0.18.0'
    monkeypatch.setattr(install_missing,'version',lambda name:'0.18.0')
    run.reset_mock();install_missing.repair_known_torchao();run.assert_not_called()
