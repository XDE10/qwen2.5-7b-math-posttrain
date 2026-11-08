import json
import pytest
from scripts.freeze_project1_screening_v3 import screen_lines


def test_union_uses_rl_side_and_deduplicates():
    lines=[(json.dumps({'sample_id':i,'training_approved':False})+'\n').encode() for i in ['a','b','c']]
    hits=[{'source':'rl','target':'dev','source_id':'a','target_id':'dev1'},
          {'source':'sft','target':'rl','source_id':'sft1','target_id':'a'},
          {'source':'sft','target':'rl','source_id':'sft2','target_id':'b'}]
    kept,removed,heldout,sft,_=screen_lines(lines,hits)
    assert heldout=={'a'} and sft=={'a','b'}
    assert kept==lines[2:] and removed==lines[:2]
    assert all(json.loads(line)['training_approved'] is False for line in kept+removed)


def test_stale_or_duplicate_ids_rejected():
    line=b'{"sample_id":"a"}\n'
    with pytest.raises(ValueError,match='Duplicate'):screen_lines([line,line],[])
    with pytest.raises(ValueError,match='absent'):screen_lines([line],[{'source':'rl','target':'dev','source_id':'missing'}])


def test_frozen_snapshot_and_execution_interlock():
    from pathlib import Path
    from src.training.guards_v2 import approved_config
    out=Path('reports/handover_20261024_v3')
    result=json.loads((out/'freeze_record.json').read_text(encoding='utf-8'))
    assert result['excluded_count']==180 and result['remaining_count']==4630
    assert not result['intersection_ids'] and not any(result['lexical_recheck'].values())
    assert result['training_approved'] is False
    for arm in ('raw','compact'):
        config=out/'equal_exposure_frozen_v1'/f'E1_{arm}.parameters_frozen.json'
        c=json.loads(config.read_text(encoding='utf-8'))
        assert c['parameter_status']=='frozen' and c['budget_protocol']['epochs']==1
        assert c['max_steps']==94 and c['gradient_accumulation_steps']==16
        # Supply Drive env separately in the test to reach the draft interlock.
        import os
        previous=os.environ.get('PROJECT_DRIVE_ROOT')
        os.environ['PROJECT_DRIVE_ROOT']='/content/drive/MyDrive/test'
        try:
            with pytest.raises(ValueError,match='Draft'):approved_config(config)
        finally:
            if previous is None:os.environ.pop('PROJECT_DRIVE_ROOT',None)
            else:os.environ['PROJECT_DRIVE_ROOT']=previous


def test_tiny_one_epoch_is_94_updates_and_same_exposure(tmp_path):
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel,Trainer,TrainingArguments
    torch.set_num_threads(2)
    orders=[]
    for length in (3,5):
        consumed=[]
        class CountingModel(GPT2LMHeadModel):
            def forward(self,input_ids=None,attention_mask=None,labels=None,sample_id=None,**kwargs):
                consumed.extend(sample_id.tolist())
                return super().forward(input_ids=input_ids,attention_mask=attention_mask,labels=labels,**kwargs)
        model=CountingModel(GPT2Config(vocab_size=24,n_positions=8,n_embd=8,n_layer=1,n_head=1,use_cache=False))
        data=[{'input_ids':[1]+[3]*(length-2)+[2],'sample_id':i} for i in range(1500)]
        def collate(batch):
            ids=torch.tensor([r['input_ids'] for r in batch])
            return {'input_ids':ids,'labels':ids.clone(),'sample_id':torch.tensor([r['sample_id'] for r in batch])}
        args=TrainingArguments(output_dir=str(tmp_path/str(length)),use_cpu=True,max_steps=94,per_device_train_batch_size=1,gradient_accumulation_steps=16,seed=42,data_seed=42,save_strategy='no',report_to='none',disable_tqdm=True,remove_unused_columns=False,dataloader_num_workers=0)
        trainer=Trainer(model=model,args=args,train_dataset=data,data_collator=collate)
        trainer.train()
        assert trainer.state.global_step==94 and trainer.state.epoch==pytest.approx(1.0)
        assert sorted(consumed)==list(range(1500))
        orders.append(consumed)
    assert orders[0]==orders[1]
