"""Execute the pinned TRL scoring/masking and loss, not a copied formula."""
import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import GRPOConfig, GRPOTrainer
from src.losses.reference import group_advantages
from src.losses.reference_v2 import trl_policy_loss


@pytest.mark.parametrize('mode',['grpo','dr_grpo'])
@pytest.mark.parametrize('truncate',[False,True])
@pytest.mark.parametrize('beta',[0.,.04])
def test_actual_trl_advantage_mask_loss_gradient(tmp_path,mode,truncate,beta):
    torch.manual_seed(42)
    backend=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'question':3},unk_token='[UNK]'))
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
    model=GPT2LMHeadModel(GPT2Config(vocab_size=24,n_positions=32,n_embd=16,n_layer=1,n_head=2,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))
    model.generation_config.eos_token_id=2;model.generation_config.pad_token_id=1
    local=tmp_path/'base';model.save_pretrained(local);model.config._name_or_path=str(local)
    def reward(completions,**kwargs):return [1.,0.,0.,0.]
    args=GRPOConfig(output_dir=str(tmp_path/'run'),use_cpu=True,bf16=False,fp16=False,per_device_train_batch_size=4,gradient_accumulation_steps=2,num_generations=4,generation_batch_size=4,max_completion_length=4,beta=beta,loss_type=mode,scale_rewards='group' if mode=='grpo' else 'none',mask_truncated_completions=truncate,report_to='none',save_strategy='no',use_vllm=False,gradient_checkpointing=False)
    trainer=GRPOTrainer(model=model,args=args,train_dataset=Dataset.from_list([{'prompt':'question'}]*4),processing_class=tok,reward_funcs=reward)
    seqs=[[3,2],[3,3,2],[3,3,3,2],[3,3,3,3]]
    trainer._generate=lambda prompts:([[3]]*4,seqs,None,['answer']*4,None,{},None,None)
    trainer.current_gradient_accumulation_steps=2
    data=trainer._generate_and_score_completions([{'prompt':'question'}]*4)
    expected_adv=group_advantages(torch.tensor([[1.,0.,0.,0.]]),mode,eps=1e-4).flatten()
    torch.testing.assert_close(data['advantages'],expected_adv)
    expected_mask=torch.tensor([[1,1,0,0],[1,1,1,0],[1,1,1,1],[0,0,0,0] if truncate else [1,1,1,1]])
    torch.testing.assert_close(data['completion_mask'].cpu(),expected_mask)
    ids=torch.cat([data['prompt_ids'],data['completion_ids']],dim=1)
    attention=torch.cat([data['prompt_mask'],data['completion_mask']],dim=1)
    logp,_,_=trainer._get_per_token_logps_and_entropies(model,ids,attention,4,compute_entropy=True,compute_aux_loss=False)
    data['old_per_token_logps']=logp.detach()+torch.tensor([[2.,-2.,0.,0.]])*1.
    if beta:data['ref_per_token_logps']=logp.detach()+.1
    actual=trainer._compute_loss(model,data)
    reference=trl_policy_loss(logp,data['old_per_token_logps'],expected_adv,data['completion_mask'],mode,4,beta,data.get('ref_per_token_logps'),2)
    torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-5)
    parameter=model.transformer.wte.weight
    a=torch.autograd.grad(actual,parameter,retain_graph=True)[0]
    b=torch.autograd.grad(reference,parameter)[0]
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
