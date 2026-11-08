"""CPU tiny verification or explicitly requested local merged-model verification.

Pass: disabled-adapter logits before/after an RL adapter update equal the SFT
base within atol/rtol; all base weights frozen, active adapter logits change.
No remote weights are ever downloaded. Real 7B GPU verification is unverified.
"""
import argparse
import json
from pathlib import Path


def verify(model, ids, target_modules, atol=1e-6, rtol=1e-5, rank=2, alpha=4, updated_adapter=None):
    import hashlib
    import torch
    from peft import LoraConfig, get_peft_model
    model.eval()
    with torch.no_grad():
        baseline = model(input_ids=ids).logits.detach().clone()
    policy = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0, target_modules=target_modules, task_type='CAUSAL_LM'))
    policy.eval()
    def digest(p):
        return hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    frozen_before = {n: digest(p) for n, p in policy.named_parameters() if not p.requires_grad}
    with torch.no_grad(), policy.disable_adapter():
        disabled_before = policy(input_ids=ids).logits.detach().clone()
    if updated_adapter:
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        result = set_peft_model_state_dict(policy, load_peft_weights(updated_adapter, device=str(ids.device)))
        if result.unexpected_keys or any('lora_' in name for name in result.missing_keys):
            raise ValueError('Incomplete or unexpected RL adapter weights')
    else:
        # Only the CPU tiny path may perform a synthetic optimizer update.
        if model.config.model_type != 'gpt2' or ids.device.type != 'cpu':
            raise ValueError('Real model check requires an existing updated RL adapter; no implicit training')
        optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=.1)
        loss = policy(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    policy.eval()
    with torch.no_grad(), policy.disable_adapter():
        disabled_after = policy(input_ids=ids).logits.detach().clone()
    with torch.no_grad():
        active_after = policy(input_ids=ids).logits.detach().clone()
    torch.testing.assert_close(disabled_before, baseline, atol=atol, rtol=rtol)
    torch.testing.assert_close(disabled_after, baseline, atol=atol, rtol=rtol)
    assert all(digest(p) == frozen_before[n] for n, p in policy.named_parameters() if n in frozen_before)
    changed = float((active_after-baseline).abs().max())
    if not changed > 0:
        raise AssertionError('No observable adapter update')
    return {'status': 'passed', 'baseline_shape': list(baseline.shape), 'atol': atol, 'rtol': rtol,
            'fresh_disabled_max_abs': float((disabled_before-baseline).abs().max()),
            'updated_disabled_max_abs': float((disabled_after-baseline).abs().max()),
            'active_adapter_change_max_abs': changed, 'frozen_base_unchanged': True}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--tiny', action='store_true')
    group.add_argument('--config')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--rl-adapter', help='Local adapter after actual RL update; mandatory for non-tiny check')
    a = p.parse_args()
    torch.manual_seed(42)
    if a.tiny:
        model = GPT2LMHeadModel(GPT2Config(vocab_size=24, n_positions=32, n_embd=16, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).to(a.device)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=a.device)
        result = verify(model, ids, ['c_attn'])
    else:
        if not a.rl_adapter or not Path(a.rl_adapter).is_dir():
            raise ValueError('Provide an existing local updated RL adapter; no real-model training is performed here')
        from src.training.guards_v2 import approved_config
        config, _, identity = approved_config(a.config)
        if config['stage'] != 'rl' or config['initial_state'] != 'sft':
            raise ValueError('Reference verification requires a merged SFT RL config')
        adapter = Path(a.rl_adapter)
        identity_path = adapter / 'run_identity.json'
        if not identity_path.is_file():
            identity_path = adapter.parent / 'run_identity.json'
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('Updated RL adapter belongs to a different run/base identity')
        model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True, torch_dtype=torch.bfloat16).to(a.device)
        tok = AutoTokenizer.from_pretrained(config['tokenizer_path'], local_files_only=True)
        ids = tok.apply_chat_template([{'role':'user','content':'Compute 2 + 3.'}], tokenize=True, add_generation_prompt=True, return_tensors='pt').to(a.device)
        result = verify(model, ids, ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], atol=.005, rtol=.005,
                        rank=config.get('lora_r',16), alpha=config.get('lora_alpha',32), updated_adapter=a.rl_adapter)
    result.update(mode='tiny_cpu' if a.tiny else 'local_merged', real_7b_gpu='未验证' if a.tiny else 'see this run only')
    Path(a.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
