import copy
from pathlib import Path

import pytest

from math_posttrain.common import (load_config, read_json, sha256_file, sha256_json, write_json, write_jsonl)


def test_mmr_append_requires_the_original_byte_prefix_and_source_identity(tmp_path, monkeypatch):
    from math_posttrain import grpo

    old = load_config('grpo/grpo_mmr_dr150.json')
    old.update(max_steps=2, warmup_steps=0)
    old['pool'] = {'approval_manifest': 'old.json'}
    rows = [{'sample_id': str(i), 'question': f'question {i}', 'reference_answer': str(i)} for i in range(4)]
    write_jsonl(tmp_path / 'old.jsonl', rows[:2])
    write_jsonl(tmp_path / 'new.jsonl', rows)
    spec = lambda name, count: {'sha256': sha256_file(tmp_path / name), 'rows': count,
                                'source_training_approved': None}
    scope = {'run_names': [old['run_name']], 'approved_training_files': ['old.jsonl'],
             'files': {'old.jsonl': spec('old.jsonl', 2)}}
    write_json(tmp_path / 'old.json', scope)
    new = copy.deepcopy(old)
    new.update(run_name='extended', max_steps=4, retained_checkpoint_steps=[2, 4],
               extend_from={'config': 'source', 'code_bundle': 'old-code',
                            'approval_sha256': sha256_file(tmp_path / 'old.json'),
                            'data_mode': 'append_frozen_prefix'})
    new['pool'] = {'approval_manifest': 'new.json'}
    new_scope = {'run_names': ['extended'], 'approved_training_files': ['new.jsonl'],
                 'files': {'new.jsonl': spec('new.jsonl', 4)},
                 'unchanged_prefix': {'file': 'old.jsonl', **scope['files']['old.jsonl']}}
    write_json(tmp_path / 'new.json', new_scope)
    monkeypatch.setattr(grpo, 'CONFIG_DIR', tmp_path)
    monkeypatch.setattr(grpo, 'load_config', lambda _: old)
    identity = {'run': 'extended', 'code_bundle': 'new-code', 'pool_sha256': sha256_json(rows),
                'approval_sha256': sha256_file(tmp_path / 'new.json'), 'recipe_sha256': sha256_json(new)}
    source_identity = {**identity, 'run': old['run_name'], 'code_bundle': 'old-code',
                       'pool_sha256': sha256_json(rows[:2]), 'approval_sha256': sha256_file(tmp_path / 'old.json'),
                       'recipe_sha256': sha256_json(old)}
    source = tmp_path / old['run_name']
    write_json(source / 'run_identity.json', {'identity': source_identity})
    write_json(source / 'GRPO_COMPLETE.json', {'steps': 2})
    assert grpo.extension_source(new, identity, tmp_path, pool=rows, data_root=tmp_path) == (
        source, 2, sha256_json(source_identity))
    changed = copy.deepcopy(new)
    changed['learning_rate'] *= 2
    with pytest.raises(RuntimeError, match='training recipe'):
        grpo.extension_source(changed, identity, tmp_path, pool=rows, data_root=tmp_path)
    # Even a separately hashed new file may not change already-trained questions.
    rows[0]['question'] = 'changed old question'
    write_jsonl(tmp_path / 'new.jsonl', rows)
    new_scope['files']['new.jsonl'] = spec('new.jsonl', 4)
    write_json(tmp_path / 'new.json', new_scope)
    with pytest.raises(RuntimeError, match='byte prefix'):
        grpo.extension_source(new, identity, tmp_path, pool=rows, data_root=tmp_path)


def test_milestone_retention_survives_rolling_mirror_pruning(tmp_path, monkeypatch):
    from math_posttrain import checkpointing as ck

    monkeypatch.delenv('P1_RCLONE_REMOTE', raising=False)
    mirror = ck.CheckpointMirror(tmp_path / 'local', tmp_path / 'drive', keep=2,
                                mirror_optimizer_state=True, retain_steps=[2, 4], min_free_gib=0)
    for step in range(1, 8):
        path = tmp_path / 'local' / f'checkpoint-{step}'
        path.mkdir(parents=True)
        write_json(path / ck.SAVED_MARK, {'step': step})
        write_json(path / 'trainer_state.json', {'global_step': step})
        for name in ['adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
            (path / name).write_bytes(b'valid checkpoint bytes')
        mirror._mirror(path)
        assert mirror.error is None
    assert {p.name for p in mirror.mirror_dir.iterdir()} == {'checkpoint-2', 'checkpoint-4', 'checkpoint-6', 'checkpoint-7'}


def test_milestone_export_rejects_corruption_and_reuses_bound_receipt(tmp_path, monkeypatch):
    from math_posttrain import checkpointing as ck, merge_lora, milestones

    monkeypatch.delenv('P1_RCLONE_REMOTE', raising=False)
    monkeypatch.setenv('P1_LOCAL_ROOT', str(tmp_path / 'exports'))
    root = tmp_path / 'run'
    local = tmp_path / 'local' / 'checkpoint-4'
    local.mkdir(parents=True)
    write_json(local / ck.SAVED_MARK, {'step': 4})
    write_json(local / 'trainer_state.json', {'global_step': 4})
    for name in ['adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
        (local / name).write_bytes(b'correct')
    mirror = ck.CheckpointMirror(local.parent, root, mirror_optimizer_state=True, min_free_gib=0)
    mirror._mirror(local)
    job = {'source_run': str(tmp_path / 'source'), 'source_step': 2, 'drive_run': str(root),
           'final_step': 6, 'base': str(tmp_path / 'base'), 'run_identity_sha256': 'identity'}
    source = root / 'ckpt_mirror/checkpoint-4'
    (source / 'optimizer.pt').write_bytes(b'corrupt')  # same length; checksum must catch this
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        milestones.export_one(job, 4)
    (source / 'optimizer.pt').write_bytes(b'correct')
    merges = []

    def fake_merge(base, adapter, out, *, provenance):
        merges.append(provenance)
        out.mkdir(parents=True)
        (out / 'model.safetensors').write_bytes(b'merged weights')
        write_json(out / 'p1_model_identity.json', {'adapter': str(adapter)})
        write_json(out / 'p1_export_manifest.json', {'provenance': provenance,
                   'files': {'model.safetensors': {'bytes': 14}}})

    monkeypatch.setattr(merge_lora, 'merge', fake_merge)
    assert milestones.export_one(job, 4)
    assert milestones.export_one(job, 4)
    assert len(merges) == 1
    receipt = read_json(milestones._receipt(job, 4))
    assert receipt['provenance']['checkpoint_manifest_sha256'] == sha256_file(source / ck.MIRROR_MARK)
    write_json(source / ck.MIRROR_MARK, {**read_json(source / ck.MIRROR_MARK), 'changed': True})
    with pytest.raises(RuntimeError, match='different checkpoint'):
        milestones.export_one(job, 4)


def test_expanded_dataset_resumes_next_groups_and_keeps_fixed_cosine_state(tmp_path):
    torch = pytest.importorskip('torch')
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig
    from math_posttrain.mmr_grpo import MmrDRGRPOTrainer

    torch.set_num_threads(1)
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'question': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    base = tmp_path / 'base'
    GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=128, n_embd=8, n_layer=1, n_head=1,
                              pad_token_id=1, eos_token_id=2, bos_token_id=2)).save_pretrained(base)
    rows = [{'prompt': [{'role': 'user', 'content': 'question'}], 'sample_id': str(i)} for i in range(10)]
    seen, restored = [], []

    class Reward:
        __name__ = 'test_reward'
        exclude_masked_completions = False
        diagnostics = None

        def __call__(self, prompts, completions, sample_id, **kw):
            seen.append(sample_id)
            return [float(i % 2) for i in range(len(completions))]

    def args(out, steps):
        return GRPOConfig(output_dir=str(out), max_steps=steps, use_cpu=True, bf16=False,
                          per_device_train_batch_size=4, gradient_accumulation_steps=1, num_generations=2,
                          max_completion_length=4, learning_rate=1e-6, lr_scheduler_type='cosine_with_min_lr',
                          lr_scheduler_kwargs={'min_lr_rate': .1}, warmup_steps=0, shuffle_dataset=False,
                          save_steps=1, report_to='none', disable_tqdm=True, remove_unused_columns=False)

    first = MmrDRGRPOTrainer(model=str(base), args=args(tmp_path / 'old', 3), reward_funcs=Reward(),
                             train_dataset=Dataset.from_list(rows[:6]), processing_class=tok,
                             prompt_limit=64, scheduler_steps=500)
    first.train()
    checkpoint = tmp_path / 'old/checkpoint-3'
    optimizer = torch.load(checkpoint / 'optimizer.pt', weights_only=True)
    scheduler = torch.load(checkpoint / 'scheduler.pt', weights_only=True)
    assert seen == [[str(i) for i in range(start, start + 2) for _ in range(2)] for start in (0, 2, 4)]
    seen.clear()

    class VerifyResume(MmrDRGRPOTrainer):
        def _load_optimizer_and_scheduler(self, path):
            super()._load_optimizer_and_scheduler(path)
            assert self.lr_scheduler.state_dict() == scheduler
            for index, moments in optimizer['state'].items():
                for name, tensor in moments.items():
                    torch.testing.assert_close(self.optimizer.state_dict()['state'][index][name], tensor, rtol=0, atol=0)
            restored.append(path)

    second = VerifyResume(model=str(base), args=args(tmp_path / 'new', 5), reward_funcs=Reward(),
                          train_dataset=Dataset.from_list(rows), processing_class=tok,
                          prompt_limit=64, scheduler_steps=500)
    second.train(resume_from_checkpoint=str(checkpoint))
    assert restored == [str(checkpoint)]
    assert seen == [['6', '6', '7', '7'], ['8', '8', '9', '9']]
    assert second.state.global_step == 5
    assert second.lr_scheduler.last_epoch == 5
    assert second.optimizer.param_groups[0]['lr'] == pytest.approx(1e-6 * (.1 + .9 * (1 + __import__('math').cos(5 / 500 * __import__('math').pi)) / 2))


def test_real_cpu_merge_publishes_weights_and_checkpoint_provenance(tmp_path, monkeypatch):
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from math_posttrain import merge_lora

    monkeypatch.delenv('P1_RCLONE_REMOTE', raising=False)
    monkeypatch.setattr(merge_lora, 'local_root', lambda: tmp_path)
    base, adapter, out = (tmp_path / x for x in ('base', 'adapter', 'merged'))
    GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=16, n_embd=8, n_layer=1, n_head=1)).save_pretrained(base)
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0, 'x': 1}, unk_token='[UNK]')))
    tok.save_pretrained(base)
    model = get_peft_model(GPT2LMHeadModel.from_pretrained(base), LoraConfig(r=2, target_modules=['c_attn']))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.fill_(.02)
    model.eval()
    inputs = torch.tensor([[1, 1]])
    expected = model(inputs).logits
    model.save_pretrained(adapter)
    provenance = {'step': 200, 'checkpoint_manifest_sha256': 'fixture-checkpoint'}
    merge_lora.merge(base, adapter, out, smoke=True, provenance=provenance)
    manifest = read_json(out / 'p1_export_manifest.json')
    assert manifest['provenance'] == provenance
    for name, record in manifest['files'].items():
        assert sha256_file(out / name) == record['sha256']
    actual = GPT2LMHeadModel.from_pretrained(out).eval()(inputs).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
