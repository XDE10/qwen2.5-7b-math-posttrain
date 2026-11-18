import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('trl')

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, TrainerCallback
from trl import GRPOTrainer

from experiments.qfft_v93.checkpointing import CheckpointMirror, resolve_resume
from experiments.qfft_v93.common import load_config, read_json, read_jsonl
from experiments.qfft_v93.grpo import RolloutDiagnostics, frozen_mmr_pool, grpo_config, reload_mmr_checkpoint
from experiments.qfft_v93.mmr_grpo import MmrDRGRPOTrainer, MmrRecoveryCheckpoints, MmrReward, training_messages


def test_reward_groups_are_independent_and_truncated_correct_answers_do_not_count():
    config = load_config('grpo/grpo_qfft_r1_mmr_dr50.json')
    config['max_completion_length'] = 4
    tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=1)
    rank = np.arange(1, 7, dtype=np.float32)
    reward = MmrReward(tokenizer, config, lambda texts: np.stack([rank, np.ones(len(texts))], axis=1))
    correct = [{'role': 'assistant', 'content': 'work\n</think>\nThe final answer is \\boxed{2}.'}]
    wrong = [{'role': 'assistant', 'content': 'The final answer is \\boxed{3}.'}]
    metrics = {}
    shared = dict(completion_ids=[[2, 3, 4, 5]] + [[4, 0]] * 11,
                  reference_answer=['2'] * 12, sample_id=['a'] * 6 + ['b'] * 6,
                  log_metric=metrics.__setitem__)
    a = [correct, correct, wrong, wrong, wrong, wrong]
    before = reward(completions=a + [wrong] * 6, **shared)
    assert metrics['comparison/length_stops'] == 1
    assert metrics['reward/correct_frac'] == 1 / 12
    after = reward(completions=a + [correct] * 6, **shared)
    assert before[:6] == after[:6]
    assert before[6:] != after[6:]
    assert all(math.isfinite(value) for value in before + after)


@pytest.mark.parametrize('valid_count', [0, 1, 2, 5, 6])
def test_masked_completions_are_excluded_from_rewards_and_embeddings(valid_count):
    cfg = load_config('grpo/grpo_qfft_r1_mmr_dr150.json')
    cfg['max_completion_length'] = 4
    tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=1)
    # PAD and an alternate EOS are valid, even at the cap. No EOS is masked even below the cap.
    terminal = [[2, 2, 2, 0], [2, 1], [2, 3], [2, 2, 0], [2, 3], [1]]
    nonterminal = [[], [2], [2, 2, 2, 2], [], [2], [2, 2]]
    ids = terminal[:valid_count] + nonterminal[valid_count:]
    texts = [f'sample {i}\n</think>\n\\boxed{{{2 + i % 2}}}' for i in range(6)]

    def embedding(batch):
        assert list(batch) == texts[:valid_count]
        return np.eye(len(batch), dtype=np.float32)

    reward = MmrReward(tokenizer, cfg, embedding)
    reward.eos.add(3)

    def score(text, reward_func):
        return reward_func(completions=[[{'role': 'assistant', 'content': value}] for value in text],
                           completion_ids=ids, reference_answer=['2'] * 6, sample_id=['id'] * 6,
                           log_metric=lambda *_: None)

    values = score(texts, reward)
    assert score(texts[:valid_count] + ['changed masked text'] * (6 - valid_count), reward) == values
    assert all(math.isfinite(r) for r in values[:valid_count])
    assert values[valid_count:] == [None] * (6 - valid_count)
    if valid_count == 6:
        old = MmrReward(tokenizer, load_config('grpo/grpo_qfft_r1_mmr_dr50.json'), embedding)
        old.cap = cfg['max_completion_length']
        old.eos.add(3)
        assert score(texts, old) == values


def test_mmr150_inherits_recipe_and_exact_frozen_prefix():
    data = Path(__file__).resolve().parents[2] / 'data'
    folder = data / 'rl_openr1_46k_decontaminated'
    old = load_config('grpo/grpo_qfft_r1_mmr_dr50.json')
    new = load_config('grpo/grpo_qfft_r1_mmr_dr150.json')
    smoke = load_config('grpo/grpo_qfft_r1_mmr_dr150_smoke2.json')
    different = {key for key in old if old[key] != new[key]}
    assert different == {'run_name', 'pool', 'mmr_dr_grpo', 'max_steps', 'note'}
    assert new['mmr_dr_grpo'] == {**old['mmr_dr_grpo'], 'exclude_masked_completions': True}
    different = {key for key in smoke if key not in new or smoke[key] != new[key]}
    assert different == {'run_name', 'pool', 'max_steps', 'save_steps', 'merge_adapter', 'verify_final_checkpoint', 'note'}
    assert smoke['max_steps'] == smoke['save_steps'] == 2 and not smoke['merge_adapter']
    assert smoke['verify_final_checkpoint']
    if not (folder / 'mmr_dr150_frozen_manifest.json').exists():
        pytest.skip('Frozen OpenR1 files are local artifacts')
    manifest = read_json(folder / 'mmr_dr150_frozen_manifest.json')
    replacements = [row['seed_position'] for row in manifest['changes'] if row['action'] == 'replace']
    assert replacements == [position for position in range(1203, 1218) if position != 1209]
    frozen = (folder / 'mmr_dr50_train.jsonl').read_bytes()
    approved = (folder / manifest['train_file']).read_bytes()
    assert approved.startswith(frozen)
    assert (folder / manifest['smoke_prefix']['train_file']).read_bytes() == b''.join(
        frozen.splitlines(keepends=True)[:16])
    draft = folder / 'mmr_dr150_train.jsonl'
    if draft.is_file():
        changed_positions = [i for i, (before, after) in enumerate(zip(
            draft.read_bytes().splitlines(), approved.splitlines(), strict=True), 1) if before != after]
        assert changed_positions == [row['train_position'] for row in manifest['changes']]
    old_pool, _ = frozen_mmr_pool(old, data)
    new_pool, _ = frozen_mmr_pool(new, data)
    assert len(new_pool) == 1200 and new_pool[:400] == old_pool
    smoke_pool, _ = frozen_mmr_pool(smoke, data)
    assert len(smoke_pool) == 16 and smoke_pool == old_pool[:16]


def test_dr150_smoke_stage_uses_mmr_environment(monkeypatch):
    from experiments.qfft_v93.colab import stages

    calls = []
    monkeypatch.setattr(stages, 'gpu_banner', lambda: None)
    ctx = SimpleNamespace(mmr_module=lambda *args: ['mmr', *args],
                          run=lambda command, name: calls.append((command, name)))
    stages._dispatch(ctx, 'grpo:grpo_qfft_r1_mmr_dr150_smoke2')
    assert calls == [(['mmr', 'experiments.qfft_v93.grpo', '--config', 'grpo/grpo_qfft_r1_mmr_dr150_smoke2.json'],
                      'grpo_grpo_qfft_r1_mmr_dr150_smoke2')]


def test_mmr_trace_preserves_exact_trainer_advantages(tmp_path):
    llm = SimpleNamespace(generate=lambda: [SimpleNamespace(outputs=[
        SimpleNamespace(token_ids=[5, 0], finish_reason='stop', stop_reason=0),
        SimpleNamespace(token_ids=[], finish_reason='stop', stop_reason='marker'),
    ])])
    trainer = SimpleNamespace(accelerator=SimpleNamespace(num_processes=1), eos_token_ids=[0],
                              _tokenizer=SimpleNamespace(pad_token_id=1), vllm_generation=SimpleNamespace(llm=llm))
    diagnostics = RolloutDiagnostics(trainer, mmr_dir=tmp_path, identity={'run': 'fixture'})
    llm.generate()
    diagnostics.record_mmr(1, ['p', 'p'], [[5, 0], [2]], ['done', 'placeholder'], [
        {'trl_clipped': False, 'adjusted_reward': 2.0}, {'trl_clipped': True, 'adjusted_reward': None},
    ])
    diagnostics.finish_mmr([0.0, 0.0])
    data = read_jsonl(tmp_path / 'step-0001.jsonl')
    assert [row['advantage'] for row in data] == [0.0, 0.0]
    assert data[1]['tokens'] == 0 and data[1]['text'] == '' and data[1]['adjusted_reward'] is None


@pytest.mark.parametrize('steps', [2, 50, 150])
def test_fixed_cpu_run_uses_author_schedule_and_can_export(tmp_path, monkeypatch, request, steps):
    from experiments.qfft_v93 import merge_lora

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    raw = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'problem': 3, 'answer': 4}, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}answer"
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=5, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, eos_token_id=2, pad_token_id=1, bos_token_id=3))
    base = tmp_path / 'base'
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)

    recipe = 'dr150_smoke2' if steps == 2 else f'dr{steps}'
    config = load_config(f'grpo/grpo_qfft_r1_mmr_{recipe}.json')
    # Only fixture workload is smaller. The production scheduler and final step are exercised unchanged.
    config.update(gradient_accumulation_steps=3, max_completion_length=2)
    dataset = Dataset.from_list([
        {'prompt': training_messages('problem ' * 520), 'reference_answer': '2', 'sample_id': 'first'},
        {'prompt': training_messages('problem'), 'reference_answer': '2', 'sample_id': 'second'},
    ])
    reward = MmrReward(tokenizer, config, lambda texts: np.eye(len(texts), dtype=np.float32))

    drive = tmp_path / 'drive'

    class StopAfterMirror(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            assert state.global_step == 2
            control.should_training_stop = True

    def make_trainer(local, extra_callbacks=()):
        mirror = CheckpointMirror(local, drive, mirror_final=True, mirror_optimizer_state=True, min_free_gib=0)
        return MmrDRGRPOTrainer(
            model=str(base), processing_class=tokenizer, args=grpo_config(config, local, smoke=True, max_steps=None),
            reward_funcs=reward, train_dataset=dataset,
            peft_config=LoraConfig(r=2, lora_alpha=4, lora_dropout=0.05, target_modules=['q_proj'], task_type='CAUSAL_LM'),
            callbacks=[MmrRecoveryCheckpoints(), mirror, *extra_callbacks],
            prompt_limit=512, scheduler_steps=config['mmr_dr_grpo']['scheduler_steps'])

    trainer = make_trainer(tmp_path / 'vm1', [StopAfterMirror()])
    assert len(trainer._tokenize_prompts([dataset[0]['prompt']])[0][0]) == 512
    sampler = trainer._get_train_sampler()
    assert not sampler.shuffle
    assert list(sampler)[::trainer.args.generation_batch_size * trainer.args.steps_per_generation] == [0, 1]

    if steps == 150:
        # Drive the actual TRL mask, reward aggregation and DR-GRPO loss with known token endings, including empty.
        trainer.current_gradient_accumulation_steps = trainer.args.gradient_accumulation_steps
        for usable in (0, 1, 2, 5, 6):
            responses = [[4, 2] if i < usable else ([] if i % 2 else [4, 4]) for i in range(6)]
            with monkeypatch.context() as patch:
                patch.setattr(GRPOTrainer, '_generate_single_turn', lambda *_, **__: (responses, None, None))
                output = trainer._generate_and_score_completions([dataset[0]] * 6)
            assert output['completion_mask'].any(dim=1).tolist() == [True] * usable + [False] * (6 - usable)
            assert torch.isfinite(output['advantages']).all()
            assert torch.count_nonzero(output['advantages'][usable:]) == 0
            if usable <= 1:
                assert torch.count_nonzero(output['advantages']) == 0
                assert torch.isfinite(trainer._compute_loss(trainer.model, output))
    trainer.train()
    assert trainer.state.global_step == 2
    if steps == 2:
        assert trainer.args.save_steps == 2
        assert trainer.optimizer.state
        reload_mmr_checkpoint(trainer, tmp_path / 'vm1' / 'checkpoint-2')
        assert trainer.optimizer.state
        assert trainer.lr_scheduler.last_epoch == 2
        return
    assert trainer.args.save_steps == 50

    local = tmp_path / 'vm2'
    restored = resolve_resume(local, drive)
    assert restored == str(local / 'checkpoint-2')
    trainer = make_trainer(local)
    trainer.train(resume_from_checkpoint=restored)
    assert trainer.state.global_step == steps
    assert trainer.lr_scheduler.lr_lambdas[0](50) == pytest.approx(1)
    assert trainer.lr_scheduler.lr_lambdas[0](500) == pytest.approx(0.1)
    assert all(lr == pytest.approx(1e-6 * trainer.lr_scheduler.lr_lambdas[0](steps))
               for lr in trainer.lr_scheduler.get_last_lr())
    assert any('kl' in row for row in trainer.state.log_history)
    assert any('reward/format_frac' in row for row in trainer.state.log_history)
    assert (local / f'checkpoint-{steps}' / 'optimizer.pt').is_file()
    trainer._load_from_checkpoint(str(local / f'checkpoint-{steps}'))
    adapter = tmp_path / 'adapter'
    trainer.save_model(adapter)
    assert (adapter / 'adapter_model.safetensors').is_file()
    monkeypatch.setattr(merge_lora, 'local_root', lambda: tmp_path)
    merged = merge_lora.merge(base, adapter, tmp_path / 'merged', smoke=True)
    assert (merged / 'model.safetensors').is_file()
