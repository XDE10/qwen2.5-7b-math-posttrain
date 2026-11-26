"""Single-G4 adaptation of the 7B MMR-DR-GRPO recipe at db07f713ce67."""
from __future__ import annotations

from math import cos, pi
from pathlib import Path

import numpy as np
from trl import GRPOTrainer
from transformers import TrainerCallback

from math_posttrain.grading import _math_verify, _parse_gold, clean_reference


# From the pinned upstream 7B YAML, including its literal doubled braces.
SYSTEM_PROMPT = (
    'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
    'The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, '
    'and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within '
    '<think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> '
    '<answer> answer here </answer>. Note that respond by English, NOT use other languages.'
)


def training_messages(question):
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': question}]


def training_template(tokenizer):
    # The upstream DeepSeek template starts the assistant continuation inside <think>.
    return tokenizer.chat_template + "{% if add_generation_prompt %}{{ '<think>\\n' }}{% endif %}"


class MmrRecoveryCheckpoints(TrainerCallback):
    """Persist resumable state on Colab between the scheduled final save and VM loss."""

    def __init__(self, retained_steps=()):
        self.retained_steps = frozenset(retained_steps)

    def on_step_end(self, args, state, control, **kwargs):
        # The G4 completed two steps within 40 minutes before the first VM disappeared at one hour.
        if (state.global_step in self.retained_steps
                or 0 < state.global_step < state.max_steps and state.global_step % 2 == 0):
            control.should_save = True


class MmrDRGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, prompt_limit, scheduler_steps, **kwargs):
        self.prompt_limit = prompt_limit
        self.scheduler_steps = scheduler_steps
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError('MMR requires complete prompt groups on the single rollout process')
        self.chat_template = training_template(self.processing_class)

    def create_scheduler(self, num_training_steps, optimizer=None):
        # The fixed export step is independent of the original 500-step scheduler horizon.
        return super().create_scheduler(self.scheduler_steps, optimizer=optimizer)

    def _tokenize_prompts(self, prompts):
        ids, images, fields = super()._tokenize_prompts(prompts)
        # TRL uses the returned IDs both for vLLM and for the policy/reference forwards.
        self._log_metric('mmr/prompt_truncated_frac', sum(len(row) > self.prompt_limit for row in ids) / len(ids))
        return [row[-self.prompt_limit:] for row in ids], images, fields

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        ids, logprobs, sampling_mask = super()._generate_single_turn(
            prompt_ids, images, multimodal_fields, num_generations, has_tool_images,
        )
        if self.reward_funcs[0].exclude_masked_completions and any(not row for row in ids):
            # TRL 1.17.1 indexes ids[-1] before reward scoring. Represent an empty generation
            # with a non-terminal token so the same TRL completion mask excludes it completely.
            endings = {*self.eos_token_ids, self._tokenizer.pad_token_id}
            token = next(token for token in self._tokenizer.get_vocab().values() if token not in endings)
            for index, row in enumerate(ids):
                if not row:
                    ids[index] = [token]
                    if logprobs is not None:
                        logprobs[index] = [0.0]
                    if sampling_mask is not None:
                        sampling_mask[index] = [0]
        return ids, logprobs, sampling_mask

    def _generate_and_score_completions(self, inputs):
        scored = super()._generate_and_score_completions(inputs)
        diagnostics = self.reward_funcs[0].diagnostics
        if diagnostics is not None:
            diagnostics.finish_mmr(scored['advantages'].tolist())
        return scored

    def _compute_loss(self, model, inputs):
        # Upstream DR-GRPO uses exp(logp - detach(logp)). Keep the separately
        # computed vLLM token IS weights and reference KL when using TRL's loss.
        if 'old_per_token_logps' in inputs:
            inputs = {k: v for k, v in inputs.items() if k != 'old_per_token_logps'}
        return super()._compute_loss(model, inputs)


class JinaEmbeddings:
    """Author-selected Jina checkpoint, using its official mean-pooled ONNX export."""

    def __init__(self, cfg):
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        model = cfg['embedding_model']
        revision = cfg['embedding_revision']
        path = hf_hub_download(model, 'model-w-mean-pooling.onnx', revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.max_length = cfg['embedding_max_length']

    def __call__(self, texts):
        tokens = self.tokenizer(texts, truncation=True, max_length=self.max_length)
        # Avoid padding five short responses to the longest one. The CPU ONNX graph
        # materializes attention matrices, so encode each response before grouping.
        embeddings = []
        for index in range(len(texts)):
            inputs = {name: np.asarray([tokens[name][index]], dtype=np.int64) for name in self.input_names}
            embeddings.append(self.session.run(None, inputs)[0][0])
        return np.stack(embeddings)


def reweight_group(rewards, embeddings, *, order=None):
    """Algorithm 1 with the pinned code's population std and nonnegative similarity floor."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(rewards) or not np.isfinite(embeddings).all():
        raise ValueError('MMR needs one finite embedding per completion')
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    unit = embeddings / norms
    similarity = unit @ unit.T
    weight = 1 / (1 + np.exp(-rewards.std()))
    first = int(rewards.argmax())
    if order is not None:
        order.append(first)
    chosen = np.zeros(len(rewards), dtype=bool)
    chosen[first] = True
    adjusted = rewards.copy()
    best_similarity = np.maximum(0, similarity[:, first])
    for _ in range(1, len(rewards)):
        scores = weight * rewards - (1 - weight) * best_similarity
        scores[chosen] = -np.inf
        index = int(scores.argmax())
        if order is not None:
            order.append(index)
        adjusted[index] = scores[index]
        chosen[index] = True
        best_similarity = np.maximum(best_similarity, similarity[:, index])
    return adjusted


def cosine_reward(correct, length, cap):
    # The 7B script defaults are correct [0.5,1] and wrong [0,-0.5].
    mix = (1 + cos(pi * min(length, cap) / cap)) / 2
    return 0.5 + 0.5 * mix if correct else -0.5 + 0.5 * mix


class MmrReward:
    __name__ = 'mmr_format_cosine'

    def __init__(self, tokenizer, cfg, embed):
        from latex2sympy2_extended import NormalizationConfig

        self.embed = embed
        self.cap = cfg['max_completion_length']
        self.group = cfg['num_generations']
        self.exclude_masked_completions = cfg['mmr_dr_grpo'].get('exclude_masked_completions', False)
        if self.exclude_masked_completions and not cfg['mask_truncated_completions']:
            raise ValueError('MMR exclusion requires the TRL truncated-completion loss mask')
        self.diagnostics = None
        self.eos = {tokenizer.eos_token_id, tokenizer.pad_token_id}
        self.mv = _math_verify()
        self.answer_extraction = self.mv.LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False, malformed_operators=False, basic_latex=True, equations=True, boxed='all', units=True),
            boxed_match_priority=0, try_extract_without_anchor=False)

    def __call__(self, completions, completion_ids, reference_answer, sample_id, log_metric, **kwargs):
        texts = [row[0]['content'] for row in completions]
        masked = [not ids or ids[-1] not in self.eos for ids in completion_ids]
        length_stops = [len(ids) >= self.cap and mask for ids, mask in zip(completion_ids, masked, strict=True)]
        formats = [text.count('\n</think>\n') == 1 for text in texts]
        correct = [False] * len(texts)
        adjusted = [None] * len(texts)
        diagnostic_rows = []
        signal_groups = mixed_groups = 0
        groups = sample_id[::self.group]
        if not groups or sample_id != [sid for sid in groups for _ in range(self.group)]:
            raise ValueError('Each MMR group must contain exactly six consecutive completions for one question')
        for start in range(0, len(texts), self.group):
            end = start + self.group
            usable = [i for i in range(start, end) if not masked[i]]
            eligible = usable if self.exclude_masked_completions else list(range(start, end))
            if not eligible and not self.diagnostics:
                continue
            base, values, order = [], np.array([], dtype=np.float32), []
            if eligible:
                base, values = self._score_group(eligible, texts, completion_ids, reference_answer[start],
                                                 sample_id[start], formats, length_stops, correct, order)
            baseline = float(values.mean()) if len(values) else None
            for i, value in zip(eligible, values.tolist(), strict=True):
                adjusted[i] = value
            signal_groups += any(adjusted[i] != baseline for i in usable)
            mixed_groups += len({correct[i] for i in usable}) == 2
            if self.diagnostics:
                base_by_index = dict(zip(eligible, base, strict=True))
                ranks = {eligible[i]: rank for rank, i in enumerate(order, 1)}
                for i in range(start, end):
                    diagnostic_rows.append({
                        'trl_clipped': masked[i], 'valid_for_mmr': i in base_by_index,
                        'base_reward': base_by_index.get(i), 'adjusted_reward': adjusted[i],
                        'group_mean': baseline, 'mmr_rank': ranks.get(i),
                        'reward_status': 'masked' if masked[i] else ('correct' if correct[i] else 'wrong'),
                    })
        if self.diagnostics:
            self.diagnostics.record_mmr(kwargs['trainer_state'].global_step + 1, sample_id,
                                        completion_ids, texts, diagnostic_rows)
        for key, value in {
            'reward/format_frac': sum(formats) / len(texts), 'reward/correct_frac': sum(correct) / len(texts),
            'comparison/completions': len(texts), 'comparison/groups': len(groups),
            'comparison/groups_with_reward_signal': signal_groups,
            'comparison/mixed_unmasked_reward_groups': mixed_groups,
            'comparison/length_stops': sum(length_stops),
        }.items():
            log_metric(key, value)
        return adjusted

    def _score_group(self, eligible, texts, completion_ids, reference, sid, formats, length_stops, correct, order):
        gold = _parse_gold(self.mv, clean_reference(reference), 5)
        if not gold:
            raise ValueError(f'MMR cannot parse the reference for {sid}')
        for i in eligible:
            answer = [] if length_stops[i] else self.mv.parse(
                texts[i], extraction_mode='first_match', extraction_config=[self.answer_extraction],
                raise_on_error=True)
            correct[i] = bool(answer and self.mv.verify(gold, answer, raise_on_error=True))
        base = [float(formats[i]) + 2 * cosine_reward(correct[i], len(completion_ids[i]), self.cap)
                for i in eligible]
        if len(base) == 1:
            order.append(0)
            return base, np.asarray(base, dtype=np.float32)
        return base, reweight_group(base, self.embed([texts[i] for i in eligible]), order=order)


def check_local(cfg, load_pool, make_args):
    from transformers import AutoTokenizer

    data = Path(__file__).resolve().parents[2] / 'data'
    pool, _ = load_pool(cfg, data)
    tok = AutoTokenizer.from_pretrained(data / 'tokenizers' / 'Qwen2.5-7B-Instruct', local_files_only=True)
    args = make_args(cfg, data / cfg['run_name'], smoke=True, max_steps=None)
    steps = args.max_steps
    questions = args.generation_batch_size // args.num_generations
    prompts = tok.apply_chat_template([training_messages(row['question']) for row in pool], tokenize=True,
                                     return_dict=False, add_generation_prompt=True, chat_template=training_template(tok))
    mv = _math_verify()
    failures = [row['sample_id'] for row in pool if not _parse_gold(mv, clean_reference(row['reference_answer']), 5)]
    if failures:
        raise ValueError(f'Unparseable references in the frozen draw: {failures}')
    truncated = sum(len(ids) > cfg['max_prompt_tokens'] for ids in prompts)
    print(f'OPENR1_OK rows={len(pool)} selected={len(pool)} parseable={len(pool)} order=frozen '
          f'prompt_truncated={truncated}')
    print(f'MMR_CONFIG_OK groups={questions} samples_per_group={args.num_generations} '
          f'rollouts={args.generation_batch_size} completion_tokens={args.max_completion_length} '
          f'scheduler_steps={cfg["mmr_dr_grpo"]["scheduler_steps"]} warmup_steps={args.warmup_steps} '
          f'final_step={steps} beta={args.beta} loss={args.loss_type}')
