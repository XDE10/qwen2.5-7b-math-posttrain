"""MMR-DR-GRPO algorithm components: Jina ONNX embeddings, Algorithm 1 reweighting, cosine reward, and trainer."""
from __future__ import annotations

from math import cos, pi

import numpy as np
from trl import GRPOTrainer

from math_posttrain.grading import _math_verify, _parse_gold, clean_reference

SYSTEM_PROMPT = (
    'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
    'The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, '
    'and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within '
    '<think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> '
    '<answer> answer here </answer>. Note that respond by English, NOT use other languages.'
)


def training_messages(question: str) -> list[dict]:
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': question}]


def training_template(tokenizer) -> str:
    return tokenizer.chat_template + "{% if add_generation_prompt %}{{ '<think>\\n' }}{% endif %}"


class MmrDRGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, prompt_limit: int, scheduler_steps: int, **kwargs):
        self.prompt_limit = prompt_limit
        self.scheduler_steps = scheduler_steps
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError('MMR-DR-GRPO requires complete prompt groups on a single process')
        self.chat_template = training_template(self.processing_class)

    def create_scheduler(self, num_training_steps, optimizer=None):
        return super().create_scheduler(self.scheduler_steps, optimizer=optimizer)

    def _tokenize_prompts(self, prompts):
        ids, images, fields = super()._tokenize_prompts(prompts)
        self._log_metric('mmr/prompt_truncated_frac', sum(len(row) > self.prompt_limit for row in ids) / len(ids))
        return [row[-self.prompt_limit:] for row in ids], images, fields

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        ids, logprobs, sampling_mask = super()._generate_single_turn(
            prompt_ids, images, multimodal_fields, num_generations, has_tool_images,
        )
        if self.reward_funcs[0].exclude_masked_completions and any(not row for row in ids):
            endings = {*self.eos_token_ids, self._tokenizer.pad_token_id}
            token = next(t for t in self._tokenizer.get_vocab().values() if t not in endings)
            for index, row in enumerate(ids):
                if not row:
                    ids[index] = [token]
                    if logprobs is not None:
                        logprobs[index] = [0.0]
                    if sampling_mask is not None:
                        sampling_mask[index] = [0]
        return ids, logprobs, sampling_mask

    def _compute_loss(self, model, inputs):
        if 'old_per_token_logps' in inputs:
            inputs = {k: v for k, v in inputs.items() if k != 'old_per_token_logps'}
        return super()._compute_loss(model, inputs)


class JinaEmbeddings:
    """CPU ONNX mean-pooled Jina embeddings used for MMR group diversity reweighting."""

    def __init__(self, cfg: dict):
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        model = cfg['embedding_model']
        revision = cfg.get('embedding_revision')
        path = hf_hub_download(model, 'model-w-mean-pooling.onnx', revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.max_length = cfg['embedding_max_length']

    def __call__(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts, truncation=True, max_length=self.max_length)
        embeddings = []
        for index in range(len(texts)):
            inputs = {name: np.asarray([tokens[name][index]], dtype=np.int64) for name in self.input_names}
            embeddings.append(self.session.run(None, inputs)[0][0])
        return np.stack(embeddings)


def reweight_group(rewards, embeddings, *, order: list[int] | None = None) -> np.ndarray:
    """MMR-GRPO Algorithm 1: diversity-penalized greedy reweighting within one rollout group."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(rewards) or not np.isfinite(embeddings).all():
        raise ValueError('MMR requires one finite embedding per completion')
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


def cosine_reward(correct: bool, length: int, cap: int) -> float:
    """Cosine length-shaped correctness reward: correct in [0.5, 1.0], wrong in [-0.5, 0.0]."""
    mix = (1 + cos(pi * min(length, cap) / cap)) / 2
    return 0.5 + 0.5 * mix if correct else -0.5 + 0.5 * mix


class MmrReward:
    __name__ = 'mmr_format_cosine'

    def __init__(self, tokenizer, cfg: dict, embed):
        from latex2sympy2_extended import NormalizationConfig

        self.embed = embed
        self.cap = cfg['max_completion_length']
        self.group = cfg['num_generations']
        self.exclude_masked_completions = cfg['mmr_dr_grpo'].get('exclude_masked_completions', False)
        if self.exclude_masked_completions and not cfg.get('mask_truncated_completions', True):
            raise ValueError('MMR exclusion requires mask_truncated_completions=True')
        self.eos = {tokenizer.eos_token_id, tokenizer.pad_token_id}
        self.mv = _math_verify()
        self.answer_extraction = self.mv.LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False, malformed_operators=False, basic_latex=True, equations=True, boxed='all', units=True,
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )

    def __call__(self, completions, completion_ids, reference_answer, sample_id, log_metric=None, **kwargs):
        texts = [row[0]['content'] for row in completions]
        masked = [not ids or ids[-1] not in self.eos for ids in completion_ids]
        length_stops = [len(ids) >= self.cap and mask for ids, mask in zip(completion_ids, masked, strict=True)]
        formats = [text.count('\n</think>\n') == 1 for text in texts]
        correct = [False] * len(texts)
        adjusted = [None] * len(texts)
        groups = sample_id[::self.group]
        if not groups or sample_id != [sid for sid in groups for _ in range(self.group)]:
            raise ValueError(f'Each MMR group must contain {self.group} consecutive completions for one prompt')
        for start in range(0, len(texts), self.group):
            end = start + self.group
            usable = [i for i in range(start, end) if not masked[i]]
            eligible = usable if self.exclude_masked_completions else list(range(start, end))
            if not eligible:
                continue
            values = self._score_group(
                eligible, texts, completion_ids, reference_answer[start], sample_id[start],
                formats, length_stops, correct,
            )
            for i, value in zip(eligible, values.tolist(), strict=True):
                adjusted[i] = value
        if callable(log_metric):
            log_metric('reward/format_frac', sum(formats) / len(texts))
            log_metric('reward/correct_frac', sum(correct) / len(texts))
            log_metric('reward/length_stops', sum(length_stops))
        return adjusted

    def _score_group(self, eligible, texts, completion_ids, reference, sid, formats, length_stops, correct):
        gold = _parse_gold(self.mv, clean_reference(reference), 5)
        if not gold:
            raise ValueError(f'MMR cannot parse the reference for {sid}')
        for i in eligible:
            answer = [] if length_stops[i] else self.mv.parse(
                texts[i], extraction_mode='first_match', extraction_config=[self.answer_extraction],
                raise_on_error=True,
            )
            correct[i] = bool(answer and self.mv.verify(gold, answer, raise_on_error=True))
        base = [
            float(formats[i]) + 2 * cosine_reward(correct[i], len(completion_ids[i]), self.cap)
            for i in eligible
        ]
        if len(base) == 1:
            return np.asarray(base, dtype=np.float32)
        return reweight_group(base, self.embed([texts[i] for i in eligible]))
