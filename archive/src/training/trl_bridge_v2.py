"""TRL 1.16 has no max_prompt_length; add enforced project budget fields.

This adapter only checks inputs/outputs; policy loss remains TRL's implementation.
"""
import json
from dataclasses import dataclass
from trl import GRPOConfig, GRPOTrainer


@dataclass
class ProjectGRPOConfig(GRPOConfig):
    max_prompt_length: int = 1752
    max_length: int = 3800

    def __post_init__(self):
        super().__post_init__()
        if self.max_length != self.max_prompt_length + self.max_completion_length:
            raise ValueError('Total budget must equal prompt + completion budget')


class ProjectGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs):
        for row in inputs:
            prompt = row['prompt']
            n = len(self._tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)) if isinstance(prompt, list) else len(self._tokenizer.encode(prompt, add_special_tokens=False))
            if n > self.args.max_prompt_length:
                raise ValueError('Prompt budget exceeded; no silent truncation')
        output = super()._generate_and_score_completions(inputs)
        print(json.dumps({'before_policy_loss': True, 'prompt_effective_lengths': output['prompt_mask'].sum(-1).tolist(),
                          'completion_loss_lengths': output['completion_mask'].sum(-1).tolist()}))
        if output['prompt_ids'].shape[-1] > self.args.max_prompt_length or output['completion_ids'].shape[-1] > self.max_completion_length:
            raise ValueError('Backend exceeded configured context budget')
        if not output['completion_mask'].any():
            raise ValueError('All completions masked; no effective policy tokens')
        return output
