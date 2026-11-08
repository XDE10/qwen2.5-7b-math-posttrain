"""Pinned project-two prompt and student-rollout input contract, not a trainer."""
import json
from pathlib import Path

CONFIG = Path('project_blueprints/configs/project2_prompt_contract.json')


def load_contract(path=CONFIG):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def render_student_prompt(tokenizer, problem, domain, contract=None):
    contract = contract or load_contract()
    if domain not in {'math', 'code'} or not isinstance(problem, str) or not problem.strip():
        raise ValueError('Expected nonempty math/code problem')
    system = contract[domain + '_system']
    messages = ([{'role': 'system', 'content': system}] if system is not None else [])
    messages.append({'role': 'user', 'content': problem})
    # No teacher tokenizer is accepted by this interface.
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(ids) + contract[domain + '_max_new_tokens'] > contract['context_limit']:
        raise ValueError('Full prompt exceeds reserved generation budget; never truncate question')
    return ids


def teacher_batch_from_student(input_ids, attention_mask, prompt_lengths, expected_prefixes, common_support_size):
    """Pass tensors by identity; verify prefixes and return a TARGET-position mask.

    No packing. Supports left/right padding. Teacher receives exactly these
    tensors on its device, without an intervening decode/tokenize round trip.
    """
    import torch
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError('Expected matching [batch, sequence] tensors')
    if input_ids.dtype != torch.long or not ((attention_mask == 0) | (attention_mask == 1)).all():
        raise ValueError('Invalid IDs or attention mask')
    if len(prompt_lengths) != len(input_ids) or len(expected_prefixes) != len(input_ids):
        raise ValueError('Missing per-row prefix evidence')
    response_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for i, (n, prefix) in enumerate(zip(prompt_lengths, expected_prefixes)):
        positions = attention_mask[i].bool().nonzero().flatten()
        if not len(positions) or int(positions[-1]-positions[0]+1) != len(positions):
            raise ValueError('Packed or discontinuous attention is unsupported')
        valid = input_ids[i, positions]
        if n != len(prefix) or n < 1 or n >= len(valid):
            raise ValueError('Empty response or incorrect prefix length')
        if valid[:n].tolist() != list(prefix):
            raise ValueError('Teacher input is not the exact student prefix')
        if (valid < 0).any() or (valid >= common_support_size).any():
            raise ValueError('Sampled IDs outside audited common support')
        response_mask[i, positions[n:]] = True
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, response_mask
