"""Single source of truth for the v93 prompt, the s1/QFFT response layout and SFT/QFFT encodings.

Response layout (both teachers, both training formats)::

    <reasoning.strip()> "\\n<|im_start|>answer\\n" <final_answer.strip()>

This is byte-identical to the official QFFT S1 data (lwl-uestc/S1_QFFT@4a1b8ad, 1000/1000 rows
reproduce from s1K-1.1 with this join). The two training formats differ only in what surrounds it:

* SFT : Qwen chat prompt (system + user question) -> response + <|im_end|>; loss on the response only.
* QFFT: the response alone, no chat markup, no EOS (official `qwen_qfft` template); loss on every token.

Evaluation and RL always use the same chat prompt as SFT, for every model.
"""
from __future__ import annotations

SYSTEM_PROMPT = r'Please reason step by step, and put your final answer within \boxed{}.'
ANSWER_TAG = '<|im_start|>answer'
ANSWER_SEP = '\n' + ANSWER_TAG + '\n'
ASSISTANT_PREFIX = '<|im_start|>assistant\n'
IM_END = '<|im_end|>'
IGNORE_INDEX = -100


def canonical_response(reasoning: str, final: str) -> str:
    reasoning, final = (reasoning or '').strip(), (final or '').strip()
    if not reasoning or not final:
        raise ValueError('Both the reasoning and the final answer must be non-empty')
    if ANSWER_TAG in reasoning or ANSWER_TAG in final:
        raise ValueError('The answer separator may appear only once, between reasoning and final answer')
    return reasoning + ANSWER_SEP + final


def split_response(response: str):
    head, sep, tail = response.rpartition(ANSWER_SEP)
    if not sep:
        raise ValueError('Response has no answer separator')
    return head, tail


def messages(question: str) -> list:
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': question}]


def render_prompt(question: str) -> str:
    """Exactly what the Qwen2.5 chat template renders for [system, user] + generation prompt."""
    return (f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'
            f'<|im_start|>user\n{question}<|im_end|>\n' + ASSISTANT_PREFIX)


def token_ids(value) -> list:
    """Normalize list / dict / BatchEncoding / single-row nested outputs to a flat list of ints."""
    if hasattr(value, 'keys') and 'input_ids' in value:
        value = value['input_ids']
    value = list(value)
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not all(isinstance(i, int) and i >= 0 for i in value):
        raise TypeError('Expected a flat sequence of non-negative token ids')
    return value


def prompt_ids(tokenizer, question: str) -> list:
    """Chat-template prompt ids; fails loudly if a tokenizer renders anything but the frozen prompt."""
    ids = token_ids(tokenizer.apply_chat_template(messages(question), tokenize=True,
                                                  add_generation_prompt=True, return_dict=False))
    if tokenizer.decode(ids, skip_special_tokens=False) != render_prompt(question):
        raise ValueError('Chat template output differs from the frozen v93 prompt; check the tokenizer')
    return ids


def im_end_id(tokenizer) -> int:
    i = tokenizer.convert_tokens_to_ids(IM_END)
    if not isinstance(i, int) or i < 0 or tokenizer.convert_ids_to_tokens(i) != IM_END:
        raise ValueError('Tokenizer has no <|im_end|> token')
    return i


def response_ids(tokenizer, response: str) -> list:
    return token_ids(tokenizer(response, add_special_tokens=False)['input_ids'])


def encode_sft(tokenizer, question: str, response: str, max_length: int) -> dict:
    prefix = prompt_ids(tokenizer, question)
    target = response_ids(tokenizer, response) + [im_end_id(tokenizer)]
    ids = prefix + target
    if len(ids) > max_length:
        raise OverflowError(f'SFT example has {len(ids)} tokens > {max_length}; never truncated')
    return {'input_ids': ids, 'labels': [IGNORE_INDEX] * len(prefix) + target,
            'prompt_tokens': len(prefix), 'supervised_tokens': len(target)}


def encode_qfft(tokenizer, response: str, max_length: int) -> dict:
    ids = response_ids(tokenizer, response)
    if len(ids) > max_length:
        raise OverflowError(f'QFFT example has {len(ids)} tokens > {max_length}; never truncated')
    return {'input_ids': ids, 'labels': list(ids), 'prompt_tokens': 0, 'supervised_tokens': len(ids)}


def encode(tokenizer, fmt: str, question: str, response: str, max_length: int) -> dict:
    if fmt == 'sft':
        return encode_sft(tokenizer, question, response, max_length)
    if fmt == 'qfft':
        return encode_qfft(tokenizer, response, max_length)
    raise ValueError(f'Unknown training format: {fmt}')


def final_segment(text: str, strict: bool = False) -> str:
    """Text after the last answer tag (or </think>); empty when strict=True and neither tag exists."""
    i = text.rfind(ANSWER_TAG)
    if i >= 0:
        return text[i + len(ANSWER_TAG):].lstrip('\n')
    j = text.rfind('</think>')
    if j >= 0:
        return text[j + len('</think>'):].lstrip('\n')
    return "" if strict else text
