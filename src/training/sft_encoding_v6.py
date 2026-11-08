"""Explicit chat-template output contract; original encoders remain unchanged."""
from collections.abc import Mapping
from numbers import Integral

from src.training.approved_hf import encode_sft as _original_encode_sft


def token_ids(value):
    """Normalize one token sequence, rejecting batches and malformed IDs."""
    if isinstance(value, Mapping):
        if 'input_ids' not in value:
            raise ValueError('Chat template output has no input_ids')
        value = value['input_ids']
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError('Chat template must return a single token sequence')
    # No tensors are requested. Unexpected batch axes are deliberately rejected.
    if not value or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0
                        for x in value):
        raise ValueError('Chat template token IDs must be a nonempty flat integer sequence')
    return [int(x) for x in value]


class _TokenListTemplate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, **kwargs):
        # 4.x defaults to token IDs; 5.20.2 defaults to BatchEncoding. Set the
        # contract explicitly, without altering template, EOS, or content.
        kwargs.update(return_dict=False, return_tensors=None,
                      padding=False, truncation=False)
        return token_ids(self.tokenizer.apply_chat_template(messages, **kwargs))


def encode_sft(rows, tokenizer, max_length):
    """Use the original prefix/length/assistant-only mask checks on real IDs."""
    if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 1:
        raise ValueError('max_length must be a positive integer')
    return _original_encode_sft(rows, _TokenListTemplate(tokenizer), int(max_length))
