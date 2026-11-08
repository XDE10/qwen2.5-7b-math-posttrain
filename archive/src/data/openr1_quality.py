"""Conservative noise signals, not a proof of coherent or correct reasoning."""
import re
from collections import Counter

RULE_VERSION = 'openr1-noise-v1'


def quality_signals(text):
    reject, review = [], []
    if '\ufffd' in text or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        reject.append('replacement_or_surrogate_character')
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
        reject.append('unexpected_control_character')
    if re.search(r'<\|(?:im_start|im_end|endoftext)\|>', text):
        reject.append('embedded_chat_control_token')
    lines = [' '.join(x.split()) for x in text.splitlines() if x.strip()]
    for line, count in Counter(lines).items():
        if len(line) >= 80 and count >= 4:
            reject.append('long_line_repeated_four_times')
            break
    # Detect periodic terminal words; require actual linguistic content, so a
    # repeated equality symbol or aligned TeX environment is not mistaken for prose.
    words = text[-6000:].split()
    for width in (8, 16, 32, 64):
        for end in range(len(words), max(0, len(words)-16), -1):
            if end < width*4:
                continue
            block = words[end-width:end]
            if len(set(block)) < 3 or sum(c.isalpha() for c in ' '.join(block)) < 16:
                continue
            if all(words[end-width*(i+1):end-width*i] == block for i in range(1, 4)):
                reject.append('terminal_periodic_word_loop')
                break
        if 'terminal_periodic_word_loop' in reject:
            break
    if re.search(r'([^\w\s])\1{127,}', text):
        reject.append('symbol_run_128')
    if len(text) >= 500:
        meaningful = sum(c.isalnum() for c in text)
        if meaningful/len(text) < .03:
            reject.append('almost_all_symbols')
    if sum(text.count(x) for x in ('Ã', 'Â', 'â€', '锟斤拷')) >= 3:
        review.append('possible_mojibake')
    if len(re.findall(r'\b(?:wait|let me reconsider|start over|I am confused)\b', text, re.I)) >= 10:
        review.append('many_restarts_not_proven_incoherent')
    if len(re.findall(r'\b(?:not sure|cannot solve|unable to solve|give up)\b', text, re.I)) >= 2:
        review.append('uncertainty_or_abandonment')
    return {'reject': sorted(set(reject)), 'review': sorted(set(review)),
            'automatic_pass': not reject and not review}
