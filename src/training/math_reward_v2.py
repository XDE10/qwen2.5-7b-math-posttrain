"""Auditable statuses. Infrastructure failures abort, never become wrong answers."""
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ('equivalent', 'mismatch', 'unsupported_relation', 'multiple_boxes', 'unclosed_box', 'parse_failure', 'timeout', 'verifier_error')


def category(result):
    status = result.get('status', '')
    if status in ('equivalent', 'mismatch', 'unclosed_box', 'timeout'):
        return status
    if status.startswith('relational_answer'):
        return 'unsupported_relation'
    if status.startswith('multiple_distinct_boxes'):
        return 'multiple_boxes'
    if status in ('unparsed_reference', 'unparsed_prediction', 'missing_box'):
        return 'parse_failure'
    return 'verifier_error'


def check_math(text, answer):
    try:
        p = subprocess.run([sys.executable, '-m', 'scripts.math_verify_v3_strict_worker'], cwd=ROOT,
                           input=json.dumps({'output': text, 'reference': str(answer)}),
                           text=True, encoding='utf-8', capture_output=True, timeout=20)
        if p.returncode:
            return {'status': 'verifier_error', 'returncode': p.returncode, 'stderr': p.stderr[-2000:]}
        return json.loads(p.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}
    except Exception as error:
        return {'status': 'verifier_error', 'error_type': type(error).__name__}


class AuditedMathReward:
    __name__ = 'audited_math_v2'

    def __init__(self, output_dir, checker=check_math):
        self.path = Path(output_dir) / 'reward_status.jsonl'
        self.checker = checker
        self.trainer = None

    def __call__(self, completions, reference_answer, prompts=None, **kwargs):
        texts = [c[-1]['content'] if isinstance(c, list) else c for c in completions]
        if len(texts) != len(reference_answer):
            raise ValueError('Reward batch alignment mismatch')
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda p: self.checker(*p), zip(texts, reference_answer)))
        counts = Counter(category(r) for r in results)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        step = self.trainer.state.global_step if self.trainer else None
        record = {'step': step, 'counts': {k: counts[k] for k in CATEGORIES}, 'results': results}
        if self.trainer:
            tok = self.trainer.processing_class
            def length(text):
                return len(tok.encode(text, add_special_tokens=False))
            record['completion_effective_lengths'] = [length(t) for t in texts]
            if prompts:
                record['prompt_effective_lengths'] = [len(tok.apply_chat_template(p, tokenize=True, add_generation_prompt=True)) if isinstance(p, list) else length(p) for p in prompts]
            mode = 'train' if self.trainer.model.training else 'eval'
            for k in CATEGORIES:
                self.trainer._metrics[mode]['reward_status/' + k].append(counts[k])
        print(json.dumps(record, ensure_ascii=False))
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        if counts['timeout'] or counts['verifier_error']:
            raise RuntimeError('Verifier infrastructure failure; logged and aborted, not scored incorrect')
        return [1.0 if category(r) == 'equivalent' else 0.0 for r in results]
