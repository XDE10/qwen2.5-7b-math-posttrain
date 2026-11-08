"""Replace the six already excluded s1 rows from the existing approved raw pool.

CPU preparation only. Never launches training or calls a teacher API. Existing
data and approval manifests remain unchanged. Retain the original s1 row order.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/romescape_reproduction_20261028_v57'
SYSTEM = r'Please reason step by step, and put your final answer within \boxed{}.'
# (removed s1 index, existing approved raw-pool row, category, checked gold basis)
REPLACEMENTS = [
    (68, 26, 'Algebra/logarithmic equations', 'Put u=sqrt(2x-7)>=0; (u+1)^2=u+7 gives u=2 and x=11/2.'),
    (104, 772, 'Algebra/polynomial roots', 'P=(x-r)(x-s)^2, r+2s=-4; P(-2s)=36s^2=324 gives s=+/-3, c=90 or -18; sum |c|=108.'),
    (131, 71, 'Algebra/positive real equation systems', 'u=sqrt(a),v=sqrt(b); (u+v)^3=134+3*126=512, uv=126/8; u=9/2,v=7/2.'),
    (136, 934, 'Algebra/complex numbers', 'Conjugates a+bi,a-bi; b^2=3 and imaginary part of (a+bi)^3 vanishes, so a^2=1 and modulus=2.'),
    (157, 693, 'Algebra/function iteration', 'f(x)=1/(1-x) has f^3(x)=x on this orbit; 2005 mod 3=1, so result=-1/2004.'),
    (505, 592, 'Combinatorics/probability with country groups', 'C(9,3)=84, all same=5, all different=24; exactly two same=55/84; 55+84=139.'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def normalize(text):
    return ''.join(text.split())


def wordset(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    import pandas as pd
    from tokenizers import Tokenizer
    source = ROOT / 'artifacts/research_s1_7b/s1K-1.1.parquet'
    raw_path = ROOT / 'data/portable_release/project1_raw_paired.jsonl'
    old_manifest = ROOT / 'data/portable_release/manifest.json'
    manifest = json.loads(old_manifest.read_text(encoding='utf-8'))
    assert sha(raw_path) == manifest['files'][raw_path.name]
    frame = pd.read_parquet(source)
    raw = rows(raw_path)
    assert len(frame) == 1000 and len(raw) == 1500
    checks = []
    for name in ['math_dev.jsonl', 'math500_test.jsonl', 'gsm8k_test.jsonl']:
        checks.extend((name, r.get('sample_id', r.get('id')), r['problem']) for r in rows(ROOT / 'data/math' / name))
    checks.extend(('s1K-1.1', str(i), str(r['question'])) for i, r in frame.iterrows())
    protected = {normalize(q) for _, _, q in checks}
    tok = Tokenizer.from_file(str(ROOT / 'data/tokenizers/Qwen2.5-7B-Instruct/tokenizer.json'))
    records, audit = [], []
    replacements = {}
    for removed, index, category, basis in REPLACEMENTS:
        r = raw[index]
        assert r['training_approved'] is True and r['split'] == 'train' and r['arm'] == 'raw'
        assert r['source_position']['complete'] is True
        assert normalize(r['problem']) not in protected
        words = wordset(r['problem'])
        nearest = max(((len(words & wordset(q)) / max(1, len(words | wordset(q))), pool, sid)
                       for pool, sid, q in checks), key=lambda x: x[0])
        assert nearest[0] < 0.85, (r['sample_id'], nearest)
        solution = r['solution']
        assert solution.startswith('<think>') and solution.count('</think>') == 1
        cut = solution.index('</think>')
        thinking, attempt = solution[len('<think>'):cut], solution[cut + len('</think>'):]
        assert thinking.strip() and attempt.strip()
        replacement = {'sample_id': 'romescape-topup/' + r['sample_id'], 'question': r['problem'],
                       'deepseek_thinking_trajectory': thinking, 'deepseek_attempt': attempt,
                       'reference_answer': r['reference_answer'], 'source': 'existing approved OpenR1 raw pool',
                       'source_sample_id': r['sample_id'], 'replaces_s1_row': removed}
        replacements[removed] = replacement
        audit.append({'replaced_s1_row': removed, 'old_question': str(frame.iloc[removed]['question']),
                      'new_sample_id': replacement['sample_id'], 'new_problem': r['problem'],
                      'category': category, 'reference_answer': r['reference_answer'],
                      'source_row': index, 'source_revision': r['source_revision'],
                      'source_position': r['source_position'], 'old_approval_method': r['review_method'],
                      'independent_answer_check': basis, 'nearest_question_word_jaccard': nearest,
                      'format_change': 'Remove the outer XML think delimiters only; retain reasoning and final text verbatim, separated by the s1 answer delimiter.'})
    for i, r in frame.iterrows():
        if i in replacements:
            row = replacements[i]
        else:
            row = {'sample_id': f's1k11/row/{i:04d}', 'question': str(r['question']),
                   'deepseek_thinking_trajectory': str(r['deepseek_thinking_trajectory']),
                   'deepseek_attempt': str(r['deepseek_attempt']), 'source': 's1K-1.1', 'original_s1_row': i}
        prompt = ('<|im_start|>system\n' + SYSTEM + '<|im_end|>\n<|im_start|>user\n' +
                  row['question'] + '<|im_end|>\n<|im_start|>assistant\n')
        response = row['deepseek_thinking_trajectory'] + '<|im_start|>answer\n' + row['deepseek_attempt']
        a = tok.encode(prompt, add_special_tokens=False).ids
        b = tok.encode(response + '<|im_end|>\n', add_special_tokens=False).ids
        assert len(a) * 2 < 16384
        target = b[:16384-len(a)]
        records.append({**row, 'input_ids': a+target, 'labels': [-100]*len(a)+target,
                        'prompt_tokens': len(a), 'original_tokens': len(a)+len(b),
                        'truncated_at_16384': len(a)+len(b)>16384})
    assert len({r['sample_id'] for r in records}) == 1000
    OUT.mkdir(exist_ok=True, parents=True)
    data_path = OUT / 'train1000_prepared.jsonl'
    data_path.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n' for r in records), encoding='utf-8')
    save(OUT / 'replacement6_checks.json', audit)
    # Check the final original teacher answer text for these six, not just their references.
    spec = importlib.util.spec_from_file_location('judge_v57', ROOT/'tools/evaluate_sft_ds_v29.py')
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    grades = []
    for row in replacements.values():
        grade = judge.grade_bounded(row['deepseek_attempt'], str(row['reference_answer']), ROOT)
        assert grade['correct'], (row['sample_id'], grade)
        grades.append({'sample_id': row['sample_id'], 'grade': grade})
    save(OUT/'replacement6_native_checks.json', grades)
    result = {'stage': 'CPU data preparation; no training launched', 'count': 1000,
              's1_retained': 994, 'approved_raw_pool_replacements': 6,
              'dataset_revision': '96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d',
              's1_source_sha256': sha(source), 'raw_pool_sha256': sha(raw_path),
              'prepared_file': data_path.name, 'prepared_sha256': sha(data_path),
              'same_s1_positions_replaced': sorted(replacements),
              'token_sum_after_cutoff': sum(len(r['input_ids']) for r in records),
              'truncated_at_16384': sum(r['truncated_at_16384'] for r in records),
              'updates_per_epoch': 125, 'epochs_proposed': 6, 'updates_proposed': 750, 'warmup_steps_proposed': 75,
              'source_approval_note': 'Six existing raw rows carry prior itemwise training approval; latest user requests matching-category replacement. Old manifests untouched.',
              'formal_training_launcher_ready': False, 'training_executed': False,
              'checks': 'Only six new rows checked against s1, dev, MATH500, GSM8K; no new global audit or API generation.'}
    save(OUT/'prepared_data_manifest.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
