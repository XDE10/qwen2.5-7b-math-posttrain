"""Read-only handover audit. Run from project root; no model/network/training."""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.approved_training_inputs import load_approved, model_view


def fingerprint(text):
    # Exact normalization from src/data/prepare_math_data.py:21-28;
    # keep this read-only audit standard-library-only (that module imports SymPy).
    text = unicodedata.normalize('NFKC', text).casefold()
    text = text.replace('\\left', '').replace('\\right', '').replace('\\!', '')
    return hashlib.sha256(' '.join(re.findall(r'\w+|[^\w\s]', text)).encode()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    release = ROOT / 'data/portable_release'
    manifest = read(release / 'manifest.json')
    arms = {a: load_approved(release / 'manifest.json', f'project1_{a}_paired.jsonl') for a in ['raw', 'compact']}
    a, b = arms['raw'], arms['compact']
    by_id = {r['pair_id']: r for r in b}
    protected = set(read(release / manifest['protected_dev_ids_file'])['sample_ids'])
    dev = rows(ROOT / 'data/math/math_dev.jsonl')
    candidates = rows(release / 'project1_rl_priority_not_rollout_selected.jsonl')
    dev_hashes = {fingerprint(r['problem']) for r in dev}
    test_hashes = {fingerprint(r['problem']) for r in rows(ROOT / 'data/math/math500_test.jsonl') + rows(ROOT / 'data/math/gsm8k_test.jsonl')}
    result = {
        'python': sys.executable, 'python_version': sys.version,
        'approved_training_files': manifest['approved_training_files'],
        'approved_evaluation_files': manifest['approved_evaluation_files'],
        'release_digests': {f: sha(release / f) == h for f, h in manifest['files'].items()},
        'pair_count': len(a), 'unique_pairs': {k: len({r['pair_id'] for r in v}) for k, v in arms.items()},
        'same_pair_order': [r['pair_id'] for r in a] == [r['pair_id'] for r in b],
        'pair_field_mismatches': [r['pair_id'] for r in a if r['pair_id'] not in by_id or any(r[k] != by_id[r['pair_id']][k] for k in ['sample_id', 'problem', 'reference_answer', 'split', 'problem_type'])],
        'types': dict(Counter(r['problem_type'] for r in a)),
        'protected_ids': len(protected), 'dev_rows': len(dev),
        'dev_ids_not_protected': [r['sample_id'] for r in dev if r['sample_id'] not in protected],
        'candidate_rows': len(candidates),
        'candidate_is_approved_file': 'project1_rl_priority_not_rollout_selected.jsonl' in manifest['approved_training_files'],
        'overlap_scope': 'IDs and normalized exact problem hash only; no semantic/near-duplicate absence claim',
        'overlap': {}, 'arms': {},
    }
    for name, data in {**arms, 'rl_candidates': candidates}.items():
        result['overlap'][name] = {
            'protected_sample_ids': [r['sample_id'] for r in data if r['sample_id'] in protected],
            'dev_normalized_exact': [r['sample_id'] for r in data if fingerprint(r['problem']) in dev_hashes],
            'test_normalized_exact': [r['sample_id'] for r in data if fingerprint(r['problem']) in test_hashes],
        }
    math_manifest = read(ROOT / 'data/math_manifest.json')
    result['evaluation_file_identity'] = {k: {'rows': len(rows(ROOT / math_manifest['files'][k]['path'])), 'sha_matches': sha(ROOT / math_manifest['files'][k]['path']) == math_manifest['files'][k]['sha256']} for k in ['math_dev', 'math500_test', 'gsm8k_test']}
    result['configs'] = {p.name: read(p) for p in sorted((ROOT / 'configs').glob('*.json'))}
    for arm, data in arms.items():
        result['arms'][arm] = {'rows': len(data), 'all_train': all(r['split'] == 'train' and r['training_approved'] is True for r in data), 'human_reviewed': sum(r.get('itemwise_human_review_performed') is True for r in data), 'model_view_keys': list(model_view(data[0], 'math_sft')['input']), 'relation_references': sum(bool(re.search(r'[<>≤≥≠=]|\\(?:leq?|geq?|neq?|lt|gt)\b', str(r['reference_answer']))) for r in data)}
    if args.tokens:
        from transformers import AutoTokenizer
        from src.training.approved_hf import encode_sft, CompletionCollator, check_math
        tok = AutoTokenizer.from_pretrained(str(ROOT / 'data/tokenizers/Qwen2.5-7B-Instruct'), local_files_only=True)
        result['tokenizer'] = {'eos': tok.eos_token_id, 'pad': tok.pad_token_id, 'files': {p.name: sha(p) for p in (ROOT / 'data/tokenizers/Qwen2.5-7B-Instruct').glob('*') if p.is_file()}}
        for arm, data in arms.items():
            limit = result['configs'][f'project1_{arm}_sft_smoke.json']['max_length']
            enc = []
            violations = []
            for row in data:
                messages = [{'role':'system','content':r'Please reason step by step, and put your final answer within \boxed{}.'}, {'role':'user','content':row['problem']}]
                prefix = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                full = tok.apply_chat_template(messages + [{'role':'assistant','content':row['solution']}], tokenize=True, add_generation_prompt=False)
                result['tokenizer']['apply_chat_template_return_type'] = type(full).__name__
                # Transformers 5 returns BatchEncoding by default. Normalize only
                # this diagnostic; the unchanged production entry is probed below.
                if hasattr(prefix, 'keys'):
                    prefix = prefix['input_ids']
                if hasattr(full, 'keys'):
                    full = full['input_ids']
                if full[:len(prefix)] != prefix or len(full) > limit or len(full) <= len(prefix):
                    violations.append({'sample_id':row['sample_id'], 'total':len(full), 'prefix':len(prefix), 'prefix_matches':full[:len(prefix)] == prefix})
                enc.append({'input_ids':full, 'labels':[-100]*len(prefix)+full[len(prefix):]})
            try:
                encode_sft(data, tok, limit)
                entry_result = 'passed'
            except ValueError as e:
                entry_result = str(e)
            result['arms'][arm].update(entry_encode_sft=entry_result, violations=violations)
            class ListTemplateAPI:
                def apply_chat_template(self, *args, **kwargs):
                    return tok.apply_chat_template(*args, **kwargs, return_dict=False)
            assert encode_sft(data, ListTemplateAPI(), limit) == enc
            result['arms'][arm]['encode_with_explicit_list_api'] = 'passed; audit wrapper only, production unchanged'
            lengths = [len(r['input_ids']) for r in enc]
            supervised = [sum(x != -100 for x in r['labels']) for r in enc]
            longest = max(range(len(enc)), key=lambda i: lengths[i])
            shortest = min(range(len(enc)), key=lambda i: lengths[i])
            batch = CompletionCollator(tok.pad_token_id)([enc[longest], enc[shortest]])
            assert (batch['labels'][batch['attention_mask'] == 0] == -100).all()
            assert all(tok.eos_token_id in e['labels'] for e in enc)
            result['arms'][arm].update(max_total_tokens=max(lengths), max_supervised_tokens=max(supervised), total_sequence_tokens=sum(lengths), total_supervised_tokens=sum(supervised), max_prompt_tokens=max(n-s for n,s in zip(lengths,supervised)), max_length=limit, longest_id=data[longest]['sample_id'], longest_tail=enc[longest]['input_ids'][-5:], longest_prompt_tokens=lengths[longest]-supervised[longest], batch_shape=list(batch['input_ids'].shape), eos_supervised_all=True, count_over_3800=sum(x > 3800 for x in lengths))
        probes = [(r'\boxed{2}', '2'), (r'\boxed{3}', '2'), ('2', '2'), (r'\boxed{x>1}', 'x>1'), (r'\boxed{1}\boxed{2}', '1'), (r'\boxed{', '2')]
        result['grader_probes'] = [{'output': x, 'reference': y, 'result': check_math(x, y)} for x, y in probes]
    tracked = ['AGENTS.md', 'README.md', 'src/training/approved_hf.py', 'src/data/approved_training_inputs.py', 'src/losses/reference.py', 'scripts/evaluate_approved_math.py', 'scripts/math_verify_v3_strict_worker.py', 'data/portable_release/manifest.json']
    result['source_sha256'] = {p: sha(ROOT / p) for p in tracked}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': args.output, 'pair_count': len(a), 'tokens': args.tokens}))


if __name__ == '__main__':
    main()
