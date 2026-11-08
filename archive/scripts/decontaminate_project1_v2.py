"""No download, no deletion, no approval changes. Output hit lists only."""
import argparse
import json
from collections import Counter
from pathlib import Path
from src.data.decontamination_v2 import Index, calibrate, exact
from src.training.guards_v2 import sha


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--compat-output')
    args = p.parse_args()
    paths = {'sft': 'data/portable_release/project1_raw_paired.jsonl',
             'rl': 'data/portable_release/project1_rl_priority_not_rollout_selected.jsonl',
             'dev': 'data/math/math_dev.jsonl', 'math500': 'data/math/math500_test.jsonl',
             'gsm8k': 'data/math/gsm8k_test.jsonl'}
    missing = [path for path in paths.values() if not Path(path).is_file()]
    if missing:
        Path(args.output).write_text(json.dumps({'missing': missing, 'status': '未验证'}, ensure_ascii=False, indent=2), encoding='utf-8')
        return
    data = {k: read_rows(v) for k, v in paths.items()}
    compact = read_rows('data/portable_release/project1_compact_paired.jsonl')
    calibration = calibrate(data['sft'],compact)
    threshold = calibration['threshold']
    checks = [('sft', t) for t in ('dev', 'math500', 'gsm8k', 'rl')] + [('rl', t) for t in ('dev', 'math500', 'gsm8k')]
    indexes = {key: Index(value) for key, value in data.items() if key != 'sft'}
    hits = []
    for source, target in checks:
        for row in data[source]:
            for i, similarity in indexes[target].query(row['problem'], threshold):
                other = data[target][i]
                hits.append({'source': source, 'target': target, 'source_id': row['sample_id'], 'target_id': other['sample_id'],
                             'normalized_exact': exact(row['problem']) == exact(other['problem']), 'char5_jaccard': similarity,
                             'interpretation': 'manual_review_required_not_automatic_deletion'})
        print(source, target, 'completed', flush=True)
    sft_hit = {r['source_id'] for r in hits if r['source'] == 'sft' and r['target'] != 'rl'}
    overlap_sft_rl = {r['source_id'] for r in hits if r['source'] == 'sft' and r['target'] == 'rl'}
    embedding_inventory = []
    cache = Path.home()/'.cache/huggingface/hub'
    for name in ('models--sentence-transformers--all-MiniLM-L6-v2','models--BAAI--bge-small-zh-v1.5'):
        for snapshot in (cache/name/'snapshots').glob('*'):
            weights = [str(p) for p in snapshot.rglob('*') if p.is_file() and (p.suffix in ('.safetensors','.bin','.onnx'))]
            embedding_inventory.append({'snapshot':str(snapshot),'weight_files':weights,'metadata_only':not weights})
    result = {'status': '前两层已执行；embedding未验证', 'inputs': {k: {'path': paths[k], 'rows': len(v), 'sha256': sha(paths[k])} for k, v in data.items()},
              'missing_files': missing, 'calibration': calibration, 'hits': hits,
              'counts': {s+'->'+t:sum(r['source']==s and r['target']==t for r in hits) for s,t in checks},
              'exact_counts': {s+'->'+t:sum(r['source']==s and r['target']==t and r['normalized_exact'] for r in hits) for s,t in checks},
              'embedding': {'status': '未验证', 'reason': 'Existing MiniLM/bge-small snapshots contain metadata/tokenizer but no weights; no downloads permitted', 'local_inventory':embedding_inventory},
              'hypothetical_heldout_exclusion_impact': {'paired_ids': sorted(sft_hit), 'pairs': len(sft_hit), 'rows_across_both_arms': 2*len(sft_hit),
                                                       'six_type_counts_at_risk': dict(Counter(r['problem_type'] for r in data['sft'] if r['sample_id'] in sft_hit))},
              'sft_rl_overlap': {'sft_ids': sorted(overlap_sft_rl), 'sft_count': len(overlap_sft_rl),
                                 'six_type_counts_at_risk_if_train_overlap_disallowed':dict(Counter(r['problem_type'] for r in data['sft'] if r['sample_id'] in overlap_sft_rl)),
                                 'pairing_effect':'If later excluded, both raw/compact rows move together in a new approved version; no action this round',
                                 'policy': 'train-to-train overlap allowed only if predeclared; report separately'},
              'rl_candidate_heldout_unique_ids':sorted({r['source_id'] for r in hits if r['source']=='rl'}),
              'rl_candidate_heldout_unique_count':len({r['source_id'] for r in hits if r['source']=='rl'}),
              'action_taken': 'None; sealed data and quotas unchanged; similarity alone is not contamination proof'}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.compat_output:
        Path(args.compat_output).write_bytes(Path(args.output).read_bytes())


if __name__ == '__main__':
    main()
