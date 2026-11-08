"""Summaries for fixed training-side hard probes; no model imports."""
from collections import Counter, defaultdict


def hard_summary(records, expected_ids, num_samples=4):
    groups = defaultdict(list)
    seen = set()
    for row in records:
        key = (row['sample_id'], row['trial'])
        if key in seen:
            raise ValueError('Duplicate hard-probe trial')
        seen.add(key)
        groups[row['sample_id']].append(row)
    expected = {(sid, t) for sid in expected_ids for t in range(num_samples)}
    if seen != expected:
        raise ValueError('Incomplete or foreign hard-probe trials')
    valid = Counter(); reasons = Counter(); any_correct = 0
    for rs in groups.values():
        any_correct += any(r['category'] == 'equivalent' for r in rs)
        if any(r['category'] not in ('equivalent', 'mismatch') for r in rs):
            reasons['unscorable_group'] += 1
        elif any(r['finish_reason'] != 'stop' for r in rs):
            reasons['truncated_or_aborted_group'] += 1
        else:
            valid[str(sum(r['category'] == 'equivalent' for r in rs))] += 1
    return {'candidate_count': len(expected_ids), 'trial_count': len(records), 'num_samples': num_samples,
            'questions_with_any_verified_correct': any_correct,
            'valid_group_correct_count_distribution': {str(k):valid[str(k)] for k in range(num_samples+1)},
            'group_rejections': dict(reasons),
            'potential_curriculum_count_1to3': sum(valid[str(k)] for k in range(1,num_samples)),
            'trial_status_counts':dict(Counter(r['category'] for r in records)),
            'finish_reason_counts':dict(Counter(r['finish_reason'] for r in records)),
            'training_approved':False,
            'scope':'fixed training-side E0-0of4 candidate probe, not an independent held-out benchmark or new approval'}
