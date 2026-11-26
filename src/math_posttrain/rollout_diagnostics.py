"""Compare strict GRPO rewards with a last-box fallback on the same saved rollouts."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from math_posttrain.common import read_jsonl
from math_posttrain.grading import Grader, last_box_without_final_tag


def compare(rows, grader):
    strict = grader.grade_many([(r['reward_text'], r['reference_answer']) for r in rows], strict=True)
    counts = Counter()
    fallback_rows = []
    for row, grade in zip(rows, strict, strict=True):
        if (grade['status'], grade.get('how')) != (row['strict_status'], row['how']):
            raise ValueError(f"Strict grade changed for rollout {row['call']}:{row['index']}")
        clipped = 'length' if row['truncated'] else 'non_length'
        counts['total', clipped] += 1
        if row['trl_clipped']:
            counts['trl_clipped', clipped] += 1
        counts['finish', str(row['finish_reason']), str(row['stop_reason'])] += 1
        counts['strict', clipped, grade['status'], grade.get('how') or '-'] += 1
        text = row['reward_text']
        if grade.get('how') == 'missing_answer_tag':
            candidate = last_box_without_final_tag(text)
            if candidate is not None:
                fallback_rows.append((clipped, candidate, row['reference_answer']))
            else:
                counts['missing_tag_no_box', clipped] += 1
        if grade['status'] == 'no_answer':
            counts['no_answer_stop', clipped, str(row['stop_reason']), grade.get('how') or '-'] += 1
    fallback = grader.grade_many([(text, ref) for _, text, ref in fallback_rows], strict=True)
    for (clipped, _, _), grade in zip(fallback_rows, fallback, strict=True):
        counts['last_box_fallback', clipped, grade['status']] += 1
    return {' / '.join(key): count for key, count in sorted(counts.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', help='Path to rollout_diagnostics.jsonl')
    args = parser.parse_args()
    with Grader(workers=4) as grader:
        print(json.dumps(compare(read_jsonl(args.journal), grader), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
