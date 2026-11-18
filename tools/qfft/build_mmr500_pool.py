"""Reproduce the authorized 1,200 + 2,800 OpenR1 draw without editing the old pool.

Review decisions are an input artifact. This command writes a draft and its provenance;
it never marks a file approved and never launches training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path
import re

from trl.trainer.utils import RepeatSampler

from experiments.qfft_v93 import decontam
from experiments.qfft_v93.common import ROOT, read_json, read_jsonl, sha256_file, sha256_text, write_json


def review_flags(question: str, reference: str, prompt_tokens: int) -> list[str]:
    flags = []
    if prompt_tokens > 512:
        flags.append('exceeds_unchanged_prompt_cap')
    if re.search(r'(?im)(?:^|\n)\s*(?:#{1,3}\s*)?(?:answer|solution|analysis)\s*[:.]', question):
        flags.append('answer_or_solution_block')
    if '\\boxed{' in question or '[hide=Solution]' in question:
        flags.append('boxed_or_hidden_solution')
    if re.search(r'(?i)(?:<img\b|!\[|\[img\]|\[asy\])', question):
        flags.append('diagram_needs_review')
    if re.search(r'(?i)(?:as shown|in (?:the )?(?:figure|diagram)|graph to the|see the)', question):
        flags.append('figure_reference_needs_review')
    if re.search(r'(?i)(?:therefore|thus|hence|we obtain|we get|from which|then from|from the condition)', question):
        flags.append('possible_worked_solution')
    if re.search(r'(?i)(?:translate (?:the |this |above)|translation result|保留.*(?:原文|源文本)|保持.*格式)', question):
        flags.append('translation_instruction')
    if len(re.findall(r'(?im)(?:^|\n)\s*\([1-9a-d]\)', question)) > 1 and not re.fullmatch(r'\s*[A-Ea-e]\s*', reference):
        flags.append('multiple_outputs_or_numbered_conditions')
    return flags


def build(folder: Path, reviews_path: Path, output: Path) -> dict:
    prior_path = folder / 'mmr_dr150_frozen_manifest.json'
    prior = read_json(prior_path)
    source_path, prefix_path = folder / prior['source_file'], folder / prior['train_file']
    if sha256_file(source_path) != prior['source_sha256'] or sha256_file(prefix_path) != prior['train_sha256']:
        raise ValueError('The immutable source or the 1,200-question prefix changed')
    source_rows = read_jsonl(source_path)
    source_lines = source_path.read_bytes().splitlines(keepends=True)
    prefix_rows = read_jsonl(prefix_path)
    if len(prefix_rows) != 1200:
        raise ValueError('Expected the original frozen 1,200-question prefix')
    reviews = {row['sample_id']: row for row in read_json(reviews_path)['decisions']}
    last = max(row['seed_position'] for row in prior['appended'])
    sampler = iter(RepeatSampler(source_rows, mini_repeat_count=1, batch_size=8, seed=prior['source_seed']))
    consumed = {source_rows[i]['sample_id'] for i in islice(sampler, last)}
    if not {r['sample_id'] for r in prefix_rows}.issubset(consumed):
        raise ValueError('The saved seed cursor does not include every prefix row')
    seen = {decontam.question_key(r['question']) for r in prefix_rows}
    skipped, original_draw = [], []
    cursor = last
    # Freeze the original 2,800 slots first; replacements come strictly after this draw.
    while len(original_draw) < 2800:
        index = next(sampler)
        cursor += 1
        row = source_rows[index]
        key = decontam.question_key(row['question'])
        if key in seen:
            skipped.append({'seed_position': cursor, 'sample_id': row['sample_id'], 'reason': 'normalized duplicate in original draw'})
        else:
            seen.add(key)
            original_draw.append((index, cursor))
    draw_end = cursor
    # Rebuild the final keys after reviewed removals, so unused draft questions do not block replacements.
    seen = {decontam.question_key(r['question']) for r in prefix_rows}
    final_lines, appended, changes = [], [], []
    for position, (original_index, original_seed) in enumerate(original_draw, 1201):
        index, seed = original_index, original_seed
        while True:
            original = source_rows[index]
            decision = reviews.get(original['sample_id'], {'action': 'unreviewed'})
            row = {**original}
            if decision['action'] == 'repair':
                row['question'] = decision['question']
                if 'reference_answer' in decision:
                    row['reference_answer'] = decision['reference_answer']
            key = decontam.question_key(row['question'])
            if decision['action'] != 'replace' and key not in seen:
                break
            skipped.append({'seed_position': seed, 'sample_id': original['sample_id'],
                            'reason': decision['reason'] if decision['action'] == 'replace'
                            else 'normalized duplicate after repair'})
            index, cursor = next(sampler), cursor + 1
            seed = cursor
        seen.add(key)
        line = (json.dumps(row, ensure_ascii=False) + '\n').encode() if row != original else source_lines[index]
        record = {'train_position': position, 'seed_position': seed, 'sample_id': row['sample_id'],
                  'source_row': row['source_row'], 'original_seed_position': original_seed,
                  'original_sample_id': source_rows[original_index]['sample_id'],
                  'source_line_sha256': sha256_text(source_lines[index].decode().rstrip('\r\n')),
                  'train_line_sha256': sha256_text(line.decode().rstrip('\r\n')),
                  'review_action': decision['action']}
        if index != original_index or row != original:
            changes.append({**record, 'reason': decision.get('reason'),
                            'original_review_reason': reviews.get(source_rows[original_index]['sample_id'], {}).get('reason'),
                            'replaced_draft_row': index != original_index, 'repaired_source_row': row != original})
        appended.append(record)
        final_lines.append(line)
    body = prefix_path.read_bytes() + b''.join(final_lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(body)
    if not output.read_bytes().startswith(prefix_path.read_bytes()):
        raise RuntimeError('The 1,200-row byte prefix changed')
    manifest = {'status': 'reviewed_draft_not_training_approval', 'rows': 4000, 'unchanged_prefix_rows': 1200,
                'source_file': source_path.name, 'source_sha256': sha256_file(source_path),
                'source_seed': prior['source_seed'], 'prefix_file': prefix_path.name,
                'prefix_sha256': sha256_file(prefix_path), 'prefix_manifest_sha256': sha256_file(prior_path),
                'train_file': output.name, 'train_sha256': sha256_file(output),
                'continued_after_seed_position': last, 'original_draw_end_seed_position': draw_end,
                'last_seed_position': cursor, 'review_file_sha256': sha256_file(reviews_path),
                'sft_overlap_filter': False, 'all_wrong_difficulty_filter': False,
                'changes': changes, 'skipped_seed_rows': skipped, 'appended': appended,
                'review_action_counts': dict(Counter(r['review_action'] for r in appended))}
    write_json(output.with_suffix('.manifest.json'), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build(ROOT / 'data/rl_openr1_46k_decontaminated', args.review, args.out)
    print(json.dumps({k: v for k, v in result.items() if k not in ('changes', 'skipped_seed_rows', 'appended')}, indent=2))


if __name__ == '__main__':
    main()
