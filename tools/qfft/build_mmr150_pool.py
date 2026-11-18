"""Rebuild the reviewed MMR-150 pool and its separate two-step smoke prefix."""
from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

from trl.trainer.utils import RepeatSampler

from experiments.qfft_v93 import decontam
from experiments.qfft_v93.common import read_json, read_jsonl, sha256_file, sha256_text, write_json
from experiments.qfft_v93.grading import _math_verify, _parse_gold, clean_reference


# One-based positions in the 1,200-row draft. Only the appended OpenR1 tail was reviewed.
STRIP_SUFFIX = {
    413: ('23649', '\n\nANSWER: '),
    498: ('17114', '\n\nSolution:\n'),
    694: ('11409', ' Answer: '),
    912: ('24620', '\n\nAnswer: '),
    994: ('15010', ' Answer: '),
    1180: ('29811', '\nAnswer: '),
    1183: ('18047', '\r\n[asy]'),
}
REPLACE = {
    477: ('15569', 'solution only'),
    492: ('15413', 'solution only'),
    599: ('07616', 'solution only'),
    635: ('02797', 'two requested answers but only one is keyed'),
    646: ('07734', 'missing figure and pasted translation instruction'),
    674: ('37001', 'solution only, with external image'),
    691: ('42444', 'missing diagram specifying the coordinate-axis alignment'),
    831: ('39880', 'missing diagram specifying the geometry arrangement'),
    878: ('26047', 'full worked solution and answer in the prompt'),
    908: ('30508', 'missing diagram specifying the paths of the arrows'),
    915: ('44223', 'missing diagram specifying point order and orientation'),
    930: ('20727', 'answer in prompt and inconsistent balance measurements and arm ratio'),
    949: ('14308', 'missing diagram defining points E and F'),
    979: ('01075', 'solution only'),
}
SOURCE_ID = 'openr1_46k_8192/train/'
SKIP_REPLACEMENTS = {
    1209: ('33794', '654 prompt tokens with the frozen template; the 512-token cap removes the question setup'),
}


def freeze(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Existing frozen output has different content: {path}')
        return
    path.write_bytes(data)


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / 'data/rl_openr1_46k_decontaminated'
    old_manifest = folder / 'mmr_dr50_manifest.json'
    source = read_json(old_manifest)
    original_path = folder / source['train_file']
    source_path = folder / source['source_file']
    if sha256_file(source_path) != source['source_sha256'] or sha256_file(original_path) != source['train_sha256']:
        raise ValueError('The source or frozen 400-prompt draw differs from the step-50 manifest')

    prior = read_jsonl(original_path)
    rows = read_jsonl(source_path)
    source_lines = source_path.read_bytes().splitlines(keepends=True)
    assert len(rows) == len(source_lines) == source['source_rows']
    assert [row['sample_id'] for row in prior] == source['frozen_sample_ids']
    sampler = iter(RepeatSampler(rows, mini_repeat_count=1, batch_size=source['questions_per_step'],
                                 seed=source['source_seed']))
    old_seed_ids = [rows[i]['sample_id'] for i in islice(sampler, len(prior))]
    assert old_seed_ids == source['original_sample_ids']

    replacements = {change['seed_position']: change for change in source['changes'] if change['action'] == 'replace'}
    last_position = max(replacements, default=len(prior))
    skipped = {row['position']: row['sample_id'] for change in replacements.values()
               for row in change['skipped_seed_rows']}
    for pos in range(len(prior) + 1, last_position + 1):
        expected = replacements[pos]['final_id'] if pos in replacements else skipped[pos]
        assert rows[next(sampler)]['sample_id'] == expected

    draft_indices = list(islice(sampler, 800))
    prefix = original_path.read_bytes()
    draft = folder / 'mmr_dr150_train.jsonl'
    if draft.exists() and draft.read_bytes() != prefix + b''.join(source_lines[i] for i in draft_indices):
        raise RuntimeError('The original 1,200-row draft differs from the reproduced seed draw')

    tail, records, changes, final_lines = [], [], [], []
    skipped_replacements = []
    replacement_position = last_position + len(draft_indices)
    for train_position, draft_index in enumerate(draft_indices, len(prior) + 1):
        index = draft_index
        seed_position = last_position + train_position - len(prior)
        repair, replace = STRIP_SUFFIX.get(train_position), REPLACE.get(train_position)
        if repair or replace:
            source_id = SOURCE_ID + (repair or replace)[0]
            if rows[draft_index]['sample_id'] != source_id:
                raise ValueError(f'Train row {train_position} has a different source ID than {source_id}')
        if replace:
            index = next(sampler)
            replacement_position += 1
            while replacement_position in SKIP_REPLACEMENTS:
                suffix, reason = SKIP_REPLACEMENTS[replacement_position]
                if rows[index]['sample_id'] != SOURCE_ID + suffix:
                    raise ValueError(f'Seed replacement position {replacement_position} has changed identity')
                skipped_replacements.append({
                    'seed_position': replacement_position, 'sample_id': rows[index]['sample_id'], 'reason': reason,
                    'source_line_sha256': sha256_text(source_lines[index].decode('utf-8').rstrip('\r\n')),
                })
                index = next(sampler)
                replacement_position += 1
            seed_position = replacement_position
        row, line = rows[index], source_lines[index]
        if repair:
            question, delimiter, suffix = row['question'].rpartition(repair[1])
            if not question.strip() or not delimiter or not suffix.strip():
                raise ValueError(f'Train row {train_position} no longer has its reviewed suffix')
            row = {**row, 'question': question.rstrip()}
            line = (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')
        record = {
            'train_position': train_position, 'seed_position': seed_position,
            'sample_id': row['sample_id'], 'source_row': row['source_row'],
            'source_line_sha256': sha256_text(source_lines[index].decode('utf-8').rstrip('\r\n')),
            'train_line_sha256': sha256_text(line.decode('utf-8').rstrip('\r\n')),
        }
        if repair or replace:
            changes.append({
                **record, 'action': 'strip_suffix' if repair else 'replace',
                'reason': 'answer or solution encoded in a trailing block' if repair else replace[1],
                'original_seed_position': last_position + train_position - len(prior),
                'original_sample_id': rows[draft_index]['sample_id'],
                'original_line_sha256': sha256_text(source_lines[draft_index].decode('utf-8').rstrip('\r\n')),
            })
        tail.append(row)
        records.append(record)
        final_lines.append(line)

    assert len(changes) == len(STRIP_SUFFIX) + len(REPLACE)
    assert len(skipped_replacements) == len(SKIP_REPLACEMENTS)
    all_rows = prior + tail
    assert len({row['sample_id'] for row in all_rows}) == len(all_rows) == 1200
    assert len({decontam.question_key(row['question']) for row in all_rows}) == len(all_rows)
    verifier = _math_verify()
    unparseable = [row['sample_id'] for row in tail if not row['question'].strip()
                   or not _parse_gold(verifier, clean_reference(row['reference_answer']), 5)]
    if unparseable:
        raise ValueError(f'Check the seed-ordered new rows before freezing: {unparseable}')
    candidates = [(row['sample_id'], row['question']) for row in tail]
    protected = {}
    for filename, info in source['protected_files'].items():
        path = root / filename
        assert sha256_file(path) == info['sha256']
        prompts = [(row['sample_id'], row.get('problem') or row['question']) for row in read_jsonl(path)]
        flags = decontam.find_flags(candidates, prompts)
        if flags:
            raise ValueError(f'New rows flagged against {filename}: {flags}')
        protected[filename] = {**info, 'flagged_in_append': len(flags)}

    final_path = folder / 'mmr_dr150_frozen_train.jsonl'
    final = prefix + b''.join(final_lines)
    assert prefix.endswith(b'\n') and final.endswith(b'\n')
    freeze(final_path, final)
    smoke_path = folder / 'mmr_dr150_smoke2_train.jsonl'
    smoke_rows = 2 * source['questions_per_step']
    freeze(smoke_path, b''.join(prefix.splitlines(keepends=True)[:smoke_rows]))
    write_json(folder / 'mmr_dr150_frozen_manifest.json', {
        'run': 'grpo_qfft_r1_mmr_dr150_v93', 'rows': len(all_rows),
        'source_file': source_path.name, 'source_sha256': source['source_sha256'],
        'source_seed': source['source_seed'], 'source_sampler': source['source_sampler'],
        'prefix_file': original_path.name, 'prefix_rows': len(prior), 'prefix_sha256': source['train_sha256'],
        'prefix_manifest_sha256': sha256_file(old_manifest), 'protected_files': protected,
        'sft_overlap_filter': False, 'train_file': final_path.name, 'train_sha256': sha256_file(final_path),
        'smoke_prefix': {'train_file': smoke_path.name, 'rows': smoke_rows, 'sha256': sha256_file(smoke_path)},
        'changes': changes, 'skipped_replacement_seed_rows': skipped_replacements, 'appended': records,
    })
    print(f'{final_path}: {len(all_rows)} rows, {sha256_file(final_path)}')
    print(f'{smoke_path}: {smoke_rows} rows, {sha256_file(smoke_path)}')


if __name__ == '__main__':
    main()
