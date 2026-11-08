"""Version 2: preserve v1 digest gates, enforce train/dev and paired identities."""
import hashlib
import json
from pathlib import Path
from src.data.approved_training_inputs import load_approved as load_v1, model_view

PAIR_FILES = ('project1_raw_paired.jsonl', 'project1_compact_paired.jsonl')


def _guard(rows, protected):
    if any(r.get('split') != 'train' for r in rows):
        raise ValueError('Training requires split == train')
    if {r['sample_id'] for r in rows} & protected:
        raise ValueError('Protected dev ID in training')


def assert_pairs(raw, compact):
    def index(rows):
        ids = [r['pair_id'] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate pair_id')
        return {r['pair_id']: r for r in rows}
    a, b = index(raw), index(compact)
    if a.keys() != b.keys():
        raise ValueError('raw/compact pair_id sets differ')
    for key in a:
        if any(a[key][f] != b[key][f] for f in ('sample_id', 'problem', 'reference_answer', 'split', 'problem_type')):
            raise ValueError('Paired task identity differs')
        if a[key].get('arm') != 'raw' or b[key].get('arm') != 'compact':
            raise ValueError('Paired arm identity differs')


def load_approved(manifest_path, filename):
    root = Path(manifest_path).resolve().parent
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    name = manifest['protected_dev_ids_file']
    protected_path = (root / name).resolve()
    if not protected_path.is_relative_to(root):
        raise ValueError('Protected IDs path escapes release')
    expected = manifest['files'][name]
    expected = expected['sha256'] if isinstance(expected, dict) else expected
    if hashlib.sha256(protected_path.read_bytes()).hexdigest() != expected:
        raise ValueError('Protected IDs digest changed')
    protected = set(json.loads(protected_path.read_text(encoding='utf-8'))['sample_ids'])
    rows = load_v1(manifest_path, filename)
    _guard(rows, protected)
    if filename in PAIR_FILES:
        paired = [rows if name == filename else load_v1(manifest_path, name) for name in PAIR_FILES]
        for arm in paired:
            _guard(arm, protected)
        assert_pairs(*paired)
    return rows
