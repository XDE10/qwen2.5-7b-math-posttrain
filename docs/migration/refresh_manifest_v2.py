# coding: utf-8
"""Refresh docs/metadata/migration_manifest.json after the v2 directory move.

1. Rewrite every recorded path through the v2 move mapping (longest prefix wins).
2. Recompute bytes/sha256 for files whose size changed (edited code, docs, .gitignore).
3. Append the files created by the refactor itself.
4. Record the v2 move summary under refactor_20261108.

Run with --dry to only report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'docs/metadata/migration_manifest.json'
PLAN = ROOT / 'docs/migration/restructure_v2_plan.json'
NEW_FILES = [
    'archive/mapping/restructure_v2_mapping.json',
    'docs/migration/restructure_v2_plan.json',
    'docs/migration/restructure_v2_move.py',
    'docs/migration/fix_paths_v2.py',
    'docs/migration/refresh_data_records_v2.py',
    'docs/migration/refresh_manifest_v2.py',
    'docs/migration/重构记录_20261108_v2.md',
    'docs/migration/pytest_baseline_before_restructure.txt',
    'docs/migration/pytest_after_restructure.txt',
    'docs/status/current_status.md',
    'docs/status/history.md',
    'tests/conftest.py',
    'pyproject.toml',
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    moves = sorted(((m['from'], m['to']) for m in plan['moves']),
                   key=lambda pair: len(pair[0]), reverse=True)

    def remap(path: str) -> str | None:
        for old, new in moves:
            if path == old:
                return new
            if path.startswith(old + '/'):
                return new + path[len(old):]
        return None

    # Post-plan corrections: files the static closure missed and were moved back
    # to the current tree after the bulk move (see 重构记录 实际落位差异).
    extra_remap = {
        'archive/tools/start_aimo_sft_v26.py': 'tools/aimo/start_aimo_sft_v26.py',
        'archive/tools/start_aimo_sft_v26_notify.py': 'tools/aimo/start_aimo_sft_v26_notify.py',
        'archive/scripts/verifier_stream_worker_v4.py': 'scripts/evaluation/verifier_stream_worker_v4.py',
    }

    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    missing = []
    removed = []
    moved = 0
    resized = []
    kept_entries = []
    for entry in manifest['files']:
        target = remap(entry['path']) or extra_remap.get(entry['path'])
        if target is not None:
            entry['path'] = target
            moved += 1
        path = ROOT / entry['path']
        if not path.is_file():
            if entry['path'] == 'pytest.ini':
                # merged into pyproject.toml [tool.pytest.ini_options] by the refactor
                removed.append(entry['path'])
                continue
            missing.append(entry['path'])
            kept_entries.append(entry)
            continue
        kept_entries.append(entry)
        actual = path.stat().st_size
        if actual != entry['bytes']:
            entry['bytes'] = actual
            entry.pop('source_sha256', None)
            entry['sha256'] = sha(path)
            entry['refactor_note'] = '2026-11-08 目录重整 v2：路径/文本更新后重算'
            resized.append(entry['path'])
    manifest['files'] = kept_entries

    known = {entry['path'] for entry in manifest['files']}
    added = []
    for relative in NEW_FILES:
        path = ROOT / relative
        if not path.is_file():
            continue
        if relative in known:
            continue
        manifest['files'].append({'bytes': path.stat().st_size, 'changed_for_portability': False,
                                  'path': relative, 'sha256': sha(path),
                                  'refactor_note': '2026-11-08 目录重整 v2 新增'})
        known.add(relative)
        added.append(relative)
    # Keep the manifest's original entry order (pre-move order remapped, newest appended at the
    # end) so the refactor diff only shows real path/hash changes, not a 4000-line re-sort.

    # Content edits that keep the byte size identical are invisible to the size check
    # above, so rehash every current-tree entry to keep `verify_migration.py --full` green.
    rehash_prefixes = ('configs/', 'docs/', 'experiments/', 'scripts/', 'src/', 'tests/', 'tools/',
                       'data/aimo_sft_v26/', 'data/tokenizers/')
    root_files = {'README.md', 'AGENTS.md', '.gitignore', 'pyproject.toml', 'data/README.md'}
    rehashed = 0
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file():
            continue
        if not (entry['path'].startswith(rehash_prefixes) or entry['path'] in root_files):
            continue
        digest = sha(path)
        if digest != entry.get('sha256'):
            entry['sha256'] = digest
            entry['bytes'] = path.stat().st_size
            entry['refactor_note'] = '2026-11-08 目录重整 v2：内容更新后重算'
            rehashed += 1

    manifest['refactor_20261108']['restructure_v2'] = {
        'date': '2026-11-08',
        'moves': 570,
        'moved_paths_in_manifest': moved,
        'removed_entries': removed,
        'doc': 'docs/migration/重构记录_20261108_v2.md',
        'mapping': 'archive/mapping/restructure_v2_mapping.json',
        'plan': 'docs/migration/restructure_v2_plan.json',
    }

    print(json.dumps({'moved': moved, 'resized': len(resized), 'added': added,
                      'missing': missing, 'total': len(manifest['files'])},
                     ensure_ascii=False, indent=2))
    if args.dry or missing:
        if missing:
            raise SystemExit('missing targets; fix before writing')
        return
    MANIFEST.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
