# coding: utf-8
"""Refresh path-like fields inside data/aimo_sft_v26 after the v2 directory move.

- evaluation_sources.json: file pointers -> data/aimo_sft_v26/<name> (sha untouched, verified).
- validation.json: recorded pytest target -> tests/training/test_aimo_sft_v26.py.
- bundle_checksums.json: keep it a live ledger for experiments/aimo_sft_v26/run.py by
  rebuilding the name list exactly like tools/aimo/package_aimo_sft_v26.py and recomputing
  sha256 from the current tree; the pre-refactor copy is preserved under
  archive/reports/_loose/audits/ as historical evidence.

Left untouched on purpose: bundle_info.json / delivery.json / patch_manifest.json are
delivery snapshots of the pre-refactor Colab bundle (see docs/status/current_status.md).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'data/aimo_sft_v26'
OLD_PREFIX = 'reports/aimo_sft_20261026_v26/'
NEW_PREFIX = 'data/aimo_sft_v26/'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str):
    return json.loads((REPORT / name).read_text(encoding='utf-8'))


def save(name: str, value) -> None:
    (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def refresh_evaluation_sources() -> None:
    data = load('evaluation_sources.json')
    for bench, entry in data.items():
        old = entry['file']
        if old.startswith(OLD_PREFIX):
            new = NEW_PREFIX + old[len(OLD_PREFIX):]
            if not (ROOT / new).is_file():
                raise SystemExit('new evaluation source missing: ' + new)
            # note: entry['sha256'] records the original source artifact, not this local jsonl
            lines = sum(bool(line.strip()) for line in (ROOT / new).read_text(encoding='utf-8').splitlines())
            if 'count' in entry and lines != entry['count']:
                raise SystemExit('evaluation source row count changed: ' + new)
            entry['file'] = new
    save('evaluation_sources.json', data)
    print('evaluation_sources: ' + ', '.join(e['file'] for e in data.values()))


def refresh_validation() -> None:
    data = load('validation.json')
    command = data['cpu_tests']['command']
    data['cpu_tests']['command'] = command.replace(
        'tests/test_aimo_sft_v26.py', 'tests/training/test_aimo_sft_v26.py')
    save('validation.json', data)
    print('validation: ' + data['cpu_tests']['command'])


def checksum_names() -> list[str]:
    """Same name list construction as tools/aimo/package_aimo_sft_v26.py."""
    names = [
        'configs/project1_aimo_sft_v26.json',
        'tests/training/test_aimo_sft_v26.py',
        'tools/aimo/prepare_aimo_sft_v26.py',
        'src/training/frozen_verifier_v4.py',
        'src/training/math_reward_v2.py',
        'scripts/evaluation/math_verify_v3_strict_worker.py',
        'scripts/evaluation/math_verify_pilot_worker.py',
        'scripts/evaluation/verifier_stream_worker_v4.py',
    ]
    names += [p.relative_to(ROOT).as_posix()
              for p in sorted((ROOT / 'experiments/aimo_sft_v26').glob('*.py'))]
    names += ['experiments/aimo_sft_v26/start_colab.sh']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / '.pilot_deps').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    names += [p.relative_to(ROOT).as_posix() for p in sorted(REPORT.glob('*'))
              if p.is_file() and p.suffix != '.zip'
              and p.name not in ['bundle_checksums.json', 'bundle_info.json']]
    names += ['data/tokenizers/Qwen2.5-7B-Instruct/' + f
              for f in ['tokenizer.json', 'tokenizer_config.json']]
    return names


def refresh_bundle_checksums() -> None:
    checksums = REPORT / 'bundle_checksums.json'
    keep = ROOT / 'archive/reports/_loose/audits/bundle_checksums_pre_refactor_20261108.json'
    keep.parent.mkdir(parents=True, exist_ok=True)
    if checksums.is_file() and not keep.exists():
        keep.write_text(checksums.read_text(encoding='utf-8'), encoding='utf-8')
    names = checksum_names()
    missing = [n for n in names if not (ROOT / n).is_file()]
    if missing:
        raise SystemExit('missing bundle files: ' + ', '.join(missing))
    checks = {name: sha(ROOT / name) for name in names}
    save('bundle_checksums.json', checks)
    print('bundle_checksums: %d files refreshed; pre-refactor copy -> %s'
          % (len(checks), keep.relative_to(ROOT).as_posix()))


if __name__ == '__main__':
    refresh_evaluation_sources()
    refresh_validation()
    refresh_bundle_checksums()
