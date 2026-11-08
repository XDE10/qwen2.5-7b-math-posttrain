# coding: utf-8
"""One-off restructure: v2 directory layout.

Rules
-----
* Tracked files are moved with ``git mv`` so history follows them.
* Ignored artifacts (data/, reports/, artifacts/) are moved with ``shutil.move``.
* Everything that is not reachable from the current entry points is moved to
  ``archive/`` (kept, not deleted).  The archive is frozen: it is *not*
  guaranteed to run from its new location.
* A JSON mapping of every move is written to ``archive/mapping/`` and
  ``docs/migration/``.

Usage:  python docs/migration/restructure_v2_move.py --dry
        python docs/migration/restructure_v2_move.py --run
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = '--run' in sys.argv
MOVES = []
SKIPPED = []

# ---------------------------------------------------------------- keep sets
KEEP_SCRIPTS = {
    'train_project1_sft.py': 'scripts/training/train_project1_sft.py',
    'train_project1_rl.py': 'scripts/training/train_project1_rl.py',
    'evaluate_approved_math.py': 'scripts/evaluation/evaluate_approved_math.py',
    'math_verify_pilot_worker.py': 'scripts/evaluation/math_verify_pilot_worker.py',
    'math_verify_v3_strict_worker.py': 'scripts/evaluation/math_verify_v3_strict_worker.py',
}
KEEP_TOOLS = {
    # AMO SFT / Colab delivery chain
    'start_aimo_sft_v27.py': 'tools/aimo/start_aimo_sft_v27.py',
    'start_aimo_sft_v28.py': 'tools/aimo/start_aimo_sft_v28.py',
    'prepare_aimo_sft_v26.py': 'tools/aimo/prepare_aimo_sft_v26.py',
    'package_aimo_sft_v26.py': 'tools/aimo/package_aimo_sft_v26.py',
    # evaluation
    'evaluate_sft_ds_v29.py': 'tools/evaluation/evaluate_sft_ds_v29.py',
    'evaluate_math500_final_answer_v41.py': 'tools/evaluation/evaluate_math500_final_answer_v41.py',
    'review_math500_answer_v42.py': 'tools/evaluation/review_math500_answer_v42.py',
    # QFFT dataset + monitoring
    'build_qfft_ds_dataset_v66.py': 'tools/qfft/build_qfft_ds_dataset_v66.py',
    'qfft_ds_common_v66.py': 'tools/qfft/qfft_ds_common_v66.py',
    'qfft_stream_transport_v79.py': 'tools/qfft/qfft_stream_transport_v79.py',
    'continue_qfft_all_v71.py': 'tools/qfft/continue_qfft_all_v71.py',
    'archive_qfft_followups_v70.py': 'tools/qfft/archive_qfft_followups_v70.py',
    'retry_length_max_v86.pyw': 'tools/qfft/retry_length_max_v86.pyw',
    'supplement_incomplete_high_v91.pyw': 'tools/qfft/supplement_incomplete_high_v91.pyw',
    'start_qfft_supervisor_v73.pyw': 'tools/qfft/start_qfft_supervisor_v73.pyw',
    'watch_qfft_progress_v72.pyw': 'tools/qfft/watch_qfft_progress_v72.pyw',
    'watch_qfft_length_retry_v86.pyw': 'tools/qfft/watch_qfft_length_retry_v86.pyw',
    'watch_incomplete_high_v91.pyw': 'tools/qfft/watch_incomplete_high_v91.pyw',
    # review + verification
    'build_external_review_zip_v92.py': 'tools/review/build_external_review_zip_v92.py',
    'verify_migration.py': 'tools/verification/verify_migration.py',
}
KEEP_TESTS = {
    'test_approved_training_inputs.py': 'tests/data/test_approved_training_inputs.py',
    'test_geometry_model_serialization.py': 'tests/data/test_geometry_model_serialization.py',
    'test_perception_v4.py': 'tests/data/test_perception_v4.py',
    'test_project3_perception_prototype.py': 'tests/data/test_project3_perception_prototype.py',
    'test_project3_perception_v2.py': 'tests/data/test_project3_perception_v2.py',
    'test_losses_and_masks.py': 'tests/losses/test_losses_and_masks.py',
    'test_math_verifier.py': 'tests/evaluation/test_math_verifier.py',
    'test_math500_answer_v41.py': 'tests/evaluation/test_math500_answer_v41.py',
    'test_native_ds_v29.py': 'tests/evaluation/test_native_ds_v29.py',
    'test_math_verify_v3_type_gate.py': 'tests/evaluation/test_math_verify_v3_type_gate.py',
    'test_aimo_sft_v26.py': 'tests/training/test_aimo_sft_v26.py',
    'test_colab_compat_v27.py': 'tests/training/test_colab_compat_v27.py',
    'test_swanlab_autoauth_v28.py': 'tests/training/test_swanlab_autoauth_v28.py',
    'test_qfft_atomic_save_v88.py': 'tests/qfft/test_qfft_atomic_save_v88.py',
    'test_qfft_circuit_review_v80.py': 'tests/qfft/test_qfft_circuit_review_v80.py',
    'test_qfft_ds_dataset_v63.py': 'tests/qfft/test_qfft_ds_dataset_v63.py',
    'test_qfft_ds_dataset_v66.py': 'tests/qfft/test_qfft_ds_dataset_v66.py',
    'test_qfft_length_retry_v86.py': 'tests/qfft/test_qfft_length_retry_v86.py',
    'test_qfft_process_v78.py': 'tests/qfft/test_qfft_process_v78.py',
    'test_qfft_resume_v73.py': 'tests/qfft/test_qfft_resume_v73.py',
    'test_qfft_retry_resume_v87.py': 'tests/qfft/test_qfft_retry_resume_v87.py',
    'test_qfft_server_errors_v74.py': 'tests/qfft/test_qfft_server_errors_v74.py',
    'test_qfft_stream_v79.py': 'tests/qfft/test_qfft_stream_v79.py',
    'test_qfft_supplement_v91.py': 'tests/qfft/test_qfft_supplement_v91.py',
    'test_qfft_transport_v77.py': 'tests/qfft/test_qfft_transport_v77.py',
}
# src: closure of the kept entry points + the core library covered by kept tests
KEEP_SRC = {
    '__init__.py', 'data/__init__.py', 'data/approved_training_inputs.py',
    'data/geometry_model_serialization.py', 'data/perception_text_guards.py',
    'data/perception_v4.py', 'data/project3_perception_prototype.py',
    'data/project3_perception_v2.py', 'evaluation/__init__.py',
    'evaluation/math_verifier.py', 'losses/__init__.py', 'losses/reference.py',
    'training/approved_hf.py', 'training/frozen_verifier_v4.py',
    'training/math_reward_v2.py', 'training/perception_reward_v4.py',
}
KEEP_CONFIGS = {'project1_aimo_sft_v26.json'}
KEEP_DATA = {'math', 'portable_release', 'tokenizers', 'math_manifest.json'}
# reports/ that are operational inputs or the current review snapshot
REPORTS_RUNS = {'qfft_ds_dataset_20261029_v66', 'qfft_incomplete_high_20261031_v91',
                'qfft_followups_20261029_v70', 'romescape_reproduction_20261028_v57',
                'colab_compat_20261026_v27'}
REPORTS_CURRENT = {'claude_project1_review_20261031_v90', 'external_review_20261103_v92'}
REPORTS_TO_DATA = {'aimo_sft_20261026_v26': 'data/aimo_sft_v26',
                   'aimo_sft_20261026_v25': 'archive/data/aimo_sft_20261026_v25'}
ROOT_TOOL_CMD = '查看造数进度.cmd'


def is_tracked(rel):
    return subprocess.run(['git', 'ls-files', '--error-unmatch', str(rel)],
                          cwd=ROOT, capture_output=True).returncode == 0


def move(src, dst, kind='auto'):
    src, dst = Path(src), Path(dst)
    a, b = ROOT / src, ROOT / dst
    if not a.exists():
        SKIPPED.append({'from': str(src), 'reason': 'source missing'})
        return
    if b.exists():
        SKIPPED.append({'from': str(src), 'reason': 'destination exists: ' + str(dst)})
        return
    tracked = is_tracked(src) if kind == 'auto' else kind == 'git'
    MOVES.append({'from': src.as_posix(), 'to': dst.as_posix(), 'tracked': tracked})
    if not RUN:
        return
    b.parent.mkdir(parents=True, exist_ok=True)
    if tracked:
        subprocess.run(['git', 'mv', str(src), str(dst)], cwd=ROOT, check=True)
    else:
        shutil.move(str(a), str(b))


def move_dir(src, dst):
    """Directory move: git can only move tracked files; git ignores are untracked."""
    move(src, dst, kind='git' if is_tracked(src) else 'file')


# ------------------------------------------------------------- experiments
for d in sorted((ROOT / 'experiments').iterdir()):
    if d.is_dir() and d.name != 'aimo_sft_v26':
        move_dir(f'experiments/{d.name}', f'archive/experiments/{d.name}')

# ------------------------------------------------------------------ scripts
for p in sorted((ROOT / 'scripts').glob('*')):
    if p.name in ('README.md',):
        continue
    if p.name in KEEP_SCRIPTS:
        move(f'scripts/{p.name}', KEEP_SCRIPTS[p.name])
    elif p.is_file():
        move(f'scripts/{p.name}', f'archive/scripts/{p.name}')

# -------------------------------------------------------------------- tools
for p in sorted((ROOT / 'tools').glob('*')):
    if p.name == 'README.md':
        continue
    if p.name in KEEP_TOOLS:
        move(f'tools/{p.name}', KEEP_TOOLS[p.name])
    elif p.is_file():
        move(f'tools/{p.name}', f'archive/tools/{p.name}')
move(ROOT_TOOL_CMD, 'tools/qfft/' + ROOT_TOOL_CMD)

# -------------------------------------------------------------------- tests
for p in sorted((ROOT / 'tests').glob('*')):
    if p.name == 'README.md':
        continue
    if p.name in KEEP_TESTS:
        move(f'tests/{p.name}', KEEP_TESTS[p.name])
    elif p.is_file():
        move(f'tests/{p.name}', f'archive/tests/{p.name}')

# --------------------------------------------------------------------- src
for p in sorted((ROOT / 'src').rglob('*.py')):
    rel = p.relative_to(ROOT / 'src').as_posix()
    if rel in KEEP_SRC:
        continue
    move(f'src/{rel}', f'archive/src/{rel}')

# ----------------------------------------------------------------- configs
for p in sorted((ROOT / 'configs').glob('*')):
    if p.name == 'README.md':
        continue
    if p.name in KEEP_CONFIGS:
        continue
    if p.is_file():
        move(f'configs/{p.name}', f'archive/configs/{p.name}')

# -------------------------------------------------------------------- data
for p in sorted((ROOT / 'data').iterdir()):
    if p.name in KEEP_DATA:
        continue
    if p.is_file():
        move(f'data/{p.name}', f'archive/data/{p.name}')
    else:
        move_dir(f'data/{p.name}', f'archive/data/{p.name}')

# ----------------------------------------------------------------- reports
for p in sorted((ROOT / 'reports').iterdir()):
    if p.name == 'README.md':
        continue
    if p.is_dir():
        if p.name in REPORTS_TO_DATA:
            move_dir(f'reports/{p.name}', REPORTS_TO_DATA[p.name])
        elif p.name in REPORTS_RUNS:
            move_dir(f'reports/{p.name}', f'reports/runs/{p.name}')
        elif p.name in REPORTS_CURRENT:
            move_dir(f'reports/{p.name}', f'reports/current/{p.name}')
        elif p.name.startswith('handover_'):
            move_dir(f'reports/{p.name}', f'docs/handover/{p.name}')
        else:
            m = re.search(r'(\d{8})', p.name)
            day = f'{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}' if m else 'undated'
            move_dir(f'reports/{p.name}', f'archive/reports/{day}/{p.name}')
    else:
        suffix = p.suffix.lower()
        bucket = ('logs' if suffix == '.log' else
                  'audits' if suffix == '.json' else 'snapshots')
        move(f'reports/{p.name}', f'archive/reports/_loose/{bucket}/{p.name}')

# ------------------------------------------------------------------ report
plan = {
    'generated_at': '2026-11-08',
    'run': RUN,
    'policy': 'current mainline = import closure of active entry points + documented references; everything else archive/ (frozen, not runnable)',
    'keep': {
        'scripts': sorted(KEEP_SCRIPTS.values()),
        'tools': sorted(KEEP_TOOLS.values()),
        'tests': sorted(KEEP_TESTS.values()),
        'src': sorted(KEEP_SRC),
        'configs': sorted(KEEP_CONFIGS),
        'data': sorted(KEEP_DATA),
        'reports_runs': sorted(REPORTS_RUNS),
        'reports_current': sorted(REPORTS_CURRENT),
        'experiments': ['aimo_sft_v26'],
    },
    'moves': MOVES,
    'skipped': SKIPPED,
}
out1 = ROOT / 'docs/migration/restructure_v2_plan.json'
out2 = ROOT / 'archive/mapping/restructure_v2_mapping.json'
if RUN:
    for out in (out1, out2):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
from collections import Counter
buckets = Counter()
for m in MOVES:
    parts = m['to'].split('/')
    buckets['/'.join(parts[:2]) if parts[0] == 'archive' else '/'.join(parts[:1])] += 1
print(json.dumps({'run': RUN, 'moves': len(MOVES), 'tracked': sum(m['tracked'] for m in MOVES),
                  'skipped': len(SKIPPED), 'buckets': dict(sorted(buckets.items()))}, ensure_ascii=False))
for s in SKIPPED:
    print('SKIP', s['from'], '->', s['reason'])
