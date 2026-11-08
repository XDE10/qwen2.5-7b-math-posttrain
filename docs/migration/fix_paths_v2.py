# coding: utf-8
"""One-shot: rewrite path constants in kept code after the v2 directory move.

Each replacement is literal and asserted, so re-running is safe (it fails loudly
instead of silently rewriting nothing). Kept for the migration record.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EDITS = [
    # configs: training data now lives in data/aimo_sft_v26
    ('configs/project1_aimo_sft_v26.json',
     '"reports/aimo_sft_20261026_v26/train.jsonl"', '"data/aimo_sft_v26/train.jsonl"'),
    ('configs/project1_aimo_sft_v26.json',
     '"reports/aimo_sft_20261026_v26/manifest.json"', '"data/aimo_sft_v26/manifest.json"'),
    # qfft common: one directory deeper, reports split into current/ + runs/
    ('tools/qfft/qfft_ds_common_v66.py',
     'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[2]'),
    ('tools/qfft/qfft_ds_common_v66.py',
     "REPORT = ROOT / 'reports/qfft_ds_dataset_20261029_v66'",
     "REPORT = ROOT / 'reports/runs/qfft_ds_dataset_20261029_v66'"),
    ('tools/qfft/qfft_ds_common_v66.py',
     "PREPARED = ROOT / 'reports/romescape_reproduction_20261028_v57/train1000_prepared.jsonl'",
     "PREPARED = ROOT / 'reports/runs/romescape_reproduction_20261028_v57/train1000_prepared.jsonl'"),
    ('tools/qfft/qfft_ds_common_v66.py',
     "read(ROOT / 'reports/romescape_reproduction_20261028_v57/replacement6_checks.json')",
     "read(ROOT / 'reports/runs/romescape_reproduction_20261028_v57/replacement6_checks.json')"),
    # qfft follow-up archiver
    ('tools/qfft/archive_qfft_followups_v70.py',
     'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[2]'),
    ('tools/qfft/archive_qfft_followups_v70.py',
     "RECIPE = ROOT / 'reports/qfft_ds_dataset_20261029_v66'",
     "RECIPE = ROOT / 'reports/runs/qfft_ds_dataset_20261029_v66'"),
    ('tools/qfft/archive_qfft_followups_v70.py',
     "OUT = ROOT / 'reports/qfft_followups_20261029_v70'",
     "OUT = ROOT / 'reports/runs/qfft_followups_20261029_v70'"),
    # qfft supervisor: child script paths
    ('tools/qfft/continue_qfft_all_v71.py',
     "'tools/archive_qfft_followups_v70.py'", "'tools/qfft/archive_qfft_followups_v70.py'"),
    ('tools/qfft/continue_qfft_all_v71.py',
     "'tools/build_qfft_ds_dataset_v66.py'", "'tools/qfft/build_qfft_ds_dataset_v66.py'"),
    # qfft builder: dynamic judge module load
    ('tools/qfft/build_qfft_ds_dataset_v66.py',
     "'tools/evaluate_sft_ds_v29.py'", "'tools/evaluation/evaluate_sft_ds_v29.py'"),
    # AIMO data preparation / packaging
    ('tools/aimo/prepare_aimo_sft_v26.py',
     'ROOT=Path(__file__).resolve().parents[1]', 'ROOT=Path(__file__).resolve().parents[2]'),
    ('tools/aimo/prepare_aimo_sft_v26.py',
     "report=ROOT/'reports/aimo_sft_20261026_v26'", "report=ROOT/'data/aimo_sft_v26'"),
    ('tools/aimo/package_aimo_sft_v26.py',
     'ROOT=Path(__file__).resolve().parents[1]', 'ROOT=Path(__file__).resolve().parents[2]'),
    ('tools/aimo/package_aimo_sft_v26.py',
     "report=ROOT/'reports/aimo_sft_20261026_v26'", "report=ROOT/'data/aimo_sft_v26'"),
    ('tools/aimo/package_aimo_sft_v26.py',
     "'tests/test_aimo_sft_v26.py'", "'tests/training/test_aimo_sft_v26.py'"),
    ('tools/aimo/package_aimo_sft_v26.py',
     "'tools/prepare_aimo_sft_v26.py'", "'tools/aimo/prepare_aimo_sft_v26.py'"),
    ('tools/aimo/package_aimo_sft_v26.py',
     "'scripts/math_verify_v3_strict_worker.py','scripts/math_verify_pilot_worker.py','scripts/verifier_stream_worker_v4.py'",
     "'scripts/evaluation/math_verify_v3_strict_worker.py','scripts/evaluation/math_verify_pilot_worker.py',"
     "'archive/scripts/verifier_stream_worker_v4.py'"),
    ('tools/aimo/package_aimo_sft_v26.py',
     "names.append('reports/aimo_sft_20261026_v26/bundle_checksums.json')",
     "names.append('data/aimo_sft_v26/bundle_checksums.json')"),
    ('tools/aimo/start_aimo_sft_v27.py',
     "root/'reports/colab_compat_20261026_v27/patch_manifest.json'",
     "root/'reports/runs/colab_compat_20261026_v27/patch_manifest.json'"),
    # evaluation entries
    ('tools/evaluation/evaluate_sft_ds_v29.py',
     "project_root / 'reports/aimo_sft_20261026_v26/evaluation_sources.json'",
     "project_root / 'data/aimo_sft_v26/evaluation_sources.json'"),
    ('tools/evaluation/evaluate_math500_final_answer_v41.py',
     "project / 'reports/aimo_sft_20261026_v26/math500.jsonl'",
     "project / 'data/aimo_sft_v26/math500.jsonl'"),
    ('tools/evaluation/review_math500_answer_v42.py',
     'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[2]'),
    # external review package builder
    ('tools/review/build_external_review_zip_v92.py',
     'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[2]'),
    ('tools/review/build_external_review_zip_v92.py',
     "OUT = ROOT / 'reports' / 'external_review_20261103_v92'",
     "OUT = ROOT / 'reports' / 'current' / 'external_review_20261103_v92'"),
    ('tools/review/build_external_review_zip_v92.py',
     "OLD = ROOT / 'reports' / 'claude_project1_review_20261031_v90' / '项目一_Claude审查包_v90.zip'",
     "OLD = ROOT / 'reports' / 'current' / 'claude_project1_review_20261031_v90' / '项目一_Claude审查包_v90.zip'"),
    ('tools/review/build_external_review_zip_v92.py',
     "(ROOT / 'reports/qfft_incomplete_high_20261031_v91/README.md', 'v91/补跑方案与边界.md')",
     "(ROOT / 'reports/runs/qfft_incomplete_high_20261031_v91/README.md', 'v91/补跑方案与边界.md')"),
    # migration verifier moved one level deeper
    ('tools/verification/verify_migration.py',
     'ROOT=Path(__file__).resolve().parents[1]', 'ROOT=Path(__file__).resolve().parents[2]'),
    # native windows frontends
    ('tools/qfft/start_qfft_supervisor_v73.pyw',
     'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[2]'),
    ('tools/qfft/watch_qfft_progress_v72.pyw',
     'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[2]'),
    ('tools/qfft/watch_qfft_length_retry_v86.pyw',
     "OUT = Path(__file__).resolve().parents[1] / 'artifacts/qfft_length_retry_v86'",
     "OUT = Path(__file__).resolve().parents[2] / 'artifacts/qfft_length_retry_v86'"),
    ('tools/qfft/watch_incomplete_high_v91.pyw',
     "OUT = Path(__file__).resolve().parents[1] / 'artifacts/qfft_incomplete_high_v91'",
     "OUT = Path(__file__).resolve().parents[2] / 'artifacts/qfft_incomplete_high_v91'"),
    # scripts/evaluation: .pilot_deps is at the repo root, and the sibling import gains a package level
    ('scripts/evaluation/math_verify_pilot_worker.py',
     "Path(__file__).resolve().parents[1]/'.pilot_deps'",
     "Path(__file__).resolve().parents[2]/'.pilot_deps'"),
    ('scripts/evaluation/math_verify_v3_strict_worker.py',
     "Path(__file__).resolve().parents[1]/'.pilot_deps'",
     "Path(__file__).resolve().parents[2]/'.pilot_deps'"),
    ('scripts/evaluation/math_verify_v3_strict_worker.py',
     'from scripts.math_verify_pilot_worker import boxes',
     'from scripts.evaluation.math_verify_pilot_worker import boxes'),
    # experiments read the bundle index next to the training data
    ('experiments/aimo_sft_v26/evaluate.py',
     "ROOT/'reports/aimo_sft_20261026_v26/evaluation_sources.json'",
     "ROOT/'data/aimo_sft_v26/evaluation_sources.json'"),
    ('experiments/aimo_sft_v26/run.py',
     "ROOT/'reports/aimo_sft_20261026_v26/bundle_checksums.json'",
     "ROOT/'data/aimo_sft_v26/bundle_checksums.json'"),
    # double-click launcher now lives next to its watcher
    ('tools/qfft/查看造数进度.cmd',
     '%~dp0tools\\watch_qfft_progress_v72.pyw', '%~dp0watch_qfft_progress_v72.pyw'),
]


def main():
    errors = []
    for relative, old, new in EDITS:
        path = ROOT / relative
        text = path.read_text(encoding='utf-8')
        if old not in text:
            errors.append(f'{relative}: pattern not found: {old!r}')
            continue
        path.write_text(text.replace(old, new), encoding='utf-8')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'applied {len(EDITS)} replacements')


if __name__ == '__main__':
    main()
