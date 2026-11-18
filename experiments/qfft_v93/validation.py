"""Independent MMR development evaluation; never load inference on the training GPU."""
from __future__ import annotations

from pathlib import Path
import os
from contextlib import contextmanager

from experiments.qfft_v93.common import (drive_layout, load_config, now_iso, read_json,
                                         read_jsonl, sha256_file, sha256_json)

PLAN = 'grpo/mmr_validation25.json'


def result_root(layout):
    return Path(os.environ.get('P1_VALIDATION_ROOT', str(layout.evals.parent / 'mmr_validation25')))


@contextmanager
def result_drive():
    """A separate writable result drive may be used when the input drive is rate-limited."""
    saved = {key: os.environ.get(key) for key in ('P1_DRIVE_ROOT', 'P1_RCLONE_REMOTE', 'RCLONE_CONFIG')}
    for key in saved:
        replacement = os.environ.get('P1_VALIDATION_' + key.removeprefix('P1_'))
        if replacement:
            os.environ[key] = replacement
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def put_result(path, value):
    from experiments.qfft_v93 import drive_io
    with result_drive():
        drive_io.put_json(path, value)


def scheduled_steps(plan):
    first, final, every = (plan[k] for k in ('first_step', 'final_step', 'every_steps'))
    if any(type(x) is not int or x <= 0 for x in (first, final, every)) or first % every or final % every:
        raise ValueError('Invalid global validation cadence')
    return list(range(first, final + 1, every))


def protocol(plan):
    from experiments.qfft_v93.formats import SYSTEM_PROMPT
    from experiments.qfft_v93.grading import GRADER_VERSION
    dev = load_config(plan['dev_config'])
    evaluation = load_config('pipeline.json')['eval']
    return {'dev_sha256': dev['sha256'], 'rows': dev['rows'],
            'spec': evaluation['suites'][plan['suite']][plan['benchmark']],
            'temperature': evaluation['temperature'], 'top_p': evaluation['top_p'],
            'seed': evaluation['seed'], 'max_model_len': evaluation['max_model_len'],
            'system_prompt': SYSTEM_PROMPT, 'grader': GRADER_VERSION,
            'selection_metric': plan['selection_metric'], 'tie_break': plan['tie_break']}


def select_best(receipts, expected_protocol):
    """Reject mixed protocols and duplicate identities before comparing dev accuracy."""
    if not receipts:
        return None
    steps = set()
    for row in receipts:
        if row['protocol_sha256'] != expected_protocol or row['step'] in steps:
            raise ValueError('Development records have mixed protocols or duplicate steps')
        steps.add(row['step'])
        if row['paired_problems'] != 249 or not 0 <= row['mean_at_n'] <= 1:
            raise ValueError('Development result is incomplete or invalid')
    return max(receipts, key=lambda r: (r['mean_at_n'], -r['step']))


def checkpoint_model(plan, layout, step):
    """Bind exported weights to the exact full checkpoint, not merely a step label."""
    from experiments.qfft_v93 import models
    if step in plan['backfill_steps']:
        run = layout.runs / f'grpo_qfft_r1_mmr_dr{step}_eval_v93'
        checkpoint, model = run / f'checkpoint-{step}', run / 'merged'
        source_identity = read_json(run / 'evaluation_identity.json')['source_run_identity_sha256']
    elif step == plan['first_step']:
        run = layout.runs / plan['source_run']
        checkpoint, model = run / 'ckpt_mirror' / f'checkpoint-{step}', run / 'merged'
        if read_json(run / 'GRPO_COMPLETE.json')['steps'] != step:
            raise RuntimeError('Source training has not completed its final export')
        source_identity = sha256_json(read_json(run / 'run_identity.json')['identity'])
    else:
        run = layout.runs / plan['run']
        checkpoint = run / 'ckpt_mirror' / f'checkpoint-{step}'
        receipt = read_json(run / 'milestones' / f'step-{step}' / 'EXPORT_COMPLETE.json')
        source_identity = sha256_json(read_json(run / 'run_identity.json')['identity'])
        expected = {'step': step, 'run_identity_sha256': source_identity,
                    'checkpoint': str(checkpoint),
                    'checkpoint_manifest_sha256': sha256_file(checkpoint / 'MIRROR_COMPLETE.json')}
        if receipt['provenance'] != expected:
            raise RuntimeError('Validation model belongs to a different checkpoint')
        model = Path(receipt['model'])
        if sha256_file(model / models.MARK) != receipt['model_identity_sha256']:
            raise RuntimeError('Validation model identity changed')
        if sha256_file(model / 'p1_export_manifest.json') != receipt['export_manifest_sha256']:
            raise RuntimeError('Validation model export manifest changed')
    marker = read_json(checkpoint / 'MIRROR_COMPLETE.json')
    required = {'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json'}
    if marker['artifact_type'] != 'full_state' or not required <= marker['file_manifest'].keys():
        raise RuntimeError('A validation checkpoint must preserve all resume state')
    if read_json(checkpoint / 'trainer_state.json')['global_step'] != step or not models._complete(model):
        raise RuntimeError('Validation checkpoint step or model completeness differs')
    return model, {'checkpoint': str(checkpoint), 'checkpoint_step': step,
                   'checkpoint_manifest_sha256': sha256_file(checkpoint / 'MIRROR_COMPLETE.json'),
                   'source_run_identity_sha256': source_identity,
                   'model_identity_sha256': sha256_file(model / models.MARK)}


def publish_selection(root, plan):
    protocol_hash = sha256_json(protocol(plan))
    receipts = []
    expected = scheduled_steps(plan)
    for step in sorted(set(expected + plan['backfill_steps'])):
        path = root / 'reports' / f'step-{step}' / 'VALIDATION_COMPLETE.json'
        if path.exists():
            receipts.append(read_json(path))
    best = select_best(receipts, protocol_hash)
    ledger = {'protocol_sha256': protocol_hash, 'expected_steps': expected,
              'completed_steps': [r['step'] for r in receipts], 'records': receipts,
              'best': best, 'updated': now_iso()}
    put_result(root / 'VALIDATION_INDEX.json', ledger)
    if best is not None:
        put_result(root / 'BEST_VALIDATION.json', best)
    if set(expected) <= {r['step'] for r in receipts}:
        put_result(root / 'VALIDATION_COMPLETE.json', ledger)
    return ledger


def evaluate(ctx, step):
    from experiments.qfft_v93 import drive_io, models
    plan = load_config(PLAN)
    if step not in scheduled_steps(plan) + plan['backfill_steps']:
        raise ValueError('Step is outside the authorized validation schedule')
    layout = drive_layout()
    root = result_root(layout)
    bench_dir = Path(os.environ.get('P1_VALIDATION_BENCH_DIR', str(layout.data / 'r1_grpo_dev')))
    bench = bench_dir / f"{plan['benchmark']}.jsonl"
    dev = load_config(plan['dev_config'])
    if sha256_file(bench) != dev['sha256'] or len(read_jsonl(bench)) != dev['rows']:
        raise RuntimeError('Frozen development data changed')
    model_dir, binding = checkpoint_model(plan, layout, step)
    model = 'run:' + str(model_dir.relative_to(layout.runs))
    tag = f'step{step}'
    out = root / 'generations'
    report = root / 'reports' / f'step-{step}'
    identity = {'step': step, 'model': model, **binding,
                'validation_code_sha256': sha256_file(Path(__file__)),
                'protocol_sha256': sha256_json(protocol(plan))}
    marker = report / 'VALIDATION_COMPLETE.json'
    if marker.exists():
        old = read_json(marker)
        if any(old.get(k) != v for k, v in identity.items()):
            raise RuntimeError('Completed validation differs from its checkpoint or protocol')
        publish_selection(root, plan)
        return
    # generate verifies complete cached baseline identity, prompt and draw metadata as well.
    for name, spec in [('r1', plan['baseline']), (tag, model)]:
        ctx.run(ctx.py_module('infer', 'experiments.qfft_v93.generate', '--model', spec,
                             '--tag', name, '--suite', plan['suite'], '--bench-dir', str(bench_dir),
                             '--evals-root', str(out)), f'validate_mmr_{name}')
        status = read_json(out / name / plan['benchmark'] / 'generation_status.json')
        if not status['complete'] or status['saved'] != dev['rows'] * 2:
            raise RuntimeError('Validation must contain exactly 498 completed generations')
    ctx.run(ctx.py_module('infer', 'experiments.qfft_v93.score', '--bench-dir', str(bench_dir),
                         '--benchmarks', plan['benchmark'], '--require-gradable',
                         '--evals-root', str(out), '--report-dir', str(report),
                         '--tags', f'{tag},r1', '--compare', f'{tag}:r1', '--per-problem'),
            f'score_mmr_dev_{step}')
    results = read_json(report / 'results.json')
    comparison, = results['comparisons']
    if comparison['problems'] != dev['rows'] or len(read_jsonl(report / 'per_problem_pairs.jsonl')) != dev['rows']:
        raise RuntimeError('Validation paired scores are incomplete')
    score = results['results'][tag][plan['benchmark']]
    receipt = {**identity, 'paired_problems': dev['rows'],
               'mean_at_n': score['mean_at_n'], 'baseline_mean_at_n': results['results']['r1'][plan['benchmark']]['mean_at_n'],
               'diff': comparison['diff'], 'ci95': comparison['ci95'],
               'mean_tokens': score['mean_tokens'], 'truncation_rate': score['truncation_rate'],
               'model': model, 'completed': now_iso(), 'report': str(report)}
    put_result(marker, receipt)
    publish_selection(root, plan)
    print((report / 'results.md').read_text(), flush=True)


def install(stages):
    """Permit the first backfill using the unchanged, already-running source code bundle."""
    dispatch = stages._dispatch

    def validation_dispatch(ctx, stage):
        if stage.startswith('mmr-validation:'):
            return evaluate(ctx, int(stage.split(':')[1]))
        return dispatch(ctx, stage)

    stages._dispatch = validation_dispatch
