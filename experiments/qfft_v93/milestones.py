"""Retained full checkpoints and standalone model exports, without using the training GPU."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.qfft_v93 import drive_io
from experiments.qfft_v93.checkpointing import MIRROR_MARK, _resumable, read_state
from experiments.qfft_v93.common import (local_root, log, now_iso, read_json, sha256_file, sha256_json, write_json)
from experiments.qfft_v93.models import _complete


def _checkpoint(job, step):
    root = Path(job['source_run']) if step == job['source_step'] else Path(job['drive_run'])
    return root / 'ckpt_mirror' / f'checkpoint-{step}'


def _export_root(job, step):
    root = Path(job['drive_run'])
    return root / 'merged' if step == job['final_step'] else root / 'milestones' / f'step-{step}' / 'merged'


def _receipt(job, step):
    return Path(job['drive_run']) / 'milestones' / f'step-{step}' / 'EXPORT_COMPLETE.json'


def _expected(job, step, checkpoint):
    return {'step': step, 'run_identity_sha256': job['run_identity_sha256'],
            'checkpoint': str(checkpoint), 'checkpoint_manifest_sha256': sha256_file(checkpoint / MIRROR_MARK)}


def exported(job, step):
    checkpoint, path = _checkpoint(job, step), _receipt(job, step)
    if not path.exists():
        return False
    receipt = read_json(path)
    if receipt['provenance'] != _expected(job, step, checkpoint):
        raise RuntimeError(f'Milestone {step} belongs to a different checkpoint or run')
    model = Path(receipt['model'])
    if not _complete(model) or sha256_file(model / 'p1_model_identity.json') != receipt['model_identity_sha256']:
        raise RuntimeError(f'Milestone {step} model is incomplete or its identity changed')
    if step != job['source_step']:
        manifest = model / 'p1_export_manifest.json'
        if sha256_file(manifest) != receipt['export_manifest_sha256']:
            raise RuntimeError(f'Milestone {step} export manifest changed')
        files = read_json(manifest)['files']
        if any((model / name).stat().st_size != record['bytes'] for name, record in files.items()):
            raise RuntimeError(f'Milestone {step} weights are incomplete')
    return True


def export_one(job, step):
    """Return False until a full checkpoint is published; verify all checkpoint bytes before merging."""
    checkpoint = _checkpoint(job, step)
    if not (checkpoint / MIRROR_MARK).exists():
        return False
    if not _resumable(checkpoint, MIRROR_MARK):
        raise RuntimeError(f'Milestone {step} is not a full resumable checkpoint')
    if exported(job, step):
        return True
    expected = _expected(job, step, checkpoint)
    if step == job['source_step']:
        source = Path(job['source_run'])
        if read_json(source / 'GRPO_COMPLETE.json')['steps'] != step:
            raise RuntimeError('Source final export has no matching completion receipt')
        if sha256_json(read_json(source / 'run_identity.json')['identity']) != job['source_identity_sha256']:
            raise RuntimeError('Source identity changed before registering its milestone')
        out = source / 'merged'
        if not _complete(out):
            raise RuntimeError('The source final merged model is incomplete')
        # Old exporters record a local adapter path; its run name survives VM changes.
        adapter = Path(read_json(out / 'p1_model_identity.json')['adapter'])
        if adapter.parent.name != source.name or adapter.name != 'adapter':
            raise RuntimeError('The source model was merged from another adapter')
        for name, record in read_json(checkpoint / MIRROR_MARK)['file_manifest'].items():
            if name.startswith('adapter_') and sha256_file(source / 'adapter' / name) != record['sha256']:
                raise RuntimeError('The source final adapter differs from its retained checkpoint')
    else:
        staging = local_root() / f'checkpoint-{step}'
        metadata = read_json(checkpoint / MIRROR_MARK)
        manifest = metadata.get('file_manifest')
        if not manifest:
            raise RuntimeError('Milestone export requires the full checksum manifest')
        drive_io.download_tree(checkpoint, staging)
        for name, record in manifest.items():
            if sha256_file(staging / name) != record['sha256']:
                raise RuntimeError(f'Milestone {step} checksum mismatch: {name}')
        if read_state(staging)['global_step'] != step:
            raise RuntimeError('Milestone Trainer state has the wrong global step')
        from experiments.qfft_v93.merge_lora import merge
        out = _export_root(job, step)
        merge(Path(job['base']), staging, out, provenance=expected)
        shutil.rmtree(staging)
    if not _complete(out):
        raise RuntimeError(f'Milestone {step} merge did not publish complete weights')
    receipt = {'provenance': expected, 'model': str(out), 'completed': now_iso(),
               'model_identity_sha256': sha256_file(out / 'p1_model_identity.json')}
    if step != job['source_step']:
        receipt['export_manifest_sha256'] = sha256_file(out / 'p1_export_manifest.json')
    drive_io.put_json(_receipt(job, step), receipt)
    log(f'Milestone {step}: full checkpoint retained; standalone model -> {out}')
    return True


def worker(job):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Milestone exports must hide all CUDA devices')
    local_root().mkdir(parents=True, exist_ok=True)
    with (local_root() / 'worker.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A milestone export worker is already running on this VM')
        errors = 0
        while True:
            try:
                done = [step for step in job['steps'] if export_one(job, step)]
                errors = 0
                drive_io.put_json(Path(job['drive_run']) / 'milestones' / 'status.json',
                                  {'state': 'exporting', 'completed_steps': done, 'checked': now_iso()})
                if done == job['steps']:
                    drive_io.put_json(Path(job['drive_run']) / 'MILESTONES_COMPLETE.json', {
                        'run_identity_sha256': job['run_identity_sha256'], 'steps': done,
                        'models': {str(step): read_json(_receipt(job, step))['model'] for step in done},
                        'completed': now_iso()})
                    return
            except Exception as error:
                errors += 1
                log(f'Milestone export error ({errors}/3): {type(error).__name__}: {error}')
                drive_io.put_json(Path(job['drive_run']) / 'milestones' / 'status.json', {
                    'state': 'retrying' if errors < 3 else 'failed', 'error': str(error), 'checked': now_iso()})
                if errors >= 3:
                    raise
            time.sleep(30)


def start_worker(cfg, identity, base, drive_run, source):
    job = {'steps': cfg['retained_checkpoint_steps'], 'final_step': cfg['max_steps'],
           'source_step': identity['extension']['source_step'], 'source_run': str(source),
           'source_identity_sha256': identity['extension']['source_identity_sha256'],
           'drive_run': str(drive_run), 'base': str(base), 'run_identity_sha256': sha256_json(identity)}
    job_path = local_root() / 'runs' / cfg['run_name'] / 'milestone_job.json'
    write_json(job_path, job)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '',
           'P1_LOCAL_ROOT': str(local_root() / 'milestone_exports' / cfg['run_name']),
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    log_path = job_path.parent / 'milestone_exports.log'
    with log_path.open('a') as output:
        process = subprocess.Popen([sys.executable, '-u', '-m', __name__, '--job', str(job_path)],
                                   env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    return process, job


def wait_worker(process, job):
    code = process.wait()
    if code or not all(exported(job, step) for step in job['steps']):
        raise RuntimeError(f'Milestone export worker did not complete (exit={code}); full checkpoints remain retained')
    return read_json(Path(job['drive_run']) / 'MILESTONES_COMPLETE.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    worker(read_json(args.job))


if __name__ == '__main__':
    main()
