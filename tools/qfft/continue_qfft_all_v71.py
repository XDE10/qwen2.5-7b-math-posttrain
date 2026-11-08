"""Supervise authorized completion of the frozen 1000 questions, preserving v66 outputs.

Only time-window pauses resume automatically. All other failures stop for review.
No model training, resampling, prompt edits, or extra questions are performed.
"""
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import subprocess
import sys
import time

from build_qfft_ds_dataset_v66 import run_lock, reviewed_unknowns
from qfft_ds_common_v66 import ROOT, OUT, REPORT, read, save, digest, TZ, check_time

CONTROL = ROOT / 'artifacts/qfft_all_supervisor_v71'
WORKERS = 2


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(TZ)


def next_allowed():
    candidate = now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(60 * 48):
        try:
            check_time(candidate)
            return candidate
        except RuntimeError:
            candidate += timedelta(minutes=1)
    raise RuntimeError('No permitted window within 48 hours; stop instead of bypassing the gate.')


def status(state, **extra):
    save(CONTROL / 'status.json', {'state': state, 'updated_at': now().isoformat(),
         'target_questions': 1000, 'source_version': 66, 'max_concurrency': WORKERS,
         'official_cap_usd': 5.0, 'automatic_resampling': False, 'gpu_training': False, **extra})


def saved_count():
    return len(list((OUT / 'candidates').glob('*.json')))


def background_subprocess_kwargs():
    if sys.platform == 'win32':
        return {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def record_process_exit(code, log_path, archive_code):
    """Don't show an older row-level network pause as the current process exit."""
    unsigned = code & 0xffffffff
    message = ('Windows process interrupted (0xC000013A); source of signal undetermined.'
               if unsigned == 0xC000013A else f'Generation process exited with code {code}.')
    pause_path = OUT / 'last_pause.json'
    previous = read(pause_path) if pause_path.exists() else None
    if unsigned == 0xC000013A or not pause_path.exists() or pause_path.stat().st_mtime < log_path.stat().st_ctime:
        stamp = now().strftime('%Y%m%d_%H%M%S_%f')
        if previous is not None:
            save(CONTROL / 'pause_history' / (stamp + '.json'), previous)
        save(pause_path, {'scope': 'supervisor_process_exit', 'observed_at': now().isoformat(),
             'failures': [{'sample_id': None, 'error_type': 'WindowsProcessInterrupted' if unsigned == 0xC000013A else 'ProcessExit',
                           'message': message}], 'saved_candidates': saved_count(),
             'child_exit_code': code, 'archive_exit_code': archive_code,
             'automatic_retry': False, 'prior_pause_archived': previous is not None})
    return message


def verify_complete(baseline):
    rows = read(REPORT / 'questions1000.json')
    identity = digest(read(REPORT / 'protocol.json'))
    assert len(rows) == len({r['sample_id'] for r in rows}) == 1000
    deferred = reviewed_unknowns(OUT, identity)
    for row in rows:
        if row['index'] in deferred:
            assert deferred[row['index']]['sample_id'] == row['sample_id']
            continue
        result = read(OUT / 'candidates' / f'{row["index"]:04d}.json')
        assert result['identity'] == identity and result['sample_id'] == row['sample_id']
    for rel, expected in baseline.items():
        assert file_sha(ROOT / rel) == expected
    for resumed_baseline in CONTROL.glob('resume_v*_existing_candidate_hashes.json'):
        for rel, expected in read(resumed_baseline).items():
            assert file_sha(ROOT / rel) == expected


def accounted_count():
    return saved_count() + len(reviewed_unknowns(OUT, digest(read(REPORT / 'protocol.json'))))


def archive_existing():
    # Local read/export only, including exceptional-length cases. No API requests.
    with (CONTROL / 'archive.log').open('a', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, str(ROOT / 'tools/archive_qfft_followups_v70.py')],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **background_subprocess_kwargs())
    return result.returncode


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    with run_lock(CONTROL):
        baseline_path = CONTROL / 'initial_existing_candidate_hashes.json'
        baseline = (read(baseline_path) if baseline_path.exists() else
                    {p.relative_to(ROOT).as_posix(): file_sha(p)
                     for p in sorted((OUT / 'candidates').glob('*.json'))})
        if not baseline_path.exists():
            save(baseline_path, baseline)
        save(CONTROL / 'authorization.json', {
            'recorded_at': now().isoformat(),
            'user_request': '现在直接跑完1000题吧，可以了吗',
            'scope': 'Complete original frozen 1000, reusing all existing candidates; no resampling/training.',
            'offpeak_only': True, 'official_estimated_cap_usd': 5.0,
            'automatic_resume': 'Time-window pauses only; no automatic retry of any other failure.',
        })
        attempt = 0
        while True:
            if accounted_count() == 1000:
                verify_complete(baseline)
                archive_rc = archive_existing()
                count = saved_count()
                state = 'generation_complete' if count == 1000 else 'finished_with_deferred_requests'
                status(state, saved_candidates=count, deferred_unknown_requests=1000-count, original_candidates_unchanged=True,
                       followup_archive_exit_code=archive_rc, training_approved=False)
                print('QFFT_ALL_V71_GENERATION_COMPLETE_NOT_TRAINING_APPROVAL' if count == 1000 else
                      'QFFT_ALL_V73_FINISHED_WITH_DEFERRED_REQUESTS_NOT_ALL_RESPONSES_RECEIVED', flush=True)
                return
            try:
                check_time()
            except RuntimeError:
                resume_at = next_allowed()
                status('waiting_offpeak', saved_candidates=saved_count(), resume_at=resume_at.isoformat())
                print('WAITING_OFFPEAK until ' + resume_at.isoformat(), flush=True)
                while now() < resume_at:
                    time.sleep(min(30, max(0.1, (resume_at - now()).total_seconds())))
                continue
            attempt += 1
            log_path = CONTROL / ('generation_' + now().strftime('%Y%m%d_%H%M%S_%f') + '.log')
            command = [sys.executable, '-u', str(ROOT / 'tools/build_qfft_ds_dataset_v66.py'),
                       '--run', '--all', '--workers', str(WORKERS), '--official-cap-usd', '5']
            with log_path.open('w', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         **background_subprocess_kwargs())
                status('running', child_pid=child.pid, saved_candidates=saved_count(),
                       log_path=log_path.relative_to(ROOT).as_posix(), invocation=attempt)
                while child.poll() is None:
                    status('running', child_pid=child.pid, saved_candidates=saved_count(),
                           log_path=log_path.relative_to(ROOT).as_posix(), invocation=attempt)
                    time.sleep(10)
                code = child.returncode
            count = saved_count()
            archive_rc = archive_existing()
            if code == 0 and accounted_count() == 1000:
                continue
            failures_path = OUT / 'last_pause.json'
            failures = read(failures_path).get('failures', []) if failures_path.exists() else []
            peak_text = 'Peak window: pause before new requests.'
            tail = log_path.read_text(encoding='utf-8')[-64000:]
            # Resume only when this invocation explicitly stopped at the time gate.
            paused_here = 'Paused; all received outputs saved.' in tail
            all_peak = bool(failures) and all(peak_text in f.get('message', '') for f in failures)
            initial_peak = ('RuntimeError: ' + peak_text) in tail
            if (paused_here and all_peak) or (initial_peak and not paused_here):
                status('paused_for_peak', saved_candidates=count, followup_archive_exit_code=archive_rc,
                       log_path=log_path.relative_to(ROOT).as_posix())
                continue
            current_reason = record_process_exit(code, log_path, archive_rc)
            status('stopped_requires_review', saved_candidates=count, child_exit_code=code,
                   log_path=log_path.relative_to(ROOT).as_posix(),
                   followup_archive_exit_code=archive_rc,
                   reason=current_reason + ' No automatic retry.')
            raise SystemExit(code or 1)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('supervisor_error', error_type=type(error).__name__, saved_candidates=saved_count())
        raise
