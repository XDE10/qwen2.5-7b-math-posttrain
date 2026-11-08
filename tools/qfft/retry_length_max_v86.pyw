"""User-authorized bounded max -> high repair; original 1000 stays read-only."""
from pathlib import Path
import copy
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import qfft_ds_common_v66 as c
import build_qfft_ds_dataset_v66 as b

OUT = c.ROOT / 'artifacts/qfft_length_retry_v86'
PLAN = OUT / 'plan.json'


def select(rows):
    exhausted = [r for r in rows if r['finish_reason'] == 'length']
    return (sorted(r['index'] for r in exhausted if r['reasoning_effort_requested'] == 'max'),
            sorted(r['index'] for r in exhausted if r['reasoning_effort_requested'] == 'high'))


def cascade(index, attempt):
    for effort in ('max', 'high'):
        result = attempt(index, effort)
        if result['success_structural']:
            return {'index': index, 'outcome': 'complete_pending_review', 'selected_effort': effort,
                    'selected_candidate': result['candidate_path'], 'training_approved': False}
    return {'index': index, 'outcome': 'both_failed_skip', 'training_approved': False}


def prepare():
    original = [c.read(p) for p in sorted((c.OUT / 'candidates').glob('*.json'))]
    assert len(original) == 1000
    selected, skipped = select(original)
    plan = {'version': 86, 'identity': c.digest(c.read(c.REPORT / 'protocol.json')),
            'authorization': 'User: skip original high; original max gets one max, then one high only on failure; two concurrent questions.',
            'selected': selected, 'skipped_original_high': skipped,
            'original_length_count': len(selected) + len(skipped),
            'workers': 2, 'max_attempts_per_selected_question': 2,
            'success': 'finish_reason=stop and nonempty reasoning and final; not mathematical correctness',
            'complete_over_16k': 'preserve separately; structurally successful, never hard truncate',
            'transport_unknown': 'save evidence; one distinct authorized high attempt only if circuit remains closed',
            'original_candidate_hashes': {p.relative_to(c.ROOT).as_posix(): c.sha_file(p)
                                          for p in sorted((c.OUT / 'candidates').glob('*.json'))},
            'training_approved': False}
    c.freeze(PLAN, plan)
    return plan


def verify_original(plan):
    assert all(c.sha_file(c.ROOT / p) == sha for p, sha in plan['original_candidate_hashes'].items())


def run():
    with b.run_lock(OUT), b.run_lock(c.OUT):
        plan = c.read(PLAN)
        verify_original(plan)
        c.freeze(OUT / 'identity.json', {'identity': plan['identity'], 'retry_plan_sha256': c.digest(plan)})
        rows = {r['index']: r for r in c.read(c.REPORT / 'questions1000.json')}
        # One shared router for both tiers: official cost/reservations and circuit are global.
        prior_spend = c.read(c.OUT / 'routing.json')['official_estimated_usd']
        assert 0 <= prior_spend <= 5
        allowed_slots = set(plan['selected']) | {10000 + i for i in plan['selected']}
        router = b.Router(OUT, plan['identity'], 5 - prior_spend, allowed_deferred_indices=allowed_slots)
        tok = b.tokenizer()
        stop = threading.Event()
        lock = threading.RLock()
        iterator = iter(plan['selected'])
        state = {'state': 'running', 'target': len(plan['selected']), 'original_high_skipped': len(plan['skipped_original_high']),
                 'workers': 2, 'completed': 0, 'complete_pending_review': 0, 'both_failed_skip': 0,
                 'active': {}, 'official_prior_estimated_usd': prior_spend,
                 'official_total_estimated_cap_usd': 5, 'training_approved': False}

        def persist():
            state['updated_at'] = datetime.now(c.TZ).isoformat()
            state['official_new_estimated_usd'] = router.spent
            c.save(OUT / 'status.json', state)

        def wait_window():
            while not stop.is_set():
                try:
                    c.check_time()
                    return
                except RuntimeError as error:
                    if 'Peak window' not in str(error):
                        raise
                    with lock:
                        state['state'] = 'waiting_offpeak'; persist()
                    stop.wait(20)
            raise RuntimeError('Run stopped; no further request')

        def attempt(index, effort):
            # Separate dispatch slot for high, confined to this retry run (not main indices).
            slot = index if effort == 'max' else 10000 + index
            folder = OUT / 'attempts' / f'{index:04d}' / effort
            summary = folder / 'summary.json'
            if summary.exists():
                return c.read(summary)
            wait_window()
            with lock:
                state['state'] = 'running'; state['active'][str(index)] = effort; persist()
            row = copy.deepcopy(rows[index]); row['difficulty']['reasoning_effort'] = effort
            dispatch_row = copy.deepcopy(row); dispatch_row['index'] = slot
            c.save(folder / 'provenance.json', {'original_index': index, 'dispatch_slot': slot,
                   'effort': effort, 'source_candidate': str(c.OUT / 'candidates' / f'{index:04d}.json'),
                   'authorization': str(PLAN), 'training_approved': False})
            try:
                if slot in router.deferred:
                    result = {'success_structural': False, 'failure': 'saved_transport_unknown_no_repeat'}
                else:
                    record = router.generate(dispatch_row)
                    result = b.candidate(row, record, tok)
                    result['dispatch_slot'] = slot
                    result['authorization'] = str(PLAN)
                    candidate_path = folder / 'candidate.json'
                    c.save(candidate_path, result)
                    result = {'success_structural': result['structurally_complete'],
                              'candidate_path': str(candidate_path), 'flags': result['structural_flags']}
            except b.DeferredGoServerFailure as error:
                if error.circuit_open:
                    raise
                result = {'success_structural': False, 'failure': 'transport_unknown',
                          'raw_calls_directory': str(OUT / 'calls' / f'{slot:04d}')}
            c.save(summary, result)
            print(f'ATTEMPT index={index:04d} effort={effort} success={result["success_structural"]}', flush=True)
            return result

        def worker():
            while not stop.is_set():
                with lock:
                    index = next(iterator, None)
                if index is None:
                    return
                try:
                    path = OUT / 'results' / f'{index:04d}.json'
                    result = c.read(path) if path.exists() else cascade(index, attempt)
                    c.save(path, result)
                    with lock:
                        state['completed'] += 1; state[result['outcome']] += 1
                        state['active'].pop(str(index), None); persist()
                    print(f'RESULT index={index:04d} outcome={result["outcome"]}', flush=True)
                except Exception as error:
                    stop.set()
                    with lock:
                        state['state'] = 'paused_requires_review'
                        state['error'] = {'index': index, 'type': type(error).__name__, 'message': str(error)[:500]}
                        persist()
                    return

        persist()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker) for _ in range(2)]
            for future in futures:
                future.result()
        verify_original(plan)
        with lock:
            if not stop.is_set():
                state['state'] = 'complete'
                state['active'] = {}
            state['original_1000_hashes_verified'] = True
            persist()


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    if '--run' not in sys.argv:
        plan = prepare()
        print('PREPARED', len(plan['selected']), 'SKIPPED_HIGH', len(plan['skipped_original_high']))
    else:
        sys.executable = str(Path(sys.executable).with_name('python.exe'))
        with (OUT / 'run.log').open('a', encoding='utf-8', buffering=1) as log:
            sys.stdout = log; sys.stderr = log
            try:
                run()
            except Exception as error:
                import traceback
                traceback.print_exc()
                c.save(OUT / 'fatal_error.json', {'type': type(error).__name__, 'message': str(error)[:500]})
                state = c.read(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
                state.update(state='paused_requires_review', error={'type': type(error).__name__, 'message': str(error)[:500]})
                c.save(OUT / 'status.json', state)
                raise
