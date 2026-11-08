"""Explicitly authorized one-pass high repair of the effective 62 incomplete questions."""
from pathlib import Path
import copy
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import qfft_ds_common_v66 as c
import build_qfft_ds_dataset_v66 as b

OUT = c.ROOT / 'artifacts/qfft_incomplete_high_v91'
PREVIOUS = c.ROOT / 'artifacts/qfft_length_retry_v86'
ARGON = Path('D:/Users/Mr.Z/Desktop/项目一审查_Argon.md')
OLD_LINES = c.SYSTEM.splitlines()
assert OLD_LINES[4].startswith('Use alternative approaches')
assert OLD_LINES[5].startswith('Before giving the final answer')
SYSTEM = '\n'.join(OLD_LINES[:4] + [
    'Once a rigorous derivation reaches a verified result, proceed directly to the final solution without repeating full-problem recalculations or exploring redundant secondary methods. Avoid repetitive self-doubt loops.',
    'If an approach fails, identify and correct the specific error, resolve contradictions, and continue with a viable method. Keep all essential derivations and checks needed to justify the result.',
] + OLD_LINES[6:])


def effective_view(original, repairs):
    view = dict(original)
    for index, candidate in repairs.items():
        if not candidate['structurally_complete']:
            raise ValueError('A repair may replace an original only if structurally complete')
        view[index] = candidate
    return view


def outcome(candidate):
    if not candidate['structurally_complete']:
        return 'incomplete'
    return 'complete_within_16k' if candidate['fits_current_training_window'] else 'complete_over_16k'


def wire_payload(row, provider):
    body = c.payload(row, provider)
    body['messages'][0]['content'] = SYSTEM
    body['reasoning_effort'] = 'high'
    body.update(stream=True, stream_options={'include_usage': True})
    return body


def transport(provider, body, secret, session):
    return b.http_post(provider, body, secret, 'project1-qfft-v91-' + session.rsplit('-', 1)[-1])


def prepare():
    original_paths = sorted((c.OUT / 'candidates').glob('*.json'))
    original = {int(p.stem): c.read(p) for p in original_paths}
    assert len(original) == 1000 and c.read(PREVIOUS / 'status.json')['state'] == 'complete'
    prior_plan = c.read(PREVIOUS / 'plan.json')
    assert all(c.sha_file(c.ROOT / p) == sha for p, sha in prior_plan['original_candidate_hashes'].items())
    repairs, repair_paths = {}, []
    result_paths = sorted((PREVIOUS / 'results').glob('*.json'))
    assert len(result_paths) == 41
    for path in result_paths:
        result = c.read(path)
        if result.get('selected_candidate'):
            candidate_path = Path(result['selected_candidate'])
            repairs[result['index']] = c.read(candidate_path)
            repair_paths.append(candidate_path)
    view = effective_view(original, repairs)
    selected = sorted(i for i, item in view.items() if not item['structurally_complete'])
    assert len(repairs) == 27 and len(selected) == 62
    plan = {
        'version': 91, 'selected': selected, 'workers': 2, 'effort': 'high', 'max_new_attempts_per_question': 1,
        'authorization': 'User asks to rerun incomplete questions following Argon, explicitly selects high plus anti-redundancy prompt, two concurrent questions, once per question.',
        'parent_protocol_identity': c.digest(c.read(c.REPORT / 'protocol.json')),
        'already_repaired_27_skipped': sorted(repairs), 'complete_over_16k_not_selected': 180,
        'system_prompt': SYSTEM, 'system_prompt_sha256': c.digest(SYSTEM),
        'prompt_variant': 'argon_anti_redundancy_high_v91', 'teacher_output_limit': 32768,
        'student_window': 16384, 'thinking': 'enabled', 'official_cumulative_cap_usd': 5,
        'argon_source_sha256': c.sha_file(ARGON),
        'source_file_hashes': {p.relative_to(c.ROOT).as_posix(): c.sha_file(p)
                               for p in original_paths + result_paths + repair_paths},
        'source_original_candidates': {str(i): (c.OUT / 'candidates' / f'{i:04d}.json').relative_to(c.ROOT).as_posix()
                                       for i in selected},
        'quality_review_complete': False, 'training_approved': False,
        'reasoning_effort_note': 'Official DeepSeek docs map medium to high; user explicitly chose high, not low.',
        'docs_url': 'https://api-docs.deepseek.com/guides/thinking_mode/',
    }
    c.freeze(OUT / 'plan.json', plan)
    c.freeze(OUT / 'identity.json', {'identity': c.digest(plan), 'protocol': plan})
    (OUT / 'teacher_system_prompt.txt').write_text(SYSTEM + '\n', encoding='utf-8')
    return plan


def verify_sources(plan):
    assert all(c.sha_file(c.ROOT / p) == sha for p, sha in plan['source_file_hashes'].items())


def run():
    with b.run_lock(OUT), b.run_lock(c.OUT), b.run_lock(PREVIOUS):
        plan = c.read(OUT / 'plan.json')
        verify_sources(plan)
        assert plan['effort'] == 'high' and plan['system_prompt'] == SYSTEM
        rows = {r['index']: r for r in c.read(c.REPORT / 'questions1000.json')}
        prior_spend = sum(c.read(p / 'routing.json')['official_estimated_usd'] for p in (c.OUT, PREVIOUS))
        assert 0 <= prior_spend <= 5
        # Isolated process: use this branch for new requests and exact cached-request checks.
        b.wire_payload = wire_payload
        router = b.Router(OUT, c.digest(plan), 5 - prior_spend, transport=transport)
        tok = b.tokenizer()
        lock, stop = threading.RLock(), threading.Event()
        iterator = iter(plan['selected'])
        state = {'state': 'running', 'target': 62, 'workers': 2, 'effort': 'high', 'completed': 0,
                 'complete_within_16k': 0, 'complete_over_16k': 0, 'incomplete': 0, 'transport_unknown_skip': 0,
                 'active': [], 'official_prior_estimated_usd': prior_spend,
                 'official_total_estimated_cap_usd': 5, 'training_approved': False}

        def persist():
            state['updated_at'] = datetime.now(c.TZ).isoformat()
            state['official_new_estimated_usd'] = router.spent
            c.save(OUT / 'status.json', state)

        def wait_window():
            while not stop.is_set():
                try:
                    c.check_time(); return
                except RuntimeError as error:
                    if 'Peak window' not in str(error): raise
                    with lock:
                        state['state'] = 'waiting_offpeak'; persist()
                    stop.wait(20)
            raise RuntimeError('Run stopped; no further request')

        def worker():
            while not stop.is_set():
                with lock:
                    index = next(iterator, None)
                if index is None: return
                try:
                    path = OUT / 'results' / f'{index:04d}.json'
                    if path.exists():
                        result = c.read(path)
                    else:
                        wait_window()
                        with lock:
                            state['state'] = 'running'; state['active'].append(index); persist()
                        row = copy.deepcopy(rows[index]); row['difficulty']['reasoning_effort'] = 'high'
                        result = {'index': index, 'sample_id': row['sample_id'], 'effort': 'high',
                                  'prompt_variant': plan['prompt_variant'], 'authorization': str(OUT / 'plan.json'),
                                  'training_approved': False}
                        if index in router.deferred:
                            result['outcome'] = 'transport_unknown_skip'
                        else:
                            try:
                                record = router.generate(row)
                                candidate = b.candidate(row, record, tok)
                                candidate.update(prompt_variant=plan['prompt_variant'], authorization=str(OUT / 'plan.json'),
                                                 parent_original_candidate=plan['source_original_candidates'][str(index)])
                                candidate_path = OUT / 'candidates' / f'{index:04d}.json'
                                c.save(candidate_path, candidate)
                                result.update(outcome=outcome(candidate), candidate_path=str(candidate_path),
                                              structural_flags=candidate['structural_flags'])
                            except b.DeferredGoServerFailure as error:
                                if error.circuit_open: raise
                                result['outcome'] = 'transport_unknown_skip'
                        c.save(path, result)
                        print(f'RESULT index={index:04d} outcome={result["outcome"]} effort=high', flush=True)
                    with lock:
                        state['completed'] += 1; state[result['outcome']] += 1
                        if index in state['active']: state['active'].remove(index)
                        persist()
                except Exception as error:
                    stop.set()
                    with lock:
                        state['state'] = 'paused_requires_review'
                        state.setdefault('error', {'index': index, 'type': type(error).__name__, 'message': str(error)[:500]})
                        persist()
                    return

        persist()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker) for _ in range(2)]
            for future in futures: future.result()
        verify_sources(plan)
        with lock:
            if not stop.is_set(): state.update(state='complete', active=[])
            state['all_parent_sources_sha_verified'] = True
            persist()


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    if '--run' not in sys.argv:
        print('PREPARED_ONLY', len(prepare()['selected']), 'questions; no API calls')
    else:
        sys.executable = str(Path(sys.executable).with_name('python.exe'))
        with (OUT / 'run.log').open('a', encoding='utf-8', buffering=1) as log:
            sys.stdout = log; sys.stderr = log
            try:
                run()
            except Exception as error:
                import traceback
                traceback.print_exc()
                s = c.read(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
                s.update(state='paused_requires_review', error={'type': type(error).__name__, 'message': str(error)[:500]})
                c.save(OUT / 'status.json', s)
                raise
