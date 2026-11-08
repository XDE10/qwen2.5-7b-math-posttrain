"""Run final-model MATH-500 v30, then the existing 637-question E0 0/4 probe."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('v30_helper', SCRIPT.with_name('evaluate_last_e0_protocol_v30.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[record['sample_id']].append(record)
    complete = [rs for rs in groups.values() if len(rs) == 4 and {r['trial'] for r in rs} == set(range(4))]
    any_correct = sum(any(r['category'] == 'equivalent' for r in rs) for rs in complete)
    eligible = [rs for rs in complete if all(r['category'] in ('equivalent', 'mismatch') and
                                           r['finish_reason'] == 'stop' for r in rs)]
    return {'questions': 637, 'expected_responses': 2548, 'responses': len(records),
        'complete_groups': len(complete), 'complete': len(complete) == 637,
        'questions_with_at_least_one_correct': any_correct,
        'valid_intermediate_1_to_3_of_4': sum(1 <= sum(r['category'] == 'equivalent' for r in rs) <= 3 for rs in eligible),
        'valid_all_wrong': sum(all(r['category'] == 'mismatch' for r in rs) for rs in eligible),
        'valid_all_correct': sum(all(r['category'] == 'equivalent' for r in rs) for rs in eligible),
        'groups_with_unsupported_or_abnormal_output': len(complete) - len(eligible),
        'status_counts': dict(Counter(r['category'] for r in records)),
        'finish_reason_counts': dict(Counter(r['finish_reason'] for r in records)),
        'response_accuracy': sum(r['category'] == 'equivalent' for r in records) / len(records) if records else None,
        'scope': 'Conditional E0 0/4 training-side difficulty probe; NOT an independent test set or training approval'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('/content/workspaces/aimo-sft-v26'))
    parser.add_argument('--run-root', type=Path, default=Path('/content/drive/MyDrive/runs/project1/aimo_sft_v26'))
    parser.add_argument('--runs-root', type=Path, default=Path('/content/drive/MyDrive/runs/project1'))
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--sampling-root', type=Path)
    parser.add_argument('--grade-only', action='store_true')
    parser.add_argument('--worker', choices=['generate', 'grade'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is None:
        command = [sys.executable, '-u', str(SCRIPT.with_name('evaluate_last_e0_protocol_v30.py')),
            '--project-root', str(args.project_root), '--run-root', str(args.run_root)]
        if args.grade_only:
            command.append('--grade-only')
        subprocess.run(command, check=True)
    project = args.project_root.resolve()
    os.chdir(project)
    sys.path.insert(0, str(project))
    from src.training.frozen_verifier_v4 import digest, grader_identity
    from src.training.math_reward_v2 import category
    baseline = args.baseline_root
    if baseline is None:
        found = sorted(p for p in args.runs_root.glob('E0_dev_v9_*')
                       if (p / 'hard_probe_identity.json').exists() and (p / 'hard_candidates_e0_0of4.jsonl').exists())
        if not found:
            raise RuntimeError('Saved E0 637-probe directory not found; no new screening will be run')
        baseline = found[-1]
    sampling = args.sampling_root or args.runs_root / 'rl_screening_v8_n4'
    protocol = helper.read(sampling / 'sampling_identity.json')
    probe = helper.read(baseline / 'hard_probe_identity.json')
    hard_path = baseline / 'hard_candidates_e0_0of4.jsonl'
    if helper.sha(hard_path) != probe['output_sha256'] or helper.sha(sampling / 'sampling_identity.json') != probe['source_sampling_identity_sha256']:
        raise RuntimeError('Historical hard-probe source identity differs')
    if protocol['num_samples'] != 4 or digest(grader_identity()) != protocol['grader_identity_sha256']:
        raise RuntimeError('Historical four-sample protocol/scorer differs')
    hard = helper.journal(hard_path)
    if len(hard) != 637 or {r['sample_id'] for r in hard} != set(probe['sample_ids']):
        raise RuntimeError('Expected the original 637-question probe')
    if any(r['training_approved'] is not False for r in hard):
        raise RuntimeError('Probe must remain unapproved for training')
    by_id = {r['sample_id']: r for r in hard}
    target = args.run_root / 'evaluations/final_hard637_e0_protocol_v31'
    target.mkdir(parents=True, exist_ok=True)
    model_identity = args.run_root / 'merged/sft_identity.json'
    identity = {'script_sha256': helper.sha(SCRIPT), 'helper_sha256': helper.sha(helper.SCRIPT),
        'model': helper.read(model_identity), 'hard_probe': probe, 'sampling_protocol': protocol,
        'grader_identity_sha256': digest(grader_identity()), 'samples_per_question': 4}
    if (target / 'identity.json').exists() and helper.read(target / 'identity.json') != identity:
        raise RuntimeError('Existing hard-probe identity differs')
    helper.save(target / 'identity.json', identity)
    raw_path = target / 'predictions.jsonl'
    raw = helper.journal(raw_path)
    pairs = {(r['sample_id'], r['trial']) for r in raw}
    if len(pairs) != len(raw) or any(sid not in by_id or t not in range(4) for sid, t in pairs):
        raise RuntimeError('Invalid hard-probe generation journal')
    if args.worker == 'generate':
        seal = helper.read(sampling / 'preparation_seal.json')
        token_path = sampling / 'prompt_tokens.jsonl'
        if helper.sha(token_path) != seal['prompt_tokens_sha256']:
            raise RuntimeError('Original prompt tokens changed')
        tokens = {r['sample_id']: r['prompt_token_ids'] for r in helper.journal(token_path) if r['sample_id'] in by_id}
        if set(tokens) != set(by_id):
            raise RuntimeError('Missing historical prompt tokens')
        model = Path('/content/models') / ('aimo_sft_v26_merged--' + helper.sha(model_identity)[:12])
        if not (model / 'model.safetensors.index.json').exists():
            raise RuntimeError('Final model SSD cache missing; run MATH-500 v30 first')
        from vllm import LLM, SamplingParams
        engine = LLM(model=str(model), tokenizer=str(project / 'data/tokenizers/Qwen2.5-7B-Instruct'),
            dtype='bfloat16', tensor_parallel_size=1, max_model_len=max(map(len, tokens.values())) + protocol['max_new_tokens'],
            max_num_seqs=128, gpu_memory_utilization=0.90, enable_prefix_caching=True,
            enable_chunked_prefill=True, max_num_batched_tokens=8192, seed=42, generation_config='vllm')
        # Same four per-ID seeds and 64-question batches as the old hard-probe evaluation.
        for start in range(0, len(hard), 64):
            work = [(r, trial) for r in hard[start:start + 64] for trial in range(4)
                    if (r['sample_id'], trial) not in pairs]
            if not work:
                continue
            seeds = [int(digest([protocol['seed'], r['sample_id'], t])[:8], 16) & 0x7fffffff for r, t in work]
            params = [SamplingParams(n=1, temperature=protocol['temperature'], top_p=protocol['top_p'],
                top_k=-1, max_tokens=protocol['max_new_tokens'], seed=seed) for seed in seeds]
            outputs = engine.generate([{'prompt_token_ids': tokens[r['sample_id']]} for r, _ in work], params, use_tqdm=True)
            if len(outputs) != len(work):
                raise RuntimeError('Hard-probe generation count differs')
            for (row, trial), response, seed in zip(work, outputs, seeds):
                output = response.outputs[0]
                helper.append(raw_path, {'sample_id': row['sample_id'], 'trial': trial, 'text': output.text,
                    'generated_tokens': len(output.token_ids), 'finish_reason': output.finish_reason, 'seed': seed})
            print(f'Hard probe saved: {start + len(hard[start:start + 64])}/637 questions', flush=True)
        print('HARD_GENERATION_COMPLETE: GPU child exits; CPU scoring follows.', flush=True)
        return
    if args.worker == 'grade':
        if len(raw) != 2548:
            raise RuntimeError('Hard-probe generation incomplete')
        path = target / 'graded_predictions.jsonl'
        graded = helper.journal(path)
        done = {(r['sample_id'], r['trial']) for r in graded}
        if len(done) != len(graded) or not done <= pairs:
            raise RuntimeError('Invalid hard-probe scored journal')
        checker = helper.PooledVerifier(4)
        def grade(row):
            result = checker.check(row['text'], by_id[row['sample_id']]['reference_answer'])
            return {**row, 'verification': result, 'category': category(result)}
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for row in pool.map(grade, [r for r in raw if (r['sample_id'], r['trial']) not in done]):
                    if row['category'] in ('timeout', 'verifier_error'):
                        raise RuntimeError('Scorer failure; saved raw retained; rerun to resume CPU grading')
                    helper.append(path, row)
                    graded.append(row)
        finally:
            checker.close()
        result = summarize(graded)
        helper.save(target / 'summary.json', result)
        helper.save(target / 'COMPLETE.json', {'summary_sha256': helper.sha(target / 'summary.json')})
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print('MATH500_AND_HARD637_COMPLETE: both tasks finished; GPU can be stopped.', flush=True)
        return
    import fcntl
    lock_path = args.runs_root / 'projection_pilot_v21/running.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another GPU task is running') from None
        if (target / 'COMPLETE.json').exists():
            print(helper.read(target / 'summary.json'))
            print('MATH500_AND_HARD637_COMPLETE: already complete; no GPU model loaded.')
            return
        command = [sys.executable, '-u', str(SCRIPT), '--project-root', str(project),
            '--run-root', str(args.run_root), '--runs-root', str(args.runs_root),
            '--baseline-root', str(baseline), '--sampling-root', str(sampling)]
        if len(raw) < 2548:
            if args.grade_only:
                raise RuntimeError('Hard-probe generation still requires GPU')
            subprocess.run(command + ['--worker', 'generate'], check=True,
                           env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn'))
        subprocess.run(command + ['--worker', 'grade'], check=True)


if __name__ == '__main__':
    main()
