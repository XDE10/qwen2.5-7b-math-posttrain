"""32-question inference-only adaptation of aakaran/finetuning-with-sampling.

Uses the upstream mean-logprob hill-climb approximation, NOT exact MH.
No optimizer, SFT, training approval, or paid API is used.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def append(path, obj):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


QUESTION_SUFFIX = '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
PROPOSAL = '''You are given a mathematics problem, an expert solution, and a partial response.
Use the expert solution to continue the partial response in your own words.
Preserve necessary conditions, calculations, and cases. Do not merely copy the final answer.
Finish with the correct final answer inside \\boxed{{}}.

Problem:
{question}

Expert solution:
{solution}

Partial response:
{partial}

Continue the partial response; do not restart it.'''


class ContextOverflow(Exception):
    pass


class BudgetPause(Exception):
    pass


class Backend:
    def __init__(self, model_path, config):
        import torch
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.config = config
        self.llm = LLM(model=str(model_path), tokenizer=str(model_path), dtype='bfloat16',
                       tensor_parallel_size=1, max_model_len=config['max_model_len'],
                       max_num_seqs=1, max_num_batched_tokens=2048,
                       gpu_memory_utilization=config['gpu_memory_utilization'],
                       enable_chunked_prefill=True, enable_prefix_caching=True,
                       generation_config='vllm', seed=config['seed'])
        self.tok = self.llm.get_tokenizer()
        self.eos = self.tok.eos_token_id
        self.generated = 0
        self.scored = 0
        self.input_tokens = 0
        self.calls = 0
        self.torch = torch

    def prompt(self, text):
        # Only render the user prompt. Rendering a full assistant message with the
        # R1 template may REMOVE its thought span. Continuation IDs stay separate.
        result = self.tok.apply_chat_template([{'role': 'user', 'content': text}],
                    tokenize=True, add_generation_prompt=True, return_dict=False)
        if isinstance(result, dict):
            result = result['input_ids']
        return list(result)

    def score(self, question, ids):
        if not ids:
            raise ValueError('Cannot score an empty response')
        prefix = self.prompt(question + QUESTION_SUFFIX)
        full = prefix + ids
        if len(full) + 1 > self.config['max_model_len']:
            raise ContextOverflow('question-only scoring context exceeds limit')
        # Crucial: NO expert solution in this scoring prompt.
        result = self.llm.generate([{'prompt_token_ids': full}],
                    self.SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0), use_tqdm=False)[0]
        values = result.prompt_logprobs[len(prefix):]
        if len(values) != len(ids):
            raise RuntimeError('Prompt logprob coverage mismatch')
        lp = []
        for token, value in zip(ids, values):
            if value is None or token not in value:
                raise RuntimeError('Missing observed-token prompt logprob')
            number = float(value[token].logprob)
            if not math.isfinite(number):
                raise RuntimeError('Non-finite logprob')
            lp.append(number)
        self.scored += len(ids)
        self.input_tokens += len(full)
        self.generated += len(result.outputs[0].token_ids)
        self.calls += 1
        return sum(lp) / len(lp)

    def propose(self, row, prefix, limit, seed):
        partial = self.tok.decode(prefix, skip_special_tokens=False)
        text = PROPOSAL.format(question=row['question'], solution=row['solution'], partial=partial)
        prompt = self.prompt(text) + prefix
        # As in upstream, the partial trace appears in the instruction AND as the
        # actual assistant continuation. Never truncate the expert or prefix.
        if len(prompt) + limit > self.config['max_model_len']:
            raise ContextOverflow(f'proposal context {len(prompt)}+{limit} exceeds limit')
        output = self.llm.generate([{'prompt_token_ids': prompt}],
                    self.SamplingParams(max_tokens=limit, temperature=0.6, top_p=1.0,
                        repetition_penalty=1.1, stop_token_ids=[self.eos], seed=seed), use_tqdm=False)[0].outputs[0]
        new = list(output.token_ids)
        stop = output.finish_reason == 'stop'
        if stop and (not new or new[-1] != self.eos):
            # EOS can be omitted from returned text/IDs by a backend; represent
            # the successful stop explicitly, so partial blocks are not extended.
            new.append(self.eos)
        self.generated += len(output.token_ids)
        self.input_tokens += len(prompt)
        self.calls += 1
        return prefix + new

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def response_ids(self, text):
        # The fixed R1 generation prefix already includes <think>\n. Do not
        # score an expert's leading tag for a second time.
        text = text.lstrip()
        if text.startswith('<think>'):
            text = text[len('<think>'):].lstrip('\r\n')
        ids = self.tok.encode(text, add_special_tokens=False)
        if not ids or ids[-1] != self.eos:
            ids.append(self.eos)
        return ids


def sample(row, backend, config, updates, output, state=None, checkpoint=None, deadline=None):
    rng = random.Random()
    if state:
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, list) else value
        rng.setstate(tuples(state['rng']))
        ids, block_start, events = state['ids'], state['next_block'], state['events']
        generated, scored, inputs, calls = state['counts']
        accumulated_seconds = state['seconds']
    else:
        rng.seed(config['seed'] + int(row['sample_id'][:8], 16))
        ids, block_start, events = [], 0, []
        generated = scored = inputs = calls = 0
        accumulated_seconds = 0.0
    start_counts = (backend.generated, backend.scored, backend.input_tokens, backend.calls)
    started = time.monotonic()
    width = config['max_new_tokens'] // config['block_num']
    score = backend.score(row['question'], ids) if ids else None
    status = 'length'
    for block in range(block_start, config['block_num']):
        if ids and ids[-1] == backend.eos:
            status = 'stop'
            break
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetPause()
        try:
            ids = backend.propose(row, ids, min(width, config['max_new_tokens'] - len(ids)), rng.randrange(2**31))
            if not ids:
                status = 'empty_output'
                break
            score = backend.score(row['question'], ids)
            for update in range(updates):
                cut = rng.randrange(len(ids))
                proposal = backend.propose(row, ids[:cut], len(ids) - cut, rng.randrange(2**31))
                new_score = backend.score(row['question'], proposal)
                accepted = new_score > score
                events.append({'block': block, 'update': update, 'cut': cut,
                               'old_mean_logprob': score, 'candidate_mean_logprob': new_score,
                               'accepted': accepted, 'candidate_tokens': len(proposal)})
                if accepted:
                    ids, score = proposal, new_score
        except ContextOverflow as error:
            status = 'context_overflow'
            events.append({'block': block, 'error': str(error)})
            break
        counts = [old + now - base for old, now, base in zip(
            (generated, scored, inputs, calls),
            (backend.generated, backend.scored, backend.input_tokens, backend.calls), start_counts)]
        current = {'ids': ids, 'next_block': block + 1, 'events': events,
                   'rng': rng.getstate(), 'counts': counts,
                   'seconds': accumulated_seconds + time.monotonic() - started}
        if checkpoint:
            checkpoint(current)
        print(f"  block {block+1}/{config['block_num']} tokens={len(ids)} mean_logprob={score:.4f}", flush=True)
        if ids[-1] == backend.eos:
            status = 'stop'
            break
    if ids and ids[-1] == backend.eos and status == 'length':
        status = 'stop'
    counts = [old + now - base for old, now, base in zip(
        (generated, scored, inputs, calls),
        (backend.generated, backend.scored, backend.input_tokens, backend.calls), start_counts)]
    attempts = [e for e in events if 'accepted' in e]
    return {'sample_id': row['sample_id'], 'method': f'projection_{updates}', 'updates_per_block': updates,
            'text': backend.decode(ids), 'token_ids': ids, 'finish_reason': status,
            'final_mean_logprob': score, 'attempts': len(attempts),
            'acceptances': sum(e['accepted'] for e in attempts),
            'generation_tokens_spent': counts[0], 'scored_response_tokens': counts[1],
            'input_tokens_spent': counts[2], 'engine_calls': counts[3],
            'runtime_seconds': accumulated_seconds + time.monotonic() - started,
            'trace': events, 'training_approved': False}


def summarize(records, selected, completed=False):
    result = {'scope': 'training-side 32-question sampling pilot; not benchmark or SFT gain',
              'training_approved': False, 'completed': completed, 'methods': {}}
    for method in sorted({r['method'] for r in records}):
        group = [r for r in records if r['method'] == method]
        attempts = sum(r['attempts'] for r in group)
        result['methods'][method] = {'count': len(group),
            'grade_categories': dict(Counter(r['grade_category'] for r in group)),
            'finish_reasons': dict(Counter(r['finish_reason'] for r in group)),
            'completed_verified_equivalent': sum(r['grade_category'] == 'equivalent' and r['finish_reason'] == 'stop' for r in group),
            'mean_output_tokens': sum(len(r['token_ids']) for r in group)/len(group),
            'mean_logprob_gain_over_original': sum(r['final_mean_logprob']-r['original_mean_logprob'] for r in group if r['final_mean_logprob'] is not None)/len(group),
            'acceptance_rate': sum(r['acceptances'] for r in group)/attempts if attempts else None,
            'runtime_seconds': sum(r['runtime_seconds'] for r in group),
            'generation_tokens_spent': sum(r['generation_tokens_spent'] for r in group),
            'input_tokens_spent': sum(r['input_tokens_spent'] for r in group)}
    result['reference_sources'] = dict(Counter(r['reference_source'] for r in selected))
    result['interpretation'] = 'Higher likelihood is not a proof of correctness, retained reasoning, or downstream improvement.'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/project1_projection_pilot_v21.json')
    p.add_argument('--model-path', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--time-budget-minutes', type=float, default=90)
    a = p.parse_args()
    c = read(ROOT/a.config)
    data = ROOT/c['data_file']
    if sha(data) != c['data_sha256']:
        raise ValueError('Selected input bytes differ from frozen pilot')
    selected = rows(data)
    if len(selected) != 32 or len({r['sample_id'] for r in selected}) != 32:
        raise ValueError('Expected 32 unique fixed questions')
    out = Path(a.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from src.training.frozen_verifier_v4 import Worker, grader_identity
    from src.training.math_reward_v2 import category
    # Load the vendored dependency path before recording package versions, just
    # as the isolated grading subprocess does.
    from scripts import math_verify_v3_strict_worker as frozen_rules
    identity = {'config': c, 'sampler_sha256': sha(__file__), 'grader': grader_identity(),
                'model_path': str(Path(a.model_path).resolve()),
                'model_identity': read(Path(a.model_path)/'projection_model_identity.json')}
    if identity['model_identity']['revision'] != c['model_revision']:
        raise ValueError('Wrong model revision')
    for name, expected in c['tokenizer_sha256'].items():
        if sha(Path(a.model_path)/name) != expected:
            raise ValueError(f'Model tokenizer differs: {name}')
    if (out/'run_identity.json').exists():
        if read(out/'run_identity.json') != identity:
            raise ValueError('Output belongs to a different protocol; use a new path')
    else:
        save(out/'run_identity.json', identity)
    # OS lock is released even after a killed runtime; it is never a stale PID file.
    if os.name == 'nt':
        raise RuntimeError('Run the GPU sampler on Linux/Colab; local preparation is CPU only')
    import fcntl
    lock = (out/'running.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = rows(out/'results.jsonl') if (out/'results.jsonl').exists() else []
    done = {(r['sample_id'], r['updates_per_block']) for r in records}
    if len(done) != len(records):
        raise ValueError('Duplicate completed records')
    if len(records) == len(selected)*len(c['updates_per_block']):
        print('Already complete; no GPU model loaded.', flush=True)
        return
    print('Loading fixed R1-Distill-7B; no training. Outputs: '+str(out), flush=True)
    backend = Backend(a.model_path, c)
    versions = {}
    for name in ('torch', 'transformers', 'vllm', 'tokenizers'):
        try: versions[name] = version(name)
        except PackageNotFoundError: versions[name] = 'missing'
    save(out/'environment.json', {'versions': versions,
        'gpu': backend.torch.cuda.get_device_name(0),
        'gpu_total_bytes': backend.torch.cuda.get_device_properties(0).total_memory})
    worker = Worker()
    started = time.monotonic()
    try:
        for index, row in enumerate(selected):
            original_ids = backend.response_ids(row['solution'])
            original_score = None
            for updates in c['updates_per_block']:
                if (row['sample_id'], updates) in done:
                    continue
                if time.monotonic()-started >= a.time_budget_minutes*60:
                    save(out/'summary.json', summarize(records, selected))
                    print('Time budget reached at question boundary; rerun the same command to resume.', flush=True)
                    return
                print(f'QUESTION {index+1}/32 method=projection_{updates} reference={row["reference_source"]}', flush=True)
                rawfile = out/f'raw_{row["sample_id"]}_{updates}.json'
                if rawfile.exists():
                    result = read(rawfile)
                else:
                    if original_score is None:
                        original_score = backend.score(row['question'], original_ids)
                    statefile = out/f'state_{row["sample_id"]}_{updates}.json'
                    state = read(statefile) if statefile.exists() else None
                    try:
                        result = sample(row, backend, c, updates, out, state, lambda s: save(statefile,s),
                                        deadline=started+a.time_budget_minutes*60)
                    except BudgetPause:
                        save(out/'summary.json', summarize(records, selected))
                        print('Time budget reached at block boundary; saved state. Rerun the same command to resume.',flush=True)
                        return
                    result['original_mean_logprob'] = original_score
                    result['reference_source'] = row['reference_source']
                    result['reference_answer'] = row['reference_answer']
                    # Persist raw output BEFORE grading. Reuse it after a grader
                    # failure; never spend GPU tokens regenerating a saved answer.
                    save(rawfile, result)
                result['grade'] = worker.check(result['text'], row['reference_answer'])
                result['grade_category'] = category(result['grade'])
                if result['grade_category'] in ('verifier_error','timeout'):
                    save(out/'grading_failure.json', result)
                    raise RuntimeError('Grader infrastructure issue saved; rerun to regrade the existing raw answer')
                append(out/'results.jsonl', result)
                done.add((row['sample_id'], updates)); records.append(result)
                save(out/'summary.json', summarize(records, selected))
                print(f'  grade={result["grade_category"]} finish={result["finish_reason"]} seconds={result["runtime_seconds"]:.1f}', flush=True)
        report = summarize(records, selected, completed=True)
        save(out/'summary.json', report)
        save(out/'COMPLETE.json', {'results_sha256': sha(out/'results.jsonl'), 'records': len(records)})
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print('采样结束，结果已保存：'+str(out)+'；可以终止GPU运行时。', flush=True)
    finally:
        save(out/f'session_meter_{time.time_ns()}.json', {
            'includes_original_scoring_and_all_proposals': True,
            'generation_tokens_spent': backend.generated, 'scored_response_tokens': backend.scored,
            'input_tokens_spent': backend.input_tokens, 'engine_calls': backend.calls,
            'runtime_seconds_after_model_load': time.monotonic()-started,
            'peak_allocated_bytes': backend.torch.cuda.max_memory_allocated(),
            'note': 'vLLM worker process allocations may not be fully reflected by this parent-process torch counter'})
        worker.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
