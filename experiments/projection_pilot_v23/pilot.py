"""Qwen-Instruct projection pilot: fixed expert prefix, four independent jobs."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.projection_pilot_v21.pilot import read, rows, save, sha, append, ContextOverflow
from experiments.projection_pilot_v21.parallel_bench import BatchBackend

INSTRUCTIONS = '''Write a standalone solution to QUERY in your own words.
TRAJECTORY is private reference material, not a preceding part of your response.
If the assistant prefix is empty, start the reasoning from the problem itself.
If a prefix exists, continue it without restarting or repeating it.
Derive the key equations, explain the origin of any recurrence, and justify cases.
Do not refer to the reference or any earlier external solution. Do not say
"according to the above solution", "I remember a formula", "after finding the result",
or otherwise imply that an unstated derivation has already been performed.
Do not merely reveal the answer and verify a few examples. Include the reasoning
needed to obtain it. Do not imitate or add <think> tags. Give one final answer
inside \\boxed{} after the necessary derivation.'''
SUFFIX = '\n\nSolve step by step. Give one final answer inside \\boxed{}.'


def generation_user(row):
    # This entire user message remains byte-identical across all proposals.
    return '<QUERY>\n'+row['question']+'\n</QUERY>\n<TRAJECTORY>\n'+row['solution']+'\n</TRAJECTORY>\n\n'+INSTRUCTIONS


def render(tokenizer, user):
    result = tokenizer.apply_chat_template([{'role':'user', 'content':user}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    if isinstance(result, dict):
        result = result['input_ids']
    return list(result)


def proposal_ids(tokenizer, row, prefix):
    return render(tokenizer, generation_user(row)) + list(prefix)


def scoring_prefix(tokenizer, question):
    return render(tokenizer, question + SUFFIX)


class Backend(BatchBackend):
    def __init__(self, model, config):
        super().__init__(model, config, config['concurrency'])
        self.fixed = {}

    def execute(self, requests):
        prompts, params, metadata = [], [], []
        for req in requests:
            if req[0] == 'score':
                _, question, ids = req
                prefix = scoring_prefix(self.tok, question)
                full = prefix + list(ids)
                if not ids or len(full)+1 > self.config['max_model_len']:
                    raise ContextOverflow('Empty or oversized score request')
                param = self.SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)
                metadata.append(('score', len(prefix), ids))
            else:
                _, row, prefix, limit, seed = req
                key = row['sample_id']
                if key not in self.fixed:
                    self.fixed[key] = render(self.tok, generation_user(row))
                full = self.fixed[key] + list(prefix)
                if len(full)+limit > self.config['max_model_len']:
                    raise ContextOverflow('Expert + assistant prefix exceeds context; no truncation')
                param = self.SamplingParams(max_tokens=limit, temperature=self.config['temperature'],
                    top_p=1.0, repetition_penalty=self.config['repetition_penalty'],
                    stop_token_ids=[self.eos], seed=seed)
                metadata.append(('propose', prefix, None))
            prompts.append({'prompt_token_ids':full})
            params.append(param)
            self.input_tokens += len(full)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError('Engine output count mismatch')
        values = []
        for result, (kind, prefix, ids) in zip(outputs, metadata):
            generated = list(result.outputs[0].token_ids)
            self.generated += len(generated)
            if kind == 'propose':
                if result.outputs[0].finish_reason == 'stop' and (not generated or generated[-1] != self.eos):
                    generated.append(self.eos)
                values.append(list(prefix)+generated)
            else:
                observed = result.prompt_logprobs[prefix:]
                if len(observed) != len(ids):
                    raise RuntimeError('Score coverage mismatch')
                scores = []
                for token, item in zip(ids, observed):
                    if item is None or token not in item:
                        raise RuntimeError('Missing observed-token logprob')
                    number = float(item[token].logprob)
                    if not math.isfinite(number):
                        raise RuntimeError('Nonfinite logprob')
                    scores.append(number)
                values.append(sum(scores)/len(scores))
                self.scored += len(ids)
        self.calls += 1
        return values


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


def trajectory(row, config, updates, state=None):
    rng = random.Random(config['seed'] + int(row['sample_id'][:8],16))
    ids, events, blocks, score, start_block = [], [], [], None, 0
    if state:
        ids, events, blocks, score = state['ids'], state['trace'], state['blocks'], state['score']
        start_block = state['next_block']
        rng.setstate(tuples(state['rng']))
    for block in range(start_block, config['block_num']):
        if ids and ids[-1] == config['eos']:
            break
        ids = yield ('propose', row, ids, min(config['block_tokens'],config['max_new_tokens']-len(ids)), rng.randrange(2**31))
        if not ids:
            break
        score = yield ('score', row['question'], ids)
        before = score
        for update in range(updates):
            cut = rng.randrange(len(ids))
            candidate = yield ('propose',row,ids[:cut],len(ids)-cut,rng.randrange(2**31))
            new_score = yield ('score',row['question'],candidate)
            accepted = new_score > score
            events.append({'block':block, 'update':update, 'cut':cut,
                'old_mean_logprob':score, 'candidate_mean_logprob':new_score,
                'accepted':accepted, 'old_tokens':len(ids), 'candidate_tokens':len(candidate)})
            if accepted:
                ids, score = candidate, new_score
        blocks.append({'block':block, 'before_updates_mean_logprob':before,
                       'after_updates_mean_logprob':score, 'tokens':len(ids)})
        yield ('checkpoint', {'ids':ids, 'trace':events, 'blocks':blocks, 'score':score,
                              'next_block':block+1, 'rng':rng.getstate()})
    return {'sample_id':row['sample_id'], 'method':f'projection_{updates}',
        'updates_per_block':updates, 'token_ids':ids, 'trace':events, 'blocks':blocks,
        'blocks_used':len(blocks), 'final_mean_logprob':score,
        'within_run_mcmc_gain':sum(e['candidate_mean_logprob']-e['old_mean_logprob'] for e in events if e['accepted']),
        'finish_reason':'stop' if ids and ids[-1]==config['eos'] else ('length' if ids else 'empty_output'),
        'training_approved':False}


def run_jobs(selected, config, updates, backend, out, completed, deadline):
    """Interleave independent trajectories; persist only consistent block boundaries."""
    pending, active = iter(selected), []
    def advance(job, value=None, first=False):
        while True:
            try:
                req = next(job['gen']) if first else job['gen'].send(value)
            except StopIteration as stop:
                completed(stop.value)
                return False
            first = False
            if req[0] != 'checkpoint':
                job['request'] = req
                return True
            save(job['path'],req[1])
            print(f"{job['id'][:8]} projection_{updates} block={req[1]['next_block']}/{config['block_num']} tokens={len(req[1]['ids'])}",flush=True)
            value = None
    while True:
        if time.monotonic() >= deadline:
            return False
        while len(active) < config['concurrency']:
            row = next(pending,None)
            if row is None:
                break
            path = out/f"state_{row['sample_id']}_{updates}.json"
            job = {'gen':trajectory(row,config,updates,read(path) if path.exists() else None),
                   'path':path, 'id':row['sample_id']}
            if advance(job,first=True):
                active.append(job)
        if not active:
            return True
        current = list(active)
        values = backend.execute([j['request'] for j in current])
        if len(values) != len(current):
            raise RuntimeError('Batch result count mismatch')
        for job,value in zip(current,values):
            if not advance(job,value):
                active.remove(job)


def mean(values):
    return sum(values)/len(values) if values else None


def summarize(records):
    methods = {}
    for method in ['projection_0','projection_4']:
        group = [r for r in records if r['method']==method]
        attempts = sum(len(r['trace']) for r in group)
        methods[method] = {'count':len(group),
            'grade_categories':dict(Counter(r['grade_category'] for r in group)),
            'finish_reasons':dict(Counter(r['finish_reason'] for r in group)),
            'mean_output_tokens':mean([len(r['token_ids']) for r in group]),
            'mean_blocks_used':mean([r['blocks_used'] for r in group]),
            'mean_within_run_mcmc_gain':mean([r['within_run_mcmc_gain'] for r in group]),
            'mean_logprob_gain_over_original':mean([r['final_mean_logprob']-r['original_mean_logprob']
                for r in group if r['final_mean_logprob'] is not None]),
            'acceptance_rate':sum(e['accepted'] for r in group for e in r['trace'])/attempts if attempts else None}
    a = {r['sample_id']:r for r in records if r['method']=='projection_0'}
    b = {r['sample_id']:r for r in records if r['method']=='projection_4'}
    paired = sorted(a.keys() & b.keys())
    diffs = [b[k]['final_mean_logprob']-a[k]['final_mean_logprob'] for k in paired
             if a[k]['final_mean_logprob'] is not None and b[k]['final_mean_logprob'] is not None]
    return {'methods':methods,'paired_count':len(diffs),
        'mean_logprob_diff_proj4_minus_proj0':mean(diffs),
        'proj4_gt_proj0_win_rate':mean([int(x>0) for x in diffs]),
        'proj4_eq_proj0_ties':sum(x==0 for x in diffs),
        'training_approved':False,
        'interpretation':'Paired final differences are not a pure causal MCMC gain; within-run gain sums accepted local mean-logprob changes, not the final-vs-initial score across block extensions.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--time-budget-minutes',type=float,default=90)
    args = p.parse_args()
    if args.time_budget_minutes <= 0:
        p.error('Positive budget required')
    c = read(ROOT/'configs/project1_projection_pilot_v23.json')
    model, out = Path(args.model_path).resolve(), Path(args.output_root).resolve()
    if sha(ROOT/c['data_file']) != c['data_sha256']:
        raise ValueError('Frozen data hash mismatch')
    selected = rows(ROOT/c['data_file'])
    if len(selected)!=32 or len({r['sample_id'] for r in selected})!=32:
        raise ValueError('Expected 32 unique questions')
    for name,expected in c['tokenizer_sha256'].items():
        if sha(model/name)!=expected:
            raise ValueError('Tokenizer mismatch: '+name)
    model_id = read(model/'projection_model_identity.json')
    if model_id['repo']!=c['model'] or model_id['revision']!=c['model_revision']:
        raise ValueError('Model identity mismatch')
    from src.training.frozen_verifier_v4 import Worker, grader_identity
    from src.training.math_reward_v2 import category
    identity = {'config':c,'model_identity':model_id,'grader':grader_identity(),
        'sources':{name:sha(ROOT/name) for name in ['experiments/projection_pilot_v23/pilot.py',
            'experiments/projection_pilot_v21/pilot.py','experiments/projection_pilot_v21/parallel_bench.py']}}
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    # A shared lock also prevents overlap with the default v21/v22 GPU jobs.
    held = []
    for path in [out/'running.lock', Path('/content/drive/MyDrive/runs/project1/projection_pilot_v21/running.lock')]:
        path.parent.mkdir(parents=True,exist_ok=True)
        handle = path.open('a')
        try:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another experiment is running; wait for it to exit')
        held.append(handle)
    if (out/'run_identity.json').exists() and read(out/'run_identity.json')!=identity:
        raise ValueError('Protocol changed; use another output-root')
    save(out/'run_identity.json',identity)
    records = rows(out/'results.jsonl') if (out/'results.jsonl').exists() else []
    done = {(r['sample_id'],r['updates_per_block']) for r in records}
    if len(done)!=len(records):
        raise ValueError('Duplicate records')
    expected = {(r['sample_id'],u) for r in selected for u in c['updates_per_block']}
    if not done <= expected:
        raise ValueError('Unexpected saved records')
    if done==expected:
        print('Already complete; no GPU loaded.',flush=True)
        return
    backend = Backend(model,c)
    c = dict(c,eos=backend.eos)
    save(out/'environment.json',{'versions':{n:version(n) for n in ['torch','transformers','vllm']},
        'gpu':backend.torch.cuda.get_device_name(0), 'enable_prefix_caching':True,
        'concurrency':c['concurrency']})
    verifier = Worker()
    started = time.monotonic()
    by_id = {r['sample_id']:r for r in selected}
    original = read(out/'original_scores.json') if (out/'original_scores.json').exists() else {}
    def collect(result):
        row = by_id[result['sample_id']]
        raw = out/f"raw_{result['sample_id']}_{result['updates_per_block']}.json"
        if 'text' not in result:
            result['text'] = backend.decode(result['token_ids'])
        result['original_mean_logprob'] = original[result['sample_id']]
        result['reference_source'] = row['reference_source']
        result['reference_answer'] = row['reference_answer']
        save(raw,result)
        result['grade'] = verifier.check(result['text'],row['reference_answer'])
        result['grade_category'] = category(result['grade'])
        if result['grade_category'] in ('verifier_error','timeout'):
            raise RuntimeError('Grader failed; saved raw answer will be reused on rerun')
        append(out/'results.jsonl',result)
        done.add((result['sample_id'],result['updates_per_block']))
        records.append(result)
        save(out/'summary.json',summarize(records))
        print(f"DONE {len(records)}/64 {result['method']} grade={result['grade_category']} blocks={result['blocks_used']}",flush=True)
    try:
        # Preserve original text including R1 think tags for this diagnostic.
        # All methods and originals are scored with the SAME question-only prefix.
        missing = [r for r in selected if r['sample_id'] not in original]
        for start in range(0,len(missing),c['concurrency']):
            if time.monotonic()-started >= args.time_budget_minutes*60:
                print('Budget reached during original scoring; rerun to resume.',flush=True)
                return
            batch = missing[start:start+c['concurrency']]
            requests = []
            for row in batch:
                ids = backend.tok.encode(row['solution'],add_special_tokens=False)
                if not ids or ids[-1]!=backend.eos:
                    ids.append(backend.eos)
                requests.append(('score',row['question'],ids))
            values = backend.execute(requests)
            original.update({r['sample_id']:v for r,v in zip(batch,values)})
            save(out/'original_scores.json',original)
        for updates in c['updates_per_block']:
            pending = []
            for row in selected:
                if (row['sample_id'],updates) in done:
                    continue
                raw = out/f"raw_{row['sample_id']}_{updates}.json"
                if raw.exists():
                    collect(read(raw))
                else:
                    pending.append(row)
            finished = run_jobs(pending,c,updates,backend,out,collect,
                                started+args.time_budget_minutes*60)
            if not finished:
                save(out/'summary.json',summarize(records))
                print('Time budget reached; last complete blocks saved. Repeat command to resume.',flush=True)
                return
        report = summarize(records)
        save(out/'summary.json',report)
        save(out/'COMPLETE.json',{'records':len(records),'results_sha256':sha(out/'results.jsonl')})
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        print('采样结束，结果已保存：'+str(out),flush=True)
    finally:
        save(out/f'session_meter_{time.time_ns()}.json',{'seconds_after_model_load':time.monotonic()-started,
            'generated_tokens':backend.generated,'input_tokens':backend.input_tokens,
            'scored_tokens':backend.scored,'batch_calls':backend.calls})
        verifier.close()


if __name__=='__main__':
    main()
