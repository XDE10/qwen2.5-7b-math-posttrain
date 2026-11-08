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
The final summary after </think> in TRAJECTORY may omit intermediate steps
developed inside <think>. Extract the necessary mathematical derivations,
algebraic steps, proofs, and case distinctions from that reasoning to build
your standalone solution, rather than merely paraphrasing the final summary.
Select correct and relevant reasoning; do not reproduce abandoned attempts
or mistakes merely because they appear in the reference thinking.
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
                metadata.append(('propose', prefix, req[3]))
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
                if len(generated) > ids:
                    raise RuntimeError('Engine exceeded candidate generation budget')
                # Keep sampled IDs within the fixed ceiling. If the engine omits
                # EOS, termination is carried separately, not appended beyond budget.
                values.append({'ids':list(prefix)+generated,
                               'stopped':result.outputs[0].finish_reason == 'stop'})
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


def score_ids(ids, stopped, eos):
    """Normalize terminal EOS for scoring only; never inflate generation length."""
    return list(ids)+([eos] if stopped and (not ids or ids[-1]!=eos) else [])


def trajectory(row, config, updates, state=None):
    rng = random.Random(config['seed'] + int(row['sample_id'][:8],16))
    ids, events, blocks, score, start_block, stopped = [], [], [], None, 0, False
    if state:
        ids, events, blocks, score = state['ids'], state['trace'], state['blocks'], state['score']
        start_block = state['next_block']
        stopped = state['stopped']
        rng.setstate(tuples(state['rng']))
    for block in range(start_block, config['block_num']):
        if stopped:
            break
        block_ceiling = min(len(ids)+config['block_tokens'],config['max_new_tokens'])
        if block_ceiling <= len(ids):
            break
        initial = yield ('propose', row, ids, block_ceiling-len(ids), rng.randrange(2**31))
        ids, stopped = initial['ids'], initial['stopped']
        if len(ids)>block_ceiling:
            raise RuntimeError('Initial extension exceeded block ceiling')
        if not ids and not stopped:
            break
        score = yield ('score', row['question'], score_ids(ids,stopped,config['eos']))
        before = score
        for update in range(updates):
            cut = rng.randrange(len(ids)) if ids else 0
            proposal = yield ('propose',row,ids[:cut],block_ceiling-cut,rng.randrange(2**31))
            candidate, candidate_stop = proposal['ids'], proposal['stopped']
            if len(candidate)>block_ceiling:
                raise RuntimeError('Candidate exceeded fixed block ceiling')
            new_score = yield ('score',row['question'],score_ids(candidate,candidate_stop,config['eos']))
            accepted = new_score > score
            events.append({'block':block, 'update':update, 'cut':cut,
                'old_mean_logprob':score, 'candidate_mean_logprob':new_score,
                'accepted':accepted, 'old_tokens':len(ids), 'candidate_tokens':len(candidate),
                'block_ceiling':block_ceiling,'candidate_stopped':candidate_stop})
            if accepted:
                ids, score, stopped = candidate, new_score, candidate_stop
        blocks.append({'block':block, 'before_updates_mean_logprob':before,
                       'after_updates_mean_logprob':score, 'tokens':len(ids),
                       'block_ceiling':block_ceiling,'stopped':stopped})
        yield ('checkpoint', {'ids':ids, 'trace':events, 'blocks':blocks, 'score':score,
                              'next_block':block+1, 'rng':rng.getstate(),'stopped':stopped})
    return {'sample_id':row['sample_id'], 'method':f'projection_{updates}',
        'updates_per_block':updates, 'token_ids':ids, 'trace':events, 'blocks':blocks,
        'blocks_used':len(blocks), 'final_mean_logprob':score,
        'length_bin':row['length_bin'], 'source_solution_tokens':row['solution_tokens'],
        'within_run_mcmc_gain':sum(e['candidate_mean_logprob']-e['old_mean_logprob'] for e in events if e['accepted']),
        'finish_reason':('stop' if stopped else 'length') if ids else 'empty_output',
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


class CostLedger:
    """Persist cumulative stage costs; repeated flushes never double count."""
    COUNTERS = {'generation_tokens_spent':'generated', 'input_tokens_spent':'input_tokens',
                'scoring_tokens_spent':'scored', 'batch_calls':'calls'}

    def __init__(self, path, backend, clock=time.monotonic):
        self.path, self.backend, self.clock = path, backend, clock
        self.totals = read(path) if path.exists() else {}
        self.key = None

    def begin(self, key):
        self.flush()
        self.key = key
        self.base = dict(self.totals.get(key, {}))
        self.start = self.clock()
        self.counters = {k:getattr(self.backend, attr) for k,attr in self.COUNTERS.items()}

    def flush(self):
        if self.key is None:
            return
        result = {'wall_clock_seconds':self.base.get('wall_clock_seconds',0.)+self.clock()-self.start}
        for key,attr in self.COUNTERS.items():
            result[key] = self.base.get(key,0)+getattr(self.backend,attr)-self.counters[key]
        self.totals[self.key] = result
        save(self.path,self.totals)

    def end(self):
        self.flush()
        self.key = None


def summarize(records, costs=None):
    costs = costs or {}
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
        methods[method].update({k:costs.get(method,{}).get(k) for k in
            ['wall_clock_seconds','generation_tokens_spent','input_tokens_spent','scoring_tokens_spent','batch_calls']})
    a = {r['sample_id']:r for r in records if r['method']=='projection_0'}
    b = {r['sample_id']:r for r in records if r['method']=='projection_4'}
    paired = sorted(a.keys() & b.keys())
    diffs = [b[k]['final_mean_logprob']-a[k]['final_mean_logprob'] for k in paired
             if a[k]['final_mean_logprob'] is not None and b[k]['final_mean_logprob'] is not None]
    buckets = {}
    for bucket in sorted({r['length_bin'] for r in records},key=lambda value:int(str(value).split('-')[0])):
        group = [r for r in records if r['length_bin']==bucket]
        bucket_methods = {}
        for method in ['projection_0','projection_4']:
            items = [r for r in group if r['method']==method]
            bucket_methods[method] = {'count':len(items),
                'mean_output_tokens':mean([len(r['token_ids']) for r in items]),
                'mean_blocks_used':mean([r['blocks_used'] for r in items]),
                'grade_categories':dict(Counter(r['grade_category'] for r in items))}
        keys = [k for k in paired if a[k]['length_bin']==bucket and
                a[k]['final_mean_logprob'] is not None and b[k]['final_mean_logprob'] is not None]
        delta = [b[k]['final_mean_logprob']-a[k]['final_mean_logprob'] for k in keys]
        buckets[str(bucket)] = {'methods':bucket_methods,'paired_count':len(delta),
            'mean_logprob_diff_proj4_minus_proj0':mean(delta),
            'proj4_gt_proj0_win_rate':mean([int(x>0) for x in delta])}
    def ratio(key):
        zero, four = costs.get('projection_0',{}).get(key), costs.get('projection_4',{}).get(key)
        return four/zero if zero is not None and zero>0 and four is not None else None
    return {'methods':methods,'paired_count':len(diffs),
        'mean_logprob_diff_proj4_minus_proj0':mean(diffs),
        'proj4_gt_proj0_win_rate':mean([int(x>0) for x in diffs]),
        'proj4_eq_proj0_ties':sum(x==0 for x in diffs),
        'by_original_length_bucket':buckets,
        'cost_ratio_proj4_over_proj0':{k:ratio(k) for k in
            ['wall_clock_seconds','generation_tokens_spent','input_tokens_spent','scoring_tokens_spent']},
        'original_scoring_cost':costs.get('original_scoring',{}),
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
    c = read(ROOT/'configs/project1_projection_pilot_v24.json')
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
        'sources':{name:sha(ROOT/name) for name in ['experiments/projection_pilot_v24/pilot.py',
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
    ledger = CostLedger(out/'costs.json',backend)
    execute = backend.execute
    def metered_execute(requests):
        values = execute(requests)
        ledger.flush()
        return values
    backend.execute = metered_execute
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
        ledger.flush()
        save(out/'summary.json',summarize(records,ledger.totals))
        print(f"DONE {len(records)}/64 {result['method']} grade={result['grade_category']} blocks={result['blocks_used']}",flush=True)
    try:
        # Preserve original text including R1 think tags for this diagnostic.
        # All methods and originals are scored with the SAME question-only prefix.
        missing = [r for r in selected if r['sample_id'] not in original]
        ledger.begin('original_scoring')
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
        ledger.end()
        for updates in c['updates_per_block']:
            ledger.begin(f'projection_{updates}')
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
                ledger.end()
                save(out/'summary.json',summarize(records,ledger.totals))
                print('Time budget reached; last complete blocks saved. Repeat command to resume.',flush=True)
                return
            ledger.end()
        report = summarize(records,ledger.totals)
        save(out/'summary.json',report)
        save(out/'COMPLETE.json',{'records':len(records),'results_sha256':sha(out/'results.jsonl')})
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        print('采样结束，结果已保存：'+str(out),flush=True)
    finally:
        ledger.end()
        save(out/'summary.json',summarize(records,ledger.totals))
        save(out/f'session_meter_{time.time_ns()}.json',{'seconds_after_model_load':time.monotonic()-started,
            'generated_tokens':backend.generated,'input_tokens':backend.input_tokens,
            'scored_tokens':backend.scored,'batch_calls':backend.calls})
        verifier.close()


if __name__=='__main__':
    main()
