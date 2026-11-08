"""Full pinned default/train read-only analysis, resumable per-shard cache.

Writes only a NEW reports directory, never SFT/RL/dev/test inputs. No weights,
GPU, network, answer regeneration, or mutation of existing source records.
Question units are normalized-problem hashes, not repeated UUIDs or source rows.
"""
import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('RAYON_NUM_THREADS', '2')
from src.data.openr1_quality import RULE_VERSION, quality_signals
from src.data.screen_openr1_sft import PROMPT, answer_is_complete, sealed_math
from src.data.prepare_math_data import fingerprint, write_jsonl
from src.data.screening_io import rows, sha, json_write, ScreeningSimilarityIndex

CAPS = (1600, 2200, 3000, 3200)


def quantiles(values):
    """Linear interpolation at (n-1)*p, equivalent to numpy default quantile."""
    values = sorted(values)
    if not values:
        return {'n': 0, 'p50': None, 'p75': None, 'p90': None}
    def q(p):
        i = (len(values)-1)*p
        lo, hi = math.floor(i), math.ceil(i)
        return round(values[lo] + (values[hi]-values[lo])*(i-lo), 2)
    return {'n':len(values), 'p50':q(.5), 'p75':q(.75), 'p90':q(.9)}


def ratio_group(correct, total):
    if total == 0:
        return 'unknown'
    if correct == 0:
        return '0'
    if correct*2 <= total:
        return '(0,0.5]'
    if correct < total:
        return '(0.5,1)'
    return '1'


def inspect_row(row, source_path, row_index, tokenizer, defer=False):
    problem = row.get('problem')
    if not isinstance(problem, str) or not problem.strip():
        return {'invalid_problem':True, 'source_row':row_index, 'source_path':source_path}
    result = {'problem':problem, 'question_hash':fingerprint(problem), 'uuid':row.get('uuid'),
              'source':row.get('source') or 'unknown', 'problem_type':row.get('problem_type') or 'unknown',
              'source_path':source_path, 'source_row':row_index, 'trajectories':[], 'aligned':False}
    generations, complete, correct = [row.get(k) for k in ('generations','is_reasoning_complete','correctness_math_verify')]
    if not all(isinstance(x,list) and x for x in (generations,complete,correct)) or not len(generations)==len(complete)==len(correct):
        result['invalid_reason'] = 'unaligned_generation_flags'
        return result
    if not all(type(x) is bool for x in complete+correct) or not all(isinstance(t,str) for t in generations):
        result['invalid_reason'] = 'invalid_flag_or_generation_type'
        return result
    finishes = row.get('finish_reasons')
    if finishes is not None and (not isinstance(finishes,list) or len(finishes)!=len(generations)):
        result['invalid_reason'] = 'unaligned_finish_reasons'
        return result
    result['aligned'] = True
    result['row_correct_count'] = sum(correct)
    result['row_generation_count'] = len(correct)
    result['source_correctness_count_disagrees'] = row.get('correctness_count') != sum(correct)
    pending = []
    for i, text in enumerate(generations):
        trajectory = {'trajectory_index':i, 'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                      'verified':correct[i], 'complete':complete[i]}
        result['trajectories'].append(trajectory)
        if not (correct[i] and complete[i]):
            continue
        finish_ok = finishes is None or finishes[i] in {None,'stop','eos','eos_token','stop_sequence'}
        if not finish_ok or not answer_is_complete(text):
            trajectory['structural_rejection'] = 'non_stop_finish' if not finish_ok else 'incomplete_final_structure'
            continue
        trajectory['quality'] = quality_signals(text)
        pending.append((trajectory, text))
    if pending:
        prompt_messages = [{'role':'system','content':PROMPT},{'role':'user','content':problem}]
        prefix = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        # Real tokenizer of the whole template, not sum of assumed component lengths.
        rendered = [tokenizer.apply_chat_template(prompt_messages+[{'role':'assistant','content':text}],
                    tokenize=False, add_generation_prompt=False) for _, text in pending]
        strings = [text for _, text in pending] + rendered + [prefix]
        result['_encoding'] = (strings, [t for t,_ in pending])
        if not defer:
            finish_tokenization([result],tokenizer)
    return result


def finish_tokenization(records,tokenizer):
    pending = [r for r in records if '_encoding' in r]
    all_strings = [s for r in pending for s in r['_encoding'][0]]
    if not all_strings:
        return
    encoded = tokenizer.backend_tokenizer.encode_batch(all_strings,add_special_tokens=False)
    offset = 0
    for r in pending:
        strings, trajectories = r.pop('_encoding')
        n = len(trajectories)
        prefix_ids = encoded[offset+2*n].ids
        for j,trajectory in enumerate(trajectories):
            full = encoded[offset+n+j].ids
            if full[:len(prefix_ids)] != prefix_ids:
                raise ValueError('Template boundary does not match response target mask')
            trajectory.update(answer_tokens=len(encoded[offset+j].ids),total_tokens=len(full),
                              supervised_tokens=len(full)-len(prefix_ids),prompt_tokens=len(prefix_ids))
        offset += len(strings)


def collect_questions(cache_files):
    groups, counts = {}, Counter()
    for cache in cache_files:
        for row in rows(cache):
            counts['raw_rows'] += 1
            if row.get('invalid_problem'):
                counts['invalid_problem_rows'] += 1
                continue
            key = row['question_hash']
            q = groups.setdefault(key, {'question_hash':key, 'problem':row['problem'], 'sources':set(),
                'types':set(), 'uuids':set(), 'row_count':0, 'all_rows_aligned':True, 'trajectories':{}})
            q['row_count'] += 1
            q['sources'].add(row['source']); q['types'].add(row['problem_type']); q['uuids'].add(row['uuid'])
            q['all_rows_aligned'] &= row['aligned']
            if not row['aligned']:
                counts[row['invalid_reason']] += 1
                continue
            counts['source_correctness_count_disagrees'] += row['source_correctness_count_disagrees']
            for t in row['trajectories']:
                counts['raw_trajectories'] += 1
                h = t['text_sha256']
                if h in q['trajectories']:
                    counts['duplicate_trajectory_occurrences'] += 1
                    prev = q['trajectories'][h]
                    if prev['verified'] != t['verified'] or prev['complete'] != t['complete']:
                        prev['flag_conflict'] = True
                    prev['verified'] &= t['verified']; prev['complete'] &= t['complete']
                    continue
                q['trajectories'][h] = {**t, 'source_path':row['source_path'], 'source_row':row['source_row'],
                                        'source_uuid':row['uuid']}
    for q in groups.values():
        q['source'] = next(iter(q['sources'])) if len(q['sources'])==1 else 'mixed'
        q['problem_type'] = next(iter(q['types'])) if len(q['types'])==1 else 'mixed'
        q['sources'] = sorted(q['sources']); q['types'] = sorted(q['types']); q['uuids'] = sorted(q['uuids'])
        trajectories = list(q['trajectories'].values())
        n = len(trajectories)
        k = sum(t['verified'] for t in trajectories)
        q['verified_count'], q['unique_trajectory_count'] = k, n
        q['verification_ratio'] = k/n if n and q['all_rows_aligned'] else None
        q['ratio_group'] = ratio_group(k,n) if q['all_rows_aligned'] else 'unknown'
        q['flag_conflict'] = any(t.get('flag_conflict') for t in trajectories)
        # Conflicting labels or unaligned records are quarantined, not silently
        # assigned a numerical difficulty/solvability label.
        q['audit_usable'] = q['all_rows_aligned'] and not q['flag_conflict']
        q['trajectories'] = trajectories
        candidates = [t for t in trajectories if q['audit_usable'] and t['verified'] and t['complete']
                      and 'answer_tokens' in t]
        quality = [t for t in candidates if t['quality']['automatic_pass']]
        q['shortest_source_eligible'] = min(candidates,key=lambda t:(t['answer_tokens'],t['text_sha256'])) if candidates else None
        q['shortest_quality_pass'] = min(quality,key=lambda t:(t['answer_tokens'],t['text_sha256'])) if quality else None
    counts['unique_questions'] = len(groups)
    counts['repeated_question_rows'] = counts['raw_rows']-counts['invalid_problem_rows']-len(groups)
    return list(groups.values()), counts


def grouped_quantiles(questions, key):
    result = {}
    for dimension in ('source','problem_type','ratio_group'):
        groups = defaultdict(list)
        for q in questions:
            groups[q[dimension]].append(q)
        result[dimension] = {}
        for name, qs in sorted(groups.items()):
            eligible = [q[key] for q in qs if q[key]]
            result[dimension][name] = {'questions':len(qs), 'with_eligible_trace':len(eligible),
                'answer_tokens':quantiles([t['answer_tokens'] for t in eligible]),
                'total_tokens':quantiles([t['total_tokens'] for t in eligible])}
    return result


def choose_for_budget(q, cap, total_limit=3800, quality=True):
    if not q['audit_usable']:
        return None
    candidates = [t for t in q['trajectories'] if t['verified'] and t['complete'] and 'answer_tokens' in t
        and 500<=t['answer_tokens']<=cap and t['total_tokens']<=total_limit
        and (not quality or t['quality']['automatic_pass'])]
    return min(candidates,key=lambda t:(t['answer_tokens'],t['text_sha256'])) if candidates else None


def retention(questions, cap, total_limit=3800, clean=False):
    groups = defaultdict(list)
    for q in questions:
        groups[q['problem_type']].append(q)
    result = {}
    for name, qs in sorted(groups.items()):
        picked = [choose_for_budget(q,cap,total_limit) for q in qs
                  if not clean or not q.get('evaluation_overlap') and not q.get('internal_near_duplicate')]
        picked = [t for t in picked if t]
        denom = sum(q['shortest_quality_pass'] is not None for q in qs)
        result[name] = {'all_questions':len(qs), 'with_quality_complete_correct':denom,
            'retained':len(picked), 'retained_over_all_questions':len(picked)/len(qs),
            'retained_over_quality_eligible':len(picked)/denom if denom else None,
            'answer_tokens_sum':sum(t['answer_tokens'] for t in picked),
            'supervised_tokens_sum':sum(t['supervised_tokens'] for t in picked),
            'total_context_tokens_sum':sum(t['total_tokens'] for t in picked)}
    return result


def analyze_shard(task):
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    i,item,output = task
    path = Path(item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('Source shard changed')
    dest = Path(output)/f'shard_{i:02d}.jsonl'
    if dest.exists():
        print('Reusing completed shard',i+1,flush=True)
        return str(dest)
    tokenizer = AutoTokenizer.from_pretrained('data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    tokenizer.model_max_length = 10**9
    needed = ['uuid','problem','source','problem_type','generations','is_reasoning_complete',
              'correctness_math_verify','correctness_count','finish_reasons']
    count,started = 0,time.monotonic()
    temp = dest.with_suffix('.partial')
    with temp.open('w',encoding='utf-8',newline='\n') as stream:
        parquet = pq.ParquetFile(path)
        columns = [c for c in needed if c in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=16,columns=columns):
            records = [inspect_row(row,item['path'],count+j,tokenizer,defer=True)
                       for j,row in enumerate(batch.to_pylist())]
            finish_tokenization(records,tokenizer)
            for record in records:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                count += 1
            if count%1024==0:
                print(f'shard={i+1}/10 rows={count} elapsed={time.monotonic()-started:.0f}s',flush=True)
    if count != parquet.metadata.num_rows:
        raise ValueError('Incomplete shard analysis')
    temp.replace(dest)
    print('Completed shard',i+1,'rows',count,flush=True)
    return str(dest)


def run(output,workers=3):
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    output = Path(output)
    if output.parent.resolve() != Path('reports').resolve():
        raise ValueError('Output must be a dedicated direct child of reports/')
    output.mkdir(exist_ok=True)
    source = json.loads(Path('data/screening_sources/openr1_source.json').read_text(encoding='utf-8'))
    frozen = Path('data/project1_sft/sft_train.jsonl')
    frozen_sha = sha(frozen)
    if frozen_sha != '3163633d56893aaed4278c538229be6cc7553909146ca367bf0521d08fd6edf5':
        raise ValueError('Sealed compact SFT identity changed')
    math_manifest, protected = sealed_math()
    folder = Path('data/tokenizers/Qwen2.5-7B-Instruct')
    pinned = json.loads(Path('data/project1_sft/manifest.json').read_text(encoding='utf-8'))
    tokenizer_files = {name:sha(folder/name) for name in pinned['tokenizer_files']}
    if tokenizer_files != pinned['tokenizer_files']:
        raise ValueError('Target tokenizer identity changed')
    identity = {'source_revision':source['revision'], 'source_files':source['files'],
        'script_sha256':sha(__file__), 'quality_rules_sha256':sha('src/data/openr1_quality.py'),
        'template_system':PROMPT, 'template_sha256':sha(folder/'tokenizer_config.json'),
        'tokenizer_files':tokenizer_files, 'compact_sft_sha256':frozen_sha,
        'protected_files':{k:v['sha256'] for k,v in math_manifest['files'].items()}}
    identity_path = output/'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('Changed statistics identity: choose a new output directory')
    else:
        json_write(identity_path,identity)
    from concurrent.futures import ProcessPoolExecutor
    started = time.monotonic()
    tasks = [(i,item,str(output)) for i,item in enumerate(source['files'])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cache_files = [Path(p) for p in pool.map(analyze_shard,tasks)]
    questions, counts = collect_questions(cache_files)
    if counts['raw_rows'] != 93733:
        raise ValueError('Unexpected full default/train row count')
    # Test data is consulted only for exclusion; no heldout answer/difficulty is
    # used for training selection. Shared math TRAIN supplies genuine level labels.
    evaluation, rl, seen = ScreeningSimilarityIndex(.85), ScreeningSimilarityIndex(.85), ScreeningSimilarityIndex(.85)
    train_lookup = {}
    for key in ('math_dev','math500_test','gsm8k_test'):
        for r in protected[key]:
            evaluation.add(r)
    for r in protected['math_train']:
        rl.add(r); train_lookup[r['sample_id']] = r
    eligible = sorted((q for q in questions if choose_for_budget(q,3200)),
                      key=lambda q:hashlib.sha256(('stats-v1/'+q['question_hash']).encode()).hexdigest())
    for i,q in enumerate(eligible):
        q['evaluation_overlap'] = evaluation.match(q['problem'])
        q['rl_train_overlap'] = rl.match(q['problem'])
        hit = q['rl_train_overlap']
        q['linked_train_level'] = train_lookup[hit['matched_id']].get('level') if hit else None
        q['level_link_kind'] = hit['kind'] if hit else None
        # Only exact train links can be used as level labels. Near matches are
        # evidence for review, never automatic difficulty transfer.
        if hit and hit['kind'] == 'jaccard_3gram':
            q['linked_train_level'] = None
        q['internal_near_duplicate'] = None
        if not q['evaluation_overlap']:
            q['internal_near_duplicate'] = seen.match(q['problem'])
            if not q['internal_near_duplicate']:
                seen.add({'sample_id':q['question_hash'],'problem':q['problem']})
        if (i+1)%1000==0:
            print('Exclusion and train-level links',i+1,'/',len(eligible),flush=True)
    write_jsonl(output/'questions.jsonl',questions)
    quality_counts = Counter()
    for q in questions:
        for t in q['trajectories']:
            if 'quality' in t:
                quality_counts.update('reject/'+x for x in t['quality']['reject'])
                quality_counts.update('review/'+x for x in t['quality']['review'])
    stats = {'population':dict(counts), 'quantile_method':'linear interpolation (n-1)*p',
        'question_unit':'normalized-problem hash; unique generation-text hashes; repeated flags not independent samples',
        'ratio_definition':'count of unique math-verify-true trajectories / unique aligned trajectories; completion not part of numerator; conflicts quarantined',
        'difficulty_warning':'source and source-teacher verification ratio are proxies, not student difficulty or step correctness',
        'source_complete_correct':grouped_quantiles(questions,'shortest_source_eligible'),
        'quality_pass':grouped_quantiles(questions,'shortest_quality_pass'),
        'quality_rule_version':RULE_VERSION, 'quality_signals':dict(quality_counts),
        'quality_warning':'Automatic pass is not semantic/step-level approval; review flags excluded pending adjudication.',
        'retention_policy':{'answer_min':500,'answer_caps':list(CAPS),'total_max':3800,
            'supervised_tokens':'full assistant target including template EOS/newline, excluding prompt; assistant-only mask',
            'denominators':'all unique questions and questions with any quality-approved source-complete-correct trace, independently reported',
            'rl_compatibility':'Reference trace length is not generated rollout length. Dev pilot must measure actual truncation.'},
        'retention_before_exclusion':{str(cap):retention(questions,cap) for cap in CAPS},
        'retention_after_eval_and_internal_dedup':{str(cap):retention(questions,cap,clean=True) for cap in CAPS},
        'rl_budget_scenarios':{str(cap):retention(questions,cap,total_limit=3800,clean=True) for cap in (2048,2560,3072)},
        'overlap_counts':{'budget3200_candidates':len(eligible),
            'evaluation_overlaps':sum(bool(q.get('evaluation_overlap')) for q in eligible),
            'internal_near_duplicates':sum(bool(q.get('internal_near_duplicate')) for q in eligible),
            'rl_train_overlaps':sum(bool(q.get('rl_train_overlap')) for q in eligible),
            'exact_training_level_links':sum(q.get('linked_train_level') is not None for q in eligible)},
        'frozen_compact_sha256_before':frozen_sha, 'frozen_compact_sha256_after':sha(frozen),
        'elapsed_seconds':round(time.monotonic()-started,2), 'gpu_operations':0,'paid_api_calls':0}
    if stats['frozen_compact_sha256_after'] != frozen_sha:
        raise ValueError('Sealed data changed during read-only analysis')
    json_write(output/'statistics.json',stats)
    files = [output/'questions.jsonl',output/'statistics.json'] + cache_files
    json_write(output/'manifest.json',{'identity':identity, 'config':'default','split':'train','scan_complete':True,
        'files':{p.name:{'path':p.as_posix(),'sha256':sha(p)} for p in files},
        'sft_data_written':False,'model_weights_loaded':False})
    print('FULL AUDIT COMPLETE',json.dumps(dict(counts)), 'seconds',stats['elapsed_seconds'],flush=True)
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/openr1_full_audit_final_20261023')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    run(args.output,args.workers)
