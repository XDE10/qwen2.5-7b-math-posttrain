"""Local DS review of public P-ALIGN 7B v47 native failures; preserve full original and final output."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    j = helper('evaluate_sft_ds_v29')
    g = helper('evaluate_public_palign_math500_v47')
    directory = ROOT / 'artifacts/final_eval_local_snapshot/evaluations/public_palign_7b_math500_v47_rp105'
    review = directory / 'native_ds_v48'
    review.mkdir(exist_ok=True)
    attempts = review / 'api_attempts'
    attempts.mkdir(exist_ok=True)
    j.SYSTEM += ('\nFor forced_answer=true, the output contains the complete original reasoning, an '
                 'intervention, then FINAL_ANSWER_CONTINUATION. Judge ONLY the intended answer in '
                 'FINAL_ANSWER_CONTINUATION. Earlier reasoning is context, never a substitute for '
                 'a missing or wrong continuation answer. Correct verdict evidence must be an exact '
                 'short substring of FINAL_ANSWER_CONTINUATION. Keep evidence under 200 characters. '
                 'For forced_answer=false, judge the original output normally.')
    data, source_digest = g.math_data(ROOT)
    source = {r['sample_id']: r for r in data}
    expected = set(source)
    original = g.by_id(g.rows(directory / 'reasoning.jsonl'), expected)
    final = g.by_id(g.rows(directory / 'raw.jsonl'), expected)
    auto = g.by_id(g.read(directory / 'native_scores.json'), expected)
    complete = g.read(directory / 'GENERATION_COMPLETE.json')
    if set(original) != expected or set(final) != expected or set(auto) != expected:
        raise RuntimeError('Incomplete native results')
    if complete['raw_sha256'] != g.sha(directory / 'raw.jsonl') or complete['reasoning_sha256'] != g.sha(directory / 'reasoning.jsonl'):
        raise RuntimeError('Raw hash differs')
    identity = {'raw_sha256': complete['raw_sha256'], 'reasoning_sha256': complete['reasoning_sha256'],
                'native_scores_sha256': g.sha(directory / 'native_scores.json'), 'source_digest': source_digest,
                'script_sha256': g.sha(__file__), 'judge_helper_sha256': g.sha(j.__file__),
                'system_sha256': j.digest(j.SYSTEM), 'model': 'deepseek-flash',
                'scope': 'Only native final failures. Full original+continuation; forced final answer judged only from continuation.'}
    if (review / 'identity.json').exists() and g.read(review / 'identity.json') != identity:
        raise RuntimeError('Review identity changed')
    g.save(review / 'identity.json', identity)
    cache = g.by_id(g.rows(review / 'results.jsonl'), expected)
    key = j.credential(Path('C:/Users/Mr.Z/.codex/private/project1/deepseek.json'))

    def classify(sid):
        row = final[sid]
        if row['forced']:
            full = ('ORIGINAL_REASONING:\n' + original[sid]['text'] + '\nINTERVENTION:\n' + row['intervention_text']
                    + '\nFINAL_ANSWER_CONTINUATION:\n' + row['text'])
        else:
            full = original[sid]['text']
        payload = j.judge_payload(source[sid]['problem'], str(source[sid]['answer']), full,
                                  auto[sid]['final_grade'], row['finish_reason'])
        payload['forced_answer'] = row['forced']
        result = j.call_ds(payload, key, 'deepseek-flash', attempts / (j.digest(sid) + '.jsonl'))
        verdict = result['judgment']
        if verdict['verdict'] == 'correct' and verdict['evidence'] not in row['text']:
            raise RuntimeError('Correct evidence is not from final answer')
        return {'sample_id': sid, 'forced': row['forced'], 'native_grade': auto[sid]['final_grade'],
                'verdict': verdict['verdict'], 'ds_review': result, 'payload_sha256': j.digest(payload)}

    errors = []
    failure_count = sum(not r['final_grade']['correct'] for r in auto.values())
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(classify, sid): sid for sid in expected
                 if not auto[sid]['final_grade']['correct'] and sid not in cache}
        for future in as_completed(tasks):
            sid = tasks[future]
            try:
                result = future.result()
                g.append(review / 'results.jsonl', result)
                cache[sid] = result
            except Exception as error:
                errors.append({'sample_id': sid, 'error': type(error).__name__})
            if len(cache) % 10 == 0 or not tasks:
                print(f'DS_MATH500 {len(cache)}/{failure_count} saved', flush=True)
    if errors:
        g.save(review / 'errors.json', errors)
        raise RuntimeError(f'{len(errors)} failed; successful results saved, rerun retries only missing')
    native_correct = sum(r['final_grade']['correct'] for r in auto.values())
    counts = Counter(r['verdict'] for r in cache.values())
    usage = Counter()
    for path in attempts.glob('*.jsonl'):
        for row in g.rows(path):
            for name, value in row.get('response', {}).get('usage', {}).items():
                if isinstance(value, int):
                    usage[name] += value
    if len(cache) != 500 - native_correct:
        raise RuntimeError('Incomplete DS review')
    summary = {'count': 500, 'native_correct': native_correct, 'native_accuracy': native_correct / 500,
               'ds_reviewed': len(cache), 'ds_recovered': counts['correct'],
               'reviewed_correct': native_correct + counts['correct'],
               'reviewed_accuracy': (native_correct + counts['correct']) / 500,
               'incorrect': counts['incorrect'], 'uncertain': counts['uncertain'],
               'usage_including_retries': dict(usage),
               'note': 'DS answer-equivalence review, not human gold or proof validity. Native pass outputs not DS reviewed.'}
    g.save(review / 'summary.json', summary)
    g.save(review / 'COMPLETE.json', {'results_sha256': g.sha(review / 'results.jsonl')})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
