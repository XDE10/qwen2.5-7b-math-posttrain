"""Post-hoc format robustness review; never modifies either frozen pilot."""
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from src.synthesis.compact_format_review import VERSION, review_admission
from src.synthesis.compact_v2 import box_spans
from src.synthesis.deepseek_pilot import dump
from scripts.compact_distillation_v2 import ROOT, REPORT, DATA, OLD_DATA, read, file_sha


def box_values(s):
    return [s[a:b] for a,b in box_spans(s)]


def run():
    out=REPORT/'format_robustness_review'
    out.mkdir(exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    scenarios={}
    oldrefs={r['sample_id']:r for r in read(OLD_DATA/'verifier_references.json')}
    oldattempts=read(REPORT/'cache_reaudit/attempts.json')
    grouped={}
    for a in oldattempts:
        ref=oldrefs[a['sample_id']]
        grouped.setdefault(a['sample_id'],{'sample_id':a['sample_id'],'problem_type':ref['selection']['problem_type'],
            'problem':ref['problem'],'reference_answer':ref['reference_answer'],'raw_solution':ref['raw_solution'],
            'attempts':[]})['attempts'].append({'attempt':a['attempt'],'text':a['strict']['text'],
                'finish_reason':'non_stop' if 'non_stop_finish' in a['strict']['errors'] else 'stop',
                'math_verify':a['strict']['math_verify']})
    scenarios['old_cache']=list(grouped.values())
    newrefs={r['sample_id']:r for r in read(DATA/'verifier_references.json')}
    scenarios['new_pilot']=[]
    for r in read(REPORT/'results.json'):
        ref=newrefs[r['sample_id']]
        scenarios['new_pilot'].append({'sample_id':r['sample_id'],'problem_type':r['problem_type'],
            'problem':ref['problem'],'reference_answer':ref['reference_answer'],'raw_solution':ref['raw_solution'],
            'original_math_verify':r['original_math_verify'],
            'attempts':[{'attempt':a['attempt'],'text':a['raw_text'],
                'finish_reason':'non_stop' if 'non_stop_finish' in a['strict']['errors'] else 'stop',
                'math_verify':a['strict']['math_verify']} for a in r.get('attempts',[])]})
    summaries={}
    for name,rs in scenarios.items():
        attempts=[];pairs=[];strict_pairs=[]
        for r in rs:
            first=None;firststrict=None
            for a in r['attempts']:
                v=review_admission(a['text'],r['problem'],a['finish_reason'],tokenizer,a['math_verify'])
                if v['normalization']['repairable']:
                    assert box_values(a['text'])==box_values(v['text']), 'Normalization changed boxed mathematics'
                entry={'sample_id':r['sample_id'],'attempt':a['attempt'],'review':v}
                attempts.append(entry)
                if v['accepted_candidate'] and first is None:first=entry
                if v['strict_existing_wrapper_candidate'] and firststrict is None:firststrict=entry
            if name=='old_cache':
                rawchecks=[x for x in read(REPORT/'cache_reaudit/normalized_pairs.json') if x['sample_id']==r['sample_id']]
                # Original 50 references were freshly reverified in cache_reaudit;
                # use their result from the per-question file, never source labels.
                q=next(x for x in read(REPORT/'cache_reaudit/questions.json') if x['sample_id']==r['sample_id'])
                rawok=q['original_verified']
            else:rawok=r['original_math_verify'].get('correct',False)
            base={k:r[k] for k in ['sample_id','problem_type','problem','reference_answer','raw_solution']}
            if first and rawok:
                pairs.append({**base,'compact_solution':first['review']['text'],'attempt':first['attempt'],
                    'answer_tokens':first['review']['answer_tokens'],'normalization':first['review']['normalization'],
                    'human_review':'pending','ai_process_review':'pending','training_approved':False})
            if firststrict and rawok:strict_pairs.append(r['sample_id'])
        summary={'questions':len(rs),'attempts':len(attempts),'strict_existing_wrapper_common_pairs':len(strict_pairs),
            'normalized_common_pairs':len(pairs),'failure_signals_nonexclusive':dict(Counter(e for a in attempts for e in a['review']['errors'])),
            'normalization_failures':dict(Counter(str(a['review']['normalization'].get('reason')) for a in attempts if not a['review']['normalization']['repairable'])),
            'per_type_candidates':dict(Counter(r['problem_type'] for r in pairs)),
            'training_approved':False,'human_review_completed':False,
            'scope':'Post-hoc implementation/format robustness review, not a passed frozen pilot or process proof'}
        dump(out/(name+'_attempts.json'),attempts);dump(out/(name+'_pairs.json'),pairs)
        dump(out/(name+'_summary.json'),summary);summaries[name]=summary
    dump(out/'summary.json',{'version':VERSION,'scenarios':summaries,
        'amendment':'Frozen parser treated ordinary LaTeX display closers and prose after </think> before final boxed as invalid. This follow-up accepts documented serialization normalization, never repairs mathematics. Nested/unclosed tags and text after final answer remain isolated.',
        'independent_validation_passed':False,'new_api_calls':0})
    dump(out/'manifest.json',{'code':{p:file_sha(ROOT/p) for p in ['src/synthesis/compact_format_review.py','scripts/review_compact_format.py']},
        'files':{p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}})
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':run()
