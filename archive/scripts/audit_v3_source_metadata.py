"""Additional source-metadata review queue; no changes to the frozen v3 pool."""
import json
import re
from collections import Counter
from pathlib import Path


def flags(problem,reference):
    result=[];a=str(reference)
    if re.search(r'(?im)^\s*#{0,6}\s*(?:solution|answer|proof)\s*[:：]',problem):
        result.append('solution_or_answer_heading_in_problem_requires_review')
    if re.search(r'(?i)(?:above|following) text.{0,50}(?:translated|translation)|preserving.{0,50}line breaks|output the translation',problem):
        result.append('translation_metadata_remains')
    if re.search(r'(^|\n)\s*#{1,6}\s*$',problem):result.append('empty_markdown_heading')
    if a.lstrip().startswith(('=','>','<')):result.append('reference_starts_with_relation_or_missing_lhs')
    if re.search(r'\\mathrm\{(?:\s|~)*\}',a):result.append('reference_empty_unit_markup')
    return result


def run():
    root=Path(__file__).resolve().parents[1]
    rows=[json.loads(s) for s in (root/'data/project1_textonly_v3/native_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    flagged=[]
    for r in rows:
        f=flags(r['problem'],r['reference_answer'])
        if f:flagged.append({'sample_id':r['sample_id'],'flags':f,'human_review':'pending','training_approved':False})
    result={'scope':'supplemental source review overlay; frozen pool and pilot unchanged',
        'pool_count':len(rows),'flagged_questions':len(flagged),'counts':dict(Counter(x for r in flagged for x in r['flags'])),
        'rows':flagged,'bulk_generation_enabled':False,'training_approved':False,
        'interpretation':'Format/source warning, not a claim that every flagged problem is mathematically invalid.'}
    out=root/'reports/compact_distillation_v3_20261023/source_review_flags.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['pool_count','flagged_questions','counts']},ensure_ascii=False))


if __name__=='__main__':run()
