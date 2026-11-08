"""Documented format repair for cached written solutions, no mathematical edits.

Kept separate from the frozen v2 pilot so its original admission is reproducible.
This is a post-hoc format robustness analysis, not a new independent pilot.
"""
import re
from src.synthesis.compact_v2 import box_spans, inspect_structure, admit

VERSION = 'compact-format-review-v2.1-20261023'


def normalize_written_solution(text):
    original=text; text=text.strip();ops=[]
    if text.startswith('```') and text.endswith('```'):
        lines=text.splitlines()
        if len(lines)>2 and lines[0] in ('```','```text','```markdown'):
            text='\n'.join(lines[1:-1]).strip();ops.append('remove_outer_markdown_fence')
    opened,closed=text.count('<think>'),text.count('</think>')
    if (opened,closed) not in ((0,0),(1,1)) or (opened and text.find('<think>')>text.find('</think>')):
        return {'text':original,'operations':ops,'repairable':False,'reason':'malformed_or_multiple_think_tags'}
    # These are published solution wrappers. Preserve all mathematical/prose
    # content before and after them, rather than discarding an outside derivation.
    if opened:
        text=text.replace('<think>','').replace('</think>','').strip()
        ops.append('unwrap_existing_written_solution_tags')
    try:spans=box_spans(text)
    except ValueError as e:return {'text':original,'operations':ops,'repairable':False,'reason':str(e)}
    if not spans:return {'text':original,'operations':ops,'repairable':False,'reason':'missing_box'}
    a,b=spans[-1];prefix=text[:a].rstrip();tail=text[b:].strip();answer=text[a:b]
    # Only remove paired DISPLAY delimiters and final punctuation. Never delete
    # explanatory prose or silently repair a conflicting answer.
    tail=tail.lstrip('.').strip()
    pairs=[('\\[','\\]'),('\\(','\\)'),('$$','$$'),('$','$')]
    for left,right in pairs:
        if prefix.endswith(left) and tail.startswith(right):
            prefix=prefix[:-len(left)].rstrip();tail=tail[len(right):].strip()
            ops.append('remove_final_box_math_display_delimiters');break
    if tail.strip(' \t\r\n.'):
        return {'text':original,'operations':ops,'repairable':False,'reason':'content_after_final_answer_needs_review'}
    canonical='<think>\n'+prefix+'\n</think>\n'+answer
    check=inspect_structure(canonical)
    return {'text':canonical,'operations':ops+['canonical_wrap_existing_derivation'],
            'repairable':check['structure_pass'],'reason':None if check['structure_pass'] else check['errors'],
            'mathematical_content_added':False,'process_correctness':'not_certified'}


def review_admission(text,problem,finish_reason,tokenizer,math_result):
    norm=normalize_written_solution(text)
    strict_base=inspect_structure(text)
    # Strict means already has one closed wrapper plus final boxed outside it;
    # normal TeX display delimiters are allowed around the final boxed answer.
    strict_existing=(text.count('<think>')==text.count('</think>')==1
                     and text.find('<think>')<text.find('</think>'))
    try:last_start=box_spans(text)[-1][0]
    except (ValueError,IndexError):last_start=-1
    strict_existing=strict_existing and last_start>text.find('</think>')
    result=admit(norm['text'],problem,finish_reason,tokenizer,math_result,False)
    result['accepted_candidate']=result['accepted_candidate'] and norm['repairable']
    result['normalization']=norm
    result['strict_existing_wrapper_candidate']=result['accepted_candidate'] and strict_existing
    result['rule_version']=VERSION
    result['assessment_scope']='post_hoc_format_robustness_not_independent_gate'
    return result
