"""V4 bounded numeric normalization and explicit source/scope triage."""
import json
import re
from decimal import Decimal
from src.synthesis.compact_v3 import SYSTEM as V3_SYSTEM,serialize_response as v3_serialize,admission as v3_admit,payload as v3_payload
from src.synthesis.compact_v2 import box_spans

VERSION='compact-v4-numeric-only-20261024'
SYSTEM=V3_SYSTEM+' Preserve all legitimate cases. Simplify exponential coefficients carefully before dividing. A short complete solution is preferable to padding. Never encode control characters in mathematical answers.'
SOURCE_ISOLATIONS={
 '826e786b':'damaged_reference_answer', '8937d875':'reference_exponents_and_relation_conflict',
 '57a1fd84':'circle_internal_external_tangency_unspecified','846b4c1b':'average_price_semantics_ambiguous'}
KNOWN_WRONG_OUTPUTS={'b800f924':'independent_algebra_check: both sides factor to -20 times base^x, answer x>0; teacher introduced spurious 4/5',
 '0e10fb1c':'independent_process_check: teacher incorrectly excluded legitimate a=b branch'}
SIMPLE_REVIEW_IDS=('8058c171','badc3743','f179832f')


def starts(sid,mapping):return next((v for k,v in mapping.items() if sid.startswith(k)),None)


def clean_source(problem):
    s=problem;edits=[]
    # Anchored metadata-only tails; never cut arbitrary solution-like mathematical prose.
    t=re.sub(r'(?is)\n\s*(?:##\s*SOLUTION:\s*)?The above text has been translated into English[^\n]*(?:\n\s*)*\Z','',s)
    if t!=s:edits.append('strip_known_translation_metadata_tail');s=t
    t=re.sub(r'(?im)^\s*#{1,6}\s*(?:SOLUTION:)?\s*$(?:\n)?','',s)
    if t!=s:edits.append('strip_empty_markdown_heading');s=t
    return {'problem':s.strip(),'edits':edits,'mathematical_content_added':False}


def media_class(problem):
    count=len(re.findall(r'\[asy\]',problem,re.I));ends=len(re.findall(r'\[/asy\]',problem,re.I))
    if count:
        if count!=ends:return 'depends_on_missing_image_or_incomplete_asy'
        prose=re.sub(r'(?is)\[asy\].*?\[/asy\]','',problem)
        if re.search(r'(?i)\b(?:shown|diagram|figure|pictured|shaded)\b',prose):return 'requires_drawing_code_review'
        return 'prose_sufficiency_pending_review'
    if re.search(r'(?i)\b(?:figure|diagram|pictured|shown below|shown above|as shown|graph below)\b|如图|图中|<image|!\[|https?://',problem):return 'depends_on_missing_image_or_external_material'
    return 'prose_only_screen_pass_semantics_pending'


def boxes_kind(solution):
    try:values=[solution[a:b][solution[a:b].index('{')+1:-1] for a,b in box_spans(solution)]
    except (ValueError,IndexError,TypeError):return {'kind':'malformed_boxes','count':0}
    if not values:return {'kind':'no_box','count':0}
    compact=[re.sub(r'\s+','',str(x)) for x in values]
    if len(values)==1:return {'kind':'single_box','count':1}
    if len(set(compact))==1:return {'kind':'repeated_identical_answer','count':len(values)}
    return {'kind':'distinct_boxes_components_collection_or_intermediate_requires_review','count':len(values)}


def source_triage(row):
    sid=row['sample_id'];c=clean_source(row['problem']);p=c['problem'];ref=str(row['reference_answer']).strip()
    reasons=[];scope=[];manual=[]
    known=starts(sid,SOURCE_ISOLATIONS)
    if known:reasons.append(known)
    if not ref or ref.startswith('=') or re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',ref):reasons.append('damaged_or_empty_reference')
    if re.fullmatch('[A-Ea-e]',ref):scope.append('open_solution_experiment_excludes_option_letter_answer_not_quality_defect')
    elif re.search(r'\([A-E]\).*\([A-E]\)',p,re.S):scope.append('open_solution_experiment_excludes_complete_multiple_choice_not_quality_defect')
    media=media_class(p)
    if media.startswith('depends_on_missing'):reasons.append(media)
    elif '[asy]' in p.lower():manual.append(media)
    if any(sid.startswith(x) for x in SIMPLE_REVIEW_IDS):scope.append('predeclared_known_elementary_arithmetic_scope_exclusion_not_wrong_math')
    if re.search(r'(?i)\b(?:parts? \(a\)|answer both|find both.*and.*respectively)\b',p):manual.append('possible_multiple_questions_answer_structure_review')
    b=boxes_kind(row.get('solution',row.get('raw_solution','')))
    if b['kind'] in ('malformed_boxes','no_box') or b['kind'].startswith('distinct_boxes'):manual.append(b['kind'])
    return {'sample_id':sid,'cleaning':c,'quality_isolation_reasons':sorted(set(reasons)),
        'experiment_scope_exclusions':scope,'manual_review_flags':manual,'media_class':media,'boxes':b,
        'automatic_pool_eligible':not(reasons or scope or manual),'difficulty':'not_labeled; token length is not a difficulty label',
        'human_review':'pending','training_approved':False}


def serialize_response(content):
    try:d=json.loads(content,parse_float=Decimal,parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite_json_number')))
    except (ValueError,TypeError):return {'valid':False,'errors':['invalid_json_no_blind_repair'],'text':'','fields':None,'normalization_operations':[]}
    operations=[]
    if isinstance(d,dict) and 'final_answer' in d:
        v=d['final_answer']
        if type(v) is int:
            d['final_answer']=str(v);operations.append('finite_integer_final_answer_to_string')
        elif isinstance(v,Decimal) and v.is_finite():
            d['final_answer']=str(v);operations.append('finite_decimal_final_answer_to_string_preserve_decimal_value')
    try:normalized=json.dumps(d,ensure_ascii=False)
    except TypeError:return {'valid':False,'errors':['non_string_field'],'text':'','fields':None,'normalization_operations':operations}
    result=v3_serialize(normalized);result['normalization_operations']=operations
    result['global_backslash_repair']=False
    return result


def payload(problem):
    p=v3_payload(problem);p['messages'][0]['content']=SYSTEM;return p


def admission(parsed,problem,finish,tokenizer,math_result):
    a=v3_admit(parsed,problem,finish,tokenizer,math_result);a['rule_version']=VERSION;return a
