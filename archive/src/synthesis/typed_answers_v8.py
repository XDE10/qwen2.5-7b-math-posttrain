"""Strict contextual one-variable solution sets. Unsupported is never pass.

This compares answers, not the validity of a mathematical derivation. Production
admission still requires source scope, length and full-trajectory review.
"""
import re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'.pilot_deps'))
import sympy as sp
from math_verify import parse,LatexExtractionConfig

class Unsupported(ValueError):pass

def clean(s):
    if not isinstance(s,str) or len(s)>1600:raise Unsupported('answer_size_or_type')
    s=s.strip().strip('$').strip();s=re.sub(r'\\(?:left|right)\b','',s)
    s=re.sub(r'\\(?:,|;|!|quad\b|qquad\b)',' ',s)
    return s.strip()

def top_split(s,separators):
    parts=[];start=0;depth=0;i=0
    while i<len(s):
        if depth==0:
            sep=next((x for x in separators if s.startswith(x,i)),None)
            if sep:parts.append(s[start:i]);i+=len(sep);start=i;continue
        if s[i] in '([{':depth+=1
        elif s[i] in ')]}':depth-=1
        if depth<0:raise Unsupported('unbalanced_delimiters')
        i+=1
    if depth:raise Unsupported('unbalanced_delimiters')
    parts.append(s[start:]);return parts

def scalar(s,allow_infinity=False):
    s=clean(s)
    if '^' in s:raise Unsupported('power_endpoint_outside_initial_safe_grammar')
    if re.search(r'\d{19,}|\d\s+\d',s):raise Unsupported('oversized_or_ambiguous_number_token')
    # Check original variable-bearing syntax BEFORE SymPy can cancel x/x or
    # discard domains. Endpoints must be ground constants, not expressions in x.
    bare=re.sub(r'\\(?:dfrac|tfrac|frac|sqrt|infty|pi)\b','',s)
    if re.search(r'[A-Za-z\\]',bare):raise Unsupported('nonconstant_or_unsupported_endpoint')
    if not re.fullmatch(r'[0-9{}().+*/^\s\-]*',bare):raise Unsupported('endpoint_grammar')
    xs=parse(r'\boxed{'+s+'}',extraction_config=[LatexExtractionConfig()],extraction_mode='first_match',parsing_timeout=0)
    values=[x for x in xs if not isinstance(x,str)]
    if len(values)!=1:raise Unsupported('scalar_not_parsed_uniquely')
    v=values[0]
    if not isinstance(v,sp.Expr) or v.free_symbols or v.is_extended_real is not True:raise Unsupported('endpoint_not_ground_real')
    if v.has(sp.nan,sp.zoo):raise Unsupported('nonfinite_endpoint')
    if v in (sp.oo,-sp.oo):
        if not allow_infinity:raise Unsupported('infinity_not_finite_element')
    elif v.is_finite is not True:raise Unsupported('endpoint_finiteness_unknown')
    # Exact decimal semantics; no approximate/tolerance comparison.
    v=v.xreplace({x:sp.Rational(str(x)) for x in v.atoms(sp.Float)})
    if sp.count_ops(v)>60:raise Unsupported('endpoint_expression_complexity')
    return sp.simplify(v)

def solution_set(text,profile):
    if profile.get('kind')!='one_variable_solution_set' or profile.get('domain') not in {'real','integer'} or not re.fullmatch('[a-z]',profile.get('variable','')):raise Unsupported('unsupported_answer_profile')
    s=clean(text);var=profile['variable'];domain=sp.S.Reals if profile['domain']=='real' else sp.S.Integers
    if s in [r'\mathbb{R}',r'\mathbb R']:
        if profile['domain']!='real':raise Unsupported('explicit_domain_notation_mismatch')
        return sp.S.Reals
    if s in [r'\mathbb{Z}',r'\mathbb Z']:return sp.S.Integers
    if s in [r'\emptyset',r'\varnothing']:return sp.S.EmptySet
    def finite(values):
        if profile['domain']=='integer' and any(v.is_integer is not True for v in values):raise Unsupported('noninteger_explicit_solution')
        return sp.FiniteSet(*values)
    membership=re.fullmatch(r'([a-z])\s*\\in\s*(.+)',s,re.S)
    if membership:
        if membership.group(1)!=var:raise Unsupported('wrong_answer_variable')
        s=membership.group(2).strip()
    if profile.get('enumerated_roots') and ';' in s:
        parts=top_split(s,[';']);values=[]
        for i,p in enumerate(parts,1):
            m=re.fullmatch(re.escape(var)+r'_\{?([0-9]+)\}?\s*=\s*(.+)',p.strip(),re.S)
            if not m or int(m.group(1))!=i:raise Unsupported('enumerated_root_role_or_index')
            values.append(scalar(m.group(2)))
        if len(set(values))!=len(values):raise Unsupported('duplicate_enumerated_roots')
        return finite(values)
    union=top_split(s,[r'\cup',r'\lor',r'\vee'])
    if len(union)>1:
        if len(union)>16:raise Unsupported('too_many_set_components')
        return sp.Union(*(solution_set(x,profile) for x in union))
    intersection=top_split(s,[r'\cap',r'\land',r'\wedge'])
    if len(intersection)>1:
        if len(intersection)>16:raise Unsupported('too_many_set_components')
        return sp.Intersection(*(solution_set(x,profile) for x in intersection))
    if s.startswith((r'\{','{')) and s.endswith((r'\}','}')):
        body=s[2:-2] if s.startswith(r'\{') and s.endswith(r'\}') else s[1:-1]
        parts=top_split(body,[','])
        if len(parts)>32:raise Unsupported('too_many_finite_elements')
        return finite([scalar(x) for x in parts])
    if s.startswith(('(', '[')) and s.endswith((')',']')):
        parts=top_split(s[1:-1],[',',';'])
        if len(parts)!=2:raise Unsupported('interval_arity')
        a,b=(scalar(x,True) for x in parts)
        if a==sp.oo or b==-sp.oo or (a-b).is_positive is True:raise Unsupported('invalid_interval_order')
        if (a==-sp.oo and s[0]=='[') or (b==sp.oo and s[-1]==']'):raise Unsupported('closed_infinity_endpoint')
        return sp.Intersection(sp.Interval(a,b,left_open=s[0]=='(',right_open=s[-1]==')'),domain)
    ops={r'\leqslant':'<=',r'\geqslant':'>=',r'\leq':'<=',r'\geq':'>=',r'\le':'<=',r'\ge':'>=',r'\lt':'<',r'\gt':'>',r'\neq':'!=',r'\ne':'!='}
    for old,new in ops.items():s=s.replace(old,new)
    parts=re.split(r'(<=|>=|!=|=|<|>)',s)
    if len(parts)>1:
        if len(parts)>17:raise Unsupported('too_many_relations')
        x=sp.Symbol(var,real=True);terms=[]
        for term in parts[::2]:terms.append(x if term.strip()==var else scalar(term,allow_infinity=True))
        sets=[];construct={'<':sp.Lt,'<=':sp.Le,'>':sp.Gt,'>=':sp.Ge,'=':sp.Eq,'!=':sp.Ne}
        for i,op in enumerate(parts[1::2]):
            a,b=terms[i:i+2]
            if a!=x and b!=x:raise Unsupported('relation_without_requested_variable')
            sets.append(construct[op](a,b).as_set())
        return sp.Intersection(domain,*sets)
    pieces=top_split(s,[','])
    if len(pieces)>1:
        if len(pieces)>32:raise Unsupported('too_many_finite_elements')
        return finite([scalar(x) for x in pieces])
    return finite([scalar(s)])

def equivalent(prediction,reference,profile):
    try:
        gold=solution_set(reference,profile);pred=solution_set(prediction,profile)
        difference=sp.SymmetricDifference(gold,pred)
        correct=difference==sp.S.EmptySet
        return {'correct':bool(correct),'status':'exact_typed_set_equivalent' if correct else 'typed_set_mismatch','gold_set':str(gold),'prediction_set':str(pred),'domain':profile['domain'],'variable':profile['variable'],'process_correctness':'not_certified'}
    except Exception as e:return {'correct':False,'status':'unsupported_typed_answer','reason':str(e),'error_type':type(e).__name__}

def answer_boxes(text):
    boxes=[]
    for m in re.finditer(r'\\boxed\s*\{',text):
        depth=1;i=m.end()
        while i<len(text) and depth:
            if text[i]=='{' and (i==0 or text[i-1]!='\\'):depth+=1
            elif text[i]=='}' and (i==0 or text[i-1]!='\\'):depth-=1
            i+=1
        if depth:raise Unsupported('unclosed_final_box')
        value=text[m.end():i-1];before=text[max(0,m.start()-100):m.start()];after=text[i:i+8]
        quoted=re.search(r'([`\"\'])\s*$',before)
        literal=not value.strip() and bool(quoted) and bool(re.match(r'\s*'+re.escape(quoted.group(1)),after)) and bool(re.search(r'(?i)format|template|notation|instruction|within|example|格式|示例',before))
        boxes.append({'value':value,'start':m.start(),'end':i,'literal_quoted_format_example':literal})
    return boxes

def check_roles(text,reference,profile):
    try:
        if text.count('<think>')!=text.count('</think>') or text.count('</think>')>1:raise Unsupported('ambiguous_think_wrapper')
        zone=text.split('</think>',1)[1] if '</think>' in text else text
        raw_boxes=answer_boxes(zone);boxes=[b for b in raw_boxes if not b['literal_quoted_format_example']]
        if not boxes or any(not b['value'].strip() for b in boxes):raise Unsupported('missing_or_empty_actual_answer_box')
        results=[equivalent(b['value'],reference,profile) for b in boxes]
        if all(r['correct'] for r in results):return {'correct':True,'status':'all_final_zone_boxes_semantically_equivalent','boxes':results,'ignored_quoted_literal_examples':len(raw_boxes)-len(boxes),'ignored_think_zone_boxes_for_answer_extraction_only':'</think>' in text,'raw_process_still_requires_full_review':True}
        gold=solution_set(reference,profile)
        if isinstance(gold,sp.FiniteSet) and len(boxes)>1:
            gaps=''.join(zone[a['end']:b['start']] for a,b in zip(boxes,boxes[1:]))
            if not re.fullmatch(r'[\s,;.\[\]\\]*(?:(?:and)[\s,;.\[\]\\]*)*',gaps):raise Unsupported('nonterminal_collection_or_intermediate_boxes')
            values=[scalar(b['value']) for b in boxes];combined=sp.FiniteSet(*values)
            if combined==gold:return {'correct':True,'status':'explicit_finite_solution_collection','collection':str(combined),'raw_process_still_requires_full_review':True}
        return {'correct':False,'status':'final_boxes_not_all_equivalent','boxes':results}
    except Exception as e:return {'correct':False,'status':'unsupported_answer_roles','reason':str(e),'error_type':type(e).__name__}
