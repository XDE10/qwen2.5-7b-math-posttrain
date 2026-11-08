"""Bounded answer extraction and safe expression equivalence, no eval/sympify.

Only final markers or a standalone expression are accepted. Units may be stripped
from an allowlist, never converted. Not a complete MATH benchmark verifier.
"""
import ast
import re
from dataclasses import dataclass
import sympy as sp


class InvalidAnswer(ValueError):
    pass


def _braced(text, start):
    if start >= len(text) or text[start] != '{':
        raise InvalidAnswer('missing_brace')
    depth = 1
    for i in range(start+1,len(text)):
        depth += (text[i]=='{')-(text[i]=='}')
        if depth > 12:
            raise InvalidAnswer('nesting_limit')
        if depth == 0:
            return text[start+1:i], i+1
    raise InvalidAnswer('unclosed_brace')


def _latex(text):
    text = text.replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
    for _ in range(20):
        m = re.search(r'\\(frac|sqrt|boxed)\s*\{',text)
        if not m:
            break
        first,end = _braced(text,m.end()-1)
        if m[1] == 'frac':
            while end<len(text) and text[end].isspace(): end += 1
            second,end = _braced(text,end)
            value = '(('+_latex(first)+')/('+_latex(second)+'))'
        elif m[1] == 'sqrt':
            value = 'sqrt('+_latex(first)+')'
        else:
            value = '('+_latex(first)+')'
        text = text[:m.start()]+value+text[end:]
    return text


def expression(text):
    text = text.strip()
    if len(text) > 512 or not text:
        raise InvalidAnswer('expression_length')
    text = re.sub(r'^(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*)?:?\s*','',text,flags=re.I)
    text = text.strip('$ ').replace('−','-').replace('π','pi')
    text = re.sub(r'\\(?:text|mathrm)\{(cm|mm|m|km|hours?|minutes?|seconds?|dollars?|ounces?|degrees?)\}',r' \1',text)
    text = re.sub(r'\s*(?:cm|mm|km|m|hours?|minutes?|seconds?|dollars?|ounces?|degrees?)(?:\^?2|²|\^?3|³)?\.?$','',text,flags=re.I)
    text = text.removesuffix('.').strip('$ ')
    # Validate thousands separators before removing, rather than joining 1,2.
    for match in re.finditer(r'\d[\d,]*',text):
        if ',' in match[0] and not re.fullmatch(r'\d{1,3}(?:,\d{3})+',match[0]):
            raise InvalidAnswer('bad_separator')
    text = text.replace(',','').replace('\\left','').replace('\\right','')
    text = _latex(text).replace('\\cdot','*').replace('\\times','*').replace('\\pi','pi')
    text = text.replace('^','**').replace('{','(').replace('}',')').strip()
    if len(text)>512 or text.count('(')>20:
        raise InvalidAnswer('complexity_limit')
    try: tree = ast.parse(text,mode='eval')
    except (SyntaxError,RecursionError) as e: raise InvalidAnswer('unsupported_expression') from e
    if sum(1 for _ in ast.walk(tree))>80:
        raise InvalidAnswer('complexity_limit')
    def convert(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            if len(str(node.value))>40: raise InvalidAnswer('numeric_limit')
            return sp.Rational(str(node.value))
        if isinstance(node,ast.Name) and (node.id in {'pi'} or re.fullmatch('[a-z]',node.id)):
            return sp.pi if node.id=='pi' else sp.Symbol(node.id,real=True)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            v = convert(node.operand)
            return -v if isinstance(node.op,ast.USub) else v
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='sqrt' and len(node.args)==1 and not node.keywords:
            return sp.sqrt(convert(node.args[0]))
        if isinstance(node,ast.BinOp):
            a,b = convert(node.left),convert(node.right)
            if isinstance(node.op,ast.Add): return a+b
            if isinstance(node.op,ast.Sub): return a-b
            if isinstance(node.op,ast.Mult): return a*b
            if isinstance(node.op,ast.Div):
                if b==0: raise InvalidAnswer('zero_denominator')
                return a/b
            if isinstance(node.op,ast.Pow) and b.is_number and abs(b)<=12:
                if a.is_Rational and b.is_Integer and (abs(int(a.p)).bit_length()+int(a.q).bit_length())*max(1,abs(int(b)))>4096:
                    raise InvalidAnswer('numeric_power_limit')
                return a**b
        raise InvalidAnswer('unsupported_expression')
    value = convert(tree.body)
    if value.has(sp.zoo,sp.nan,sp.oo,-sp.oo) or value.is_real is False:
        raise InvalidAnswer('nonfinite_or_complex')
    if len(value.free_symbols)>2 or sp.count_ops(value)>40:
        raise InvalidAnswer('symbolic_complexity_limit')
    return value


def equivalent(a,b):
    return sp.simplify(expression(a)-expression(b)) == 0


def extract(output, truncated=False):
    if truncated:
        raise InvalidAnswer('truncated')
    values = []
    for match in re.finditer(r'\\boxed\s*\{',output):
        body,_ = _braced(output,match.end()-1)
        values.append(body)
    if '\\boxed' in output and not values:
        raise InvalidAnswer('malformed_box')
    for section in output.split('####')[1:]:
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if not lines: raise InvalidAnswer('empty_final')
        values.append(lines[0])
    if not values:
        values = [output.strip()]
    parsed = [expression(v) for v in values]
    if any(sp.simplify(v-parsed[0]) != 0 for v in parsed[1:]):
        raise InvalidAnswer('conflicting_answers')
    return values[0]


def verify(output, reference, truncated=False):
    try:
        answer = extract(output,truncated)
        correct = equivalent(answer, reference)
        return {'correct':bool(correct),'status':'ok' if correct else 'incorrect','extracted':answer}
    except (InvalidAnswer,ValueError,TypeError,OverflowError,RecursionError) as e:
        return {'correct':False,'status':str(e),'extracted':None}
