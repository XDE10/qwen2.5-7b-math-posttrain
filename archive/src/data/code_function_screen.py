"""Conservative standalone-function contracts for AceCode and Magicoder.

Static inspection only: no reference or test code runs in the host interpreter.
This first version deliberately excludes methods, custom objects and rich tests.
"""
import ast
import copy
import hashlib
import inspect
import builtins
import re
import sys
import math
import json
from pathlib import Path
from src.data.screen_code_interfaces import InterfaceError, python_code

FORBIDDEN_IMPORTS={'os','sys','subprocess','socket','pathlib','shutil','ctypes','pickle',
    'marshal','multiprocessing','threading','resource','signal','importlib','builtins'}
FORBIDDEN_CALLS={'eval','exec','compile','open','input','__import__','globals','locals','breakpoint'}
QUARANTINE=Path('data/code_v2_sources/pilot_quarantine.json')


def literal(node):
    if len(list(ast.walk(node)))>5000:raise InterfaceError('test_literal_size')
    try:return ast.literal_eval(node)
    except (ValueError,SyntaxError,RecursionError):raise InterfaceError('nonliteral_test_value')


def serializable_literal(value,depth=0):
    if depth>32:raise InterfaceError('test_literal_depth')
    if isinstance(value,float) and not math.isfinite(value):raise InterfaceError('nonfinite_test_literal')
    if isinstance(value,(list,tuple)):
        for child in value:serializable_literal(child,depth+1)
    elif isinstance(value,dict):
        if not all(isinstance(k,str) for k in value):raise InterfaceError('nonstring_test_mapping_key')
        for child in value.values():serializable_literal(child,depth+1)
    elif value is not None and not isinstance(value,(str,int,float,bool)):
        raise InterfaceError('unsupported_serialized_test_literal')


def simple_tests(texts):
    """Require each test to compare one direct target call against a literal."""
    if not isinstance(texts,list) or not 3<=len(texts)<=200:raise InterfaceError('test_count_outside_3_200')
    names=set();parsed=[]
    for text in texts:
        if not isinstance(text,str) or len(text)>20000:raise InterfaceError('test_text_size')
        try:tree=ast.parse(text)
        except (SyntaxError,ValueError,RecursionError):raise InterfaceError('test_syntax')
        if not tree.body or not all(isinstance(n,ast.Assert) for n in tree.body):raise InterfaceError('nonassert_test_statement')
        for statement in tree.body:
            expr=statement.test
            if not (isinstance(expr,ast.Compare) and len(expr.ops)==1 and isinstance(expr.ops[0],ast.Eq)
                    and isinstance(expr.left,ast.Call) and isinstance(expr.left.func,ast.Name)):
                raise InterfaceError('unsupported_assert_shape')
            call=expr.left;names.add(call.func.id)
            if any(k.arg is None for k in call.keywords):raise InterfaceError('expanded_test_keywords')
            if len({k.arg for k in call.keywords})!=len(call.keywords):raise InterfaceError('duplicate_test_keywords')
            args=[literal(a) for a in call.args]
            kwargs={k.arg:literal(k.value) for k in call.keywords}
            expected=literal(expr.comparators[0])
            for value in args+list(kwargs.values())+[expected]:serializable_literal(value)
            parsed.append({'args':args,'kwargs':kwargs,'expected_repr':repr(expected),
                           'assertion':ast.unparse(statement)})
    if len(names)!=1:raise InterfaceError('multiple_test_entry_points')
    return next(iter(names)),parsed


def question_names(question,names):
    """The public specification must explicitly name the selected callable."""
    hits=[]
    for name in names:
        # Accept an explicit signature, a backtick identifier, or a named function.
        patterns=[rf'\bdef\s+{re.escape(name)}\s*\(',rf'`{re.escape(name)}(?:\([^`]*\))?`',
                  rf'\bfunction\s+(?:called\s+|named\s+)?{re.escape(name)}\b']
        if any(re.search(p,question,re.I) for p in patterns):hits.append(name)
    return hits


def inspect_reference(text,question,target=None,test_records=None):
    code,tree=python_code(text)
    if len(code)>100000:raise InterfaceError('reference_size')
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    if len(functions)!=sum(isinstance(n,ast.FunctionDef) for n in tree.body):raise InterfaceError('duplicate_function_definition')
    if any(isinstance(n,(ast.ClassDef,ast.AsyncFunctionDef)) for n in tree.body):raise InterfaceError('class_or_async_reference')
    names=question_names(question,functions)
    if target is None:
        if len(names)!=1:raise InterfaceError('ambiguous_public_function_name')
        target=names[0]
    elif target not in names:raise InterfaceError('test_entry_not_named_in_question')
    if target not in functions:raise InterfaceError('target_missing_from_reference')
    fn=functions[target]
    if fn.decorator_list or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
        raise InterfaceError('unsupported_signature')
    parameters=list(fn.args.posonlyargs)+list(fn.args.args)
    if not 1<=len(parameters)<=8 or parameters[0].arg in {'self','cls'}:raise InterfaceError('unsupported_parameters')
    if re.search(r'(?:should|must|does)\s+not\s+return|returns?\s+nothing',question,re.I):
        raise InterfaceError('nonreturning_public_contract')
    if re.search(r'\bdef\s+'+re.escape(target)+r'\([^\n]*?\)\s*->\s*None\b',question):
        raise InterfaceError('public_none_return_contract')
    # A reference solution cannot silently add parameters to an explicit public
    # signature. This catches real Magicoder examples such as getDigest(input)
    # paired with getDigest(input, algorithm).
    public=[]
    for match in re.finditer(r'\b'+re.escape(target)+r'\s*\(([^()\n]*)\)',question):
        try:stub=ast.parse('def '+target+'('+match.group(1)+'):\n pass').body[0]
        except SyntaxError:continue  # A concrete example call is not a declaration.
        public.append(([a.arg for a in list(stub.args.posonlyargs)+list(stub.args.args)],len(stub.args.defaults)))
    actual=([p.arg for p in parameters],len(fn.args.defaults))
    if public and any(p!=actual for p in public):raise InterfaceError('public_reference_signature_conflict')
    if not any(isinstance(n,ast.Return) and n.value is not None for n in fn.body) and not any(
            isinstance(n,ast.Return) and n.value is not None for s in fn.body if not isinstance(s,ast.FunctionDef) for n in ast.walk(s)):
        raise InterfaceError('no_value_return')
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [(node.module or '').split('.')[0]]
            if any(m not in sys.stdlib_module_names or m in FORBIDDEN_IMPORTS for m in modules):raise InterfaceError('unsupported_import')
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
            if name in FORBIDDEN_CALLS or name.startswith('__'):raise InterfaceError('host_io_or_dynamic_call')
        if isinstance(node,ast.Attribute) and node.attr.startswith('__'):raise InterfaceError('dunder_attribute')
    for statement in tree.body:
        if isinstance(statement,(ast.FunctionDef,ast.Import,ast.ImportFrom)):continue
        if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Constant) and isinstance(statement.value.value,str):continue
        if isinstance(statement,(ast.Assign,ast.AnnAssign)):
            literal(statement.value);continue
        raise InterfaceError('executable_module_statement')
    defined=set(dir(builtins))|set(functions)|{n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    defined|={a.arg for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) for a in list(n.args.args)+list(n.args.posonlyargs)}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):defined|={a.asname or a.name.split('.')[0] for a in node.names}
        elif isinstance(node,ast.ImportFrom):defined|={a.asname or a.name for a in node.names}
    if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id not in defined for n in ast.walk(tree)):
        raise InterfaceError('undefined_callable_dependency')
    # Keep the exact default/positional-only contract but remove optional type hints
    # from the instruction to avoid introducing undefined annotation dependencies.
    arguments=copy.deepcopy(fn.args)
    for p in list(arguments.posonlyargs)+list(arguments.args):p.annotation=None
    defaults=[literal(d) for d in arguments.defaults]
    argspec=[inspect.Parameter(p.arg,inspect.Parameter.POSITIONAL_ONLY if i<len(arguments.posonlyargs)
             else inspect.Parameter.POSITIONAL_OR_KEYWORD,default=defaults[i-(len(parameters)-len(defaults))]
             if i>=len(parameters)-len(defaults) else inspect.Parameter.empty) for i,p in enumerate(parameters)]
    contract=inspect.Signature(argspec)
    for test in test_records or []:
        try:contract.bind(*test['args'],**test['kwargs'])
        except TypeError as exc:raise InterfaceError('test_signature_binding') from exc
    signature='def '+target+'('+ast.unparse(arguments)+'):'
    return {'entry_point':target,'function_signature':signature,'reference_code':code,
            'reference_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
            'reference_code_sha256':hashlib.sha256(code.encode()).hexdigest()}


def extract(row,source):
    source_id=source+'/'+str(row.get('id') if source=='ace' else row.get('raw_index'))
    if QUARANTINE.exists() and source_id in json.loads(QUARANTINE.read_text(encoding='utf-8'))['excluded']:
        raise InterfaceError('pilot_observed_specification_conflict')
    question=row.get('question') if source=='ace' else row.get('problem')
    if not isinstance(question,str) or not 30<=len(question)<=20000:raise InterfaceError('question_size')
    if source=='magicoder' and row.get('lang') not in {'python','python3'}:raise InterfaceError('not_python')
    if source=='ace':
        target,tests=simple_tests(row.get('test_cases'))
        solutions=[r['completion'] for r in (row.get('inferences') or [])
                   if r.get('pass_rate')==1 and isinstance(r.get('completion'),str)]
        if not solutions:raise InterfaceError('no_source_full_pass_completion')
    else:
        target=None;tests=[];solutions=[row.get('solution')]
    errors=[]
    for solution in solutions[:4]:
        try:
            reference=inspect_reference(solution,question,target,tests)
            return {**reference,'problem':question,'interface':'function','test_cases':tests,
                    'verification_status':'static_only' if source=='ace' else 'prompt_only_no_tests',
                    'source_category':row.get('source',row.get('lang'))}
        except InterfaceError as exc:errors.append(str(exc))
    raise InterfaceError(errors[0] if errors else 'missing_reference')
