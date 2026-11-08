"""Static, whole-test-list interface screening; never executes dataset code.

Native stdin, independent functions and Solution methods are distinguished.
Execution acceptance and human test-coverage audit remain separate gates.
"""
import argparse
import ast
import hashlib
import json
import keyword
import io
import math
import random
import re
import tokenize
import warnings
from collections import Counter
from pathlib import Path
from src.data.prepare_math_data import fingerprint, write_jsonl
from src.data.screening_io import rows, sha, json_write, source_files, parquet_rows, ScreeningSimilarityIndex as SimilarityIndex
from src.data.screen_openr1_sft import sealed_math


class InterfaceError(ValueError): pass


def python_code(text):
    if not isinstance(text,str) or not text.strip(): raise InterfaceError('missing_reference_code')
    fences = re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S|re.I)
    code = max(fences,key=len) if fences else text
    if len(code)>1_000_000: raise InterfaceError('reference_code_size_limit')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',SyntaxWarning)
            tree=ast.parse(code)
        return code,tree
    except (SyntaxError,ValueError,RecursionError) as exc: raise InterfaceError('reference_not_python_ast') from exc


def _json(value):
    if isinstance(value,str):
        if len(value)>100_000: raise InterfaceError('function_literal_size_limit')
        try:
            decoded=json.loads(value,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
        except (ValueError,RecursionError):
            try:
                tree=ast.parse(value.strip(),mode='eval')
                nodes=list(ast.walk(tree))
                if len(nodes)>20_000: raise ValueError('too many nodes')
                decoded=ast.literal_eval(tree)
            except (ValueError,SyntaxError,RecursionError,MemoryError) as exc:
                raise InterfaceError('non_json_function_io') from exc
        def check(v,depth=0):
            if depth>64: raise InterfaceError('function_literal_depth_limit')
            if isinstance(v,float) and not math.isfinite(v): raise InterfaceError('nonfinite_function_value')
            if isinstance(v,(list,tuple)):
                for child in v: check(child,depth+1)
            elif isinstance(v,dict):
                for k,child in v.items():check(k,depth+1);check(child,depth+1)
            elif v is not None and not isinstance(v,(str,int,float,bool)):
                raise InterfaceError('unsupported_function_value')
        check(decoded)
        return decoded
    if isinstance(value,(list,dict,int,float,bool)) or value is None: return value
    raise InterfaceError('unsupported_function_value')


def named_arguments(raw, names, minimum):
    """Parse explicit name=literal bindings without executing any source code."""
    if len(raw)>100_000: raise InterfaceError('function_literal_size_limit')
    try:
        depth=0;tokens=[]
        for t in tokenize.generate_tokens(io.StringIO(raw.strip()).readline):
            if t.type==tokenize.OP:
                if t.string in '([{':depth+=1
                elif t.string in ')]}':depth-=1
            if t.type in {tokenize.NL,tokenize.NEWLINE} and depth==0:
                if tokens and tokens[-1]!=(tokenize.OP,','):tokens.append((tokenize.OP,','))
            elif t.type not in {tokenize.ENDMARKER,tokenize.INDENT,tokenize.DEDENT}:
                tokens.append((t.type,t.string))
        expr=ast.parse('__arguments__('+tokenize.untokenize(tokens)+')',mode='eval').body
        if expr.args or not expr.keywords:raise ValueError('not named bindings')
        bindings={}
        for kw in expr.keywords:
            if kw.arg not in names or kw.arg in bindings:raise ValueError('unknown or repeated binding')
            bindings[kw.arg]=_json(ast.unparse(kw.value))
        # Positional invocation is safe only when supplied names form a prefix.
        supplied=names[:len(bindings)]
        if set(supplied)!=set(bindings) or len(bindings)<minimum:raise ValueError('missing parameter')
        return [bindings[name] for name in supplied]
    except (ValueError,SyntaxError,RecursionError,tokenize.TokenError,IndentationError) as exc:
        raise InterfaceError('invalid_named_function_arguments') from exc


def function_arguments(raw, minimum, maximum, argument_names=None):
    """Decode explicit JSON-lines positional args; reject ambiguous nesting.

    A single JSON list can be one list argument or a packed argument vector.
    For >1 required args only a length-matched vector is unpacked. Optional
    multi-parameter ambiguity is isolated rather than guessed.
    """
    if isinstance(raw,str):
        if argument_names is not None:
            try:
                return named_arguments(raw,argument_names,minimum)
            except InterfaceError:
                pass
        lines = [s for s in raw.strip().split('\n') if s.strip()]
        if len(lines)>1:
            args = [_json(s) for s in lines]
        else:
            value = _json(raw)
            if maximum==0 and value in [[],None]: args=[]
            elif maximum==1: args=[value]
            elif isinstance(value,list) and minimum==maximum and len(value)==maximum: args=value
            else: raise InterfaceError('ambiguous_function_arguments')
    elif isinstance(raw,list):
        if maximum==1: args=[raw]
        elif minimum==maximum and len(raw)==maximum: args=raw
        else: raise InterfaceError('ambiguous_function_arguments')
    else:
        args=[_json(raw)]
    if not minimum<=len(args)<=maximum: raise InterfaceError('argument_arity_mismatch')
    return args


def classify(row):
    info = row.get('verification_info')
    if not isinstance(info,dict) or str(info.get('language','')).lower() not in {'python','python3'}:
        raise InterfaceError('unsupported_language_or_verification')
    cases = info.get('test_cases')
    if not isinstance(cases,list) or not cases or not all(isinstance(c,dict) for c in cases):
        raise InterfaceError('missing_tests')
    types = {c.get('type') for c in cases}
    names = {c.get('fn_name') for c in cases}
    if len(types)!=1 or len(names)!=1: raise InterfaceError('inconsistent_all_test_interfaces')
    code,tree = python_code(row.get('gold_standard_solution'))
    signature = None
    if types=={'stdin_stdout'} and names=={None}:
        if not all(isinstance(c.get('input'),str) and isinstance(c.get('output'),str) for c in cases):
            raise InterfaceError('invalid_stdio_test_schema')
        interface,entry = 'stdin_stdout',None
    elif types=={'function_call'}:
        name = next(iter(names))
        if not isinstance(name,str) or not name.isidentifier() or keyword.iskeyword(name):
            raise InterfaceError('invalid_function_name')
        top = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        methods = [n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='Solution'
                   for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(top)+len(methods)!=1: raise InterfaceError('missing_or_ambiguous_entry')
        interface = 'function' if top else 'solution_method'
        fn = (top or methods)[0]
        if fn.decorator_list or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
            raise InterfaceError('complex_function_signature')
        args = list(fn.args.posonlyargs)+list(fn.args.args)
        if interface=='solution_method':
            if not args or args[0].arg!='self': raise InterfaceError('unsupported_method_binding')
            args=args[1:]
        # Reject graph/node/tree/custom-class serialization in the first adapter.
        allowed_types={'List','Tuple','Dict','Set','Optional','Union','Any','int','str','float','bool','list','tuple','dict','set','None'}
        for a in args:
            if a.annotation and any(isinstance(n,ast.Name) and n.id not in allowed_types for n in ast.walk(a.annotation)):
                raise InterfaceError('custom_object_argument_type')
        if not any(isinstance(n,ast.Return) and n.value is not None for n in ast.walk(fn)):
            raise InterfaceError('no_value_return')
        minimum,maximum = len(args)-len(fn.args.defaults),len(args)
        if minimum<0: raise InterfaceError('invalid_defaults')
        for c in cases:
            if 'input' not in c or 'output' not in c: raise InterfaceError('missing_function_io')
            function_arguments(c['input'],minimum,maximum,[a.arg for a in args])
            _json(c['output'])
        parameters=[a.arg for a in args]
        for i,default in enumerate(fn.args.defaults,start=minimum):
            # Defaults are part of the callable contract; only literal defaults
            # enter the generated interface instruction.
            _json(ast.unparse(default))
            parameters[i]+='='+ast.unparse(default)
        signature = 'def '+name+'('+', '.join(parameters)+'):'
        entry = name if interface=='function' else 'Solution.'+name
    else:
        raise InterfaceError('unsupported_test_type')
    return {'interface':interface,'entry_point':entry,'function_signature':signature,
            'test_case_count':len(cases),'reference_code_sha256':hashlib.sha256(code.encode()).hexdigest(),
            'reference_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
            'static_status':'all_tests_schema_and_reference_ast_checked',
            'execution_status':'not_executed_requires_isolated_sandbox'}


def invocation(interface, entry_point, arguments):
    """Adapter contract for an isolated worker, not execution on this host."""
    if interface=='function': return {'kind':'call','function':entry_point,'args':arguments}
    if interface=='solution_method':
        cls,method=entry_point.split('.')
        return {'kind':'instantiate_then_call','class':cls,'method':method,'args':arguments}
    if interface=='stdin_stdout': return {'kind':'run_script','stdin':arguments}
    raise InterfaceError('Unknown interface')


def benchmark_references(root):
    from huggingface_hub import HfApi
    from datasets import load_dataset,disable_progress_bars
    disable_progress_bars()
    meta_path = root/'data/code_benchmarks/source_manifest.json'
    prior = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    references,ast_hashes,meta = [],set(),{}
    for repo in ['evalplus/humanevalplus','evalplus/mbppplus']:
        revision = prior.get(repo,{}).get('revision') or HfApi().dataset_info(repo).sha
        dataset = load_dataset(repo,revision=revision,cache_dir=str(root/'data/code_benchmarks/cache'))
        count=0
        for split,ds in dataset.items():
            for r in ds:
                sid = repo+'/'+str(r.get('task_id',count))
                prompt = r.get('prompt',r.get('text'))
                if not isinstance(prompt,str): raise InterfaceError('Unknown benchmark prompt schema')
                references.append({'sample_id':sid,'problem':prompt})
                solution = r.get('canonical_solution',r.get('code',''))
                for candidate in [solution,prompt+solution]:
                    try:
                        _,tree=python_code(candidate)
                        ast_hashes.add(hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())
                    except InterfaceError: pass
                count+=1
        meta[repo]={'revision':revision,'rows':count,'features':{k:str(v.features) for k,v in dataset.items()}}
    json_write(meta_path,meta)
    return references,ast_hashes,meta


def choose_split(candidates,target=2500,dev_size=200,seed=42):
    ordered=sorted(candidates,key=lambda r:hashlib.sha256((str(seed)+r['sample_id']).encode()).hexdigest())
    # All families deduplicated BEFORE the train/dev split, in stable order.
    seen=SimilarityIndex(.85);families=set();clean=[];duplicates=0
    for r in ordered:
        if r['source_family'] in families or seen.match(r['problem']): duplicates+=1;continue
        seen.add(r);families.add(r['source_family']);clean.append(r)
    callables=[r for r in clean if r['interface']!='stdin_stdout']
    stdio=[r for r in clean if r['interface']=='stdin_stdout']
    n_dev_call=min(dev_size//2,max(1,len(callables)//5)) if callables else 0
    dev=callables[:n_dev_call]+stdio[:dev_size-n_dev_call]
    # If stdio is scarce, fill remaining dev from unused callable families.
    dev += callables[n_dev_call:n_dev_call+max(0,dev_size-len(dev))]
    dev_ids={r['sample_id'] for r in dev}
    callables=[r for r in callables if r['sample_id'] not in dev_ids]
    stdio=[r for r in stdio if r['sample_id'] not in dev_ids]
    n_call=min(target//2,len(callables))
    train=callables[:n_call]+stdio[:target-n_call]
    if len(train)<target: train+=callables[n_call:n_call+target-len(train)]
    return [{**r,'split':'train'} for r in train],[{**r,'split':'dev'} for r in dev],duplicates


def build_math_pool(root):
    manifest,protected=sealed_math(root)
    index=SimilarityIndex(.85)
    for key in ['math_dev','math500_test','gsm8k_test']:
        for r in protected[key]:index.add(r)
    gsm_meta=manifest['files']['gsm8k_train'];gsm_path=root/gsm_meta['path']
    if sha(gsm_path)!=gsm_meta['sha256']:raise ValueError('GSM train identity changed')
    randomizer=random.Random(42)
    math_rows=list(protected['math_train']);gsm=list(rows(gsm_path))
    randomizer.shuffle(math_rows);randomizer.shuffle(gsm)
    picked=[];seen=SimilarityIndex(.85)
    for pool,wanted in [(math_rows,1500),(gsm,1000)]:
        n=0
        for r in pool:
            if n>=wanted:break
            if index.match(r['problem']) or seen.match(r['problem']):continue
            seen.add(r);n+=1
            picked.append({k:r[k] for k in ['sample_id','source','source_revision','source_split','problem','problem_hash','family_id','domain']})
    return picked,{'shared_dev_path':manifest['files']['math_dev']['path'],
                   'shared_dev_sha256':manifest['files']['math_dev']['sha256'],
                   'math_candidate_sha256':manifest['files']['math_train']['sha256'],
                   'math_count':sum(r['source'].endswith('MATH-lighteval') for r in picked),
                   'gsm_count':sum(r['source']=='openai/gsm8k' for r in picked)}


def run(root=Path('.'),target=2500,dev_size=200):
    source=source_files('code',root)
    if not all('path' in f for f in source['files']): raise ValueError('Complete source download first')
    out=root/'data/project2_code';out.mkdir(exist_ok=True)
    identity={'source_revision':source['revision'],'script_sha256':sha(__file__),
              'target':target,'dev_size':dev_size,'seed':42}
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if old['identity']==identity:
            print('Code sealed output already complete',flush=True);return old
        raise ValueError('Use a fresh output version for changed code selection')
    refs,ast_hashes,benchmeta=benchmark_references(root)
    heldout=SimilarityIndex(.85)
    for r in refs:heldout.add(r)
    candidates=[];counts=Counter();raw_interfaces=Counter()
    audit_path=out/'interface_audit.jsonl'
    with audit_path.open('w',encoding='utf-8',newline='\n') as audit:
        for pos,row in enumerate(parquet_rows(source,root)):
            counts['source_rows']+=1
            sid='code/'+str(row.get('problem_id',pos))
            raw= row.get('verification_info') or {}
            raw_types={str(c.get('type')) for c in (raw.get('test_cases') or []) if isinstance(c,dict)}
            raw_interfaces['+'.join(sorted(raw_types)) or 'missing']+=1
            try:
                interface=classify(row)
                problem=row.get('problem_statement')
                if not isinstance(problem,str) or not problem.strip():raise InterfaceError('missing_problem')
                overlap=heldout.match(problem)
                if overlap:raise InterfaceError('benchmark_prompt_overlap')
                if interface['reference_ast_sha256'] in ast_hashes:raise InterfaceError('benchmark_exact_reference_ast')
                prompt=problem
                if interface['function_signature']:
                    if interface['interface']=='function':
                        prompt+='\nRequired Python interface (return the result; do not read stdin):\n'+interface['function_signature']
                    else:
                        signature=interface['function_signature'].replace('(', '(self, ',1).replace('(self, )','(self)')
                        prompt+='\nRequired Python interface (return the result; do not read stdin):\nclass Solution:\n    '+signature
                else:prompt+='\nWrite a complete Python program that reads standard input and writes standard output.'
                result={**interface,'sample_id':sid,'problem':problem,'prompt':prompt,'domain':'code',
                        'source':source['repo'],'source_revision':source['revision'],'source_split':'train',
                        'source_category':row.get('source'),'source_id':str(row.get('in_source_id')),
                        'source_family':str(row.get('source'))+'/'+str(row.get('in_source_id') or fingerprint(problem)),
                        'problem_hash':fingerprint(problem),'family_id':fingerprint(problem)}
                candidates.append(result);reason='structural_candidate'
                counts['eligible_'+interface['interface']]+=1
            except (InterfaceError,TypeError,RecursionError) as exc:
                reason=str(exc);counts['reject_'+reason]+=1
            audit.write(json.dumps({'sample_id':sid,'reason':reason},ensure_ascii=False)+'\n')
            if (pos+1)%2000==0:print(f'Code classified={pos+1}, structural_candidates={len(candidates)}',flush=True)
    print('Code classification complete; deduplicating families with exact prefix-filtered Jaccard',flush=True)
    write_jsonl(out/'structural_candidates.jsonl',candidates)
    train,dev,duplicates=choose_split(candidates,target,dev_size)
    write_jsonl(out/'train_prompts.jsonl',train)
    write_jsonl(out/'dev_metadata.jsonl',dev)
    # Re-read only to retain full tests/reference code for the small sealed dev.
    dev_by_id={r['sample_id']:r for r in dev}
    with (out/'dev_verification.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for pos,row in enumerate(parquet_rows(source,root)):
            sid='code/'+str(row.get('problem_id',pos))
            if sid in dev_by_id:
                f.write(json.dumps({**dev_by_id[sid],'gold_standard_solution':row.get('gold_standard_solution'),
                                    'verification_info':row.get('verification_info')},ensure_ascii=False)+'\n')
    math_pool,math_meta=build_math_pool(root)
    write_jsonl(out/'math_train_prompts.jsonl',math_pool)
    for row in train:
        assert 'verification_info' not in row and 'gold_standard_solution' not in row
    manifest={'identity':identity,'source_rows':counts['source_rows'],'raw_test_type_rows':dict(raw_interfaces),
              'counts':dict(counts),'deduplicated_families':duplicates,'candidate_count':len(candidates),
              'train_rows':len(train),'dev_rows':len(dev),'train_target_met':len(train)>=target,
              'dev_target_met':len(dev)==dev_size,'requested_callable_fraction':.5,
              'actual_train_interfaces':dict(Counter(r['interface'] for r in train)),
              'actual_dev_interfaces':dict(Counter(r['interface'] for r in dev)),
              'benchmarks':benchmeta,'math_pool':math_meta,
              'files':{p.name:{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in out.glob('*.jsonl')},
              'quality_gate':'Static classification and complete testcase schema validation only; reference execution in isolated sandbox and human coverage audit PENDING.',
              'dedup_scope':'Raw/normalized prompt matching and 0.85 lexical Jaccard; exact reference AST for heldout; not proof of semantic decontamination.',
              'api_calls':0,'executed_dataset_code':False}
    json_write(out/'manifest.json',manifest)
    json_write(root/'reports/code_interface_screening_report.json',manifest)
    print('Code finished:',json.dumps(manifest['actual_train_interfaces']),f'dev={len(dev)}',flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',type=int,default=2500);p.add_argument('--dev-size',type=int,default=200)
    args=p.parse_args();run(target=args.target,dev_size=args.dev_size)
