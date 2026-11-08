"""Additional bounded static diagnostics; never execute reference programs.

The initial interface screen checks syntax/arity. This pass catches the narrow
case of a directly iterated parameter bound to a non-iterable literal. It does
not claim general type inference or replace the isolated execution gate.
"""
import ast
import json
from collections import Counter
from pathlib import Path
from src.data.screen_code_interfaces import python_code, function_arguments, choose_split
from src.data.screening_io import rows, sha, json_write
from src.data.prepare_math_data import write_jsonl


def literal_iteration_conflicts(row):
    _, tree=python_code(row['gold_standard_solution'])
    name=row['verification_info']['test_cases'][0]['fn_name']
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    functions += [n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Solution'
                  for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(functions)!=1:return []
    fn=functions[0]; parameters=list(fn.args.posonlyargs)+list(fn.args.args)
    if parameters and parameters[0].arg=='self':parameters=parameters[1:]
    names=[a.arg for a in parameters]
    # Only top-level loops before any reassignment to the iterated parameter.
    untouched=set(names);iterated=set()
    for statement in fn.body:
        if isinstance(statement,ast.For) and isinstance(statement.iter,ast.Name):
            if statement.iter.id in untouched:iterated.add(statement.iter.id)
        for node in ast.walk(statement):
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):untouched.discard(node.id)
    findings=[]
    minimum=len(names)-len(fn.args.defaults)
    for i,case in enumerate(row['verification_info']['test_cases']):
        args=function_arguments(case['input'],minimum,len(names),names)
        for name,value in zip(names,args):
            if name in iterated and (value is None or isinstance(value,(int,float,bool))):
                findings.append({'case_index':i,'parameter':name,'literal_type':type(value).__name__,
                                 'reason':'direct_iteration_over_noniterable_literal'})
    return findings


def run(root=Path('.')):
    base=root/'data/project2_code';path=base/'manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    script_hash=sha(__file__)
    if 'literal_diagnostics' in manifest:
        assert manifest['literal_diagnostics']['script_sha256']==script_hash
        print('Literal diagnostic output already sealed');return manifest
    candidates=list(rows(base/'structural_candidates.jsonl'))
    raw={('code/'+r['problem_id']):r for r in rows(root/'data/screening_sources/code_function_rows.jsonl')}
    findings=[]
    for r in candidates:
        if r['interface']=='stdin_stdout':continue
        hits=literal_iteration_conflicts(raw[r['sample_id']])
        if hits:findings.append({'sample_id':r['sample_id'],'findings':hits})
    excluded={r['sample_id'] for r in findings}
    train,dev,duplicates=choose_split([r for r in candidates if r['sample_id'] not in excluded],
                                    manifest['identity']['target'],manifest['identity']['dev_size'])
    references={r['sample_id']:r for r in rows(base/'dev_verification.jsonl')}
    references.update(raw)
    verification=[]
    for r in dev:
        reference=references[r['sample_id']]
        verification.append({**r,**{k:reference[k] for k in ['gold_standard_solution','verification_info']}})
    # Materialize all rows successfully before replacing any selected output.
    json_write(root/'reports/backups/code_manifest_before_literal_diagnostics.json',manifest)
    write_jsonl(base/'train_prompts.jsonl',train)
    write_jsonl(base/'dev_metadata.jsonl',dev)
    write_jsonl(base/'dev_verification.jsonl',verification)
    write_jsonl(base/'literal_quarantine.jsonl',findings)
    manifest.update(actual_train_interfaces=dict(Counter(r['interface'] for r in train)),
                    actual_dev_interfaces=dict(Counter(r['interface'] for r in dev)),
                    final_candidate_count=len(candidates)-len(excluded),
                    final_deduplicated_families=duplicates,
                    literal_diagnostics={'script_sha256':script_hash,'excluded_rows':len(excluded),
                        'function_cache_sha256':sha(root/'data/screening_sources/code_function_rows.jsonl'),
                        'scope':'Direct parameter iteration and noniterable literals only; not general type or runtime validation.'})
    manifest['files']={p.name:{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in base.glob('*.jsonl')}
    json_write(path,manifest)
    json_write(root/'reports/code_interface_screening_report.json',manifest)
    print('Literal conflicts quarantined:',len(excluded),'train:',manifest['actual_train_interfaces'],
          'dev:',manifest['actual_dev_interfaces'])
    return manifest


if __name__=='__main__':run()
