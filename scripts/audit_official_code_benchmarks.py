"""Read-only supplementary decontamination against downloaded official releases."""
import ast
import hashlib
from pathlib import Path
from src.data.screening_io import rows, sha, json_write
from src.data.prepare_math_data import SimilarityIndex
from src.data.screen_code_interfaces import python_code, InterfaceError


def run(root=Path('.')):
    index=SimilarityIndex(.85);hashes=set();sources={};ids={}
    for name in ['HumanEvalPlus-v0.1.10','MbppPlus-v0.2.0']:
        path=root/'data/code_benchmarks/official'/f'{name}.jsonl'
        records=list(rows(path))
        sources[name]={'path':path.relative_to(root).as_posix(),'sha256':sha(path),'rows':len(records)}
        ids[name]=[r['task_id'] for r in records]
        for r in records:
            index.add({'sample_id':name+'/'+str(r['task_id']),'problem':r['prompt']})
            for code in [r['canonical_solution'],r['prompt']+r['canonical_solution']]:
                try:
                    _,tree=python_code(code)
                    hashes.add(hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())
                except InterfaceError:pass
    findings=[];counts={}
    for key in ['train_prompts','dev_metadata']:
        selected=list(rows(root/'data/project2_code'/f'{key}.jsonl'));counts[key]=len(selected)
        for r in selected:
            match=index.match(r['problem'])
            if match or r['reference_ast_sha256'] in hashes:
                findings.append({'sample_id':r['sample_id'],'split':key,'match':match,'ast_match':r['reference_ast_sha256'] in hashes})
    report={'sources':sources,'task_ids':ids,'checked':counts,'findings':findings,
            'passed':not findings,'scope':'Additional official-release prompt Jaccard/exact reference AST; not semantic or pretraining decontamination proof.'}
    json_write(root/'reports/official_code_overlap_audit.json',report)
    if findings:raise ValueError('Official benchmark overlap requires re-screening')
    print('Official benchmark overlap audit:',counts,'findings=0')
    return report


if __name__=='__main__':run()
