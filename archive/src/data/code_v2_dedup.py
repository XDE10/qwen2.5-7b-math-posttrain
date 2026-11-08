"""Read-only final-test/legacy-dev protection and cross-source code identity."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from src.data.screening_io import rows,sha,ScreeningSimilarityIndex
from src.data.screen_code_interfaces import python_code,InterfaceError


def ast_hash(code):
    _,tree=python_code(code)
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def function_body_hash(code,target):
    """Conservative alpha-normalized target body; collisions are exclusions."""
    _,tree=python_code(code)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==target]
    if len(functions)!=1:return None
    fn=copy.deepcopy(functions[0]);mapping={};counter=0
    class Rename(ast.NodeTransformer):
        def rename(self,name):
            if name not in mapping:mapping[name]='v'+str(len(mapping))
            return mapping[name]
        def visit_arg(self,node):node.arg=self.rename(node.arg);node.annotation=None;return node
        def visit_Name(self,node):
            if isinstance(node.ctx,ast.Store):self.rename(node.id)
            if node.id in mapping:node.id=mapping[node.id]
            return node
    fn.name='_target';fn.returns=None;fn.decorator_list=[]
    if fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str):fn.body=fn.body[1:]
    Rename().visit(fn)
    return hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()


class Protected:
    def __init__(self):
        self.index=ScreeningSimilarityIndex(.85);self.ast=set();self.body=set();self.names=set();self.files={}
        for filename in ['data/code_benchmarks/official/HumanEvalPlus-v0.1.10.jsonl',
                         'data/code_benchmarks/official/MbppPlus-v0.2.0.jsonl',
                         'data/project2_code/dev_verification.jsonl']:
            self.files[filename]=sha(filename)
            for row in rows(filename):
                sid=row.get('task_id',row.get('sample_id'))
                problem=row.get('problem',row.get('prompt'))
                self.index.add({'sample_id':str(sid),'problem':problem})
                target=row.get('entry_point')
                if target:self.names.add(target)
                solution=row.get('gold_standard_solution',row.get('canonical_solution',''))
                for code in [solution,problem+solution]:
                    try:
                        self.ast.add(ast_hash(code))
                        if target:
                            body=function_body_hash(code,target)
                            if body:self.body.add(body)
                    except InterfaceError:pass

    def match(self,row,protect_old_dev=True):
        hit=self.index.match(row['problem'])
        if hit:return {'reason':'protected_prompt_overlap','evidence':hit}
        if row.get('reference_ast_sha256') in self.ast:return {'reason':'protected_exact_ast'}
        if row.get('reference_body_sha256') in self.body:return {'reason':'protected_alpha_body'}
        # Name equality alone is not leakage proof; route collisions to review.
        if row.get('entry_point') in self.names:return {'reason':'protected_entry_name_review'}
        return None
