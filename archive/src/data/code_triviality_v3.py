"""Conservative syntactic triviality, not a claim of semantic difficulty."""
import ast


def analyze(code,entry):
    tree=ast.parse(code)
    matches=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==entry]
    if len(matches)!=1:return {'isolate':True,'reason':'unresolved_target_function','metrics':{}}
    fn=matches[0];body=fn.body[:]
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):body=body[1:]
    args={a.arg for a in fn.args.posonlyargs+fn.args.args+fn.args.kwonlyargs}
    if fn.args.vararg:args.add(fn.args.vararg.arg)
    if fn.args.kwarg:args.add(fn.args.kwarg.arg)
    nodes=[n for stmt in body for n in ast.walk(stmt)]
    metrics={'effective_top_level_statements':len(body),'control_flow':sum(isinstance(n,(ast.If,ast.For,ast.While,ast.Try,ast.Match,ast.IfExp)) for n in nodes),
        'comprehensions':sum(isinstance(n,(ast.ListComp,ast.DictComp,ast.SetComp,ast.GeneratorExp)) for n in nodes),
        'arithmetic':sum(isinstance(n,(ast.BinOp,ast.UnaryOp,ast.Compare,BoolOp)) for n in nodes),
        'calls':sum(isinstance(n,ast.Call) for n in nodes),'ast_nodes':len(nodes)}
    reason=None
    if len(body)==1 and isinstance(body[0],ast.Return):
        v=body[0].value
        if isinstance(v,ast.Name) and v.id in args:reason='identity_return_parameter'
        elif v is None:reason='constant_return'
        else:
            try:ast.literal_eval(v);reason='constant_return'
            except (ValueError,TypeError):pass
        if isinstance(v,ast.Dict) and v.values and all(isinstance(k,ast.Constant) for k in v.keys) and all(isinstance(x,ast.Name) and x.id in args for x in v.values):
            reason='pure_parameter_dictionary'
        if isinstance(v,(ast.Tuple,ast.List,ast.Set)) and v.elts and all(isinstance(x,ast.Name) and x.id in args for x in v.elts):reason='pure_parameter_sequence'
    return {'isolate':reason is not None,'reason':reason,'metrics':metrics,
        'complexity_review':'syntactic_metrics_only_not_a_difficulty_label'}


BoolOp=ast.BoolOp
