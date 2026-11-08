import json,sys
from scripts.math_verify_v3_strict_worker import check as frozen_check
from src.synthesis.typed_answers_v8 import check_roles
def check(r):
    p=r.get('profile',{})
    if p.get('kind')=='one_variable_solution_set':return check_roles(r['output'],r['reference'],p)
    if p.get('kind')=='scalar':
        text=r['output']
        if text.count('<think>')!=text.count('</think>') or text.count('</think>')>1:return {'correct':False,'status':'ambiguous_source_wrapper'}
        zone=text.split('</think>',1)[1] if '</think>' in text else text
        return frozen_check({'output':zone,'reference':r['reference']})
    return frozen_check(r)
if __name__=='__main__':
    out={}
    for r in json.load(sys.stdin):
        try:out[r['id']]=check(r)
        except Exception as e:out[r['id']]={'correct':False,'status':'verifier_error','error_type':type(e).__name__}
    print(json.dumps(out))
