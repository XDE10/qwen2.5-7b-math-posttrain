import pytest
from src.data.screen_code_interfaces import classify, InterfaceError, function_arguments, invocation, choose_split


def record(code,fn='add',kind='function_call',args='1\n2',expected='3'):
    return {'gold_standard_solution':code,'verification_info':{'language':'python','test_cases':[
        {'type':kind,'fn_name':fn,'input':args,'output':expected}]}}


def test_independent_function_and_solution_method():
    r=classify(record('def add(a,b):\n    return a+b'))
    assert r['interface']=='function' and r['function_signature']=='def add(a, b):'
    r=classify(record('class Solution:\n    def add(self,a,b):\n        return a+b'))
    assert r['interface']=='solution_method' and r['entry_point']=='Solution.add'
    assert invocation(r['interface'],r['entry_point'],[1,2])['kind']=='instantiate_then_call'


def test_solve_def_is_still_stdio_and_no_dataset_code_is_executed():
    r=classify(record('def solve():\n    print(input())\nsolve()',fn=None,kind='stdin_stdout'))
    assert r['interface']=='stdin_stdout' and r['execution_status'].startswith('not_executed')
    classify(record('raise RuntimeError("MUST NOT RUN")\ndef add(a,b):\n    return a+b'))


def test_all_test_cases_checked_not_only_first():
    r=record('def add(a,b):\n    return a+b')
    r['verification_info']['test_cases'].append({'type':'stdin_stdout','fn_name':None,'input':'','output':''})
    with pytest.raises(InterfaceError,match='inconsistent_all'):classify(r)
    r=record('def add(a,b):\n    return a+b')
    r['verification_info']['test_cases'].append({'type':'function_call','fn_name':'add','input':'1\n2\n3','output':'6'})
    with pytest.raises(InterfaceError,match='arity'):classify(r)


@pytest.mark.parametrize('code,reason',[
    ('def wrong(a,b):\n return a+b','missing_or_ambiguous_entry'),
    ('class Solution:\n def add(self,a: TreeNode,b):\n  return a','custom_object'),
    ('def add(*args):\n return args','complex_function'),
    ('def add(a,b):\n print(a+b)','no_value_return'),
])
def test_unsupported_interfaces_are_quarantined(code,reason):
    with pytest.raises(InterfaceError,match=reason):classify(record(code))


def test_native_json_arguments_and_ambiguity():
    assert function_arguments('[1,2]\n3',2,2)==[[1,2],3]
    assert function_arguments('[1,2]',1,1)==[[1,2]]
    assert function_arguments('[1,2]',2,2)==[1,2]
    with pytest.raises(InterfaceError):function_arguments('[1,2]',1,2)
    with pytest.raises(InterfaceError):function_arguments('__import__("os")',1,1)


def test_explicit_named_literals_are_bound_by_signature_order():
    assert function_arguments('b = [2, 3], a = 1',2,2,['a','b'])==[1,[2,3]]
    assert function_arguments('a = [\n1,\n2\n]\nb = True',2,2,['a','b'])==[[1,2],True]
    assert classify(record('def add(a,b):\n return a+b',args='b=2, a=1'))['interface']=='function'
    for raw in ['a=__import__("os"), b=2','a=1, a=2','wrong=1, b=2','b=2','a={1,2}, b=2']:
        with pytest.raises(InterfaceError):function_arguments(raw,2,2,['a','b'])
    r=classify(record('def add(a,b=2):\n return a+b',args='a=1'))
    assert r['function_signature']=='def add(a, b=2):'


def test_family_split_no_repeated_upsampling_and_callable_shortage():
    data=[{'sample_id':str(i),'source_family':str(i),'problem':f'Unique prompt entity_{i} operation_{i}',
           'interface':'function' if i<4 else 'stdin_stdout'} for i in range(40)]
    data.append({**data[0],'sample_id':'duplicate'})
    train,dev,duplicates=choose_split(data,target=20,dev_size=4)
    assert duplicates==1 and len(train)==20 and len(dev)==4
    assert sum(x['interface']=='function' for x in train)<10
    assert not {r['source_family'] for r in train}&{r['source_family'] for r in dev}
