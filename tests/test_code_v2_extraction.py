import pytest
from src.data.code_function_screen import extract, inspect_reference, simple_tests, InterfaceError


def test_target_is_specified_function_not_helper_and_contract_matches():
    code='def helper(x):\n return x\ndef solve(a,b=1):\n return helper(a)+b'
    q='Write the function `solve(a,b=1)` and return the sum.'
    assert inspect_reference(code,q)['function_signature']=='def solve(a, b=1):'
    with pytest.raises(InterfaceError,match='signature_conflict'):
        inspect_reference('def getDigest(input,algorithm):\n return input','Complete `getDigest(input)` to compute the digest.')
    with pytest.raises(InterfaceError,match='ambiguous'):
        inspect_reference(code,'Write a program that does the work.')


def test_tests_bind_all_cases_and_do_not_execute_payloads():
    target,tests=simple_tests(['assert add(1,b=2)==3']*3)
    assert target=='add'
    inspect_reference('def add(a,b):\n return a+b','Implement `add(a,b)` for integer addition.',target,tests)
    with pytest.raises(InterfaceError,match='binding'):
        inspect_reference('def add(a):\n return a','Implement `add(a)` for addition.',target,tests)
    with pytest.raises(InterfaceError):simple_tests(['assert add(__import__("os"))==0']*3)
    with pytest.raises(InterfaceError):simple_tests(['import os\nassert add(1)==1']*3)
    with pytest.raises(InterfaceError,match='serialized'):simple_tests(['assert add({1,2})==3']*3)


@pytest.mark.parametrize('code',[
    'def target(x):\n return input()',
    'import os\ndef target(x):\n return os.listdir(x)',
    'class Solution:\n def target(self,x):\n  return x',
    'def target(x):\n return unknown_external(x)',
    'def target(x):\n return x\nprint(target(1))',
])
def test_unsupported_reference_is_quarantined(code):
    with pytest.raises(InterfaceError):inspect_reference(code,'Implement the function `target(x)` returning the result.')


def test_ace_does_not_treat_arbitrary_generation_as_gold():
    r={'question':'Implement the function `target(x)` returning the same input.',
       'test_cases':['assert target(1)==1']*3,
       'inferences':[{'completion':'def target(x):\n return x','pass_rate':.9}]}
    with pytest.raises(InterfaceError,match='no_source_full_pass'):extract(r,'ace')
