import pytest
from src.synthesis.typed_answers_v8 import equivalent,check_roles
R={'kind':'one_variable_solution_set','domain':'real','variable':'x'}
def test_equivalent_and_strict_endpoints():
    assert equivalent('0<x<1',r'(0,1)',R)['correct']
    assert not equivalent('0<=x<1',r'(0,1)',R)['correct']
    assert equivalent(r'x\in(-\infty,2)',r'x<2',R)['correct']
    assert equivalent(r'-\infty<x<2',r'x<2',R)['correct']
    assert equivalent(r'((-3-\sqrt{41})/2,(-3+\sqrt{41})/2)',r'(-3/2-\sqrt{41}/2,-3/2+\sqrt{41}/2)',R)['correct']
    assert not equivalent(r'[-\infty,2)',r'x<2',R)['correct']
    assert equivalent(r'(-\infty,0)\cup(1,\infty)',r'x<0\lor x>1',R)['correct']
def test_no_domain_loss_or_wrong_variables():
    assert not equivalent('x/x>0',r'\mathbb{R}',R)['correct']
    assert not equivalent('y>3','x>3',R)['correct']
    assert not equivalent(r'0\leq\{x\}<1',r'0<=x<1',R)['correct']
    assert not equivalent('x^2<4','(-2,2)',R)['correct']
    assert not equivalent('f(x)=c','0<x<1',R)['correct']
    assert not equivalent('a+b<2','x<2',R)['correct']
    assert not equivalent('1<2',r'\mathbb{R}',R)['correct']
    assert not equivalent('2^999999999< x','0<x',R)['correct']
def test_integer_domain_and_enumerated_solution_roles():
    p={**R,'domain':'integer','enumerated_roots':True}
    assert equivalent(r'x_1=2;x_2=3',r'\{2,3\}',p)['correct']
    assert equivalent('(1,4)',r'\{2,3\}',p)['correct']
    assert not equivalent('(1,4)',r'\{2,3\}',R)['correct']
    assert not equivalent(r'y_1=2;y_2=3',r'\{2,3\}',p)['correct']
    assert not equivalent(r'x_1=2;x_1=3',r'\{2,3\}',p)['correct']
    assert not equivalent(r'\{2,2.5,3\}',r'\{2,3\}',p)['correct']
    assert not equivalent(r'\mathbb{R}',r'\mathbb{Z}',p)['correct']
def test_answer_roles_not_whole_string_box_guessing():
    p={**R,'variable':'k'}
    s=r'<think>Format example \boxed{}, trial \boxed{11}, explicitly corrected to k>11.</think>Final: \boxed{k>11}'
    assert check_roles(s,'k>11',p)['correct']
    assert not check_roles(r'Final \boxed{}','k>11',p)['correct']
    assert not check_roles(r'\boxed{11}; \boxed{k>11}','k>11',p)['correct']
    assert check_roles(r'\boxed{(11,\infty)}; \boxed{k>11}','k>11',p)['correct']
    assert not check_roles(r'\boxed{k>=11}; \boxed{k>11}','k>11',p)['correct']
    assert check_roles(r'Format example `\boxed{}`. Actual answer \boxed{k>11}','k>11',p)['correct']
    assert not check_roles(r'Actual answer `\boxed{}`','k>11',p)['correct']
def test_terminal_collection_versus_intermediate_values():
    p={**R,'domain':'integer'}
    assert check_roles(r'Final \boxed{3}, \boxed{4}, and \boxed{5}',r'\{3,4,5\}',p)['correct']
    assert not check_roles(r'First \boxed{3}, subtract this to get \boxed{4}',r'\{3,4\}',p)['correct']
    assert not check_roles(r'\boxed{3},\boxed{4}',r'\{3,4,5\}',p)['correct']

@pytest.mark.parametrize('pred,ref,expected',[
 (r'x>=-1/3',r'[-\frac{1}{3},\infty)',True),
 (r'x>-1/3',r'[-\frac{1}{3},\infty)',False),
 (r'x!=0',r'(-\infty,0)\cup(0,\infty)',True),
 (r'x<=0',r'(-\infty,0)',False),
 (r'x>=0',r'(0,\infty)',False),
 (r'[0,1]\cap[1,2]',r'\{1\}',True),
 (r'\{2,3,4\}',r'\{2,3\}',False),
 (r'\{3,2,2\}',r'\{2,3\}',True),
 (r'[-1,1]',r'(-1,1)',False),
 (r'1<x<3',r'(1,3)',True),
 (r'1<x<3',r'(3,1)',False),
 (r'x>2, x<3',r'(2,3)',False),
 (r'(a,b)=(-1,1)',r'(-1,1)',False),
 (r'x>0 trailing words',r'(0,\infty)',False),
 (r'\sqrt{-1}<x',r'(0,\infty)',False),
])
def test_boundary_and_type_mutations(pred,ref,expected):
    assert equivalent(pred,ref,R)['correct'] is expected
