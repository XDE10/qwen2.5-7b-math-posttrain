import pytest
from src.evaluation.math_verifier import verify, expression, InvalidAnswer


@pytest.mark.parametrize('output,gold,correct',[
 ('#### 1/2','0.5',True), (r'\boxed{\frac{1}{2}}','0.5',True),
 (r'\boxed{\sqrt{2}}','sqrt(2)',True),(r'\boxed{\frac{\sqrt{8}}{2}}','sqrt(2)',True),
 ('#### -42','-42',True),('#### −3.5','-7/2',True),
 ('#### $1,234.50','1234.5',True),('#### 25 cm','25',True),
 (r'\boxed{12\text{ ounces}}','12',False), # unknown unit spelling fails closed
 (r'\boxed{12\text{ounces}}','12',True),
 (r'\boxed{\boxed{2}}','2',True),('#### 2\n#### 3','2',False),
 (r'\boxed{2} and \boxed{3}','2',False),
 (r'\boxed{\frac{1}{2}} and \boxed{0.5}','1/2',True),
 (r'\boxed{\frac{1}{2}','1/2',False),('####','0',False),
 ('','0',False),('#### 1,2','12',False),('#### 1/0','0',False),
 ('#### 2 or 3','2',False),('The answer is 2 or 3','2',False),
 ('#### Final Answer: $26','26',True),('#### 5\nExplanation follows.','5',True),
 ('Working: 2+3=5','5',False),('#### __import__("os").system("whoami")','0',False),
 ('#### 9**999999','9',False),('#### nan','0',False),('#### 1e309','0',False),
 (r'\boxed{x^2+2*x+1}','(x+1)^2',True),
 (r'\boxed{-\frac{3}{4}}','-0.75',True),
 ('#### 50%','0.5',False),('#### 2 cm','0.02',False),
 ('#### 2','3',False),('#### 2 and 2','2',False),
 (r'\boxed{\frac{1}{\frac{2}{3}}}','3/2',True),
 ('#### True','1',False),('#### [2]','2',False),
])
def test_adversarial_answers(output,gold,correct):
    assert verify(output,gold)['correct'] is correct


def test_truncation_is_not_success():
    assert not verify('#### 2','2',truncated=True)['correct']
    assert verify(r'\boxed{2','2')['status']=='unclosed_brace'


def test_bound_and_unsupported_syntax():
    with pytest.raises(InvalidAnswer): expression('1'*513)
    assert not verify('#### (lambda:2)()','2')['correct']
    assert not verify('#### (((9**12)**12)**12)**12','2')['correct']
