from scripts.prepare_strict_inequalities_v8 import infer_profile

def test_integer_solutions_do_not_make_a_real_parameter_integer():
    assert infer_profile({'sample_id':'new-domain-case','reference_answer':r'-1 <= a < -3/4','problem':'Given ax+3 >= 0 has positive integer solutions 1,2,3, find the range of a.'}) is None

def test_explicit_integer_unknown_can_be_bound():
    p=infer_profile({'sample_id':'new-bound-case','reference_answer':r'n <= 5','problem':r'Determine all integers $n>1$ for which the condition holds.'})
    assert p['variable']=='n' and p['domain']=='integer'
