from src.data.code_triviality_v3 import analyze


def test_exact_wrappers_and_constants():
    for c,reason in [('def f(x):\n return x','identity_return_parameter'),
                     ('def f(x):\n return {"x":x}','pure_parameter_dictionary'),
                     ('def f(x):\n return [1,2]','constant_return')]:
        r=analyze(c,'f');assert r['isolate'] and r['reason']==reason


def test_one_line_computation_not_rejected_and_dead_branch_not_certified():
    for c in ['def f(x):\n return (x & (x-1))==0','def f(x):\n return sorted(set(x))',
              'def f(x):\n if False: pass\n return x']:
        r=analyze(c,'f');assert not r['isolate'];assert r['complexity_review'].endswith('not_a_difficulty_label')
