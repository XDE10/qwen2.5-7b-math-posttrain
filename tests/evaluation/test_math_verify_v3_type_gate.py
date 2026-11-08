import json
import subprocess
import sys
from pathlib import Path


def worker(output,reference):
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,'-m','scripts.math_verify_v3_strict_worker'],cwd=root,
        input=json.dumps({'output':output,'reference':reference}),capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
    return json.loads(p.stdout)


def test_wrong_relation_never_approved_by_generic_verifier():
    r=worker(r'\boxed{(1+23^{1088})/(1+23^{1880}) < (1+23^{1089})/(1+23^{1880})}',
        r'(1+23^{1988})/(1+23^{1989}) > (1+23^{1989})/(1+23^{1980})')
    assert not r['correct'] and r['status']=='relational_answer_requires_task_aware_manual_review'


def test_individual_boxes_not_mislabelled_as_wrong_set():
    r=worker(r'\boxed{1} and \boxed{2}',r'\{1,2\}')
    assert not r['correct'] and 'collection_review' in r['status']


def test_plain_numeric_equivalence_still_works():
    assert worker(r'\boxed{\frac{1}{2}}','0.5')['correct']
