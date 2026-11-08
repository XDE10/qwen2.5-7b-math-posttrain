import json
from scripts.review_geometry_grounding import parse_review


def test_all_fact_indices_must_be_classified_once_and_pass_has_no_missing_conditions():
    r={'verdict':'pass','visible_source_point_labels':['A','B'],'circle_count':1,'image_only_fact_indices':[0],
        'text_fact_indices':[1],'unverified_fact_indices':[],'critical_missing_conditions':[],'issues':[],'evidence':['AB tick and label visible','one circle visible']}
    assert parse_review(json.dumps(r),2,'stop')['verdict']=='pass'
    r['text_fact_indices']=[0];assert not parse_review(json.dumps(r),2,'stop')['valid_schema']
    r['text_fact_indices']=[1];r['critical_missing_conditions']=['angle label missing']
    assert parse_review(json.dumps(r),2,'stop')['verdict']=='uncertain'
    assert not parse_review(json.dumps(r),2,'length')['valid_schema']
