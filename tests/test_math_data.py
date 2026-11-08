import pytest
from src.data.prepare_math_data import fingerprint, SimilarityIndex, partition_math, reference_answer


def row(i,p): return {'sample_id':str(i),'problem':p,'split':'train'}


def test_exact_and_similarity_and_distinct_questions():
    assert fingerprint(' A  B ') == fingerprint('a b')
    index = SimilarityIndex(.8)
    original = 'A person buys three red balls and four blue balls at the shop today'
    index.add(row(0,original))
    assert index.match(original)['kind']=='raw_exact'
    assert index.match(original.upper())['kind']=='normalized_exact'
    assert index.match(original+' now')['kind']=='jaccard_3gram'
    assert index.match('Find the radius of a circular cone') is None


def test_partition_is_reproducible_disjoint_and_excludes_test_near_duplicates():
    rows = [row(i,f'Unique item label_{i} measures quantity_{i} times factor_{i}') for i in range(20)]
    rows += [row(30,rows[0]['problem'].upper()),row(31,'Held out reference question with enough lexical detail')]
    ref = [row('test','Held out reference question with enough lexical detail today')]
    a = partition_math(rows,ref,dev_size=4,threshold=.8)
    b = partition_math(rows,ref,dev_size=4,threshold=.8)
    assert a==b and len(a[1])==4 and len(a[2])==2
    assert {r['sample_id'] for r in a[0]}.isdisjoint(r['sample_id'] for r in a[1])
    assert {r['filter_reason'] for r in a[2]}=={'internal_duplicate','evaluation_overlap'}


def test_source_answer_nested_box_and_missing_gold():
    assert reference_answer({'solution':r'Work \boxed{1} finally \boxed{\frac{1}{2}}'})==r'\frac{1}{2}'
    assert reference_answer({'solution':r'Final $\boxed 2$.'})=='2'
    with pytest.raises(ValueError): reference_answer({'solution':'no answer'})
