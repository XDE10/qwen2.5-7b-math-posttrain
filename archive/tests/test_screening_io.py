import json
from src.data.screening_io import json_write, rows, SOURCES, ScreeningSimilarityIndex
from src.data.prepare_math_data import SimilarityIndex
import random


def test_atomic_json_and_jsonl_unicode(tmp_path):
    p = tmp_path/'nested'/'m.json'
    json_write(p,{'text':'中文'})
    assert json.loads(p.read_text(encoding='utf-8'))=={'text':'中文'}
    p.write_text('{"text":"a\u2028b"}\n',encoding='utf-8')
    assert list(rows(p))==[{'text':'a\u2028b'}]
    assert all(k.startswith('open-r1/') and prefix=='data/' for k,prefix in SOURCES.values())


def test_fast_index_matches_reference_without_false_negative_or_tie_drift():
    rng=random.Random(42)
    for threshold in [.5,.85,1.]:
        original,fast=SimilarityIndex(threshold),ScreeningSimilarityIndex(threshold)
        strings=[' '.join(rng.choices(['boilerplate','code','python']+[f'w{i}' for i in range(30)],k=rng.randrange(4,35))) for _ in range(140)]
        for i,text in enumerate(strings):
            row={'sample_id':str(i),'problem':text}
            original.add(row);fast.add(row)
        queries=strings[:30]+[t+' extra' for t in strings]+[t.upper() for t in strings[:5]]
        for query in queries:assert original.match(query)==fast.match(query)
