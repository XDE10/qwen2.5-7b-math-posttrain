import hashlib
import json
from pathlib import Path
from src.data.prepare_math_data import SimilarityIndex, fingerprint


def test_real_data_manifest_and_sealed_splits():
    m = json.loads(Path('data/math_manifest.json').read_text(encoding='utf-8'))
    assert m['sources']['gsm8k']['raw_counts']=={'train':7473,'test':1319}
    assert m['sources']['math']['raw_counts']=={'train':7500,'test':5000}
    assert m['sources']['math500']['raw_counts']=={'test':500}
    required = {'sample_id','source','source_revision','source_split','problem','reference_answer',
                'solution','domain','token_count','filter_reason','problem_hash','raw_problem_sha256'}
    groups = {}
    for name,meta in m['files'].items():
        p = Path(meta['path'])
        assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
        # JSONL is delimited by LF; str.splitlines also splits legal U+2028 in strings.
        with p.open(encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        assert len(rows)==meta['rows']
        assert all(required <= r.keys() and r['problem_hash']==fingerprint(r['problem']) for r in rows)
        groups[name] = rows
    assert len(groups['math_dev'])==256
    index = SimilarityIndex(m['dedup']['threshold'])
    for r in groups['gsm8k_test']+groups['math500_test']+groups['math_dev']: index.add(r)
    for r in groups['math_train']:
        assert index.match(r['problem']) is None
    assert sum(len(groups[k]) for k in ['math_train','math_dev','math_quarantine'])==7500
    assert not {r['sample_id'] for r in groups['math_train']} & {r['sample_id'] for r in groups['math_dev']}
