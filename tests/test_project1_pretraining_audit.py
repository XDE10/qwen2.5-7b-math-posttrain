import json
from pathlib import Path
import pytest
from scripts.finalize_project1_sft import audit_at,bucket
from src.data.screening_io import rows,sha,ScreeningSimilarityIndex
from src.data.screen_openr1_sft import sealed_math


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def test_repeated_source_uuid_is_resolved_by_scan_position():
    records=[{'sample_id':'same','marker':'first'},{'sample_id':'same','marker':'second'}]
    assert audit_at(records,0,'same')['marker']=='first'
    assert audit_at(records,1,'same')['marker']=='second'
    with pytest.raises(ValueError):audit_at(records,1,'different')


def test_formal_sft_membership_source_proofs_and_file_hashes():
    manifest=read('data/project1_sft/manifest.json')
    original={r['sample_id']:r for r in rows('data/project1_openr1/sft_train.jsonl')}
    selected=list(rows('data/project1_sft/sft_train.jsonl'))
    proofs={r['sample_id']:r for r in rows('data/project1_sft/trajectory_revalidation.jsonl')}
    assert len(selected)==len(original)==len(proofs)==1500
    assert sha('scripts/finalize_project1_sft.py')==manifest['identity']['script_sha256']
    for r in selected:
        old=original[r['sample_id']];p=proofs[r['sample_id']]
        assert r['solution']==old['solution'] and r['trajectory_index']==old['trajectory_index']
        assert 500<=r['answer_token_count']==r['response_token_count']<=1600
        assert r['total_token_count']<=4096 and p['source_flags_aligned']
        assert r['difficulty'] is None and not r['rl_train_overlap']
    for meta in manifest['files'].values():assert sha(meta['path'])==meta['sha256']


def test_formal_sft_evaluation_and_rl_pool_isolation():
    _,protected=sealed_math();index=ScreeningSimilarityIndex(.85)
    for records in protected.values():
        for r in records:index.add(r)
    for r in rows('data/project1_sft/sft_train.jsonl'):assert index.match(r['problem']) is None
    assert list(rows('data/project1_sft/rl_overlap_audit.jsonl'))==[]


def test_distributions_report_missing_difficulty_and_stratified_audit_coverage():
    d=read('data/project1_sft/distributions.json')
    assert d['scanned_questions']==28572 and d['repeated_source_uuid_rows']==206
    assert not d['difficulty_available']
    for k,v in d['before_questions'].items():assert sum(v.values())==28572
    for k,v in d['after_questions'].items():assert sum(v.values())==1500
    assert sum(d['length_before_compact_filter'].values())==d['length_before_denominator']==37580
    selected=list(rows('data/project1_sft/sft_train.jsonl'));audit=list(rows('data/project1_sft/human_audit_50.jsonl'))
    assert len(audit)==50 and all(r['human_verdict'] is None for r in audit)
    assert {(r['problem_type'],bucket(r['answer_token_count'])) for r in audit}=={(r['problem_type'],bucket(r['answer_token_count'])) for r in selected}
    assert {r['source_category'] for r in audit}=={r['source_category'] for r in selected}


def test_live_tokenizer_revalidation_matches_frozen_report():
    proof=read('reports/tokenizer_alignment_revalidation_20261023.json')
    assert proof['original_report_sha256']==sha('reports/tokenizer_alignment_report.json')
    assert proof['script_sha256']==sha('scripts/revalidate_tokenizer_alignment.py')
    assert proof['case_count']==64 and proof['case_categories']=={'math':16,'code':16,'unicode':16,'special':16}
    assert proof['all_raw_encodings_identical'] and not proof['all_chat_encodings_identical']
    assert proof['common_support_size']==proof['exclusive_upper_bound']==151665
    assert proof['mapping_difference_count']==0
    assert [m['outside_common_support_rows'] for m in proof['models']]==[271,399,399,399]
