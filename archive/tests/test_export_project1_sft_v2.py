"""Review-only export provenance and immutable identity acceptance."""
import copy
import json
from pathlib import Path
import pytest
from scripts.export_project1_sft_v2 import verify_existing
from src.data.prepare_math_data import fingerprint
from src.data.screening_io import rows, sha


BASE = Path('data/project1_sft_v2')


def test_refuses_identity_mutation_corruption_and_partial_output(tmp_path):
    identity = {'source':'pinned'}
    (tmp_path/'candidate.jsonl').write_text('original', encoding='utf-8')
    with pytest.raises(ValueError, match='unsealed'):
        verify_existing(tmp_path, identity)
    manifest = {'identity':identity,'files':{'candidate.jsonl':sha(tmp_path/'candidate.jsonl')}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    assert verify_existing(tmp_path, identity) == manifest
    with pytest.raises(ValueError, match='identity change'):
        verify_existing(tmp_path, {'source':'changed'})
    (tmp_path/'candidate.jsonl').write_text('corrupted',encoding='utf-8')
    with pytest.raises(ValueError, match='Sealed output changed'):
        verify_existing(tmp_path, identity)


def test_real_export_is_review_only_sealed_and_preserves_original():
    manifest = json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    identity = manifest['identity']
    assert sha('scripts/export_project1_sft_v2.py') == identity['builder_sha256']
    assert sha('data/project1_sft/sft_train.jsonl') == identity['compact_sft_sha256']
    verify_existing(BASE,identity)
    proposals = {'2200':'balanced','3000':'coverage'}
    for cap,name in proposals.items():
        selected = json.loads(Path('reports/project1_sft_v2_plan_20261023',name+'_proposal.json').read_text(encoding='utf-8'))
        records = list(rows(BASE/f'sft_train_{cap}.jsonl'))
        assert len(records) == len(selected) == manifest['scenarios'][cap]['count']
        assert {r['problem_hash'] for r in records} == {r['question_hash'] for r in selected}
        assert len({r['problem_hash'] for r in records}) == len(records)
        for r in records:
            assert r['training_approved'] is False and r['semantic_review']=='pending'
            assert r['local_math_verify_rechecked'] is False
            assert r['source_math_verify'] is True and r['source_reasoning_complete'] is True
            assert r['problem_hash'] == fingerprint(r['problem'])
            assert 500 <= r['answer_token_count'] <= int(cap)
            assert r['total_token_count'] <= 3800
            assert r['prompt_token_count'] + r['supervised_token_count'] == r['total_token_count']
