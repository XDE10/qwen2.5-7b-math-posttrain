"""Acceptance against completed, real v2 output. No dataset code executes here."""
import json
from collections import Counter
from pathlib import Path
from src.data.screening_io import rows,sha,ScreeningSimilarityIndex
from src.data.code_v2_dedup import Protected,ast_hash
from src.data.code_function_screen import inspect_reference,simple_tests
from src.data.build_code_v2 import check_gate


BASE=Path('data/project2_code_v2')


def test_pilot_gate_and_pinned_source_files():
    pilot=check_gate()
    assert pilot['bulk_gate_passed'] and pilot['ace_reference_passed']==6
    for source in ['ace','magicoder']:
        meta=json.loads(Path(f'data/code_v2_sources/{source}_manifest.json').read_text(encoding='utf-8'))
        for file in meta['files'][:2 if source=='ace' else 1]:
            assert sha(file['path'])==file['sha256']
    isolation=json.loads(Path('reports/code_v2_sandbox_selftest.json').read_text(encoding='utf-8'))['results']
    assert isolation[0]['passed'] and not isolation[1]['passed']


def test_balanced_v2_and_no_solution_or_tests_in_train_prompts():
    manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    assert sha('src/data/build_code_v2.py')==manifest['identity']['builder_sha256']
    assert sha('scripts/apply_code_v2_input_budget.py')==manifest['input_budget']['script_sha256']
    train=list(rows(BASE/'train_prompts.jsonl'));dev=list(rows(BASE/'dev_metadata.jsonl'))
    assert len(train)==2500 and len(dev)==200
    assert Counter(r['interface'] for r in train)=={'function':1250,'stdin_stdout':1250}
    assert Counter(r['interface'] for r in dev)=={'function':100,'stdin_stdout':100}
    assert len({r['sample_id'] for r in train+dev})==2700
    assert not {r['family_id'] for r in train}&{r['family_id'] for r in dev}
    for r in train:
        assert not {'reference_code','solution','test_cases','verification_info','gold_standard_solution'}&set(r)
        assert r['execution_status']=='OPD_prompt_only_not_a_correctness_label'
        if r['interface']=='function':assert r['function_signature'] in r['prompt']
    assert all(r['prompt_token_count']<=2560 for r in train+dev)
    assert manifest['input_budget']['max_prompt_tokens']+manifest['input_budget']['generation_reserve']==4096
    for file in manifest['files'].values():assert sha(file['path'])==file['sha256']


def test_all_dev_cases_executed_and_function_contracts_reparse():
    records=list(rows(BASE/'dev_verification.jsonl'))
    for r in records:
        assert r['execution']['passed'] and r['execution_status'].startswith('all_source_tests_passed')
        if r['interface']=='function':
            assert not any(r['execution']['constant_mutant_passes'])
            target,tests=simple_tests([t['assertion'] for t in r['test_cases']])
            parsed=inspect_reference(r['reference_code'],r['problem'],target,tests)
            assert parsed['function_signature']==r['function_signature']
            assert len({repr((t['args'],t['kwargs'])) for t in tests})>=3
        else:
            assert r['execution']['cases_checked']==len(r['verification_info']['test_cases'])>=3


def test_train_protects_every_old_dev_and_final_tests_and_new_dev():
    protected=Protected();dev_index=ScreeningSimilarityIndex(.85)
    dev=list(rows(BASE/'dev_metadata.jsonl'))
    for r in dev:dev_index.add(r)
    names={r['entry_point'] for r in dev if r['interface']=='function'}
    bodies={r['reference_body_sha256'] for r in dev if r['interface']=='function'}
    for r in rows(BASE/'train_prompts.jsonl'):
        assert protected.match(r) is None
        assert dev_index.match(r['problem']) is None
        if r['interface']=='function':
            assert r['entry_point'] not in names and r['reference_body_sha256'] not in bodies
    official=ScreeningSimilarityIndex(.85);hashes=set()
    for name in ['HumanEvalPlus-v0.1.10','MbppPlus-v0.2.0']:
        for r in rows(f'data/code_benchmarks/official/{name}.jsonl'):
            official.add({'sample_id':r['task_id'],'problem':r['prompt']})
    for r in dev:assert official.match(r['problem']) is None
