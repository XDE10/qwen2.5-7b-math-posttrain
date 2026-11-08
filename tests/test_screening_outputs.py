"""Real-data acceptance tests; run after both screening CLIs finish."""
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from src.data.screening_io import rows,sha
from src.data.prepare_math_data import SimilarityIndex
from src.data.screen_openr1_sft import sealed_math,answer_is_complete,PROMPT
from src.data.screen_code_interfaces import classify
from scripts.quarantine_code_literal_conflicts import literal_iteration_conflicts


def test_openr1_real_output_exact_tokens_source_flags_and_split_isolation():
    m=json.loads(Path('data/project1_openr1/manifest.json').read_text(encoding='utf-8'))
    selected=list(rows(m['output']['path']))
    assert sha(m['output']['path'])==m['output']['sha256']
    assert len(selected)==m['selected_rows'] and len(selected)<=1500
    assert len({r['sample_id'] for r in selected})==len(selected)
    assert len(selected)>0
    tokenizer=AutoTokenizer.from_pretrained('data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    _,protected=sealed_math()
    index=SimilarityIndex(.85)
    for k in ['math_dev','math500_test','gsm8k_test']:
        for r in protected[k]:index.add(r)
    for r in selected:
        count=len(tokenizer.encode(r['solution'],add_special_tokens=False))
        assert count==r['response_token_count'] and 500<=count<=1600
        total=len(tokenizer.apply_chat_template([{'role':'system','content':PROMPT},
            {'role':'user','content':r['problem']},{'role':'assistant','content':r['solution']}],tokenize=True,add_generation_prompt=False))
        assert total==r['total_token_count'] and total<=4096
        assert r['source_math_verify'] is True and r['source_reasoning_complete'] is True
        assert r['split']=='train' and answer_is_complete(r['solution'])
        assert index.match(r['problem']) is None


def test_code_real_interface_counts_family_split_and_dev_reclassification():
    base=Path('data/project2_code')
    m=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    train=list(rows(base/'train_prompts.jsonl'));dev=list(rows(base/'dev_verification.jsonl'))
    assert len(train)==m['train_rows'] and len(train)<=2500 and len(train)>0
    assert len(dev)==m['dev_rows']==200
    assert dict(Counter(r['interface'] for r in train))==m['actual_train_interfaces']
    assert any(r['interface']!='stdin_stdout' for r in train)
    assert not {r['source_family'] for r in train}&{r['source_family'] for r in dev}
    index=SimilarityIndex(.85)
    for r in dev:index.add(r)
    for r in train:
        assert index.match(r['problem']) is None
        assert 'gold_standard_solution' not in r and 'verification_info' not in r
        if r['interface']!='stdin_stdout':
            assert r['function_signature'] and 'return the result; do not read stdin' in r['prompt']
        assert r['execution_status'].startswith('not_executed')
    for r in dev:
        assert classify(r)['interface']==r['interface']
    functions={('code/'+r['problem_id']):r for r in rows('data/screening_sources/code_function_rows.jsonl')}
    for r in train+dev:
        if r['interface']!='stdin_stdout':assert not literal_iteration_conflicts(functions[r['sample_id']])
    for name,meta in m['files'].items():assert sha(meta['path'])==meta['sha256']


def test_shared_math_pool_identity_no_dev_regeneration():
    manifest,_=sealed_math()
    code=json.loads(Path('data/project2_code/manifest.json').read_text(encoding='utf-8'))
    assert code['math_pool']['shared_dev_sha256']==manifest['files']['math_dev']['sha256']
    assert code['math_pool']['math_candidate_sha256']==manifest['files']['math_train']['sha256']
    pool=list(rows('data/project2_code/math_train_prompts.jsonl'))
    assert len(pool)==2500
    assert Counter(r['source'] for r in pool)=={'openai/gsm8k':1000,'DigitalLearningGmbH/MATH-lighteval':1500}
    assert all('solution' not in r and 'reference_answer' not in r for r in pool)


def test_official_benchmark_overlap_and_document_rules():
    audit=json.loads(Path('reports/official_code_overlap_audit.json').read_text(encoding='utf-8'))
    assert audit['passed'] and audit['findings']==[]
    assert audit['sources']['HumanEvalPlus-v0.1.10']['rows']==164
    assert audit['sources']['MbppPlus-v0.2.0']['rows']==378
    p1=Path('docs/blueprints/项目一_7B数学后训练_详细实施方案.md').read_text(encoding='utf-8')
    # 项目二方案文档已于 2026-11-08 按用户要求移除，仅保留项目一口径检查。
    assert '500～1600' in p1 and 'generations[i]' in p1
    assert 'data/math/math_dev.jsonl' in p1
