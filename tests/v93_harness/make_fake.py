"""Fake repository root for teacher.py / data_build.py tests: 1000-question plan, 997 usable v66 candidates."""
import json
import shutil
import sys
from pathlib import Path

repo, root, tokenizer = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shutil.rmtree(root, ignore_errors=True)
shutil.copytree(repo / 'experiments', root / 'experiments', ignore=shutil.ignore_patterns('__pycache__'))
(root / 'data/tokenizers/Qwen2.5-7B-Instruct').mkdir(parents=True)
shutil.copyfile(tokenizer, root / 'data/tokenizers/Qwen2.5-7B-Instruct/tokenizer.json')
questions = [{'index': i, 'sample_id': f'test/{i:04d}', 'question': f'Question number {i}: compute 2+3.',
              'reference_answer': '5', 'reference_kind': 'metadata_answer', 'origin': 's1k11', 'ds_effort': 'high',
              'r1': {'reasoning': 'r', 'final': 'f', 'from': 'test'}} for i in range(1000)]
(root / 'data/qfft1000_v93').mkdir(parents=True)
(root / 'data/qfft1000_v93/question_plan.json').write_text(json.dumps({'version': 'v93', 'questions': questions,
                                                                     'decontamination': {}}))
cand = root / 'artifacts/qfft_ds_dataset_v66/candidates'
cand.mkdir(parents=True)
for q in questions[:997]:
    (cand / f"{q['index']:04d}.json").write_text(json.dumps({
        'sample_id': q['sample_id'], 'finish_reason': 'stop', 'reasoning_content_raw': 'Reasoning.',
        'final_content_raw': '\\boxed{5}', 'usage': {}}))
# Index 999: two old v93 attempts made for a different question at the same position (plan re-run).
for k in (1, 2):
    old = root / f'artifacts/qfft_teacher_v93/attempts/0999/a{k:02d}'
    old.mkdir(parents=True)
    (old / 'request.json').write_text(json.dumps({'sample_id': 'old/0999', 'index': 999, 'attempt': k}))
    (old / 'candidate.json').write_text(json.dumps({'sample_id': 'old/0999', 'finish_reason': 'stop',
                                                     'reasoning_content_raw': 'x', 'final_content_raw': 'y'}))
print('fake root ready:', root)
