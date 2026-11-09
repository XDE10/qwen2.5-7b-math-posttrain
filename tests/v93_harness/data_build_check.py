"""data_build build on a fake root: refusing gaps, preview builds, the paired exclusion and the DS provenance
counts in the manifest and teacher report. Needs tokenizers + math-verify and the Qwen tokenizer.json."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO = Path(os.environ.get('P1_HARNESS_REPO') or HARNESS.parents[1])
WORK = Path(os.environ.get('P1_HARNESS_WORK') or tempfile.mkdtemp())
PY = os.environ.get('P1_HARNESS_PYTHON') or sys.executable
TOK = Path(os.environ.get('P1_HARNESS_TOKENIZER') or REPO / 'data/tokenizers/Qwen2.5-7B-Instruct/tokenizer.json')
results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'   [{detail}]'))


root = WORK / 'fake_data_build'
subprocess.run([PY, str(HARNESS / 'make_fake.py'), str(REPO), str(root), str(TOK)], check=True, capture_output=True)
out_dir = root / 'data/qfft1000_v93'


def build(*args):
    p = subprocess.run([PY, '-m', 'experiments.qfft_v93.data_build', 'build', '--workers', '2', *args], cwd=root,
                       env=dict(os.environ, PYTHONPATH=str(root), PYTHONUTF8='1'), capture_output=True, text=True,
                       timeout=900)
    manifest = json.loads((out_dir / 'manifest.json').read_text()) if (out_dir / 'manifest.json').exists() else None
    report = (out_dir / 'teacher_report.md').read_text() if (out_dir / 'teacher_report.md').exists() else ''
    return p.returncode, p.stdout + p.stderr, manifest, report


def rows(name):
    return [json.loads(line) for line in (out_dir / name / 'train.jsonl').read_text().splitlines()]


# make_fake: 1000 questions, 997 with a usable v66 trace; 997-999 have none (999 has foreign attempts only)
code, out, manifest, _ = build()
check('gaps without a flag are refused and the fix is named', code != 0 and '3 questions still lack' in out
      and '--paired-exclusion' in out and manifest is None, out[-500:])

code, out, manifest, report = build('--allow-gaps')
check('preview build: R1 keeps all 1000, DS has 997', code == 0 and manifest['variants']['r1']['rows'] == 1000
      and manifest['variants']['ds']['rows'] == 997, out[-500:])
check('preview build: gaps recorded, nothing excluded', manifest['gaps'] == ['test/0997', 'test/0998', 'test/0999']
      and manifest['paired_exclusions'] == [], str(manifest.get('gaps')))
check('preview build: report warns it cannot be uploaded', '预览构建' in report and 'pipeline.sh d 会拒收' in report, report)
check('preview build: no paired agreement table (different questions)', 'agreement_r1_vs_ds' not in manifest)

code, out, _, _ = build('--allow-gaps', '--paired-exclusion')
check('--allow-gaps with --paired-exclusion is refused', code != 0 and 'cannot be combined' in out, out[-300:])

# a later low-effort v93 attempt fills question 997
attempt = root / 'artifacts/qfft_teacher_v93/attempts/0997/a01'
attempt.mkdir(parents=True)
(attempt / 'candidate.json').write_text(json.dumps({
    'sample_id': 'test/0997', 'index': 997, 'attempt': 1, 'provider': 'official', 'model_returned': 'deepseek-flash',
    'reasoning_effort_requested': 'low', 'prompt_variant': 'v91_anti_redundancy', 'finish_reason': 'stop',
    'reasoning_content_raw': 'Short reasoning.', 'final_content_raw': '\\boxed{5}', 'usage': {}}))
code, out, manifest, report = build('--ds-verified', '--paired-exclusion')
check('paired exclusion: every variant has the same 998 questions', code == 0 and all(
    v['rows'] == 998 for v in manifest['variants'].values()) and len({v['questions_sha256'] for v in
                                                                    manifest['variants'].values()}) == 1, out[-500:])
check('paired exclusion: the two remaining gaps are listed with their attempts',
      [e['sample_id'] for e in manifest['paired_exclusions']] == ['test/0998', 'test/0999'] and manifest['gaps'] == []
      and all('r1_sft_tokens' in e and 'ds_attempts' in e for e in manifest['paired_exclusions']),
      json.dumps(manifest.get('paired_exclusions'))[:400])
check('paired exclusion: excluded questions are absent from R1 too',
      not {'test/0998', 'test/0999'} & {r['sample_id'] for r in rows('r1')})
check('paired exclusion: agreement table computed on the common questions', 'agreement_r1_vs_ds' in manifest)
check('paired exclusion: report lists the excluded questions', '成对剔除 2 题' in report and 'test/0998' in report, report)
ds997 = next(r for r in rows('ds') if r['sample_id'] == 'test/0997')
check('row 997 records the low-effort v93 attempt', ds997['teacher_detail']['effort'] == 'low'
      and ds997['teacher_detail']['attempt'] == 'v93-a01' and ds997['teacher_detail']['provider'] == 'official',
      str(ds997['teacher_detail']))
prov = manifest['variants']['ds']['ds_provenance']
check('provenance counts rounds, efforts, prompts and APIs', prov['rounds'] == {'v66': 997, 'v93': 1}
      and prov['efforts'].get('low') == 1 and prov['prompt_variants'].get('v91_anti_redundancy') == 1
      and prov['providers'].get('official') == 1, json.dumps(prov))
check('report shows the provenance line', 'reasoning_effort' in report and 'low 1' in report, report)
check('R1 has no provenance block', 'ds_provenance' not in manifest['variants']['r1'])

print(f'{sum(results)}/{len(results)} checks passed')
sys.exit(0 if all(results) else 1)
