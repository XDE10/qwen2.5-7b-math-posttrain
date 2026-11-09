"""teacher.py fill scenarios against the mock DeepSeek server (teacher_mock.py). Prints PASS/FAIL lines."""
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
    results.append(ok)
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'   [{detail}]'))


def fresh(tag):
    root = WORK / f'fake_{tag}'
    subprocess.run([PY, str(HARNESS / 'make_fake.py'), str(REPO), str(root), str(TOK)], check=True,
                   capture_output=True)
    return root


def run(root, scenario, *args, key=True):
    (root / 'scenario.json').write_text(json.dumps(scenario))
    env = dict(os.environ, PYTHONPATH=str(root))
    env.pop('OPENCODE_GO_API_KEY', None)
    env.pop('DEEPSEEK_API_KEY', None)
    if key:
        env['OPENCODE_GO_API_KEY'] = 'test-key'
    p = subprocess.run([PY, str(HARNESS / 'teacher_mock.py'), str(root), str(root / 'scenario.json'), '--',
                        'fill', '--go-db', str(root / 'no.db'), '--official-key-file', str(root / 'no.json'),
                        '--workers', '1', *args], capture_output=True, text=True, env=env, timeout=300)
    out = p.stdout + p.stderr
    code = next((int(line.split()[1]) for line in out.splitlines() if line.startswith('EXIT ')), None)
    return code, out


def folders(root, index):
    base = root / f'artifacts/qfft_teacher_v93/attempts/{index:04d}'
    out = {}
    for d in sorted(base.glob('a[0-9][0-9]')) if base.exists() else []:
        if (d / 'candidate.json').exists():
            out[d.name] = 'candidate:' + json.loads((d / 'candidate.json').read_text())['finish_reason']
        elif (d / 'error.json').exists():
            out[d.name] = 'error:' + json.loads((d / 'error.json').read_text())['kind']
        else:
            out[d.name] = 'request-only'
    return out


# S1: no key anywhere -> stop before any folder is created
root = fresh('s1')
code, out = run(root, ['ok'], key=False)
check('S1 missing key exits non-zero', code not in (0, None), out[-300:])
check('S1 message says nothing was sent', 'Nothing was sent' in out, out[-300:])
check('S1 no attempt folders', all(not folders(root, i) for i in (997, 998)) and len(folders(root, 999)) == 2,
      str([folders(root, i) for i in (997, 998, 999)]))

# S2: HTTP 401 -> whole run stops, attempt not counted, re-run continues with the next number
root = fresh('s2')
code, out = run(root, ['e401'], '--efforts', 'max,high')
check('S2 401 stops the run (exit 2)', code == 2, out[-400:])
check('S2 one rejected attempt only', folders(root, 997) == {'a01': 'error:rejected'} and not folders(root, 998),
      str(folders(root, 997)) + str(folders(root, 998)))
code, out = run(root, ['ok'], '--efforts', 'max,high')
check('S2 re-run succeeds (exit 0)', code == 0, out[-400:])
check('S2 next number used, budget intact', folders(root, 997) == {'a01': 'error:rejected', 'a02': 'candidate:stop'},
      str(folders(root, 997)))
efforts = [json.loads((root / f'artifacts/qfft_teacher_v93/attempts/0997/a0{k}/request.json').read_text())['effort']
           for k in (1, 2)]
check('S2 refused request retries the same effort', efforts == ['max', 'max'], str(efforts))

# S3: five 500s in a row -> circuit breaker, no budget used
root = fresh('s3')
code, out = run(root, ['e500'])
check('S3 outage stops after 5 failures (exit 2)', code == 2 and 'TooManyErrors' in out, out[-400:])
n = sum(len(folders(root, i)) for i in (997, 998, 999)) - 2
check('S3 exactly 5 server_error folders', n == 5, str([folders(root, i) for i in (997, 998, 999)]))

# S4: no trailing blank line after [DONE], a length finish, a dropped stream; index 999 has foreign attempts
root = fresh('s4')
code, out = run(root, ['ok_noblank', 'length', 'drop', 'ok', 'ok'])
check('S4 completes (exit 0)', code == 0, out[-600:])
check('S4 [DONE] without blank line accepted', folders(root, 997) == {'a01': 'candidate:stop'}, str(folders(root, 997)))
check('S4 length then drop then ok', folders(root, 998) == {'a01': 'candidate:length', 'a02': 'error:transport',
                                                            'a03': 'candidate:stop'}, str(folders(root, 998)))
f999 = folders(root, 999)
check('S4 foreign attempts kept, new number a03', f999.get('a03') == 'candidate:stop' and len(f999) == 3, str(f999))
efforts = [json.loads((root / f'artifacts/qfft_teacher_v93/attempts/0998/a0{k}/request.json').read_text())['effort']
           for k in (1, 2, 3)]
check('S4 effort schedule follows counted attempts (high, high, max)', efforts == ['high', 'high', 'max'], str(efforts))

# S5: budget used up -> exit 3 with guidance; 429 retries do not use budget
root = fresh('s5')
code, out = run(root, ['length'], '--max-attempts', '2')
check('S5 budget exhausted exits 3', code == 3, out[-600:])
check('S5 guidance printed', 'teacher fill --max-attempts 6' in out, out[-600:])
check('S5 two counted attempts each', all(len(folders(root, i)) == 2 for i in (997, 998)) and len(folders(root, 999)) == 4,
      str([folders(root, i) for i in (997, 998, 999)]))
code, out = run(root, ['ok'], '--max-attempts', '3')
check('S5 larger budget finishes (exit 0)', code == 0, out[-400:])

# S6: quota on Go without a usable official key -> stop, quota attempt not counted
root = fresh('s6')
code, out = run(root, ['e402'], '--fallback-official')
check('S6 fallback key missing is reported', 'quota fallback will not work' in out, out[-500:])
check('S6 quota stops the run (exit 2)', code == 2, out[-500:])
check('S6 quota attempt is no-output', folders(root, 997) == {'a01': 'error:quota'}, str(folders(root, 997)))

# S7: no Go key (e.g. WSL2 without the Windows opencode.db) but an official key file -> start on official
root = fresh('s7')
(root / 'ok.json').write_text(json.dumps({'DEEPSEEK_API_KEY': 'test-key'}))
code, out = run(root, ['ok'], '--fallback-official', '--official-key-file', str(root / 'ok.json'), key=False)
check('S7 starts on the official API', code == 0 and 'Starting with the official API' in out, out[-500:])
providers = {json.loads((root / f'artifacts/qfft_teacher_v93/attempts/{i:04d}/a01/request.json').read_text())['provider']
             for i in (997, 998)}
check('S7 requests used the official provider', providers == {'official'}, str(providers))

# S8: questions whose traces keep running past the window: low effort on the official API, same prompt
root = fresh('s8')
(root / 'ok.json').write_text(json.dumps({'DEEPSEEK_API_KEY': 'test-key'}))
code, out = run(root, ['length', 'ok'], '--provider', 'official', '--official-key-file', str(root / 'ok.json'),
                '--efforts', 'low', '--max-attempts', '3', key=False)
requests = [json.loads(p.read_text()) for p in sorted((root / 'artifacts/qfft_teacher_v93/attempts').glob('*/a*/request.json'))
            if json.loads(p.read_text())['sample_id'].startswith('test/')]
check('S8 low-effort run completes (exit 0)', code == 0, out[-500:])
check('S8 every request: official API, deepseek-flash, reasoning_effort low, v91 prompt',
      len(requests) == 4 and all(r['provider'] == 'official' and r['effort'] == 'low' and r['body']['reasoning_effort'] == 'low'
                                 and r['body']['model'] == 'deepseek-flash' and r['prompt_variant'] == 'v91_anti_redundancy'
                                 for r in requests), json.dumps([(r['index'], r['effort'], r['provider']) for r in requests]))
candidate = json.loads((root / 'artifacts/qfft_teacher_v93/attempts/0997/a02/candidate.json').read_text())
check('S8 the candidate records the effort it was made with', candidate['reasoning_effort_requested'] == 'low'
      and candidate['finish_reason'] == 'stop', str({k: candidate.get(k) for k in ('reasoning_effort_requested', 'finish_reason')}))
code, out = run(fresh('s8b'), ['length'], '--max-attempts', '1')
check('S8 exhausted budget suggests the low-effort step and the paired exclusion',
      code == 3 and '--efforts low' in out and '--paired-exclusion' in out, out[-700:])

print(f'{sum(results)}/{len(results)} checks passed')
sys.exit(0 if all(results) else 1)
