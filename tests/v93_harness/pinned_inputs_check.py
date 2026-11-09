"""data_build.pinned_input / TokenCounter fallback checks on a fake root (test only)."""
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO = Path(os.environ.get('P1_HARNESS_REPO') or HARNESS.parents[1])
WORK = Path(os.environ.get('P1_HARNESS_WORK') or tempfile.mkdtemp())
TOK = Path(os.environ.get('P1_HARNESS_TOKENIZER') or REPO / 'data/tokenizers/Qwen2.5-7B-Instruct/tokenizer.json')
root = WORK / 'fake_pinned'
subprocess.run([sys.executable, str(HARNESS / 'make_fake.py'), str(REPO), str(root), str(TOK)], check=True,
               capture_output=True)
sys.path.insert(0, str(root))
results = []


def check(name, ok, detail=''):
    results.append(ok)
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'   [{detail}]'))


def fresh_module():
    import experiments.qfft_v93.data_build as db
    return importlib.reload(db)


local = root / 'data/tokenizers/Qwen2.5-7B-Instruct/tokenizer.json'
fallback = root / 'data/benchmarks_v93/_sources/Qwen2.5-7B-Instruct-tokenizer.json'

db = fresh_module()
check('local tokenizer with the pinned hash is used', db.pinned_input('tokenizer') == local)

local.write_text('{"not": "the pinned tokenizer"}')
fallback.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(TOK, fallback)
db = fresh_module()
check('differing local copy -> verified download is used', db.pinned_input('tokenizer') == fallback)
counter = db.TokenCounter()
check('TokenCounter() works through the fallback', counter.count('Hello world') > 0)

fallback.unlink()
db = fresh_module()
try:
    db.pinned_input('tokenizer')
    check('no verified copy raises', False, 'no exception')
except RuntimeError as error:
    msg = str(error)
    check('no verified copy raises with both paths and the fix', 'sha256 differs' in msg and '(missing)' in msg
          and 'fetch_benchmarks --sources' in msg, msg)

try:
    db.pinned_input('s1k_parquet')
    check('missing s1K raises', False, 'no exception')
except RuntimeError as error:
    check('missing s1K raises with guidance', 'fetch_benchmarks --sources' in str(error), str(error))

print(f'{sum(results)}/{len(results)} checks passed')
sys.exit(0 if all(results) else 1)
