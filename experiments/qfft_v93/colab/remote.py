# Bootstrap executed inside the Colab kernel by `colab exec -f`. launch.sh prepends `P1_ARGS = {...}`.
# Unpacks the uploaded code bundle, then hands control to experiments/qfft_v93/colab/stages.py.
import importlib
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

_args = globals().get('P1_ARGS')
if not _args:
    raise RuntimeError('P1_ARGS missing: start stages through experiments/qfft_v93/colab/launch.sh')
_local = Path(os.environ.get('P1_LOCAL_ROOT', '/content/p1'))
_code = _local / 'code' / _args['code_sha']
_bundle = Path(_args.get('bundle', '/content/p1_bundle.tgz'))
if not (_code / '.p1_ready').exists():
    shutil.rmtree(_code, ignore_errors=True)
    _code.mkdir(parents=True)
    with tarfile.open(_bundle) as _tar:
        _tar.extractall(_code)
    (_code / '.p1_ready').write_text(_args['code_sha'])
_drive_code = Path('/content/drive/MyDrive/projects/project1/v93/code')
if Path('/content/drive/MyDrive').is_dir():
    _drive_code.mkdir(parents=True, exist_ok=True)
    _copy = _drive_code / f"{_args['code_sha']}.tgz"
    if not _copy.exists():
        shutil.copyfile(_bundle, _copy)  # exact code of every stage stays on Drive for provenance
os.environ['P1_CODE_SHA'] = _args['code_sha']
if str(_code) not in sys.path:
    sys.path.insert(0, str(_code))
from experiments.qfft_v93.colab import stages as _stages  # noqa: E402
_stages = importlib.reload(_stages)
print(f"code bundle {_args['code_sha']} (git {_args.get('git_head')}); stages: {_args['stages']}", flush=True)
try:
    _stages.main(_args)
finally:
    if _args.get('auto_release'):
        time.sleep(3)
        try:  # best effort; launch.sh also runs `colab stop` when it is still connected
            from google.colab import runtime
            runtime.unassign()
        except Exception as _error:
            print('auto-release not available here:', _error, flush=True)
