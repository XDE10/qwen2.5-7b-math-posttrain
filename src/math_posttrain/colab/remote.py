# Bootstrap executed inside the Colab kernel by `colab exec -f`. launch.sh prepends `P1_ARGS = {...}`.
# Unpacks the uploaded code bundle, then hands control to src/math_posttrain/colab/stages.py.
#
# The job lives in the kernel, not in the client: if `colab exec` disconnects, this cell keeps running and
# releases the VM at the end. Everything printed here is appended to /content/p1/session.log, and the last
# lines are mirrored to /content/p1/session_tail.json, which `launch.sh --attach SESSION` polls with
# `colab download` (the Jupyter contents API keeps answering while the kernel is busy).
import collections
import hashlib
import importlib
import json
import os
import shutil
import sys
import tarfile
import threading
import time
from pathlib import Path

_args = globals().get('P1_ARGS')
if not _args:
    raise RuntimeError('P1_ARGS missing: start stages through scripts/colab_launch.sh')
_local = Path(os.environ.get('P1_LOCAL_ROOT', '/content/p1'))
_local.mkdir(parents=True, exist_ok=True)
# Where Drive is mounted on this VM (launch.sh --drive): rclone at /content/mydrive or Colab's own mount.
# Set before stages.py is imported: its paths are computed at import time.
if _args.get('drive_root'):
    os.environ['P1_DRIVE_ROOT'] = _args['drive_root']
os.environ['P1_RCLONE_REMOTE'] = _args.get('rclone_remote') or ''
_drive = Path(os.environ.get('P1_DRIVE_ROOT', '/content/drive/MyDrive'))
_credentials = Path((_args.get('report') or {}).get('credentials') or '/content/gh_credentials')
if _credentials.is_file():
    os.chmod(_credentials, 0o600)
# Seconds to wait before the VM releases itself, so a detached `launch.sh --attach` (polling every 60 s)
# still reads the final lines. A connected launch.sh stops the VM as soon as it sees P1_RESULT.
_RELEASE_GRACE_S = float(os.environ.get('P1_RELEASE_GRACE_S', '120'))


class _Sink:
    """Full log file plus a small JSON tail (last lines, line count, P1_RESULT) rewritten every 15 s."""

    def __init__(self, log_path: Path, tail_path: Path, keep: int = 400):
        self.file = open(log_path, 'a', encoding='utf-8', buffering=1)
        self.tail_path, self.lines, self.total = tail_path, collections.deque(maxlen=keep), 0
        self.partial, self.result, self.dirty = '', None, True
        self.lock = threading.Lock()
        self.snapshot()  # exists from the first second, so `--attach` never mistakes a fresh job for a lost VM
        threading.Thread(target=self._loop, daemon=True).start()

    def add(self, data: str) -> None:
        with self.lock:
            try:
                self.file.write(data)
            except Exception:
                pass
            *done, self.partial = (self.partial + data).split('\n')
            for line in done:
                line = line.rstrip('\r')
                self.total += 1
                self.lines.append(line)
                if line.startswith('P1_RESULT='):
                    self.result = line
            self.dirty = self.dirty or bool(done)

    def snapshot(self) -> None:
        with self.lock:
            if not self.dirty:
                return
            payload = {'lines_total': self.total, 'first_index': self.total - len(self.lines) + 1,
                       'lines': list(self.lines), 'result': self.result, 'updated': time.time()}
            self.dirty = False
        tmp = self.tail_path.with_name(self.tail_path.name + '.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, self.tail_path)

    def _loop(self) -> None:
        while True:
            time.sleep(15)
            try:
                self.snapshot()
            except Exception:
                pass

    def flush(self) -> None:
        try:
            self.file.flush()
        except Exception:
            pass


class _Tee:
    def __init__(self, stream, sink: _Sink):
        self.stream, self.sink = stream, sink

    def write(self, data):
        self.stream.write(data)
        self.sink.add(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.sink.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _assemble(path: Path) -> None:
    """Rebuild a file that launch.sh uploaded in parts (`<name>.partNNN` + `<name>.parts.json`)."""
    meta = path.with_name(path.name + '.parts.json')
    if not meta.exists():
        return
    info = json.loads(meta.read_text(encoding='utf-8'))
    parts = sorted(path.parent.glob(path.name + '.part[0-9]*'))
    if len(parts) != info['parts']:
        raise RuntimeError(f'{path}: expected {info["parts"]} uploaded parts, found {len(parts)}')
    digest = hashlib.sha256()
    with open(path, 'wb') as out:
        for part in parts:
            data = part.read_bytes()
            digest.update(data)
            out.write(data)
    if digest.hexdigest() != info['sha256']:
        raise RuntimeError(f'{path}: reassembled file does not match the uploaded SHA-256')
    for part in parts:
        part.unlink()
    meta.unlink()


_run_id = f"{time.time_ns()}-{os.getpid()}"
_run_id_file = _local / 'active_run_id'
_run_id_file.write_text(_run_id, encoding='utf-8')
_sink = _Sink(_local / 'session.log', _local / 'session_tail.json')
_stdout, _stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = _Tee(_stdout, _sink), _Tee(_stderr, _sink)
_auto_release = bool(_args.get('auto_release'))
print(f"P1_SESSION={_args.get('session')} gpu={_args.get('gpu')} auto_release={int(_auto_release)}", flush=True)
try:
    _code = _local / 'code' / _args['code_sha']
    _bundle = Path(_args.get('bundle', '/content/p1_bundle.tgz'))
    _assemble(_bundle)
    if not (_code / '.p1_ready').exists():
        shutil.rmtree(_code, ignore_errors=True)
        _code.mkdir(parents=True)
        with tarfile.open(_bundle) as _tar:
            _tar.extractall(_code)
        (_code / '.p1_ready').write_text(_args['code_sha'])
    _drive_code = _drive / 'projects/project1/v93/code'
    if _drive.is_dir():
        _drive_code.mkdir(parents=True, exist_ok=True)
        _copy = _drive_code / f"{_args['code_sha']}.tgz"
        if not _copy.exists():
            shutil.copyfile(_bundle, _copy)  # exact code of every stage stays on Drive for provenance
    os.environ['P1_CODE_SHA'] = _args['code_sha']
    os.environ['P1_SESSION'] = str(_args.get('session') or '')
    for _p in (str(_code / 'src'), str(_code)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from math_posttrain.colab import stages as _stages  # noqa: E402
    _stages = importlib.reload(_stages)
    print(f"code bundle {_args['code_sha']} (git {_args.get('git_head')}); stages: {_args['stages']}", flush=True)
    _stages.main(_args)
except BaseException as _error:
    if _sink.result is None:  # failed before stages.main could report (bad bundle, import error, ...)
        print(f'P1_RESULT=FAILED bootstrap: {type(_error).__name__}: {_error}'[:2000], flush=True)
    raise
finally:
    sys.stdout.flush()
    sys.stdout, sys.stderr = _stdout, _stderr
    try:
        _sink.snapshot()
    except Exception:
        pass
    try:
        if _drive.is_dir():
            _drive_logs = _drive / 'runs/project1/v93/logs'
            _drive_logs.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_local / 'session.log', _drive_logs / f"session_{_args.get('session') or 'unnamed'}.log")
    except Exception as _copy_error:
        print('could not copy the session log to Drive:', _copy_error, flush=True)
    if _auto_release:
        try:  # the final report was pushed before P1_RESULT; the VM is about to be released
            _credentials.unlink(missing_ok=True)
        except OSError:
            pass
        time.sleep(_RELEASE_GRACE_S)
        try:
            _current_id = _run_id_file.read_text(encoding='utf-8').strip() if _run_id_file.exists() else _run_id
        except OSError:
            _current_id = _run_id
        if _current_id == _run_id:
            try:  # best effort; a connected launch.sh has usually stopped the VM already
                from google.colab import runtime
                runtime.unassign()
            except Exception as _release_error:
                print('auto-release not available here:', _release_error, flush=True)
        else:
            print('skipping auto-release: a newer run has taken over this VM', flush=True)
