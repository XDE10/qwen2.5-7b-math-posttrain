"""Shared helpers for the v93 pipeline: paths, Drive layout, atomic IO, hashing, locks and logging.

Standard library only. Importing this module never touches the network, a GPU or Google Drive.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parents[1]
CONFIG_DIR = PKG / 'configs'
TZ = timezone(timedelta(hours=8))

BASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
BASE_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
TOKENIZER_SHA256 = {
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
}
GIB = 2 ** 30


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec='seconds')


def log(*parts) -> None:
    print(f'[{datetime.now(TZ):%m-%d %H:%M:%S}]', *parts, flush=True)


# ----------------------------------------------------------------------------- JSON / JSONL IO

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _replace_with_retry(tmp: Path, path: Path) -> None:
    # Windows virus scanners and Drive sync clients can briefly lock the destination.
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1 * (attempt + 1))


def write_json(path, value) -> None:
    """Atomic write: temp file in the same directory, fsync, then replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)


def read_jsonl(path, tolerate_torn_tail: bool = True) -> list:
    """Read JSONL; a crash during append can leave one torn final line, which is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if tolerate_torn_tail and i == len(lines) - 1:
                break
            raise
    return rows


class JsonlAppender:
    """Append-only JSONL writer that fsyncs at most every `fsync_seconds`.

    With `mirror` (a Drive path) the journal itself lives at `path` on the VM's local SSD and the Drive copy
    is replaced atomically at most every `mirror_seconds` and on close: a file kept open on a Drive mount
    (rclone or Colab's) is uploaded only when it is closed, which for a training journal means hours later
    or never. A missing local journal starts from the Drive copy, so a resumed run keeps its history.
    """

    def __init__(self, path, fsync_seconds: float = 2.0, mirror=None, mirror_seconds: float = 60.0):
        self.path = Path(path)
        self.mirror = Path(mirror) if mirror else None
        self.mirror_seconds = mirror_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.mirror is not None and not self.path.exists() and self.mirror.is_file():
            shutil.copyfile(self.mirror, self.path)
        self._repair_tail()
        self.handle = self.path.open('a', encoding='utf-8', newline='\n')
        self.fsync_seconds = fsync_seconds
        self.last_sync = time.monotonic()
        self.last_mirror = time.monotonic()
        self.lock = threading.Lock()

    def _sync_mirror(self) -> None:
        try:
            sync_file(self.path, self.mirror)
        except OSError as error:  # the local journal stays complete; the next sync retries
            log(f'WARNING could not update {self.mirror}: {error}')
        self.last_mirror = time.monotonic()

    def _repair_tail(self) -> None:
        # Drop a torn last line so the next append starts on a clean line.
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        data = self.path.read_bytes()
        if data.endswith(b'\n'):
            return
        cut = data.rfind(b'\n') + 1
        self.path.write_bytes(data[:cut])

    def write(self, row) -> None:
        with self.lock:
            self.handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            self.handle.flush()
            if time.monotonic() - self.last_sync >= self.fsync_seconds:
                os.fsync(self.handle.fileno())
                self.last_sync = time.monotonic()
            if self.mirror is not None and time.monotonic() - self.last_mirror >= self.mirror_seconds:
                self._sync_mirror()

    def close(self) -> None:
        with self.lock:
            if not self.handle.closed:
                self.handle.flush()
                os.fsync(self.handle.fileno())
                self.handle.close()
                if self.mirror is not None:
                    self._sync_mirror()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------------------------- hashing

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode('utf-8'))


def sha256_json(value) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- locks

@contextlib.contextmanager
def exclusive_lock(path):
    """Non-blocking OS lock released automatically when the process exits or crashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, 'a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f'Another process holds {path}; refusing to start the same job twice') from None
    try:
        yield
    finally:
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


# ----------------------------------------------------------------------------- Drive / local layout

@dataclass(frozen=True)
class DriveLayout:
    """Persistent locations. `models/` and credentials are shared with earlier project versions."""
    root: Path

    @property
    def project(self) -> Path:
        return self.root / 'projects/project1/v93'

    @property
    def data(self) -> Path:
        return self.project / 'data'

    @property
    def code(self) -> Path:
        return self.project / 'code'

    @property
    def envcache(self) -> Path:
        return self.project / 'envcache'

    @property
    def models(self) -> Path:
        return self.root / 'models'

    @property
    def runs(self) -> Path:
        return self.root / 'runs/project1/v93/runs'

    @property
    def evals(self) -> Path:
        return self.root / 'runs/project1/v93/evals'

    @property
    def stages(self) -> Path:
        return self.root / 'runs/project1/v93/stages'

    @property
    def credentials(self) -> Path:
        return self.root / '.project1_credentials'


def drive_layout(root=None) -> DriveLayout:
    return DriveLayout(Path(root or os.environ.get('P1_DRIVE_ROOT', '/content/drive/MyDrive')))


def local_root() -> Path:
    """Fast VM-local SSD scratch (lost when the VM is released)."""
    return Path(os.environ.get('P1_LOCAL_ROOT', '/content/p1'))


# ----------------------------------------------------------------------------- configs

def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path) -> dict:
    """JSON config with optional `"base": "other.json"` inheritance (path relative to this file)."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    config = read_json(path)
    base = config.pop('base', None)
    if base:
        config = _deep_merge(load_config(path.parent / base), config)
    return config


# ----------------------------------------------------------------------------- disk / copying

def human_bytes(n: float) -> str:
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(n) < 1024 or unit == 'TiB':
            return f'{n:.1f}{unit}'
        n /= 1024
    return f'{n:.1f}TiB'


def free_bytes(path) -> int:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def tree_bytes(path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def copy_file_throttled(src, dst, *, watch_path='/content', min_free=24 * GIB,
                        chunk=64 * 1024 * 1024, stall_seconds=3600, stop_event=None) -> None:
    """Copy in chunks; pause while local free space is low.

    Writes to the Colab Drive mount are first cached on the VM disk and uploaded in the background.
    Copying a 100 GB checkpoint faster than Drive uploads it could fill the disk, so the copy waits
    until the Drive client has drained enough of its cache.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.partial')
    with src.open('rb') as fin, tmp.open('wb') as fout:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError('copy cancelled')
            waited = 0.0
            while free_bytes(watch_path) < min_free:
                if waited >= stall_seconds:
                    raise RuntimeError(f'Local disk stayed below {human_bytes(min_free)} free for '
                                       f'{stall_seconds}s while copying {src.name}; Drive upload stalled')
                time.sleep(5)
                waited += 5
            block = fin.read(chunk)
            if not block:
                break
            fout.write(block)
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(tmp, dst)


def copy_tree_throttled(src, dst, exclude=None, **kwargs) -> None:
    """Copy a directory to `dst` via `dst.partial`, then rename; never leaves a half tree at `dst`."""
    import fnmatch
    src, dst = Path(src), Path(dst)
    staging = dst.with_name(dst.name + '.partial')
    if staging.exists():
        shutil.rmtree(staging)
    for path in sorted(p for p in src.rglob('*') if p.is_file()):
        rel = path.relative_to(src).as_posix()
        if exclude and any(fnmatch.fnmatch(path.name, pat) or fnmatch.fnmatch(rel, pat) for pat in exclude):
            continue
        copy_file_throttled(path, staging / path.relative_to(src), **kwargs)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(staging, dst)


def copy_tree_fast(src, dst) -> None:
    """Plain copy for Drive -> local restores (reads do not grow the Drive cache)."""
    src, dst = Path(src), Path(dst)
    staging = dst.with_name(dst.name + '.partial')
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(src, staging)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(staging, dst)


def sync_file(src, dst) -> None:
    """Refresh a small file on Drive (a JSONL journal, a stage log) from its local copy.

    Normally temp file + rename, so readers never see a half-written file. The rclone mount can refuse that
    rename with ENOENT while it is still uploading the temp file (seen on a G4 VM for a growing vLLM log); the
    file is then copied over the old one directly: it is only a mirror, the local copy stays complete.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f'.{os.getpid()}.{threading.get_ident()}.sync')
    shutil.copyfile(src, tmp)
    try:
        _replace_with_retry(tmp, dst)
    except FileNotFoundError:
        shutil.copyfile(src, dst)
        try:
            tmp.unlink()
        except OSError:
            pass


# ----------------------------------------------------------------------------- heartbeat

class Heartbeat(threading.Thread):
    """Prints `status()` every `interval` seconds so remote shells see progress and never idle out."""

    def __init__(self, status, interval: float = 120.0):
        super().__init__(daemon=True)
        self.status, self.interval = status, interval
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.wait(self.interval):
            try:
                log('HEARTBEAT', self.status())
            except Exception as error:  # a broken status line must never kill the job
                log('HEARTBEAT status error:', type(error).__name__, error)

    def stop(self) -> None:
        self.stopped.set()


def gpu_profile() -> dict:
    """Name / memory / capability of GPU 0 and recommended vLLM settings; {} without CUDA."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / GIB
    name = props.name
    major, minor = torch.cuda.get_device_capability(0)
    if 'H100' in name or 'H200' in name:
        kind = 'h100'
    elif major == 12 or 'RTX PRO 6000' in name or 'Blackwell' in name:
        kind = 'g4'
    elif 'A100' in name:
        kind = 'a100'
    else:
        kind = 'generic'
    return {'name': name, 'kind': kind, 'total_gib': round(total, 1), 'capability': f'{major}.{minor}',
            'vllm_max_num_seqs': 256 if total >= 70 else 96, 'vllm_gpu_memory_utilization': 0.90}
