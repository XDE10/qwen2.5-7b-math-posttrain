"""Bulk transfers between the VM's local SSD and Google Drive.

Two ways to reach Drive on a Colab VM (launch.sh --drive):

* rclone (default): `rclone mount mydrive: /content/mydrive --vfs-cache-mode writes`. Small files (JSON
  markers, status, logs) go through the mount. Large trees go through the rclone CLI instead: the mount
  uploads a file only after it is closed and first keeps the whole file in its local write cache, and a
  full-FT checkpoint contains one ~91 GB optimizer file that cannot be cached next to the 122 GB local
  checkpoint on a ~235 GB disk. `rclone copy` streams the upload instead. Superseded checkpoints are
  deleted permanently (--drive-use-trash=false), otherwise every mirror (122 GB each) would sit in the
  Drive trash for 30 days.
* colab: Colab's own Drive mount at /content/drive/MyDrive with the throttled copies from common.py.

The rclone CLI is used when P1_RCLONE_REMOTE is set (e.g. `mydrive:`); P1_DRIVE_ROOT is the mount point of
that remote. After every CLI upload we wait until the mount shows the new files, so later small writes
through the mount never create a second folder with the same name (Drive allows duplicate names).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from experiments.qfft_v93.common import copy_tree_fast, copy_tree_throttled, human_bytes, log, write_json

UPLOAD_FLAGS = ['--transfers', '4', '--checkers', '8', '--drive-chunk-size', '128M', '--retries', '10',
                '--low-level-retries', '20', '--stats', '5m', '--stats-one-line', '--stats-log-level', 'NOTICE']
DOWNLOAD_FLAGS = ['--transfers', '8', '--checkers', '8', '--multi-thread-streams', '4', '--retries', '10',
                  '--low-level-retries', '20', '--stats', '5m', '--stats-one-line', '--stats-log-level', 'NOTICE']


def rclone_remote() -> str | None:
    value = os.environ.get('P1_RCLONE_REMOTE', '').strip()
    return value or None


def drive_root() -> Path:
    return Path(os.environ.get('P1_DRIVE_ROOT', '/content/drive/MyDrive'))


def remote_path(path) -> str:
    """`mydrive:runs/...` for a path under the Drive mount point."""
    relative = Path(path).relative_to(drive_root()).as_posix()
    base = rclone_remote()
    if relative in ('', '.'):
        return base
    return base + relative if base.endswith((':', '/')) else base + '/' + relative


def _rclone(*args, timeout=None, check=True):
    started = time.monotonic()
    result = subprocess.run(['rclone', *args], capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f'rclone {args[0]} failed ({result.returncode}): {result.stderr.strip()[-1200:]}')
    if result.stderr.strip():
        tail = result.stderr.strip().splitlines()[-1]
        log(f'rclone {args[0]} ({time.monotonic() - started:.0f} s): {tail[-300:]}')
    return result


def _files(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.stat().st_size for p in folder.rglob('*') if p.is_file()}


def wait_visible(path, expect: dict | None = None, timeout: float = 900, gone: bool = False) -> bool:
    """Poll the mount until `path` (and every file in `expect` with its size) is visible, or gone."""
    path = Path(path)
    deadline = time.monotonic() + timeout
    while True:
        try:
            if gone:
                ok = not path.exists()
            else:
                ok = path.exists() and all((path / rel).is_file() and (path / rel).stat().st_size == size
                                           for rel, size in (expect or {}).items())
        except OSError:
            ok = False
        if ok:
            return True
        if time.monotonic() >= deadline:
            log(f'WARNING: {path} is {"still" if gone else "not yet"} visible through the Drive mount after '
                f'{timeout:.0f} s')
            return False
        time.sleep(5)


def upload_tree(src, dst, **throttle) -> None:
    """Publish local directory `src` as Drive directory `dst` (replacing it); never leaves a half tree at `dst`."""
    src, dst = Path(src), Path(dst)
    if not rclone_remote():
        copy_tree_throttled(src, dst, **throttle)
        return
    files = _files(src)
    size = sum(files.values())
    partial = dst.with_name(dst.name + '.partial')
    started = time.monotonic()
    _rclone('copy', str(src), remote_path(partial), *UPLOAD_FLAGS)
    purge(dst, wait=False)
    _rclone('moveto', remote_path(partial), remote_path(dst))
    seconds = time.monotonic() - started
    log(f'Uploaded {src.name} to {dst} ({human_bytes(size)} in {seconds:.0f} s, '
        f'{size / max(seconds, 1e-6) / 2 ** 20:.0f} MiB/s)')
    wait_visible(dst, files)


def upload_file(src, dst) -> None:
    src, dst = Path(src), Path(dst)
    if not rclone_remote():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + '.partial')
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return
    _rclone('copyto', str(src), remote_path(dst), *UPLOAD_FLAGS)
    wait_visible(dst.parent, {dst.name: src.stat().st_size})


def put_json(path, value) -> None:
    """A small JSON file inside a tree that was just uploaded with the CLI (marker files)."""
    if not rclone_remote():
        write_json(path, value)
        return
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(path).name
        write_json(local, value)
        upload_file(local, path)


def download_tree(src, dst, atomic: bool = True) -> None:
    """Copy Drive directory `src` to local `dst` (via `dst.partial` when atomic)."""
    src, dst = Path(src), Path(dst)
    if not rclone_remote():
        copy_tree_fast(src, dst)
        return
    target = dst.with_name(dst.name + '.partial') if atomic else dst
    if atomic and target.exists():
        shutil.rmtree(target)
    started = time.monotonic()
    _rclone('copy', remote_path(src), str(target), *DOWNLOAD_FLAGS, *([] if atomic else ['--size-only']))
    if atomic:
        if dst.exists():
            shutil.rmtree(dst)
        os.replace(target, dst)
    log(f'Downloaded {src} in {time.monotonic() - started:.0f} s')


def purge(path, wait: bool = True) -> None:
    """Delete a Drive directory. With rclone: permanently (not to the trash)."""
    path = Path(path)
    if not rclone_remote():
        if path.exists():
            shutil.rmtree(path)
        return
    result = _rclone('purge', remote_path(path), '--drive-use-trash=false', check=False)
    if result.returncode != 0 and 'not found' not in result.stderr.lower():
        raise RuntimeError(f'rclone purge failed ({result.returncode}): {result.stderr.strip()[-800:]}')
    if wait:
        wait_visible(path, gone=True, timeout=300)
