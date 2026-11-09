"""Resumable checkpoints across disposable Colab VMs: save to local SSD, mirror to Drive in the background.

Layout per run:
    <local>/runs/<run>/checkpoint-N/          written by the HF Trainer (fast SSD)
    <drive>/runs/<run>/ckpt_mirror/checkpoint-N/ + MIRROR_COMPLETE.json   (survives the VM)

On start, `resolve_resume()` picks the newest complete checkpoint: local if present, otherwise the
newest Drive mirror (copied back to local first). A mirror without MIRROR_COMPLETE.json is ignored.
Local disk holds at most one checkpoint plus the one being written: before a save we wait for the
previous mirror and delete older local checkpoints when free space would not fit another one.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path

from experiments.qfft_v93.common import (GIB, copy_tree_fast, copy_tree_throttled, free_bytes, human_bytes, log,
                                         now_iso, read_json, tree_bytes, write_json)

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object

_STEP = re.compile(r'^checkpoint-(\d+)$')
SAVED_MARK = 'P1_SAVED.json'
MIRROR_MARK = 'MIRROR_COMPLETE.json'


def _checkpoints(folder: Path, marker: str):
    out = []
    if folder.exists():
        for path in folder.iterdir():
            match = _STEP.match(path.name)
            if match and path.is_dir() and (path / marker).exists() and (path / 'trainer_state.json').exists():
                out.append((int(match.group(1)), path))
    return sorted(out)


def resolve_resume(local_dir: Path, drive_dir: Path):
    """Return a local checkpoint path to resume from, restoring from the Drive mirror if needed."""
    local = _checkpoints(Path(local_dir), SAVED_MARK)
    mirror = _checkpoints(Path(drive_dir) / 'ckpt_mirror', MIRROR_MARK)
    if local and (not mirror or local[-1][0] >= mirror[-1][0]):
        log(f'Resuming from local {local[-1][1]}')
        return str(local[-1][1])
    if mirror:
        step, src = mirror[-1]
        dst = Path(local_dir) / src.name
        log(f'Restoring checkpoint-{step} from Drive ({human_bytes(tree_bytes(src))}) ...')
        copy_tree_fast(src, dst)
        write_json(dst / SAVED_MARK, {'restored_from': str(src), 'at': now_iso()})
        log(f'Resuming from restored {dst}')
        return str(dst)
    return None


class CheckpointMirror(TrainerCallback):
    def __init__(self, local_dir: Path, drive_dir: Path, every: int = 1, keep: int = 2,
                 watch_path: str = '/content', min_free_gib: float = 24, mirror_final: bool = False):
        self.local_dir, self.mirror_dir = Path(local_dir), Path(drive_dir) / 'ckpt_mirror'
        self.every, self.keep = max(1, every), max(1, keep)
        self.mirror_final = mirror_final  # the final model export is the deliverable, not the last ckpt
        self.watch_path, self.min_free = watch_path, int(min_free_gib * GIB)
        self.saves, self.thread, self.error = 0, None, None
        self.lock = threading.Lock()
        self._save_started = None  # set when the Trainer is about to save; on_save reports the duration

    # -- before a save: make room ------------------------------------------------------------------
    def _before_save(self, step: int) -> None:
        older = [p for s, p in _checkpoints(self.local_dir, SAVED_MARK) if s < step]
        if not older:
            return
        newest = older[-1]
        need = tree_bytes(newest) * 1.15 + self.min_free
        if free_bytes(self.local_dir) < need:
            self.wait()  # never delete a checkpoint whose Drive mirror is still being written
            for p in older:
                log(f'Freeing local disk before saving step {step}: removing {p.name}')
                shutil.rmtree(p, ignore_errors=True)

    def on_step_end(self, args, state, control, **kwargs):
        if control.should_save:
            self._before_save(state.global_step)
            self._save_started = time.monotonic()

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_save:
            self._before_save(state.global_step)
            self._save_started = time.monotonic()

    def _record_save(self, ckpt: Path, step: int) -> None:
        """Local save duration and size: decides how long a 122 GB full-FT checkpoint costs per save."""
        seconds = time.monotonic() - self._save_started if self._save_started is not None else None
        self._save_started = None
        size = tree_bytes(ckpt)
        record = {'step': step, 'at': now_iso(), 'seconds': round(seconds, 1) if seconds is not None else None,
                  'bytes': size, 'gib': round(size / GIB, 2),
                  'free_gib_after': round(free_bytes(self.local_dir) / GIB, 1)}
        log(f"Saved {ckpt.name} locally: {human_bytes(size)} in {record['seconds']} s, "
            f"{record['free_gib_after']} GiB free")
        try:
            self.mirror_dir.parent.mkdir(parents=True, exist_ok=True)
            with (self.mirror_dir.parent / 'saves.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
        except OSError as error:
            log('WARNING could not record the save:', error)

    # -- after a save: mark, mirror, prune ---------------------------------------------------------
    def on_save(self, args, state, control, **kwargs):
        ckpt = self.local_dir / f'checkpoint-{state.global_step}'
        if not ckpt.exists():
            return
        write_json(ckpt / SAVED_MARK, {'step': state.global_step, 'epoch': state.epoch, 'at': now_iso()})
        self._record_save(ckpt, state.global_step)
        self.saves += 1
        final = state.global_step >= state.max_steps
        if (self.saves % self.every == 0 and not final) or (final and self.mirror_final):
            self.wait()
            self._busy = ckpt
            self.thread = threading.Thread(target=self._mirror, args=(ckpt,), daemon=True)
            self.thread.start()
        else:
            self._prune_local()

    def _prune_local(self) -> None:
        """Keep only the newest local checkpoint and the one currently being mirrored."""
        with self.lock:
            local = _checkpoints(self.local_dir, SAVED_MARK)
            keep = {local[-1][1]} if local else set()
            if self._busy is not None:
                keep.add(self._busy)
            for _, p in local:
                if p not in keep:
                    shutil.rmtree(p, ignore_errors=True)

    _busy = None

    def _mirror(self, ckpt: Path) -> None:
        try:
            dst = self.mirror_dir / ckpt.name
            size = tree_bytes(ckpt)
            log(f'Mirroring {ckpt.name} to Drive ({human_bytes(size)}) in the background')
            copy_tree_throttled(ckpt, dst, watch_path=self.watch_path, min_free=self.min_free)
            write_json(dst / MIRROR_MARK, {'bytes': size, 'files': sum(1 for p in dst.rglob('*') if p.is_file()),
                                           'at': now_iso()})
            log(f'Mirror complete: {dst}')
            mirrors = _checkpoints(self.mirror_dir, MIRROR_MARK)
            for _, old in mirrors[:-self.keep]:
                shutil.rmtree(old, ignore_errors=True)
            for partial in self.mirror_dir.glob('*.partial'):
                if partial.name != dst.name + '.partial':
                    shutil.rmtree(partial, ignore_errors=True)
        except Exception as error:  # training continues; the previous mirror stays valid
            self.error = f'{type(error).__name__}: {error}'
            log('WARNING mirror failed:', self.error)
        finally:
            self._busy = None
            self._prune_local()

    def wait(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            log('Waiting for the previous checkpoint mirror to finish ...')
            self.thread.join()
        self.thread = None

    def on_train_end(self, args, state, control, **kwargs):
        self.wait()


def latest_mirror_step(drive_dir: Path):
    mirrors = _checkpoints(Path(drive_dir) / 'ckpt_mirror', MIRROR_MARK)
    return mirrors[-1][0] if mirrors else None


def prune_all_mirrors(drive_dir: Path) -> None:
    for _, p in _checkpoints(Path(drive_dir) / 'ckpt_mirror', MIRROR_MARK):
        shutil.rmtree(p, ignore_errors=True)


def read_state(path: Path) -> dict:
    return read_json(Path(path) / 'trainer_state.json')
