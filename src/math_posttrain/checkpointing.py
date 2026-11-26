"""Resumable checkpoints across disposable Colab VMs: save to local SSD, mirror to Drive in the background.

Layout per run:
    <local>/runs/<run>/checkpoint-N/          written by the HF Trainer (fast SSD)
    <drive>/runs/<run>/ckpt_mirror/checkpoint-N/ + MIRROR_COMPLETE.json   (survives the VM)

On start, `resolve_resume()` uses the newest complete local checkpoint, then tries Drive.
Drive mirrors omit the optimizer by default and are usable only for weight initialization. Legacy
mirrors are classified using the files they actually contain.
Local disk holds at most one checkpoint plus the one being written: before a save we wait for the
previous mirror and delete older local checkpoints when free space would not fit another one.
"""
from __future__ import annotations

import re
import shutil
import threading
import time
from pathlib import Path

from math_posttrain import drive_io
from math_posttrain.common import (GIB, free_bytes, human_bytes, log, now_iso, read_json, read_jsonl, sha256_file,
                                         tree_bytes, write_json, write_jsonl)

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object

_STEP = re.compile(r'^checkpoint-(\d+)$')
SAVED_MARK = 'P1_SAVED.json'
MIRROR_MARK = 'MIRROR_COMPLETE.json'
FULL_STATE = 'full_state'
WEIGHTS_ONLY = 'weights_only'


def _checkpoints(folder: Path, marker: str):
    out = []
    if folder.exists():
        for path in folder.iterdir():
            match = _STEP.match(path.name)
            if match and path.is_dir() and (path / marker).exists() and (path / 'trainer_state.json').exists():
                out.append((int(match.group(1)), path))
    return sorted(out)


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _has_training_state(path: Path) -> bool:
    """Recognize the one-GPU ZeRO-2 and ordinary Trainer checkpoint layouts."""
    if not any(_nonempty(p) for p in path.glob('rng_state*.pth')):
        return False
    if (path / 'latest').exists() or any(path.glob('global_step*')):
        latest = path / 'latest'
        if not _nonempty(latest):
            return False
        tag = latest.read_text(encoding='utf-8').strip()
        if not re.fullmatch(r'global_step\d+', tag):
            return False
        # DeepSpeed stores the LR scheduler in the model-state shard when the engine owns it.
        shard_dir = path / tag
        return all(any(_nonempty(p) for p in shard_dir.glob(pattern))
                   for pattern in ('*_model_states.pt', '*_optim_states.pt'))
    weights = ('model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json',
               'pytorch_model.bin.index.json', 'adapter_model.safetensors', 'adapter_model.bin')
    return (all(_nonempty(path / name) for name in ('optimizer.pt', 'scheduler.pt'))
            and any(_nonempty(path / name) for name in weights))


def _resumable(path: Path, marker: str) -> bool:
    try:
        metadata = read_json(path / marker)
        # Old markers had no artifact_type. Inspect those files as well as newly labeled full mirrors.
        if (not isinstance(metadata, dict) or metadata.get('artifact_type', FULL_STATE) != FULL_STATE
                or not _has_training_state(path)
                or read_state(path)['global_step'] != int(_STEP.fullmatch(path.name)[1])):
            return False
        manifest = metadata.get('file_manifest')
        return manifest is None or (bool(manifest) and all(
            not Path(name).is_absolute() and '..' not in Path(name).parts
            and (path / name).stat().st_size == record['bytes']
            for name, record in manifest.items()
        ))
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return False


def resolve_resume(local_dir: Path, drive_dir: Path, *, exact_drive_step: int | None = None):
    """Restore a full training state or stop clearly if only incomplete checkpoints exist."""
    local = _checkpoints(Path(local_dir), SAVED_MARK)
    for _, path in reversed(local if exact_drive_step is None else []):
        if _resumable(path, SAVED_MARK):
            log(f'Resuming from local {path}')
            return str(path)
    mirror = _checkpoints(Path(drive_dir) / 'ckpt_mirror', MIRROR_MARK)
    for step, src in reversed(mirror):
        if exact_drive_step is not None and step != exact_drive_step:
            continue
        if not _resumable(src, MIRROR_MARK):
            continue
        dst = Path(local_dir) / src.name
        staged = Path(local_dir) / '.restore' / src.name
        staged.parent.mkdir(parents=True, exist_ok=True)
        log(f'Restoring checkpoint-{step} from Drive ({human_bytes(tree_bytes(src))}) ...')
        try:
            metadata = read_json(src / MIRROR_MARK)
            if exact_drive_step is not None and not metadata.get('file_manifest'):
                raise RuntimeError('an imported checkpoint requires the full checksum manifest')
            drive_io.download_tree(src, staged)
            if not _resumable(staged, MIRROR_MARK):
                raise RuntimeError('missing full training state')
            for name, record in metadata.get('file_manifest', {}).items():
                if sha256_file(staged / name) != record['sha256']:
                    raise RuntimeError(f'checksum mismatch for {name}')
        except (OSError, ValueError, RuntimeError) as error:
            shutil.rmtree(staged, ignore_errors=True)
            log(f'WARNING: cannot restore checkpoint-{step}: {error}')
            continue
        if dst.exists():
            shutil.rmtree(dst)
        staged.replace(dst)
        write_json(dst / SAVED_MARK, {'restored_from': str(src), 'at': now_iso()})
        log(f'Resuming from restored {dst}')
        return str(dst)
    if local or mirror:
        raise RuntimeError('No full-state checkpoint is available for exact resume. '
                           'Drive weights-only mirrors are initialization artifacts. '
                           'Use the original VM with a complete local checkpoint or start a new run.')
    return None


def latest_warm_start_weights(drive_dir: Path):
    """Find the newest published light mirror with a usable model and matching step."""
    for step, path in reversed(_checkpoints(drive_dir / 'ckpt_mirror', MIRROR_MARK)):
        try:
            marker = read_json(path / MIRROR_MARK)
            artifact_type = marker.get('artifact_type')
            if artifact_type != WEIGHTS_ONLY and (artifact_type is not None or _resumable(path, MIRROR_MARK)):
                continue
            state = read_state(path)
            if state['global_step'] != step:
                continue
            names = ['config.json']
            if _nonempty(path / 'generation_config.json'):
                names.append('generation_config.json')
            index = path / 'model.safetensors.index.json'
            if index.is_file():
                shards = set(read_json(index)['weight_map'].values())
                if not shards or not all(re.fullmatch(r'[\w.-]+\.safetensors', name) for name in shards):
                    continue
                names += [index.name, *sorted(shards)]
            else:
                names.append('model.safetensors')
            if all(_nonempty(path / name) for name in names):
                return step, path, state, {name: (path / name).stat().st_size for name in names}
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            continue
    raise RuntimeError(f'No completed weights-only checkpoint found in {drive_dir / "ckpt_mirror"}')


class CheckpointMirror(TrainerCallback):
    def __init__(self, local_dir: Path, drive_dir: Path, every: int = 1, keep: int = 2,
                 watch_path: str = '/content', min_free_gib: float = 24, mirror_final: bool = False,
                 mirror_optimizer_state: bool = False, retain_steps=()):
        self.local_dir, self.mirror_dir = Path(local_dir), Path(drive_dir) / 'ckpt_mirror'
        self.every, self.keep = max(1, every), max(1, keep)
        self.retain_steps = frozenset(retain_steps)
        if any(type(step) is not int or step <= 0 for step in self.retain_steps):
            raise ValueError('Retained checkpoint steps must be positive integers')
        if self.retain_steps and not mirror_optimizer_state:
            raise ValueError('Retained milestones require full optimizer-state mirrors')
        self.mirror_final = mirror_final  # the final model export is the deliverable, not the last ckpt
        self.mirror_optimizer_state = mirror_optimizer_state
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
            if self.mirror_optimizer_state and not self.has_full_mirror(int(_STEP.fullmatch(newest.name)[1])):
                raise RuntimeError(f'Insufficient local disk; retaining {newest.name} without a confirmed full mirror')
            for p in older:
                step_number = int(_STEP.fullmatch(p.name)[1])
                if step_number in self.retain_steps and not self.has_full_mirror(step_number):
                    continue
                log(f'Freeing local disk before saving step {step}: removing {p.name}')
                shutil.rmtree(p, ignore_errors=True)
            if free_bytes(self.local_dir) < need:
                raise RuntimeError('Insufficient local disk; unmirrored validation checkpoints remain protected')

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
        try:  # rewrite (tiny file) instead of appending: appends through a Drive mount upload late
            path = self.mirror_dir.parent / 'saves.jsonl'
            write_jsonl(path, read_jsonl(path) + [record])
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
            # A failed upload must not erase an odd validation step such as 175 on the next save.
            keep.update(path for step, path in local if step in self.retain_steps)
            if self._busy is not None:
                keep.add(self._busy)
            for _, p in local:
                if p not in keep:
                    shutil.rmtree(p, ignore_errors=True)

    _busy = None

    def _mirror(self, ckpt: Path) -> None:
        try:
            dst = self.mirror_dir / ckpt.name
            exclude = None if self.mirror_optimizer_state else ['*optim*']
            artifact_type = FULL_STATE if self.mirror_optimizer_state else WEIGHTS_ONLY
            if self.mirror_optimizer_state and self.has_full_mirror(int(_STEP.fullmatch(ckpt.name)[1])):
                log(f'Keeping the existing verified full-state mirror for {ckpt.name}')
                return
            if artifact_type == WEIGHTS_ONLY and _resumable(dst, MIRROR_MARK):
                log(f'Keeping the existing full-state mirror for {ckpt.name}')
                return
            log(f'Mirroring {ckpt.name} to Drive ({artifact_type}) in the background')
            if self.mirror_optimizer_state and not _resumable(ckpt, SAVED_MARK):
                raise RuntimeError(f'{ckpt.name} is missing full training state; cannot publish a resumable mirror')
            manifest = None
            if self.mirror_optimizer_state:
                manifest = {p.relative_to(ckpt).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                            for p in sorted(ckpt.rglob('*')) if p.is_file()}
            files = drive_io.upload_tree(ckpt, dst, watch_path=self.watch_path, min_free=self.min_free,
                                         exclude=exclude, verify_full=self.mirror_optimizer_state)
            marker = {'artifact_type': artifact_type, 'bytes': sum(files.values()),
                      'files': len(files), 'at': now_iso()}
            if manifest is not None:
                if files != {name: entry['bytes'] for name, entry in manifest.items()}:
                    raise RuntimeError('Full checkpoint file list changed during upload')
                if not drive_io.rclone_remote() and any(
                    sha256_file(dst / name) != entry['sha256'] for name, entry in manifest.items()
                ):
                    raise RuntimeError('Full checkpoint copy checksum mismatch')
                marker['file_manifest'] = manifest
            drive_io.put_json(dst / MIRROR_MARK, marker)
            self.error = None
            log(f'Mirror complete: {dst}')
            mirrors = _checkpoints(self.mirror_dir, MIRROR_MARK)
            full, light = [], []
            for _, path in mirrors:
                (full if _resumable(path, MIRROR_MARK) else light).append(path)
            removable = [old for old in full[:-self.keep]
                         if int(_STEP.fullmatch(old.name)[1]) not in self.retain_steps]
            for old in removable + light[:-self.keep]:
                drive_io.purge(old)
            for partial in self.mirror_dir.glob('*.partial'):
                if partial.name != dst.name + '.partial':
                    drive_io.purge(partial)
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

    def has_full_mirror(self, step: int) -> bool:
        return _resumable(self.mirror_dir / f'checkpoint-{step}', MIRROR_MARK)


def latest_mirror_step(drive_dir: Path):
    mirrors = _checkpoints(Path(drive_dir) / 'ckpt_mirror', MIRROR_MARK)
    return mirrors[-1][0] if mirrors else None


def prune_all_mirrors(drive_dir: Path) -> None:
    for _, p in _checkpoints(Path(drive_dir) / 'ckpt_mirror', MIRROR_MARK):
        drive_io.purge(p)


def read_state(path: Path) -> dict:
    return read_json(Path(path) / 'trainer_state.json')
