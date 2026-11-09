"""Online SwanLab logging that survives resume (same experiment id), plus a local metrics.jsonl and status.json.

The API key comes from the SWANLAB_API_KEY environment variable (colab/remote.py reads it from the
private Drive credential file); it is never written to configs, logs or metrics.
"""
from __future__ import annotations

import inspect
import math
import os
import time
from pathlib import Path

from experiments.qfft_v93.common import JsonlAppender, log, now_iso, read_json, write_json

try:
    from transformers import TrainerCallback
except ImportError:  # data-only environments
    TrainerCallback = object


class SwanLab:
    def __init__(self, run_dir: Path, name: str, config: dict, tags: list, project: str = 'project1-7b-math',
                 enabled: bool = True, log_dir: Path | None = None):
        """run_dir (Drive) keeps the experiment id for resume; log_dir (local SSD) holds SDK files."""
        self.run_dir, self.enabled, self.run = Path(run_dir), enabled, None
        self.log_dir = Path(log_dir) if log_dir else self.run_dir / 'swanlog'
        self.module = None
        if not enabled:
            return
        import swanlab
        key = os.environ.get('SWANLAB_API_KEY')
        if not key:
            raise RuntimeError('SWANLAB_API_KEY missing; online logging is required (pass --no-swanlab for smoke tests)')
        if swanlab.login(api_key=key, save=False) is False:
            raise RuntimeError('SwanLab login failed; no silent offline fallback')
        state_path = self.run_dir / 'swanlab.json'
        previous = read_json(state_path) if state_path.exists() else {}
        kwargs = {'project': project, 'config': config, 'tags': tags, 'reinit': True}
        # SwanLab >= 0.10 uses name/online/log_dir; older SDKs use experiment_name/cloud/logdir.
        if 'name' in inspect.signature(swanlab.init).parameters:
            kwargs.update(name=name, mode='online', log_dir=str(self.log_dir))
        else:
            kwargs.update(experiment_name=name, mode='cloud', logdir=str(self.log_dir))
        if previous.get('id'):
            kwargs.update(id=previous['id'], resume='allow')
        self.run = swanlab.init(**kwargs)
        self.module = swanlab
        run_id = getattr(self.run, 'id', None) or previous.get('id')
        url = getattr(self.run, 'url', None)
        write_json(state_path, {'id': run_id, 'url': url if isinstance(url, str) else previous.get('url'),
                                'project': project, 'name': name, 'updated': now_iso()})
        log('SwanLab run', run_id, url if isinstance(url, str) else '')

    def log(self, values: dict, step: int) -> None:
        if not self.enabled:
            return
        clean = {}
        for key, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                clean[key] = number
        if clean:
            self.module.log(clean, step=int(step))

    def finish(self) -> None:
        if self.enabled and self.module is not None:
            self.module.finish()
            self.module = None


def device_metrics() -> dict:
    out = {}
    try:
        import torch
        if torch.cuda.is_available():
            out.update({'gpu/allocated_gib': torch.cuda.memory_allocated() / 2 ** 30,
                        'gpu/peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
                        'gpu/peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30})
    except Exception:
        pass
    try:
        import psutil
        out['host/rss_gib'] = psutil.Process().memory_info().rss / 2 ** 30
        out['host/available_gib'] = psutil.virtual_memory().available / 2 ** 30
    except Exception:
        pass
    return out


class ProgressCallback(TrainerCallback):
    """Writes metrics.jsonl and status.json (local + Drive) on every log; feeds SwanLab."""

    def __init__(self, local_dir: Path, drive_dir: Path, swan: SwanLab, extra: dict | None = None):
        self.local_dir, self.drive_dir, self.swan = Path(local_dir), Path(drive_dir), swan
        # Journal on the local SSD, Drive copy replaced every 30 s (see JsonlAppender `mirror`).
        self.metrics = JsonlAppender(self.local_dir / 'metrics.jsonl', mirror=self.drive_dir / 'metrics.jsonl',
                                     mirror_seconds=30)
        self.extra = extra or {}
        self.started = None
        self.last = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()
        self.last = (time.monotonic(), state.global_step, getattr(state, 'num_input_tokens_seen', 0) or 0)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = dict(logs or {})
        t, step = time.monotonic(), state.global_step
        tokens = getattr(state, 'num_input_tokens_seen', 0) or 0
        if self.last and step > self.last[1]:
            dt = t - self.last[0]
            logs['time/seconds_per_step'] = dt / (step - self.last[1])
            if tokens > self.last[2]:
                logs['throughput/tokens_per_second'] = (tokens - self.last[2]) / dt
            remaining = max(0, state.max_steps - step)
            logs['time/eta_hours'] = remaining * logs['time/seconds_per_step'] / 3600
        self.last = (t, step, tokens)
        logs.update(device_metrics())
        logs['train/epoch'] = state.epoch or 0
        record = {'step': step, 'time': now_iso(), **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
        self.metrics.write(record)
        self.swan.log(logs, step)
        status = {'state': 'training', 'step': step, 'max_steps': state.max_steps, 'epoch': state.epoch,
                  'updated': now_iso(), 'last_log': record, **self.extra}
        write_json(self.drive_dir / 'status.json', status)
        if 'loss' in logs:
            log(f"step {step}/{state.max_steps} epoch {state.epoch:.2f} loss {logs['loss']:.4f} "
                f"lr {logs.get('learning_rate', 0):.2e} s/step {logs.get('time/seconds_per_step', 0):.1f} "
                f"eta {logs.get('time/eta_hours', 0):.2f}h peakGPU {logs.get('gpu/peak_allocated_gib', 0):.1f}GiB")

    def on_train_end(self, args, state, control, **kwargs):
        self.metrics.close()
