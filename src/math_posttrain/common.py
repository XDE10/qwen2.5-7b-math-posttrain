"""Shared helpers: portable workspace paths, atomic JSON/JSONL IO, hashing, config loading, and GPU profiling.

Standard library only. Importing this module never touches the network or a GPU.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parents[1]
REQUIREMENTS_DIR = ROOT / 'requirements'

BASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
BASE_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
GIB = 2 ** 30


@dataclass(frozen=True)
class Workspace:
    """Resolved local workspace directories (all overridable via environment variables)."""

    config_dir: Path
    data_dir: Path
    output_dir: Path

    @property
    def benchmarks(self) -> Path:
        return self.data_dir / 'benchmarks'

    @property
    def train_data(self) -> Path:
        return self.data_dir / 'train'

    @property
    def runs(self) -> Path:
        return self.output_dir / 'runs'

    @property
    def evals(self) -> Path:
        return self.output_dir / 'evals'


def get_config_dir() -> Path:
    env = os.environ.get('MATH_POSTTRAIN_CONFIG_DIR')
    if env:
        return Path(env)
    return ROOT / 'configs' if (ROOT / 'configs').is_dir() else Path.cwd() / 'configs'


def workspace() -> Workspace:
    data_dir = Path(os.environ.get('MATH_POSTTRAIN_DATA_DIR') or (ROOT / 'data'))
    output_dir = Path(os.environ.get('MATH_POSTTRAIN_OUTPUT_DIR') or (ROOT / 'outputs'))
    return Workspace(config_dir=get_config_dir(), data_dir=data_dir, output_dir=output_dir)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def log(*parts) -> None:
    print(f'[{datetime.now():%m-%d %H:%M:%S}]', *parts, flush=True)


# ----------------------------------------------------------------------------- JSON / JSONL IO

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _replace_with_retry(tmp: Path, path: Path) -> None:
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
    """Read JSONL; optionally tolerate a single torn trailing line from an interrupted write."""
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
    """Thread-safe append-only JSONL writer with periodic fsync and torn-tail repair."""

    def __init__(self, path, fsync_seconds: float = 2.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_tail()
        self.handle = self.path.open('a', encoding='utf-8', newline='\n')
        self.fsync_seconds = fsync_seconds
        self.last_sync = time.monotonic()
        self.lock = threading.Lock()

    def _repair_tail(self) -> None:
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

    def close(self) -> None:
        with self.lock:
            if not self.handle.closed:
                self.handle.flush()
                os.fsync(self.handle.fileno())
                self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------------------------- hashing & configs

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


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path) -> dict:
    """Load a JSON config (relative to CONFIG_DIR when not found directly) with optional `base` inheritance."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = get_config_dir() / path
    config = read_json(path)
    base = config.pop('base', None)
    if base:
        config = _deep_merge(load_config(path.parent / base), config)
    return config


def latest_checkpoint(run_dir: Path) -> str | None:
    """Return the highest-step `checkpoint-*` directory under `run_dir`, or None."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    ckpts = []
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith('checkpoint-'):
            suffix = p.name.split('-', 1)[1]
            if suffix.isdigit():
                ckpts.append((int(suffix), p))
    return str(max(ckpts)[1]) if ckpts else None


def gpu_profile() -> dict:
    """Hardware info and sensible vLLM defaults for GPU 0; empty dict without CUDA."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / GIB
    major, minor = torch.cuda.get_device_capability(0)
    return {
        'name': props.name,
        'total_gib': round(total, 1),
        'capability': f'{major}.{minor}',
        'vllm_max_num_seqs': 256 if total >= 70 else 64,
        'vllm_gpu_memory_utilization': 0.90,
    }
