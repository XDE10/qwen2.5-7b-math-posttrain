"""Stage runner executed on the Colab VM (inside the kernel started by `colab exec`). Standard library only.

Stages (all idempotent; re-running resumes or skips completed work):

    benchmarks                  fetch the 5 pinned benchmarks on the VM straight to Drive (no local data needed)
    sync-data                   copy the uploaded data bundle to Drive and verify manifest hashes
    models:<a,b,...>            download registry models to Drive once (cheap on a CPU VM)
    smoke-train                 2 full-FT steps on rows that fill the 32K window: s/step, peak GPU/host memory
    smoke-eval                  24 short generations of the base model + scoring
    smoke-grpo                  2 GRPO steps at the real batch shape with completions forced to 8192 tokens
    train:<config>              full SFT/QFFT run, e.g. train:qfft_ds  (configs/train/<config>.json)
    warm-start:<config>:<source>:<destination>   train remaining steps from published weights
    eval:<model>[:<suite>]      generate + score, e.g. eval:base:full, eval:run:qfft_ds_v93:full
    grpo:<config>               LoRA-GRPO + merge, e.g. grpo:grpo_qfft_ds (configs/grpo/<config>.json)
    grpo:<config>:pause-after=<step> / grpo:<config>:resume   verify a full checkpoint across VMs
    r1-is-diagnostic            capture one fixed R1 rollout and compare importance weights
    r1-grpo-dev[:step30]        paired development evaluation of R1 and the 16K run
    r1-competition-2026         evaluate R1 and the two step-10 adapters on the frozen competition set
    r1-fura50-eval              score the fixed 50-step LoRA run against R1 on the competition suite
    score[:<tags>]              re-score saved generations and rebuild results.md
    analyze:<tags>[:<grpo_runs>] figures into Drive runs/project1/v93/figures
    status                      print stage markers, run progress, eval completeness and results.md
    smoke-report                re-print the smoke summary block from Drive (also printed after smoke stages)
    shell                       do nothing (use with --keep to inspect the VM)
    ping[:<seconds>]            write a probe file to Drive and wait (end-to-end CLI check on a CPU VM)

A stage writes runs/project1/v93/stages/<stage>.json on Drive (running + heartbeat, then complete/failed);
a second VM refuses a stage whose heartbeat is fresh, so relaunching a detached job cannot duplicate it.

Environments: `train` = Colab system Python + DeepSpeed/Liger/SwanLab (Drive wheel cache);
`infer` and `rl` = uv virtualenvs restored from a Drive snapshot (built once, then ~1 minute).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path

DRIVE = Path(os.environ.get('P1_DRIVE_ROOT', '/content/drive/MyDrive'))
LOCAL = Path(os.environ.get('P1_LOCAL_ROOT', '/content/p1'))
PROJECT = DRIVE / 'projects/project1/v93'
STAGES_DIR = DRIVE / 'runs/project1/v93/stages'
LOGS = DRIVE / 'runs/project1/v93/logs'
# A 'running' stage marker whose heartbeat is older than this belongs to a VM that no longer exists.
LEASE_STALE_S = 15 * 60
HEARTBEAT_S = 5 * 60
READ_ONLY_STAGES = {'status', 'shell', 'smoke-report'}


def fresh_smoke(*paths: Path) -> None:
    """Smoke outputs are measurements, not results: start every smoke from empty folders so the numbers
    (and the run identity / completion markers) always belong to this session."""
    for path in paths:
        if path.exists():
            shutil.rmtree(path)


def say(*parts):
    print(f'[{datetime.now():%m-%d %H:%M:%S}]', *parts, flush=True)


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def assemble_parts(path: Path) -> None:
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
    say(f'Reassembled {path.name} from {len(parts)} parts')


def host_id() -> str:
    return f"{os.environ.get('P1_SESSION') or 'manual'}@{socket.gethostname()}"


def marker_path(stage: str) -> Path:
    return STAGES_DIR / (stage.replace(':', '__').replace('/', '_').replace(',', '+') + '.json')


class StageLease:
    """Per-stage Drive marker with a heartbeat.

    launch.sh no longer stops a VM when its terminal disconnects, so the same command can be started again
    while the first VM is still working. Two VMs writing the same run directory would corrupt it; the
    lease makes the second one refuse unless the first one's heartbeat is stale (or --force is given).
    """

    def __init__(self, stage: str, code_sha: str, force: bool = False):
        self.stage, self.code_sha, self.path = stage, code_sha, marker_path(stage)
        self.started = datetime.now().isoformat()
        self.stop = threading.Event()
        try:
            current = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        except (OSError, ValueError):
            current = {}
        if current.get('state') == 'running' and current.get('host') != host_id():
            age = time.time() - float(current.get('heartbeat_ts') or 0)
            if age < LEASE_STALE_S and not force:
                raise RuntimeError(
                    f"stage {stage} is already running on {current.get('host')} (heartbeat {age / 60:.0f} min ago). "
                    'Follow it with launch.sh --attach <session>, stop that VM, or relaunch with --force.')
            say(f"Taking over stage {stage} from {current.get('host')} (heartbeat {age / 60:.0f} min ago)")
        self._beat()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _beat(self) -> None:
        write_json(self.path, {'stage': self.stage, 'state': 'running', 'started': self.started,
                               'code': self.code_sha, 'host': host_id(), 'heartbeat': datetime.now().isoformat(),
                               'heartbeat_ts': time.time()})

    def _loop(self) -> None:
        while not self.stop.wait(HEARTBEAT_S):
            try:
                self._beat()
            except Exception as error:
                say('WARNING: stage heartbeat failed:', error)

    def close(self, **state) -> None:
        self.stop.set()
        self.thread.join(timeout=120)
        write_json(self.path, {'stage': self.stage, **state, 'started': self.started, 'code': self.code_sha,
                               'host': host_id()})


class Context:
    def __init__(self, code_dir: Path, code_sha: str):
        self.code, self.sha = code_dir, code_sha
        self.pkg = code_dir / 'experiments/qfft_v93'
        self.env = dict(os.environ, PYTHONPATH=str(code_dir), P1_CODE_SHA=code_sha, P1_DRIVE_ROOT=str(DRIVE),
                        P1_LOCAL_ROOT=str(LOCAL), HF_HOME=str(LOCAL / 'hf'), TOKENIZERS_PARALLELISM='false',
                        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', DS_SKIP_CUDA_CHECK='1',
                        PYTHONUNBUFFERED='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                        VLLM_USE_FLASHINFER_SAMPLER='0')
        credential = DRIVE / '.project1_credentials/swanlab.json'
        if credential.exists():
            key = json.loads(credential.read_text(encoding='utf-8-sig')).get('SWANLAB_API_KEY')
            if isinstance(key, str) and key.strip():
                self.env['SWANLAB_API_KEY'] = key.strip()
        self.pythons = {}
        self.snapshot_threads = []

    # ------------------------------------------------------------------ process execution
    def run(self, cmd, name: str, quiet_minutes: float = 10) -> None:
        """Stream a subprocess to the notebook and to a log; never kill it on a client disconnect.

        The log is written on the VM's SSD and copied to Drive every minute and at the end: a file kept
        open on a Drive mount is only uploaded when it is closed.
        """
        log_name = f'{datetime.now():%Y%m%d-%H%M%S}_{name}.log'
        log_path, drive_log = LOCAL / 'logs' / log_name, LOGS / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        say('RUN', ' '.join(str(c) for c in cmd), '| log:', drive_log)
        synced = [time.monotonic()]

        def sync_log(force: bool = False):
            if force or time.monotonic() - synced[0] >= 60:
                copy_to_drive(log_path, drive_log)
                synced[0] = time.monotonic()
        with open(log_path, 'a', encoding='utf-8') as logf:
            proc = subprocess.Popen([str(c) for c in cmd], cwd=self.code, env=self.env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            last = [time.monotonic()]

            def watchdog():
                while proc.poll() is None:
                    time.sleep(30)
                    if time.monotonic() - last[0] > quiet_minutes * 60:
                        say(f'... {name} still running (pid {proc.pid}, no output for {quiet_minutes:.0f} min)')
                        last[0] = time.monotonic()
                    sync_log()
            threading.Thread(target=watchdog, daemon=True).start()
            try:
                for line in proc.stdout:
                    last[0] = time.monotonic()
                    print(line, end='', flush=True)
                    logf.write(line)
                    logf.flush()
                    sync_log()
                code = proc.wait()
            except KeyboardInterrupt:
                say(f'Detached from {name} (pid {proc.pid}); it keeps running. Log: {log_path}')
                raise
            finally:
                logf.flush()
                sync_log(force=True)
        if code != 0:
            raise RuntimeError(f'{name} failed with exit code {code}; see {drive_log}')

    # ------------------------------------------------------------------ environments
    def python(self, env_name: str) -> str:
        if env_name not in self.pythons:
            builders = {'train': self._train_env, 'score': self._score_env}
            self.pythons[env_name] = builders[env_name]() if env_name in builders else self._venv(env_name)
        return self.pythons[env_name]

    def _score_env(self) -> str:
        """CPU-only scoring/figures: system Python + Math-Verify + matplotlib (no vLLM download)."""
        py = sys.executable
        probe = 'import math_verify, matplotlib'
        if subprocess.run([py, '-c', probe], capture_output=True).returncode != 0:
            subprocess.run([py, '-m', 'pip', 'install', '-q', 'math-verify[antlr4_13_2]==0.9.0', 'matplotlib'],
                           check=True)
        return py

    def _train_env(self) -> str:
        py = sys.executable
        req = self.pkg / 'envs/train-extra.txt'
        probe = 'import deepspeed, liger_kernel, swanlab, torch, transformers, accelerate'
        if subprocess.run([py, '-c', probe], capture_output=True).returncode != 0:
            core = subprocess.run([py, '-c', 'import json, importlib.metadata as m; print(json.dumps({n: m.version(n) '
                                   'for n in ["torch", "transformers", "accelerate", "tokenizers", "safetensors", '
                                   '"numpy", "triton"] if __import__("importlib.util").util.find_spec(n.replace("-", "_"))}))'],
                                  capture_output=True, text=True, check=True)
            pins = json.loads(core.stdout.strip().splitlines()[-1])
            constraints = LOCAL / 'train-constraints.txt'
            constraints.parent.mkdir(parents=True, exist_ok=True)
            constraints.write_text(''.join(f'{k}=={v}\n' for k, v in pins.items()))
            wheels = PROJECT / 'envcache/wheels'
            wheels.mkdir(parents=True, exist_ok=True)
            say('Installing training extras (system torch/transformers kept):', pins)
            env = dict(self.env, DS_BUILD_OPS='0')
            # Cache only the listed packages (DeepSpeed is an sdist build); dependencies resolve normally.
            subprocess.run([py, '-m', 'pip', 'wheel', '-q', '--no-deps', '-w', str(wheels), '-r', str(req)],
                           env=env, check=False)
            subprocess.run([py, '-m', 'pip', 'install', '-q', '--find-links', str(wheels), '-c', str(constraints),
                            '-r', str(req)], env=env, check=True)
        out = subprocess.run([py, '-c', probe + '; print("TRAIN_ENV", torch.__version__, transformers.__version__, '
                              'deepspeed.__version__, torch.cuda.is_available())'],
                             env=self.env, capture_output=True, text=True, check=True)
        say(out.stdout.strip().splitlines()[-1])
        return py

    def _venv(self, name: str) -> str:
        req = self.pkg / f'envs/{name}.txt'
        # The venv links to the image's Python, so the snapshot key includes that interpreter.
        base = f'{sys.executable}|{sys.version_info[0]}.{sys.version_info[1]}'
        tag = hashlib.sha256(req.read_bytes() + base.encode()).hexdigest()[:12]
        venv = LOCAL / 'venvs' / name
        py = venv / 'bin/python'
        ready = venv / '.p1_ready'
        probe = {'infer': 'import vllm, math_verify, transformers', 'rl': 'import vllm, trl, peft, math_verify'}[name]
        if ready.exists() and ready.read_text().strip() == tag:
            return str(py)
        snapshot = PROJECT / f'envcache/venvs/{name}-{tag}.tar'
        if snapshot.exists():
            say(f'Restoring {name} venv from Drive snapshot {snapshot.name}')
            shutil.rmtree(venv, ignore_errors=True)
            with tarfile.open(snapshot) as tar:
                try:  # venvs contain absolute interpreter symlinks, which the 'data' filter rejects
                    tar.extractall('/', filter='fully_trusted')
                except TypeError:
                    tar.extractall('/')
        else:
            say(f'Building {name} venv from {req.name} with {base} (first time only; snapshot saved to Drive)')
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
            uv = [sys.executable, '-m', 'uv']
            env = dict(self.env, UV_CACHE_DIR=str(LOCAL / 'uv_cache'), UV_LINK_MODE='copy')
            shutil.rmtree(venv, ignore_errors=True)
            subprocess.run(uv + ['venv', '--python', sys.executable, str(venv)], env=env, check=True)
            subprocess.run(uv + ['pip', 'install', '--python', str(py), '-r', str(req)], env=env, check=True)
            lock = subprocess.run(uv + ['pip', 'freeze', '--python', str(py)], env=env, capture_output=True,
                                  text=True, check=True).stdout
            (PROJECT / 'envcache/locks').mkdir(parents=True, exist_ok=True)
            (PROJECT / f'envcache/locks/{name}-{tag}.txt').write_text(lock)
            self._snapshot_later(venv, snapshot)
        subprocess.run([str(py), '-c', probe], env=self.env, check=True)
        ready.write_text(tag)
        return str(py)

    def _snapshot_later(self, venv: Path, snapshot: Path) -> None:
        def work():
            try:
                tmp = LOCAL / 'snapshots' / snapshot.name
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tmp, 'w') as tar:
                    tar.add(str(venv), arcname=str(venv).lstrip('/'))
                from experiments.qfft_v93 import drive_io  # rclone CLI or copy + rename (stdlib only)
                drive_io.upload_file(tmp, snapshot)
                tmp.unlink()
                say(f'Saved venv snapshot {snapshot}')
            except Exception as error:
                say('WARNING: venv snapshot failed (next VM rebuilds it):', error)
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self.snapshot_threads.append(thread)

    def finish(self) -> None:
        for t in self.snapshot_threads:
            t.join()

    # ------------------------------------------------------------------ helpers
    def py_module(self, env_name: str, module: str, *args) -> list:
        return [self.python(env_name), '-m', module, *args]


# ZeRO-2 CPU offload keeps fp32 master weights, both Adam moments and fp32 gradients of the 7.6B
# parameters in host memory: 16 bytes x 7.62e9 = 122 GB, before the Python process and data.
MIN_HOST_RAM_GIB = 128
WARN_HOST_RAM_GIB = 150


def check_host_ram() -> None:
    total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30
    if total < MIN_HOST_RAM_GIB:
        raise RuntimeError(f'Host RAM is {total:.0f} GiB; full-parameter training needs about 122 GB of optimizer '
                           f'state in CPU memory (ZeRO-2 offload), so at least {MIN_HOST_RAM_GIB} GiB. Use a '
                           'high-RAM GPU VM (Colab lists about 167 GB for A100 and 230 GB for H100 high-RAM).')
    if total < WARN_HOST_RAM_GIB:
        say(f'WARNING: host RAM {total:.0f} GiB leaves little headroom above the ~122 GB optimizer state')
    say(f'Host RAM {total:.0f} GiB (need >= {MIN_HOST_RAM_GIB})')


def drive_model_dir(ctx: Context, name: str) -> Path:
    spec = json.loads((ctx.pkg / 'configs/pipeline.json').read_text(encoding='utf-8'))['models'][name]
    path = DRIVE / 'models' / spec['drive_dir']
    if not (path / 'config.json').exists():
        raise FileNotFoundError(f'{path} is missing: run pipeline.sh 0 (downloads the pinned models to Drive)')
    return path


def gpu_banner() -> None:
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        out = ''
    say('GPU:', out or 'none (nvidia-smi not found: is this a GPU VM?)')
    try:
        import psutil
        say(f'Host RAM: {psutil.virtual_memory().total / 2**30:.0f} GiB, CPUs: {os.cpu_count()}, '
            f'local disk free: {shutil.disk_usage(LOCAL).free / 2**30:.0f} GiB')
    except ImportError:
        pass


def copy_to_drive(src: Path, dst: Path) -> None:
    """Replace a small Drive file (a stage log) with the local one; a warning on failure."""
    from experiments.qfft_v93.common import sync_file  # stdlib only; falls back to a plain copy on ENOENT
    try:
        sync_file(src, dst)
    except OSError as error:
        say('WARNING: could not copy', src.name, 'to Drive:', error)


def print_status() -> None:
    for line in status_lines():
        print(line, flush=True)


def status_lines() -> list:
    """Stage markers, run progress and evaluation completeness, read from Drive."""
    out = []
    for path in sorted(STAGES_DIR.glob('*.json')):
        s = json.loads(path.read_text(encoding='utf-8'))
        extra = s.get('hours', '')
        if s['state'] == 'running':
            age = (time.time() - float(s.get('heartbeat_ts') or 0)) / 60
            extra = f"on {s.get('host')}, heartbeat {age:.0f} min ago" + (' (stale)' if age * 60 > LEASE_STALE_S else '')
        out.append(f"stage {s['stage']:<45} {s['state']:<9} {extra} {(s.get('error') or '')[:120]}")
    runs = DRIVE / 'runs/project1/v93/runs'
    for path in sorted(runs.glob('*/status.json')) if runs.exists() else []:
        s = json.loads(path.read_text(encoding='utf-8'))
        out.append(f"run {path.parent.name:<28} {s.get('state')} step {s.get('step', s.get('steps'))}/{s.get('max_steps', '')} "
            f"updated {s.get('updated', s.get('completed', ''))}")
    evals = DRIVE / 'runs/project1/v93/evals'
    for path in sorted(evals.glob('*/*/generation_status.json')) if evals.exists() else []:
        s = json.loads(path.read_text(encoding='utf-8'))
        out.append(f"eval {path.parent.parent.name}/{path.parent.name:<10} {s['saved']}/{s['expected']} complete={s['complete']}")
    results = evals / 'results.md'
    if results.exists():
        out.append(results.read_text(encoding='utf-8'))
    return out


def stage_sync_data(ctx: Context) -> None:
    bundle = LOCAL.parent / 'p1_data.tgz'  # /content/p1_data.tgz, where launch.sh --with-data uploads it
    assemble_parts(bundle)
    if not bundle.exists():
        raise FileNotFoundError('No data bundle uploaded; run launch.sh with --with-data')
    staging = LOCAL / 'data_bundle'
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(bundle) as tar:
        tar.extractall(staging)
    staged = staging / 'data/qfft1000_v93/manifest.json'
    if staged.exists():  # checked before anything on Drive is replaced
        gaps = json.loads(staged.read_text(encoding='utf-8')).get('gaps') or []
        if gaps:
            raise RuntimeError(
                f'data/qfft1000_v93 is a preview build (data_build build --allow-gaps): {len(gaps)} questions have no '
                'DeepSeek trace, so the two teachers would train on different questions. Nothing was changed on '
                'Drive. Finish `teacher fill`, or rebuild with `data_build build --ds-verified --paired-exclusion` '
                f'to drop those questions from both teachers, then run pipeline.sh d again. First: {gaps[:3]}')
    for sub in ('qfft1000_v93', 'benchmarks_v93'):
        src = staging / 'data' / sub
        if not src.exists():
            continue
        dst = PROJECT / 'data' / sub
        if sub == 'qfft1000_v93' and dst.exists():
            shutil.rmtree(dst)  # the training sets are replaced as a whole
        # benchmarks are merged: files fetched on the VM (`benchmarks` stage) are identical pinned copies
        shutil.copytree(src, dst, dirs_exist_ok=True)
        say(f'Synced {sub} to {dst}')
    manifest = PROJECT / 'data/qfft1000_v93/manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding='utf-8'))
        for name, v in m['variants'].items():
            path = PROJECT / 'data/qfft1000_v93' / name / 'train.jsonl'
            if sha_file(path) != v['sha256']:
                raise RuntimeError(f'{path} hash differs from manifest')
            say(f'  {name}: {v["rows"]} rows verified')
        excluded = m.get('paired_exclusions') or []
        if excluded:
            say(f'  {len(excluded)} questions dropped from both teachers (paired exclusion): '
                + ', '.join(e['sample_id'] for e in excluded[:30]))


def run_stage(ctx: Context, stage: str, force: bool = False) -> None:
    kind, _, rest = stage.partition(':')
    if kind in READ_ONLY_STAGES:
        _dispatch(ctx, stage)
        return
    lease_stage = stage
    if kind == 'grpo':
        lease_stage = f"grpo:{rest.partition(':')[0]}"
    if kind == 'warm-start':
        parts = rest.split(':')
        if len(parts) != 3 or not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', part) for part in parts):
            raise ValueError('Use warm-start:<config>:<source_run>:<destination_run>')
        if parts[1] == parts[2]:
            raise ValueError('Warm-start requires a separate destination run')
        lease_stage = f'warm-start:{parts[2]}'
    lease = StageLease(lease_stage, ctx.sha, force)
    started = time.monotonic()
    try:
        _dispatch(ctx, stage)
    except BaseException as error:
        lease.close(state='failed', error=f'{type(error).__name__}: {error}'[:2000], at=datetime.now().isoformat())
        raise
    lease.close(state='complete', hours=round((time.monotonic() - started) / 3600, 3),
                finished=datetime.now().isoformat())


def _dispatch(ctx: Context, stage: str) -> None:
    kind, _, rest = stage.partition(':')
    m = 'experiments.qfft_v93.'
    if kind == 'sync-data':
        stage_sync_data(ctx)
    elif kind == 'models':
        # huggingface_hub ships with the Colab image; no venv needed on a CPU VM.
        for name in rest.split(','):
            ctx.run([sys.executable, '-m', 'experiments.qfft_v93.colab.fetch_model', name], f'model_{name}')
    elif kind == 'benchmarks':
        # Same pinned revisions and SHA-256 checks as the local fetch, so GPU stages can start before the
        # local data build is finished; `sync-data` later overwrites these files with identical copies.
        ctx.run(ctx.py_module('score', m + 'fetch_benchmarks', '--out', str(PROJECT / 'data/benchmarks_v93')),
                'benchmarks')
    elif kind == 'smoke-train':
        gpu_banner()
        check_host_ram()
        fresh_smoke(DRIVE / 'runs/project1/v93/runs/smoke_peak_v93', LOCAL / 'runs/smoke_peak_v93')
        # Worst case on purpose: 32 rows that each fill the 32K window (smoke_data.py), not the real data.
        rows = LOCAL / 'smoke/peak_qfft.jsonl'
        ctx.run([ctx.python('train'), '-m', m + 'smoke_data', '--tokenizer', str(drive_model_dir(ctx, 'base')),
                 '--out', str(rows), '--format', 'qfft'], 'smoke_data')
        torchrun = [ctx.python('train'), '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1']
        ctx.run(torchrun + ['-m', m + 'train_sft', '--config', 'train/smoke_peak.json', '--data', str(rows),
                            '--max-steps', '2', '--no-export', '--new-run'], 'smoke_train')
    elif kind == 'smoke-eval':
        gpu_banner()
        fresh_smoke(DRIVE / 'runs/project1/v93/evals/smoke_base', LOCAL / 'evals/smoke_base')
        ctx.run(ctx.py_module('infer', m + 'generate', '--model', 'base', '--suite', 'smoke', '--tag', 'smoke_base'),
                'smoke_eval_generate')
        ctx.run(ctx.py_module('infer', m + 'score', '--tags', 'smoke_base'), 'smoke_eval_score')
    elif kind == 'smoke-grpo':
        gpu_banner()
        fresh_smoke(DRIVE / 'runs/project1/v93/runs/smoke_grpo_v93', LOCAL / 'runs/smoke_grpo_v93')
        ctx.run(ctx.py_module('rl', m + 'grpo', '--config', 'grpo/smoke_grpo.json', '--no-merge'), 'smoke_grpo')
    elif kind in ('train', 'warm-start'):
        from experiments.qfft_v93.common import load_config
        name = rest
        extra = []
        log_name = f'train_{name}'
        if kind == 'warm-start':
            name, source, destination = rest.split(':')
            extra = ['--warm-start-from', source, '--run-name', destination]
            log_name = f'warm_start_{destination}'
        data = PROJECT / 'data' / load_config(ctx.pkg / 'configs/train' / f'{name}.json')['data']
        if not data.exists():
            raise FileNotFoundError(f'{data} is missing: build the data locally, then run pipeline.sh d '
                                    '(launch.sh --gpu cpu --with-data sync-data)')
        gpu_banner()
        check_host_ram()
        torchrun = [ctx.python('train'), '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1']
        ctx.run(torchrun + ['-m', m + 'train_sft', '--config', f'train/{name}.json', *extra], log_name)
    elif kind == 'eval':
        gpu_banner()
        parts = rest.split(':')
        if parts[0] == 'run':
            model, suite = 'run:' + parts[1], (parts[2] if len(parts) > 2 else 'full')
        else:
            model, suite = parts[0], (parts[1] if len(parts) > 1 else 'full')
        tag = model.replace('run:', '').replace('/', '_')
        ctx.run(ctx.py_module('infer', m + 'generate', '--model', model, '--suite', suite, '--tag', tag), f'gen_{tag}')
        ctx.run(ctx.py_module('infer', m + 'score', '--tags', tag), f'score_{tag}')
    elif kind == 'grpo':
        config, _, operation = rest.partition(':')
        extra = []
        if operation == 'resume':
            extra = ['--require-resume']
        elif re.fullmatch(r'pause-after=[1-9][0-9]*', operation):
            extra = ['--pause-after', operation.split('=', 1)[1]]
        elif operation:
            raise ValueError(f'Unknown GRPO operation: {operation}')
        gpu_banner()
        ctx.run(ctx.py_module('rl', m + 'grpo', '--config', f'grpo/{config}.json', *extra), f'grpo_{config}')
    elif kind == 'r1-is-diagnostic' and not rest:
        gpu_banner()
        ctx.run(ctx.py_module('rl', m + 'grpo_is_diagnostic', 'capture'), 'r1_is_diagnostic')
    elif kind == 'r1-grpo-dev':
        from experiments.qfft_v93.common import load_config, sha256_file
        gpu_banner()
        bench_dir = PROJECT / 'data/r1_grpo_dev'
        dev = load_config('grpo/r1_dev.json')
        if sha256_file(bench_dir / 'r1_grpo_dev.jsonl') != dev['sha256']:
            raise RuntimeError('R1 GRPO development set does not match its frozen fingerprint')
        out = DRIVE / 'runs/project1/v93/grpo_dev'
        tag = rest or 'pilot'
        if tag not in ('pilot', 'step30'):
            raise ValueError(f'Unknown R1 development evaluation target: {tag}')
        for eval_tag in ('r1', tag):
            model = dev['models'][eval_tag]
            ctx.run(ctx.py_module('infer', m + 'generate', '--model', model, '--tag', eval_tag,
                                  '--suite', 'r1_grpo_dev', '--bench-dir', str(bench_dir),
                                  '--evals-root', str(out)), f'gen_r1_dev_{eval_tag}')
        ctx.run(ctx.py_module('infer', m + 'score', '--bench-dir', str(bench_dir), '--require-gradable',
                              '--evals-root', str(out), '--tags', f'r1,{tag}', '--compare', f'{tag}:r1'),
                'score_r1_grpo_dev')
    elif kind == 'r1-competition-2026' and not rest:
        from experiments.qfft_v93.common import drive_layout, load_config, read_jsonl, sha256_file
        from experiments.qfft_v93.models import drive_path
        name = 'r1_competition_2026'
        cfg = load_config(f'grpo/{name}.json')
        bench_dir = PROJECT / 'data' / name
        benchmark = bench_dir / f'{name}.jsonl'
        if sha256_file(benchmark) != cfg['sha256'] or len(read_jsonl(benchmark)) != cfg['rows']:
            raise RuntimeError('R1 competition set does not match its frozen identity')
        out = DRIVE / 'runs/project1/v93' / name
        layout = drive_layout(DRIVE)
        for tag, model in cfg['models'].items():
            model_dir = drive_path(model['spec'], layout)
            if sha256_file(model_dir / 'p1_model_identity.json') != model['identity_sha256']:
                raise RuntimeError(f'{tag}: model identity differs from the frozen evaluation protocol')
        gpu_banner()
        for tag, model in cfg['models'].items():
            ctx.run(ctx.py_module('infer', m + 'generate', '--model', model['spec'], '--tag', tag,
                                  '--suite', name, '--bench-dir', str(bench_dir), '--evals-root', str(out)),
                    f'gen_{name}_{tag}')
        ctx.run(ctx.py_module('infer', m + 'score', '--bench-dir', str(bench_dir), '--require-gradable',
                              '--evals-root', str(out), '--tags', ','.join(cfg['models']),
                              '--compare', 'token_mask:sequence_mask', '--compare', 'token_mask:r1',
                              '--compare', 'sequence_mask:r1'), f'score_{name}')
    elif kind == 'r1-fura50-eval' and not rest:
        from experiments.qfft_v93.common import drive_layout, load_config, read_json, sha256_file
        from experiments.qfft_v93.formats import SYSTEM_PROMPT
        from experiments.qfft_v93.models import drive_path, max_model_len
        from experiments.qfft_v93.score import COMPETITION
        cfg = load_config('grpo/grpo_qfft_r1_fura_lora50.json')
        run = cfg['run_name']
        model = f'run:{run}/merged'
        layout = drive_layout(DRIVE)
        run_dir = drive_path(model, layout).parent
        if read_json(run_dir / 'GRPO_COMPLETE.json')['steps'] != cfg['max_steps']:
            raise RuntimeError(f'{run}: expected the final {cfg["max_steps"]}-step checkpoint')
        baseline_model = cfg['init_model']
        tag, baseline = f'{run}_merged', baseline_model.removeprefix('run:')
        baseline_identity = read_json(drive_path(baseline_model, layout) / 'p1_model_identity.json')
        evaluation = load_config('pipeline.json')['eval']
        suite = evaluation['suites']['competition']
        for benchmark, spec in suite.items():
            base_eval = layout.evals / baseline / benchmark
            status = read_json(base_eval / 'generation_status.json')
            if not status['complete'] or not (base_eval / 'raw.jsonl').is_file():
                raise RuntimeError(f'{baseline}/{benchmark}: missing complete reference generations')
            expected = {
                'model': baseline_model, 'model_identity': baseline_identity, 'benchmark': benchmark,
                'benchmark_sha256': sha256_file(layout.data / 'benchmarks_v93' / f'{benchmark}.jsonl'),
                'n': spec['n'], 'limit': spec.get('limit'), 'max_tokens_cap': spec.get('max_tokens'),
                'temperature': evaluation['temperature'], 'top_p': evaluation['top_p'], 'seed': evaluation['seed'],
                'max_model_len': min(max_model_len(baseline_model), evaluation['max_model_len']),
                'system_prompt': SYSTEM_PROMPT, 'engine': 'vllm',
            }
            if read_json(base_eval / 'identity.json')['protocol'] != expected:
                raise RuntimeError(f'{baseline}/{benchmark}: reference generation protocol differs')
        gpu_banner()
        ctx.run(ctx.py_module('infer', m + 'generate', '--model', model, '--tag', tag,
                              '--suite', 'competition'), f'gen_{tag}')
        report_dir = layout.evals.parent / 'r1_fura50_eval'
        ctx.run(ctx.py_module('infer', m + 'score', '--benchmarks', ','.join(COMPETITION),
                              '--tags', f'{tag},{baseline}', '--require-gradable',
                              '--report-dir', str(report_dir),
                              '--compare', f'{tag}:{baseline}'), f'score_{tag}')
        pairs = read_json(report_dir / 'results.json')['comparisons']
        if not any(row['benchmark'] == 'competition' and row['problems'] == 100 for row in pairs):
            raise RuntimeError(f'{run}: missing the 100-problem paired comparison')
    elif kind == 'score':
        args = ['--tags', rest] if rest else []
        ctx.run(ctx.py_module('score', m + 'score', *args), 'score')
    elif kind == 'analyze':
        tags, _, grpo_runs = rest.partition(':')
        out = DRIVE / 'runs/project1/v93/figures'
        ctx.run(ctx.py_module('score', m + 'analyze', '--tags', tags, '--grpo-runs', grpo_runs, '--out', str(out)),
                'analyze')
    elif kind == 'status':
        print_status()
    elif kind == 'smoke-report':
        print_smoke_summary(ctx)
    elif kind == 'shell':
        say('shell stage: nothing to run')
    elif kind == 'ping':
        seconds = float(rest or 30)
        probe = PROJECT / 'ping' / f'{host_id()}.json'
        write_json(probe, {'host': host_id(), 'at': datetime.now().isoformat()})
        say(f'Drive writable ({probe}); waiting {seconds:.0f} s')
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(min(10.0, max(0.0, deadline - time.monotonic())))
            say(f'ping: {max(0.0, deadline - time.monotonic()):.0f} s left')
    else:
        raise ValueError(f'Unknown stage {stage!r}')


def print_smoke_summary(ctx) -> None:
    """Print the smoke summary block and keep a copy on Drive (runs/project1/v93/smoke/)."""
    from experiments.qfft_v93.colab import smoke_report
    text = smoke_report.summary(DRIVE, LOCAL, dict(ctx.pythons) if ctx is not None else {})
    print(text, flush=True)
    try:
        out = DRIVE / 'runs/project1/v93/smoke'
        out.mkdir(parents=True, exist_ok=True)
        session = os.environ.get('P1_SESSION') or datetime.now().strftime('%Y%m%d-%H%M%S')
        (out / f'summary_{session}.txt').write_text(text + '\n', encoding='utf-8')
        (out / 'latest.txt').write_text(text + '\n', encoding='utf-8')
    except OSError as error:
        say('WARNING: could not save the smoke summary to Drive:', error)


def start_reporter(args: dict):
    """GitHub report pusher (colab/report_push.py) plus its periodic thread, or (None, None)."""
    cfg = args.get('report')
    if not cfg or not DRIVE.is_dir():
        return None, None
    from experiments.qfft_v93.colab.report_push import Reporter
    info = {'stages': ' '.join(args.get('stages', [])), 'gpu': args.get('gpu'), 'drive': str(DRIVE),
            'code_bundle': args.get('code_sha'), 'git_head': args.get('git_head'),
            'started': datetime.now().isoformat(timespec='seconds'), 'started_ts': time.time()}
    reporter = Reporter(cfg, DRIVE, LOCAL, args.get('session'), info)
    if not reporter.enabled:
        say(f"Reports: no credentials at {cfg.get('credentials')}; results stay on Drive")
        return None, None
    stop = threading.Event()

    def loop():
        while not stop.wait(reporter.interval):
            reporter.push(f"colab {args.get('session')}: progress")
    threading.Thread(target=loop, daemon=True).start()
    say(f"Reports: {cfg['repo']} branch {reporter.branch}, every {reporter.interval / 60:.0f} min and at the end")
    return reporter, stop


def main(args: dict) -> None:
    failed = 'not started'
    ctx = None
    smoke_failures = []
    reporter, stop_reports = start_reporter(args)
    try:
        if not DRIVE.is_dir():
            raise RuntimeError('Google Drive is not mounted; launch.sh runs `colab drivemount` before this step')
        code_sha = args['code_sha']
        ctx = Context(LOCAL / 'code' / code_sha, code_sha)
        stages = list(args['stages'])
        try:
            for i, stage in enumerate(stages):
                say(f'===== STAGE {stage} =====')
                failed = stage
                try:
                    run_stage(ctx, stage, force=bool(args.get('force')))
                except Exception as error:
                    # Smoke stages are independent measurements: on an allocated GPU VM, a failed one
                    # should not cost the numbers of the others.
                    if stage.startswith('smoke-') and any(s.startswith('smoke-') for s in stages[i + 1:]):
                        smoke_failures.append(f'{stage}: {type(error).__name__}: {error}')
                        say(f'{stage} failed ({type(error).__name__}); continuing with the remaining smoke stages')
                        continue
                    raise
            failed = '; '.join(smoke_failures) or None
        finally:
            ctx.finish()
    except BaseException as error:
        failed = '; '.join(smoke_failures + [f'{failed}: {type(error).__name__}: {error}'])
        raise
    finally:
        # Before P1_RESULT: launch.sh releases the VM as soon as it sees that line.
        requested = args.get('stages', [])
        if (any(s.startswith('smoke-') for s in requested) and 'smoke-report' not in requested
                and DRIVE.is_dir()):
            try:
                print_smoke_summary(ctx)
            except Exception as error:
                say('WARNING: smoke summary failed:', type(error).__name__, error)
        result = ('OK' if failed is None else 'FAILED ' + failed)[:2000]
        if reporter is not None:  # before P1_RESULT, which makes launch.sh release the VM
            stop_reports.set()
            try:
                reporter.push(f"colab {args.get('session')}: {result[:80]}", result=result)
            except Exception as error:
                say('WARNING: final report push failed:', type(error).__name__, error)
        print('P1_RESULT=' + result, flush=True)
