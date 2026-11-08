"""Stage runner executed on the Colab VM (inside the kernel started by `colab exec`). Standard library only.

Stages (all idempotent; re-running resumes or skips completed work):

    sync-data                   copy the uploaded data bundle to Drive and verify manifest hashes
    models:<a,b,...>            download registry models to Drive once (cheap on a CPU VM)
    smoke-train                 2 optimizer steps of qfft_ds on the real model: s/step, GPU/host memory
    smoke-eval                  24 short generations of the base model + scoring
    smoke-grpo                  2 GRPO steps from the base model (TRL + vLLM + LoRA + reward)
    train:<config>              full SFT/QFFT run, e.g. train:qfft_ds  (configs/train/<config>.json)
    eval:<model>[:<suite>]      generate + score, e.g. eval:base:full, eval:run:qfft_ds_v93:full
    grpo:<config>               LoRA-GRPO + merge, e.g. grpo:grpo_qfft_ds (configs/grpo/<config>.json)
    score[:<tags>]              re-score saved generations and rebuild results.md
    analyze:<tags>[:<grpo_runs>] figures into Drive runs/project1/v93/figures
    status                      print stage markers, run progress, eval completeness and results.md
    shell                       do nothing (use with --keep to inspect the VM)

Environments: `train` = Colab system Python + DeepSpeed/Liger/SwanLab (Drive wheel cache);
`infer` and `rl` = uv virtualenvs restored from a Drive snapshot (built once, then ~1 minute).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
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


class Context:
    def __init__(self, code_dir: Path, code_sha: str):
        self.code, self.sha = code_dir, code_sha
        self.pkg = code_dir / 'experiments/qfft_v93'
        self.env = dict(os.environ, PYTHONPATH=str(code_dir), P1_CODE_SHA=code_sha, P1_DRIVE_ROOT=str(DRIVE),
                        P1_LOCAL_ROOT=str(LOCAL), HF_HOME=str(LOCAL / 'hf'), TOKENIZERS_PARALLELISM='false',
                        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', DS_SKIP_CUDA_CHECK='1',
                        PYTHONUNBUFFERED='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
        credential = DRIVE / '.project1_credentials/swanlab.json'
        if credential.exists():
            key = json.loads(credential.read_text(encoding='utf-8-sig')).get('SWANLAB_API_KEY')
            if isinstance(key, str) and key.strip():
                self.env['SWANLAB_API_KEY'] = key.strip()
        self.pythons = {}
        self.snapshot_threads = []

    # ------------------------------------------------------------------ process execution
    def run(self, cmd, name: str, quiet_minutes: float = 10) -> None:
        """Stream a subprocess to the notebook and to a Drive log; never kill it on a client disconnect."""
        LOGS.mkdir(parents=True, exist_ok=True)
        log_path = LOGS / f'{datetime.now():%Y%m%d-%H%M%S}_{name}.log'
        say('RUN', ' '.join(str(c) for c in cmd), '| log:', log_path)
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
            threading.Thread(target=watchdog, daemon=True).start()
            try:
                for line in proc.stdout:
                    last[0] = time.monotonic()
                    print(line, end='', flush=True)
                    logf.write(line)
                    logf.flush()
                code = proc.wait()
            except KeyboardInterrupt:
                say(f'Detached from {name} (pid {proc.pid}); it keeps running. Log: {log_path}')
                raise
        if code != 0:
            raise RuntimeError(f'{name} failed with exit code {code}; see {log_path}')

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
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tmp, str(snapshot) + '.partial')
                os.replace(str(snapshot) + '.partial', snapshot)
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


def gpu_banner() -> None:
    out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True)
    say('GPU:', out.stdout.strip() or 'none')
    try:
        import psutil
        say(f'Host RAM: {psutil.virtual_memory().total / 2**30:.0f} GiB, CPUs: {os.cpu_count()}, '
            f'local disk free: {shutil.disk_usage("/content").free / 2**30:.0f} GiB')
    except ImportError:
        pass


def print_status() -> None:
    """Stage markers, run progress and evaluation completeness, read from Drive."""
    for path in sorted(STAGES_DIR.glob('*.json')):
        s = json.loads(path.read_text(encoding='utf-8'))
        say(f"stage {s['stage']:<45} {s['state']:<9} {s.get('hours', '')} {(s.get('error') or '')[:120]}")
    runs = DRIVE / 'runs/project1/v93/runs'
    for path in sorted(runs.glob('*/status.json')) if runs.exists() else []:
        s = json.loads(path.read_text(encoding='utf-8'))
        say(f"run {path.parent.name:<28} {s.get('state')} step {s.get('step', s.get('steps'))}/{s.get('max_steps', '')} "
            f"updated {s.get('updated', s.get('completed', ''))}")
    evals = DRIVE / 'runs/project1/v93/evals'
    for path in sorted(evals.glob('*/*/generation_status.json')) if evals.exists() else []:
        s = json.loads(path.read_text(encoding='utf-8'))
        say(f"eval {path.parent.parent.name}/{path.parent.name:<10} {s['saved']}/{s['expected']} complete={s['complete']}")
    results = evals / 'results.md'
    if results.exists():
        print(results.read_text(encoding='utf-8'), flush=True)


def stage_sync_data(ctx: Context) -> None:
    bundle = Path('/content/p1_data.tgz')
    if not bundle.exists():
        raise FileNotFoundError('No data bundle uploaded; run launch.sh with --with-data')
    staging = LOCAL / 'data_bundle'
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(bundle) as tar:
        tar.extractall(staging)
    for sub in ('qfft1000_v93', 'benchmarks_v93'):
        src = staging / 'data' / sub
        if not src.exists():
            continue
        dst = PROJECT / 'data' / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        say(f'Synced {sub} to {dst}')
    manifest = PROJECT / 'data/qfft1000_v93/manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding='utf-8'))
        for name, v in m['variants'].items():
            path = PROJECT / 'data/qfft1000_v93' / name / 'train.jsonl'
            if sha_file(path) != v['sha256']:
                raise RuntimeError(f'{path} hash differs from manifest')
            say(f'  {name}: {v["rows"]} rows verified')


def run_stage(ctx: Context, stage: str) -> None:
    kind, _, rest = stage.partition(':')
    marker = STAGES_DIR / (stage.replace(':', '__').replace('/', '_').replace(',', '+') + '.json')
    write_json(marker, {'stage': stage, 'state': 'running', 'started': datetime.now().isoformat(), 'code': ctx.sha})
    started = time.monotonic()
    m = 'experiments.qfft_v93.'
    if kind == 'sync-data':
        stage_sync_data(ctx)
    elif kind == 'models':
        # huggingface_hub ships with the Colab image; no venv needed on a CPU VM.
        for name in rest.split(','):
            ctx.run([sys.executable, '-m', 'experiments.qfft_v93.colab.fetch_model', name], f'model_{name}')
    elif kind == 'smoke-train':
        gpu_banner()
        torchrun = [ctx.python('train'), '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1']
        ctx.run(torchrun + ['-m', m + 'train_sft', '--config', 'train/smoke_qfft_ds.json', '--max-steps', '2',
                            '--no-export', '--new-run'], 'smoke_train')
    elif kind == 'smoke-eval':
        gpu_banner()
        ctx.run(ctx.py_module('infer', m + 'generate', '--model', 'base', '--suite', 'smoke', '--tag', 'smoke_base'),
                'smoke_eval_generate')
        ctx.run(ctx.py_module('infer', m + 'score', '--tags', 'smoke_base'), 'smoke_eval_score')
    elif kind == 'smoke-grpo':
        gpu_banner()
        ctx.run(ctx.py_module('rl', m + 'grpo', '--config', 'grpo/smoke_grpo.json', '--no-merge'), 'smoke_grpo')
    elif kind == 'train':
        gpu_banner()
        torchrun = [ctx.python('train'), '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1']
        ctx.run(torchrun + ['-m', m + 'train_sft', '--config', f'train/{rest}.json'], f'train_{rest}')
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
        gpu_banner()
        ctx.run(ctx.py_module('rl', m + 'grpo', '--config', f'grpo/{rest}.json'), f'grpo_{rest}')
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
    elif kind == 'shell':
        say('shell stage: nothing to run')
    else:
        raise ValueError(f'Unknown stage {stage!r}')
    write_json(marker, {'stage': stage, 'state': 'complete', 'hours': round((time.monotonic() - started) / 3600, 3),
                        'finished': datetime.now().isoformat(), 'code': ctx.sha})


def main(args: dict) -> None:
    if not DRIVE.is_dir():
        raise RuntimeError('Google Drive is not mounted; launch.sh runs `colab drivemount` before this step')
    code_sha = args['code_sha']
    code_dir = LOCAL / 'code' / code_sha
    ctx = Context(code_dir, code_sha)
    failed = None
    try:
        for stage in args['stages']:
            say(f'===== STAGE {stage} =====')
            try:
                run_stage(ctx, stage)
            except Exception as error:
                failed = f'{stage}: {type(error).__name__}: {error}'
                write_json(STAGES_DIR / (stage.replace(':', '__').replace('/', '_').replace(',', '+') + '.json'),
                           {'stage': stage, 'state': 'failed', 'error': failed[:2000], 'code': code_sha,
                            'at': datetime.now().isoformat()})
                raise
    finally:
        ctx.finish()
        print('P1_RESULT=' + ('OK' if failed is None else 'FAILED ' + failed), flush=True)
