"""Eight-update, single-A100 full-SFT capacity probe. Never starts the six-epoch run.

Uses frozen s1K-1.1 examples with the independent QFFT reproduction's ordinary
SFT formatting. No LoRA, no YaRN, and no benchmark generation in this entry.
"""
import argparse
import gzip
import hashlib
import importlib.util
from importlib.metadata import version, PackageNotFoundError
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve()
ASSETS = 's1_full_sft_probe_v46'
DEFAULT_ROOT = '/content/drive/MyDrive/runs/project1/s1_full_sft_probe_v46'
BASE_REV = 'a09a35458c702b33eeacc393d103063234e8bc28'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def load_assets(directory):
    directory = Path(directory)
    manifest = read(directory / (ASSETS + '.json'))
    blob = directory / (ASSETS + '.jsonl.gz')
    if sha(blob) != manifest['probe_file_sha256']:
        raise ValueError('Probe asset checksum differs')
    with gzip.open(blob, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 64 or len({r['sample_id'] for r in rows}) != 64:
        raise ValueError('Expected exactly 64 distinct approved probe examples')
    if [r['sample_id'] for r in rows] != manifest['approved_probe_sample_ids']:
        raise ValueError('Probe sample order differs')
    for row in rows:
        ids, labels = row['input_ids'], row['labels']
        if row.get('probe_training_approved') is not True:
            raise ValueError('Unapproved probe sample')
        if not 0 < len(ids) <= 16384 or len(ids) != len(labels):
            raise ValueError('Invalid encoded lengths')
        boundary = row['prompt_tokens']
        if labels[:boundary] != [-100] * boundary or labels[boundary:] != ids[boundary:]:
            raise ValueError('Assistant loss mask differs')
        if any(type(i) is not int or i < 0 for i in ids):
            raise ValueError('Invalid token IDs')
    if max(len(r['input_ids']) for r in rows) != 16384:
        raise ValueError('Missing worst-window capacity sample')
    return rows, manifest


def ds_config():
    return {
        'train_batch_size': 'auto', 'train_micro_batch_size_per_gpu': 'auto',
        'gradient_accumulation_steps': 'auto', 'gradient_clipping': 'auto',
        'bf16': {'enabled': True}, 'fp16': {'enabled': False},
        'optimizer': {'type': 'AdamW', 'params': {'lr': 'auto', 'betas': 'auto',
            'eps': 'auto', 'weight_decay': 'auto'}},
        'zero_optimization': {'stage': 2, 'offload_optimizer': {'device': 'cpu', 'pin_memory': False},
            'allgather_partitions': True, 'allgather_bucket_size': 200000000,
            'reduce_scatter': True, 'reduce_bucket_size': 200000000,
            'overlap_comm': False, 'contiguous_gradients': True},
        'steps_per_print': 1, 'wall_clock_breakdown': False,
    }


def eta_hours(step_seconds, sample_lengths, pool_tokens):
    # Exclude the first optimizer update (CPUAdam JIT/state initialization).
    # Eight microbatches form each update. Selection spans the length distribution.
    steady_seconds = sum(step_seconds[1:])
    steady_tokens = sum(sample_lengths[8:])
    if steady_seconds <= 0 or steady_tokens <= 0:
        raise ValueError('No stable training window')
    return pool_tokens * 6 * steady_seconds / steady_tokens / 3600


def no_other_gpu_job():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError('A GPU compute process is already present. No process was stopped; do not launch twice.')


def clean_install_environment():
    env = dict(os.environ, DS_BUILD_OPS='0', MAX_JOBS='2')
    for key in ['PIP_CONSTRAINT', 'PIP_BUILD_CONSTRAINT', 'PIP_NO_BINARY', 'PIP_ONLY_BINARY']:
        env.pop(key, None)
    return env


def numpy_runtime():
    # Colab may have duplicate/stale dist-info directories. The version of the
    # actually importable module takes precedence over an arbitrary metadata row.
    result = subprocess.run([sys.executable, '-c',
        'import json,numpy; print(json.dumps({"version":numpy.__version__,"file":numpy.__file__}))'],
        capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def numpy_usable(runtime, python_version=None):
    from packaging.version import Version
    py = python_version or sys.version_info[:2]
    if not runtime or not runtime.get('version'):
        return False
    try:
        v = Version(runtime['version'])
        return py < (3, 13) or v >= Version('2.1.0')
    except ValueError:
        return False


def ensure_dependencies(constraint_path=None):
    # Separate DeepSpeed's own source build from its binary runtime dependencies.
    # A blanket --no-build-isolation on the combined dependency resolution let
    # pip enter a NumPy/SciPy-style source build without its mesonpy backend.
    # We do not need to compile those libraries for this capacity probe.
    env = clean_install_environment()
    runtime = numpy_runtime()
    if not numpy_usable(runtime):
        # Confirmed missing/broken/incompatible NumPy only. 2.2.6 has a published
        # CPython3.13 manylinux x86_64 wheel and is below the NumPy2.3 boundary.
        print('V46_NUMPY_REPAIR: replace only unusable NumPy with the CPython-compatible '
              '2.2.6 wheel; preserving torch/Transformers.', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
            '--no-deps', '--force-reinstall', 'numpy==2.2.6'], env=env, check=True)
        runtime = numpy_runtime()
        if not numpy_usable(runtime):
            raise RuntimeError('NumPy still cannot import after targeted wheel repair')
    print('V46_NUMPY_RUNTIME: ' + json.dumps(runtime), flush=True)
    preserved = {}
    core = {'torch', 'torchvision', 'torchaudio', 'triton', 'transformers', 'tokenizers',
            'peft', 'accelerate', 'torchao', 'numpy', 'scipy', 'datasets',
            'huggingface-hub', 'safetensors'}
    # Query the active distribution, rather than allowing the last duplicate
    # encountered in distributions() to overwrite a valid version.
    for name in sorted(core):
        try:
            preserved[name] = version(name)
        except PackageNotFoundError:
            pass
    preserved['numpy'] = runtime['version']
    constraint = Path(constraint_path or '/content/s1-full-probe-v46-constraints.txt')
    constraint.write_text('\n'.join(f'{k}=={v}' for k, v in sorted(preserved.items())) + '\n')
    need_ds = importlib.util.find_spec('deepspeed') is None
    # These are the published v0.19.8 base runtime requirements, plus the CUDA
    # nvidia-ml-py condition; no dev/all/deepcompile/scipy extras are requested.
    wheels = (['einops', 'hjson', 'msgpack', 'ninja', 'numpy', 'packaging>=20.0',
        'psutil', 'py-cpuinfo', 'pydantic>=2.0.0', 'tqdm', 'nvidia-ml-py', 'setuptools', 'wheel']
        if need_ds else [])
    for module, package in [('ninja', 'ninja'), ('swanlab', 'swanlab==0.10.3'),
                            ('accelerate', 'accelerate')]:
        if importlib.util.find_spec(module) is None and package not in wheels:
            wheels.append(package)
    if wheels:
        # SwanLab requires pwinput>=1.0.3, whose PyPI release is source-only
        # pure Python. pip gives a named no-binary exception precedence over
        # only-binary=:all:, so permit that one package without permitting
        # NumPy/SciPy (or other compiled dependencies) to fall back to source.
        print('V46_DEPENDENCY_REPAIR: binary runtime/online telemetry packages first; '
              'source-only pure-Python pwinput allowed; core package versions '
              'preserved, no NumPy/SciPy source builds.', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
            '--no-binary=pwinput', '-c', str(constraint), *wheels], env=env, check=True)
    if need_ds:
        print('V46_DEEPSPEED_INSTALL: build DeepSpeed only, without re-resolving dependencies.', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
            '--no-binary=deepspeed', 'deepspeed==0.19.8'], env=env, check=True)
    # Verify imports in a fresh process, before model restore/load or GPU worker.
    subprocess.run([sys.executable, '-c',
        'import deepspeed, swanlab, ninja, accelerate; '
        'print("V46_DEPENDENCIES_READY", deepspeed.__version__, swanlab.__version__)'],
        env=env, check=True)


def online_log(output, payload):
    import swanlab
    from experiments.aimo_sft_v26.telemetry import OnlineLog
    if not os.environ.get('SWANLAB_API_KEY'):
        raise RuntimeError('Private SwanLab credential was not inherited')
    if swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=False) is False:
        raise RuntimeError('Online SwanLab authentication failed')
    kwargs = {'project': 'project1-7b-math', 'config': payload,
              'tags': ['s1K-1.1', 'full-SFT', 'capacity-probe', 'v46'], 'reinit': True}
    if 'name' in inspect.signature(swanlab.init).parameters:
        kwargs.update(name='Qwen25_7B_s1_full_probe_v46', mode='online', log_dir=str(output / 'swanlog'))
    else:
        kwargs.update(experiment_name='Qwen25_7B_s1_full_probe_v46', mode='cloud', logdir=str(output / 'swanlog'))
    run = swanlab.init(**kwargs)
    run_id = getattr(run, 'id', None)
    if not run_id:
        raise RuntimeError('SwanLab returned no online run ID')
    return OnlineLog(swanlab, output, str(run_id), getattr(run, 'url', None))


def worker(args):
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
    sys.path.insert(0, str(args.project_root))
    from experiments.aimo_sft_v26.telemetry import device_metrics
    rows, manifest = load_assets(SCRIPT.parent)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'script_sha256': sha(SCRIPT), 'manifest_sha256': sha(SCRIPT.with_name(ASSETS + '.json')),
        'model_revision': BASE_REV, 'stage': 'probe-only', 'full_parameter_training': True,
        'updates': 8, 'batch': 1, 'gradient_accumulation': 8, 'cutoff': 16384}
    if (output / 'PROBE_COMPLETE.json').exists():
        if read(output / 'identity.json') != identity:
            raise ValueError('Completed probe identity differs; preserve it')
        print('S1_FULL_SFT_V46_PROBE_ALREADY_COMPLETE', flush=True)
        return
    if (output / 'identity.json').exists() and read(output / 'identity.json') != identity:
        raise ValueError('Probe identity differs; use a separate output directory')
    save(output / 'identity.json', identity)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('This probe requires exactly one visible CUDA GPU')
    props = torch.cuda.get_device_properties(0)
    import psutil
    environment = {'gpu': props.name, 'gpu_memory_gib': props.total_memory / 2**30,
        'host_memory_gib': psutil.virtual_memory().total / 2**30,
        'host_available_gib': psutil.virtual_memory().available / 2**30,
        'cpu_count': os.cpu_count(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'packages': {k: version(k) for k in ['transformers', 'accelerate', 'deepspeed', 'swanlab']}}
    save(output / 'environment.json', environment)
    print(json.dumps(environment, indent=2), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    for name, expected in manifest['tokenizer_sha256'].items():
        if sha(Path(args.model_path) / name) != expected:
            raise ValueError('Frozen probe tokenizer differs: ' + name)
    tokenizer.pad_token = tokenizer.eos_token
    # Construct arguments before model loading so DeepSpeed integration sees
    # the intended optimizer offload setup throughout initialization.
    kwargs = dict(output_dir=str(output / 'scratch'), per_device_train_batch_size=1,
        gradient_accumulation_steps=8, max_steps=8, learning_rate=1e-5,
        lr_scheduler_type='cosine', warmup_steps=1, weight_decay=1e-4,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1, save_strategy='no', report_to='none',
        deepspeed=ds_config(), seed=42, data_seed=42, dataloader_num_workers=0,
        remove_unused_columns=False, disable_tqdm=False)
    training_args = TrainingArguments(**kwargs)
    log = online_log(output, {'identity': identity, 'environment': environment,
        'manifest': manifest, 'probe_note': 'Eight updates; probe warmup differs from six-epoch recipe; no quality result.'})
    step_seconds, started, max_rss = [], {}, [0]

    class Monitor(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            started['time'] = time.perf_counter()

        def on_substep_end(self, args, state, control, **kwargs):
            max_rss[0] = max(max_rss[0], psutil.Process().memory_info().rss)

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started['time']
            step_seconds.append(elapsed)
            values = {'train/update_seconds': elapsed, **device_metrics(torch),
                'host/rss_gib': psutil.Process().memory_info().rss / 2**30,
                'host/available_gib': psutil.virtual_memory().available / 2**30}
            log.log(values, state.global_step)
            save(output / 'progress.json', {'completed_updates': state.global_step,
                'update_seconds': step_seconds, **values})
            print(f'FULL_SFT_PROBE: {state.global_step}/8 updates, {elapsed:.1f}s/update, '
                  f'peak GPU {values["gpu/peak_allocated_gib"]:.2f} GiB, '
                  f'host available {values["host/available_gib"]:.1f} GiB', flush=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                log.log(logs, state.global_step)

    class OrderedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            # Exact frozen length-stratified order, including 16K in update 1.
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    def collate(batch):
        # microbatch=1, so there is no global 16K padding of shorter examples.
        row = batch[0]
        return {'input_ids': torch.tensor([row['input_ids']], dtype=torch.long),
            'attention_mask': torch.ones((1, len(row['input_ids'])), dtype=torch.long),
            'labels': torch.tensor([row['labels']], dtype=torch.long)}

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        model.config.use_cache = False
        if getattr(model.config, 'rope_scaling', None):
            raise ValueError('Unexpected RoPE override in original base model')
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable != total or total < 7_000_000_000:
            raise RuntimeError('Expected full 7B training, not an adapter or frozen model')
        print(f'FULL_PARAMETER_CHECK: {trainable:,}/{total:,} trainable', flush=True)
        trainer = OrderedTrainer(model=model, args=training_args, train_dataset=rows,
            data_collator=collate, callbacks=[Monitor()])
        result = trainer.train()
        if len(step_seconds) != 8 or trainer.state.global_step != 8:
            raise RuntimeError('Probe did not finish all eight optimizer updates')
        estimated = eta_hours(step_seconds, [len(r['input_ids']) for r in rows],
            manifest['candidate_pool_cutoff_tokens'])
        summary = {'updates': 8, 'full_parameter_count': trainable,
            'all_update_seconds': step_seconds,
            'steady_median_update_seconds': statistics.median(step_seconds[1:]),
            'estimated_six_epoch_compute_hours': estimated,
            'estimated_range_hours': [estimated * .75, estimated * 1.5],
            'estimate_note': 'Token-scaled short probe, excludes setup/save/eval; not an exact runtime forecast.',
            'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_cuda_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
            'max_sampled_process_rss_gib': max_rss[0] / 2**30,
            'train_metrics': result.metrics, 'model_saved': False,
            'quality_claim': None, 'full_training_started': False}
        save(output / 'PROBE_COMPLETE.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        print('S1_FULL_SFT_V46_PROBE_COMPLETE: eight updates only; GPU worker will exit. '
              'Six-epoch training has NOT started.', flush=True)
    except BaseException as error:
        save(output / 'PROBE_FAILED.json', {'type': type(error).__name__,
            'completed_updates': len(step_seconds), 'full_training_started': False})
        raise
    finally:
        log.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--check-local', action='store_true')
    parser.add_argument('--output-root', default=DEFAULT_ROOT)
    parser.add_argument('--project-root', default='/content/workspaces/aimo-sft-v26')
    parser.add_argument('--model-path')
    args = parser.parse_args()
    if args.check_local:
        rows, manifest = load_assets(SCRIPT.parent)
        print(json.dumps({'approved_probe_rows': len(rows), 'max_length': max(len(r['input_ids']) for r in rows),
            'candidate_pool': manifest['candidate_pool_count'], 'full_training_started': False}))
        return
    if args.worker:
        worker(args)
        return
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Mount the existing project Drive account first')
    if (Path(args.output_root) / 'PROBE_COMPLETE.json').exists():
        print('Probe already finished; read PROBE_COMPLETE.json. No GPU job started.', flush=True)
        return
    import fcntl
    lock_path = Path('/content/project1-full-sft-v46.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        no_other_gpu_job()
        load_assets(SCRIPT.parent)
        print('PROBE ONLY: restoring existing project and model; no six-epoch training.', flush=True)
        subprocess.run([sys.executable, str(SCRIPT.with_name('start_aimo_sft_v28.py')),
            '--prepare-only'], check=True)
        sys.path.insert(0, str(args.project_root))
        from experiments.aimo_sft_v26.run import prepare_model
        from experiments.aimo_sft_v26.common import config
        from experiments.aimo_sft_v26.install_missing import repair_known_torchao
        repair_known_torchao()
        ensure_dependencies()
        model = prepare_model(config())
        credential = read('/content/drive/MyDrive/.project1_credentials/swanlab.json')
        key = credential.get('SWANLAB_API_KEY')
        if not isinstance(key, str) or not key.strip():
            raise RuntimeError('Private SwanLab credential is unavailable')
        env = dict(os.environ, SWANLAB_API_KEY=key.strip(), MAX_JOBS='2',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS=str(min(8, os.cpu_count() or 2)),
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        output = Path(args.output_root)
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'launcher.log').open('a', encoding='utf-8') as logfile:
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                '--nproc_per_node=1', str(SCRIPT), '--worker', '--project-root', str(args.project_root),
                '--output-root', str(output), '--model-path', str(model)]
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    logfile.write(line)
                    logfile.flush()
                if process.wait():
                    raise RuntimeError('Probe worker failed; full training was not started. See launcher.log.')
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                raise


if __name__ == '__main__':
    main()
