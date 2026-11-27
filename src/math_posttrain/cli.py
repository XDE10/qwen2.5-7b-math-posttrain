"""Unified CLI entrypoint (`math-posttrain <command> [args...]`).

Run `math-posttrain doctor` after installation to verify configs, paths, and dependencies without a GPU.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import runpy
import sys

from math_posttrain import __version__
from math_posttrain.common import REQUIREMENTS_DIR, ROOT, load_config, workspace

COMMANDS: dict[str, tuple[str, str]] = {
    'prepare-data': ('math_posttrain.prepare_data', 'Download pinned benchmarks & build decontaminated SFT/RL datasets'),
    'infer': ('math_posttrain.infer', 'Run math reasoning inference using published HF weights (optional LoRA)'),
    'train-sft': ('math_posttrain.train_sft', 'Single-GPU full-parameter QFFT / SFT (DeepSpeed ZeRO-2 offload)'),
    'train-grpo': ('math_posttrain.grpo', 'Single-GPU LoRA-GRPO / MMR-DR-GRPO (TRL + colocated vLLM)'),
    'merge-lora': ('math_posttrain.merge_lora', 'Merge a LoRA adapter into its base checkpoint'),
    'generate': ('math_posttrain.generate', 'Run resumable 32K vLLM evaluation generation (or --engine fake)'),
    'score': ('math_posttrain.score', 'Grade generations with Math-Verify and compute paired bootstrap CIs'),
}

_DIST_NAMES: dict[str, str] = {
    'huggingface_hub': 'huggingface-hub',
    'math_verify': 'math-verify',
    'liger_kernel': 'liger-kernel',
}


def _pkg_status(module_name: str) -> str:
    dist_name = _DIST_NAMES.get(module_name, module_name)
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return 'installed' if importlib.util.find_spec(module_name) else 'missing'


def run_doctor() -> int:
    ws = workspace()
    print(f'math-posttrain {__version__}')
    print(f'Python:           {sys.version.split()[0]}')
    print(f'Repository root:  {ROOT}')
    print(f'Config dir:       {ws.config_dir} ({"ok" if ws.config_dir.is_dir() else "MISSING"})')
    print(f'Requirements dir: {REQUIREMENTS_DIR} ({"ok" if REQUIREMENTS_DIR.is_dir() else "MISSING"})')
    print(f'Data dir:         {ws.data_dir}')
    print(f'Output dir:       {ws.output_dir}')

    print('\nConfigs:')
    ok = ws.config_dir.is_dir()
    if ok:
        for path in sorted(ws.config_dir.rglob('*.json')):
            rel = path.relative_to(ws.config_dir).as_posix()
            try:
                load_config(rel)
                print(f'  [ok] {rel}')
            except Exception as exc:
                ok = False
                print(f'  [FAIL] {rel}: {exc}')

    print('\nOptional packages by stage:')
    groups = {
        'core/data': ['numpy', 'pandas', 'pyarrow', 'datasets', 'huggingface_hub'],
        'eval': ['math_verify', 'vllm', 'transformers'],
        'train': ['torch', 'accelerate', 'deepspeed', 'liger_kernel'],
        'rl/mmr': ['trl', 'peft', 'onnxruntime'],
    }
    for group, mods in groups.items():
        print(f'  {group:10s}: ' + ', '.join(f'{m}={_pkg_status(m)}' for m in mods))

    bench_count = len(list(ws.benchmarks.glob('*.jsonl'))) if ws.benchmarks.is_dir() else 0
    print(f'\nLocal benchmarks ({ws.benchmarks}): {bench_count} jsonl files')
    if not bench_count:
        print('  Tip: run `math-posttrain prepare-data --benchmarks` to download the 5 pinned math benchmarks.')
    return 0 if ok else 1


def _print_help() -> None:
    print(f'math-posttrain {__version__} — Qwen2.5-7B-Instruct mathematical reasoning post-training CLI\n')
    print('Usage:')
    print('  math-posttrain doctor')
    print('  math-posttrain <command> [command-args...]\n')
    print('Commands:')
    print(f'  {"doctor":16s} Check installation, configs, optional packages, and workspace paths')
    for name, (_, desc) in COMMANDS.items():
        print(f'  {name:16s} {desc}')
    print('\nPass `-h` / `--help` after any command for options, e.g. `math-posttrain prepare-data --help`.')


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h', '--help'):
        _print_help()
        return 0
    if args[0] in ('-V', '--version'):
        print(__version__)
        return 0
    cmd, rest = args[0], args[1:]
    if cmd == 'doctor':
        return run_doctor()
    if cmd not in COMMANDS:
        print(f'Unknown command: {cmd}\n', file=sys.stderr)
        _print_help()
        return 2
    module_name, _ = COMMANDS[cmd]
    saved_argv = sys.argv
    try:
        sys.argv = [f'math-posttrain {cmd}', *rest]
        runpy.run_module(module_name, run_name='__main__', alter_sys=True)
        return 0
    finally:
        sys.argv = saved_argv


if __name__ == '__main__':
    raise SystemExit(main())
