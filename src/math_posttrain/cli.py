"""Unified CLI entrypoint (`math-posttrain <command> [args...]`).

Run `math-posttrain doctor` right after installation to verify dependencies,
configs, and local workspace paths without requiring a GPU.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import runpy
import sys
from pathlib import Path

from math_posttrain import __version__
from math_posttrain.common import CONFIG_DIR, REQUIREMENTS_DIR, ROOT, drive_layout, load_config, local_root

COMMANDS: dict[str, tuple[str, str]] = {
    "fetch-benchmarks": ("math_posttrain.fetch_benchmarks", "Download & verify pinned math benchmarks from Hugging Face"),
    "build-data": ("math_posttrain.data_build", "Plan, inspect, or build the 1,000-problem SFT/QFFT datasets"),
    "fill-teacher": ("math_posttrain.teacher", "Generate or verify DeepSeek teacher traces via API"),
    "train-sft": ("math_posttrain.train_sft", "Run single-GPU full-parameter SFT or QFFT (DeepSpeed ZeRO-2 offload)"),
    "train-grpo": ("math_posttrain.grpo", "Run single-GPU LoRA-GRPO / MMR-DR-GRPO (TRL + colocated vLLM)"),
    "merge-lora": ("math_posttrain.merge_lora", "Merge a trained LoRA adapter into its base checkpoint"),
    "generate": ("math_posttrain.generate", "Run resumable vLLM evaluation generation"),
    "score": ("math_posttrain.score", "Grade generated outputs with Math-Verify and compute paired bootstrap CIs"),
    "analyze": ("math_posttrain.analyze", "Generate Pareto, difficulty, and pass@k analysis tables/charts"),
    "diagnose-rollouts": ("math_posttrain.rollout_diagnostics", "Replay & compare GRPO rollout grading offline"),
    "diagnose-is": ("math_posttrain.grpo_is_diagnostic", "Compare sequence vs token vLLM importance-sampling weights"),
}


_DIST_NAMES: dict[str, str] = {
    "huggingface_hub": "huggingface-hub",
    "math_verify": "math-verify",
    "liger_kernel": "liger-kernel",
}


def _pkg_status(module_name: str) -> str:
    dist_name = _DIST_NAMES.get(module_name, module_name)
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "installed" if importlib.util.find_spec(module_name) else "missing"


def run_doctor() -> int:
    print(f"math-posttrain {__version__}")
    print(f"Python:           {sys.version.split()[0]}")
    print(f"Repository root:  {ROOT}")
    print(f"Config dir:       {CONFIG_DIR} ({'ok' if CONFIG_DIR.is_dir() else 'MISSING'})")
    print(f"Requirements dir: {REQUIREMENTS_DIR} ({'ok' if REQUIREMENTS_DIR.is_dir() else 'MISSING'})")
    print(f"Workspace dir:    {drive_layout().root}")
    print(f"Cache dir:        {local_root()}")

    print("\nConfigs:")
    ok = CONFIG_DIR.is_dir()
    if ok:
        cfg_files = sorted(p for p in CONFIG_DIR.rglob("*.json") if "approvals" not in p.parts)
        for path in cfg_files:
            rel = path.relative_to(CONFIG_DIR).as_posix()
            try:
                load_config(rel)
                print(f"  [ok] {rel}")
            except Exception as exc:
                ok = False
                print(f"  [FAIL] {rel}: {exc}")

    print("\nOptional packages by stage:")
    groups = {
        "core/data": ["numpy", "pandas", "pyarrow", "tokenizers", "datasets", "huggingface_hub"],
        "eval": ["math_verify", "vllm", "transformers", "matplotlib"],
        "train": ["torch", "accelerate", "deepspeed", "liger_kernel", "swanlab"],
        "rl/mmr": ["trl", "peft", "onnxruntime"],
    }
    for group, mods in groups.items():
        summary = ", ".join(f"{m}={_pkg_status(m)}" for m in mods)
        print(f"  {group:10s}: {summary}")

    bench_dir = ROOT / "data" / "benchmarks_v93"
    bench_count = len(list(bench_dir.glob("*.jsonl"))) if bench_dir.is_dir() else 0
    print(f"\nLocal benchmarks ({bench_dir}): {bench_count} jsonl files")
    if not bench_count:
        print("  Tip: run `math-posttrain fetch-benchmarks` to download the 5 pinned math benchmarks.")
    return 0 if ok else 1


def _print_help() -> None:
    print(f"math-posttrain {__version__} — Qwen2.5-7B-Instruct mathematical reasoning post-training CLI\n")
    print("Usage:")
    print("  math-posttrain doctor")
    print("  math-posttrain <command> [command-args...]\n")
    print("Commands:")
    print(f"  {'doctor':18s} Check installation, configs, optional packages, and local data")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:18s} {desc}")
    print("\nPass `-h` / `--help` after any command to see its options, e.g. `math-posttrain score --help`.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return 0
    if args[0] in ("-V", "--version"):
        print(__version__)
        return 0
    cmd, rest = args[0], args[1:]
    if cmd == "doctor":
        return run_doctor()
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}\n", file=sys.stderr)
        _print_help()
        return 2
    module_name, _ = COMMANDS[cmd]
    saved_argv = sys.argv
    try:
        sys.argv = [f"math-posttrain {cmd}", *rest]
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        return 0
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
