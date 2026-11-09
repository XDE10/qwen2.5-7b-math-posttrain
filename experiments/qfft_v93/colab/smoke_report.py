"""One block of text that summarizes the GPU smoke stages, printed at the end of the session.

Paste the block back for analysis: it holds the numbers that size the real runs (seconds per step,
peak GPU memory, host RAM, checkpoint save time, generation throughput, GRPO step time) and, for a
failed stage, the error. Standard library only; every section tolerates missing files.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

TRAIN_RUN, GRPO_RUN, EVAL_TAG = 'smoke_peak_v93', 'smoke_grpo_v93', 'smoke_base'
BEGIN = '===== P1 SMOKE SUMMARY (copy from this line to the END line) ====='
END = '===== END P1 SMOKE SUMMARY ====='
TOKENS_PER_TRAIN_STEP = 16 * 32768  # smoke_peak: effective batch 16, every row fills the window
PACKAGES = {'train': ['torch', 'transformers', 'accelerate', 'deepspeed', 'liger-kernel'],
            'infer': ['vllm', 'torch', 'transformers'], 'rl': ['vllm', 'trl', 'peft', 'torch', 'transformers'],
            'score': ['math-verify']}
ERROR_LINE = re.compile(r'Traceback|Error|error:|Killed|out of memory|OOM|CUDA', re.I)


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list:
    rows = []
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def _num(rows: list, key: str, fn=max):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return fn(values) if values else None


def _f(value, digits: int = 1) -> str:
    if value is None:
        return '-'
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def machine(local: Path) -> str:
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout.strip() or 'no GPU'
    except (OSError, subprocess.SubprocessError):
        gpu = 'no GPU (nvidia-smi not found)'
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30
    disk = shutil.disk_usage(local if local.exists() else '/')
    return (f'machine: {gpu} | host RAM {ram:.0f} GiB | {os.cpu_count()} CPUs | '
            f'local disk {disk.total / 2 ** 30:.0f} GiB, {disk.free / 2 ** 30:.0f} GiB free now')


def versions(pythons: dict) -> list:
    out = []
    for name, py in pythons.items():
        packages = PACKAGES.get(name, [])
        code = ('import importlib.metadata as m, json\nv = {}\n'
                f'for n in {packages!r}:\n'
                '    try:\n        v[n] = m.version(n)\n    except Exception:\n        v[n] = None\n')
        if name == 'train':
            code += 'import torch\nv["cuda"] = torch.version.cuda\n'
        code += 'print(json.dumps(v))'
        try:
            text = subprocess.run([py, '-c', code], capture_output=True, text=True, timeout=120).stdout.strip()
            found = json.loads(text.splitlines()[-1])
            out.append(f'env {name}: ' + ', '.join(f'{k} {v}' for k, v in found.items()))
        except Exception as error:  # an env that failed to build is part of the report
            out.append(f'env {name}: version probe failed ({type(error).__name__})')
    return out


def stage(stages_dir: Path, name: str) -> dict:
    return _json(stages_dir / f'{name}.json') or {'state': 'not run'}


def last_error(logs_dir: Path, name: str):
    logs = sorted(logs_dir.glob(f'*_{name}.log'))
    if not logs:
        return None
    try:
        lines = logs[-1].read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return None
    hits = [line.strip() for line in lines if ERROR_LINE.search(line)]
    return hits[-1][:300] if hits else None


def _state_line(label: str, marker: dict, logs_dir: Path, log_name: str) -> list:
    state = marker.get('state')
    head = f"{label}: {state}" + (f" in {marker['hours'] * 60:.0f} min" if marker.get('hours') is not None else '')
    lines = [head]
    if state == 'failed':
        lines.append(f"  error: {str(marker.get('error'))[:300]}")
        detail = last_error(logs_dir, log_name)
        if detail:
            lines.append(f'  last error line in the log: {detail}')
    return lines


def train(root: Path, stages_dir: Path, logs_dir: Path) -> list:
    run = root / 'runs' / TRAIN_RUN
    lines = _state_line('train (full FT, 2 steps of 16 x 32,768 tokens)', stage(stages_dir, 'smoke-train'),
                        logs_dir, 'smoke_train')
    steps = [r for r in _jsonl(run / 'metrics.jsonl') if 'loss' in r]
    if steps:
        per_step = [r.get('time/seconds_per_step') for r in steps]
        last = per_step[-1]
        lines.append(f"  s/step: {' / '.join(_f(x, 0) for x in per_step)} (step 1 includes warm-up) | "
                     f"~{TOKENS_PER_TRAIN_STEP / last:.0f} tok/s at the last step" if last else
                     f"  s/step: {' / '.join(_f(x, 0) for x in per_step)}")
        lines.append(f"  GPU peak allocated {_f(_num(steps, 'gpu/peak_allocated_gib'))} GiB, reserved "
                     f"{_f(_num(steps, 'gpu/peak_reserved_gib'))} GiB | host RSS max {_f(_num(steps, 'host/rss_gib'))} "
                     f"GiB, host available min {_f(_num(steps, 'host/available_gib', min))} GiB")
        lines.append('  loss: ' + ' / '.join(_f(r.get('loss'), 4) for r in steps) +
                     f" | grad_norm: {' / '.join(_f(r.get('grad_norm'), 2) for r in steps)}")
    tok = _json(run / 'tokenization.json')
    if tok:
        lines.append(f"  rows {tok.get('examples')}, longest {tok.get('max_tokens')} tokens, "
                     f"{tok.get('supervised_tokens'):,} supervised tokens")
    for save in _jsonl(run / 'saves.jsonl')[-1:]:
        lines.append(f"  checkpoint-{save.get('step')} saved locally: {_f(save.get('gib'))} GiB in "
                     f"{_f(save.get('seconds'), 0)} s; local disk free after {_f(save.get('free_gib_after'), 0)} GiB")
    return lines


def evaluate(root: Path, stages_dir: Path, logs_dir: Path) -> list:
    lines = _state_line('eval (base model, 24 generations <= 2048 tokens)', stage(stages_dir, 'smoke-eval'),
                        logs_dir, 'smoke_eval_generate')
    tag_dir = root / 'evals' / EVAL_TAG
    for bench_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()) if tag_dir.is_dir() else []:
        status = _json(bench_dir / 'generation_status.json') or {}
        summary = _json(bench_dir / 'summary.json') or {}
        session = status.get('last_session') or {}
        raw = _jsonl(bench_dir / 'raw.jsonl')
        cut = sum(1 for r in raw if r.get('finish_reason') == 'length')
        acc = summary.get('mean_at_n')
        lines.append(f"  {bench_dir.name}: {status.get('saved', len(raw))}/{status.get('expected', '?')} saved, "
                     f"{cut} hit the token cap, mean@n {_f(acc * 100 if acc is not None else None)}% | "
                     f"{_f(session.get('output_tokens_per_s'), 0)} output tok/s over "
                     f"{_f(session.get('seconds'), 0)} s of generation; vLLM start {_f(session.get('engine_start_s'), 0)} s")
    return lines


def grpo(root: Path, stages_dir: Path, logs_dir: Path) -> list:
    run = root / 'runs' / GRPO_RUN
    lines = _state_line('grpo (LoRA, 2 steps, 8 x 8 completions forced to 8192 tokens)', stage(stages_dir, 'smoke-grpo'),
                        logs_dir, 'smoke_grpo')
    steps = [r for r in _jsonl(run / 'metrics.jsonl') if 'loss' in r or 'reward' in r]
    if steps:
        lines.append(f"  s/step: {' / '.join(_f(r.get('time/seconds_per_step'), 0) for r in steps)} | GPU peak allocated "
                     f"{_f(_num(steps, 'gpu/peak_allocated_gib'))} GiB, reserved {_f(_num(steps, 'gpu/peak_reserved_gib'))}"
                     f" GiB (training process; vLLM memory is extra) | host RSS max {_f(_num(steps, 'host/rss_gib'))} GiB")
        keys = ['completions/mean_length', 'completions/clipped_ratio', 'reward', 'frac_reward_zero_std', 'entropy']
        lines.append('  ' + ' | '.join(f"{k} {_f(steps[-1].get(k), 3)}" for k in keys if k in steps[-1]))
    calls = _jsonl(run / 'reward_status.jsonl')
    if calls:
        totals = {}
        for c in calls:
            for k, v in (c.get('counts') or {}).items():
                totals[k] = totals.get(k, 0) + v
        lines.append(f'  reward grading over {len(calls)} calls: ' + ', '.join(f'{k} {v}' for k, v in sorted(totals.items())))
    return lines


def summary(drive: Path, local: Path, pythons: dict) -> str:
    root = drive / 'runs/project1/v93'
    stages_dir, logs_dir = root / 'stages', root / 'logs'
    lines = [BEGIN]
    for section in (lambda: [machine(local)], lambda: versions(pythons), lambda: train(root, stages_dir, logs_dir),
                    lambda: evaluate(root, stages_dir, logs_dir), lambda: grpo(root, stages_dir, logs_dir)):
        try:
            lines += section()
        except Exception as error:  # the report must never hide the real result
            lines.append(f'(section failed: {type(error).__name__}: {error})')
    lines.append(END)
    return '\n'.join(lines)
