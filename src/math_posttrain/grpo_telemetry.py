"""Per-step timing and whole-device sampling for the GRPO length comparison."""
from __future__ import annotations

import subprocess
import threading
import time
from collections import defaultdict
from statistics import mean

import torch
from transformers import TrainerCallback

from math_posttrain.common import log


def gpu_sample():
    output = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
         '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=3,
    ).stdout.strip()
    used, total, utilization = (int(value) for value in output.split(','))
    return used / 1024, total / 1024, utilization


class ComparisonTelemetry(TrainerCallback):
    """Time the rollout pipeline separately from forward/backward and optimizer updates."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.phase = None
        self.samples = defaultdict(list)
        self.summary = {}

    def start(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def _sample(self):
        warned = False
        while not self.stop_event.is_set():
            with self.lock:
                phase = self.phase
            if phase is not None:
                try:
                    sample = gpu_sample()
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    if not warned:
                        log(f'WARNING: whole-GPU telemetry unavailable: {error}')
                        warned = True
                else:
                    with self.lock:
                        if self.phase == phase:
                            self.samples[phase[1]].append(sample)
            self.stop_event.wait(1)

    def _set_phase(self, step, phase):
        with self.lock:
            self.phase = (step, phase)

    def attach(self, trainer):
        rollout = trainer._generate_and_score_completions
        llm = trainer.vllm_generation.llm
        generate = llm.generate

        def timed_rollout(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.monotonic()
            step = trainer.state.global_step
            self._set_phase(step, 'rollout_other')
            result = rollout(*args, **kwargs)
            torch.cuda.synchronize()
            self.rollout_seconds += time.monotonic() - started
            self._set_phase(step, 'update')
            return result

        def timed_generate(*args, **kwargs):
            step = trainer.state.global_step
            self._set_phase(step, 'generation')
            started = time.monotonic()
            result = generate(*args, **kwargs)
            self.generation_seconds += time.monotonic() - started
            self._set_phase(step, 'rollout_other')
            return result

        trainer._generate_and_score_completions = timed_rollout
        llm.generate = timed_generate

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.started = time.monotonic()
        self.rollout_seconds = self.generation_seconds = 0.0
        self._set_phase(state.global_step, 'update')

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.monotonic() - self.started
        with self.lock:
            self.phase = None
            samples, self.samples = self.samples, defaultdict(list)
        summary = {
            'comparison/step_seconds': elapsed,
            'comparison/generation_seconds': self.generation_seconds,
            'comparison/rollout_other_seconds': self.rollout_seconds - self.generation_seconds,
            'comparison/update_seconds': elapsed - self.rollout_seconds,
        }
        samples['step'] = [row for phase_samples in samples.values() for row in phase_samples]
        for phase in ('step', 'generation', 'rollout_other', 'update'):
            rows = samples[phase]
            prefix = f'comparison/gpu_{phase}'
            summary[f'{prefix}_samples'] = len(rows)
            if rows:
                used, total, utilization = zip(*rows)
                summary[f'{prefix}_peak_used_gib'] = max(used)
                summary[f'{prefix}_total_gib'] = max(total)
                summary[f'{prefix}_mean_utilization_pct'] = mean(utilization)
        self.summary = summary

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            logs.update(self.summary)
            peak = logs.get('comparison/gpu_step_peak_used_gib')
            utilization = logs.get('comparison/gpu_step_mean_utilization_pct')
            card = (f'card peak {peak:.1f} GiB, utilization {utilization:.1f}%'
                    if peak is not None else 'no whole-card samples')
            log(f"comparison step {state.global_step}: "
                f"length stops {logs['comparison/length_stops']:.0f}/{logs['comparison/completions']:.0f}, "
                f"mixed unmasked groups {logs['comparison/mixed_unmasked_reward_groups']:.0f}/"
                f"{logs['comparison/groups']:.0f}, "
                f"generation {logs['comparison/generation_seconds']:.1f}s, "
                f"update {logs['comparison/update_seconds']:.1f}s, {card}")
