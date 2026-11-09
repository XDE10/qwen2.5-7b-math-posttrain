"""smoke_report on synthetic Drive files + CheckpointMirror save timing (test only)."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.environ.get('P1_HARNESS_REPO') or str(Path(__file__).resolve().parents[2]))
from experiments.qfft_v93.colab import smoke_report  # noqa: E402
from experiments.qfft_v93.checkpointing import CheckpointMirror  # noqa: E402

results = []


def check(name, ok, detail=''):
    results.append(ok)
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'   [{detail}]'))


tmp = Path(tempfile.mkdtemp())
drive, local = tmp / 'drive', tmp / 'local'
root = drive / 'runs/project1/v93'
(root / 'stages').mkdir(parents=True)
(root / 'logs').mkdir(parents=True)
local.mkdir()

# --- CheckpointMirror records the local save (step 2 of a 2-step smoke: final save, no mirror)
run = root / 'runs/smoke_peak_v93'
local_run = local / 'runs/smoke_peak_v93'
mirror = CheckpointMirror(local_run, run, every=1000, keep=1, watch_path=str(local))
state = SimpleNamespace(global_step=2, epoch=0.06, max_steps=2)
mirror.on_step_end(None, state, SimpleNamespace(should_save=True))
ckpt = local_run / 'checkpoint-2'
ckpt.mkdir(parents=True)
(ckpt / 'trainer_state.json').write_text('{}')
(ckpt / 'weights.bin').write_bytes(os.urandom(3 * 2 ** 20))
mirror.on_save(None, state, SimpleNamespace(should_save=False))
saves = [json.loads(line) for line in (run / 'saves.jsonl').read_text().splitlines()]
check('save recorded with duration, size and free space', len(saves) == 1 and saves[0]['step'] == 2
      and saves[0]['seconds'] is not None and saves[0]['bytes'] > 3 * 2 ** 20
      and saves[0]['free_gib_after'] > 0, str(saves))

# --- synthetic outputs of the three smoke stages
with (run / 'metrics.jsonl').open('w') as f:
    for step, sec in ((1, 812.4), (2, 401.7)):
        f.write(json.dumps({'step': step, 'loss': 1.2345 - step / 10, 'grad_norm': 3.1, 'time/seconds_per_step': sec,
                            'gpu/peak_allocated_gib': 52.5 + step, 'gpu/peak_reserved_gib': 60.0 + step,
                            'host/rss_gib': 120.0 + step, 'host/available_gib': 40.0 - step}) + '\n')
(run / 'tokenization.json').write_text(json.dumps({'examples': 32, 'max_tokens': 32768, 'supervised_tokens': 1048576}))
(root / 'stages/smoke-train.json').write_text(json.dumps({'stage': 'smoke-train', 'state': 'complete', 'hours': 0.42}))
bench = root / 'evals/smoke_base/math500'
bench.mkdir(parents=True)
(bench / 'generation_status.json').write_text(json.dumps({'expected': 16, 'saved': 16, 'complete': True, 'last_session': {
    'output_tokens_per_s': 2345.6, 'seconds': 31.0, 'engine_start_s': 95.0}}))
(bench / 'summary.json').write_text(json.dumps({'mean_at_n': 0.75}))
(bench / 'raw.jsonl').write_text('\n'.join(json.dumps({'finish_reason': 'length' if i < 3 else 'stop'}) for i in range(16)))
(root / 'stages/smoke-eval.json').write_text(json.dumps({'stage': 'smoke-eval', 'state': 'complete', 'hours': 0.05}))
(root / 'stages/smoke-grpo.json').write_text(json.dumps({'stage': 'smoke-grpo', 'state': 'failed',
                                                         'error': 'smoke_grpo: RuntimeError: smoke_grpo failed with exit code 1'}))
(root / 'logs/20261109-101010_smoke_grpo.log').write_text('loading\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB\n')

text = smoke_report.summary(drive, local, {'score': sys.executable})
print(text)
check('block has begin and end markers', text.startswith(smoke_report.BEGIN) and text.endswith(smoke_report.END))
check('machine line', 'machine:' in text and 'host RAM' in text)
check('train numbers', 's/step: 812 / 402' in text and 'GPU peak allocated 54.5 GiB, reserved 62.0 GiB' in text
      and 'host RSS max 122.0' in text and 'host available min 38.0' in text, text)
check('train throughput', '~1305 tok/s at the last step' in text, text)
check('checkpoint save line', 'checkpoint-2 saved locally:' in text, text)
check('eval line', 'math500: 16/16 saved, 3 hit the token cap, mean@n 75.0%' in text and '2346 output tok/s' in text, text)
check('failed grpo with the OOM line', 'grpo (LoRA' in text and 'failed' in text and 'CUDA out of memory' in text, text)
check('env probe line', 'env score: math-verify' in text, text)
print(f'{sum(results)}/{len(results)} checks passed')
sys.exit(0 if all(results) else 1)
