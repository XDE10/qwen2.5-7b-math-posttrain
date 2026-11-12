"""Unit checks with the fake rclone: drive_io (rclone mode), CheckpointMirror over rclone, the Drive daily upload
limit, JsonlAppender mirrors, sync_file when the mount refuses a rename, report_push helpers."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

tmp = Path(tempfile.mkdtemp())
os.environ['MOCK_ROOT'] = str(tmp)
HARNESS = Path(__file__).resolve().parent
REPO = Path(os.environ.get('P1_HARNESS_REPO') or HARNESS.parents[1])
os.environ['PATH'] = str(HARNESS / 'mockbin') + ':' + os.environ['PATH']
drive_store = tmp / 'drive'
drive_store.mkdir()
mount = tmp / 'mnt'
mount.symlink_to(drive_store)          # what the fake `rclone mount` creates
os.environ['P1_DRIVE_ROOT'] = str(mount)
os.environ['P1_RCLONE_REMOTE'] = 'mydrive:'
sys.path.insert(0, str(REPO))

from experiments.qfft_v93 import drive_io  # noqa: E402
from experiments.qfft_v93.checkpointing import (CheckpointMirror, MIRROR_MARK, SAVED_MARK, prune_all_mirrors,  # noqa: E402
                                                 resolve_resume)
from experiments.qfft_v93.common import JsonlAppender, read_jsonl  # noqa: E402
from experiments.qfft_v93.colab.report_push import redact, repo_url  # noqa: E402

results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'   [{detail}]'))


def calls():
    path = tmp / 'rclone_calls.log'
    return path.read_text().splitlines() if path.exists() else []


# --- drive_io
src = tmp / 'local/tree'
(src / 'sub').mkdir(parents=True)
(src / 'a.bin').write_bytes(os.urandom(1000))
(src / 'sub/b.txt').write_text('hello')
dst = mount / 'runs/x/final'
drive_io.upload_tree(src, dst)
check('upload_tree publishes the tree', (dst / 'a.bin').stat().st_size == 1000 and (dst / 'sub/b.txt').read_text() == 'hello')
check('upload_tree went through a .partial folder and a server-side move',
      any(c.startswith('copy ') and 'final.partial' in c for c in calls()) and any(c.startswith('moveto ') for c in calls())
      and not (mount / 'runs/x/final.partial').exists(), calls())
(src / 'a.bin').write_bytes(os.urandom(10))
drive_io.upload_tree(src, dst)
check('re-upload replaces the old tree', (dst / 'a.bin').stat().st_size == 10)
drive_io.put_json(dst / 'MARK.json', {'ok': 1})
check('put_json uploads a marker', json.loads((dst / 'MARK.json').read_text()) == {'ok': 1})
back = tmp / 'local/restored'
drive_io.download_tree(dst, back)
check('download_tree (atomic) restores', (back / 'sub/b.txt').read_text() == 'hello' and not (tmp / 'local/restored.partial').exists())
drive_io.purge(dst)
check('purge deletes permanently', not dst.exists() and any(c.startswith('purge ') and '--drive-use-trash=false' in c for c in calls()))
drive_io.purge(mount / 'runs/x/never_existed')
check('purge of a missing folder is not an error', True)

# --- CheckpointMirror over rclone: three mirrored saves, keep 1, then resume from Drive on a "new VM"
local_run, drive_run = tmp / 'vm1/runs/r', mount / 'runs/project1/v93/runs/r'
mirror = CheckpointMirror(local_run, drive_run, every=1, keep=1, watch_path=str(tmp), mirror_optimizer_state=True)
for step in (2, 4, 6):
    state = SimpleNamespace(global_step=step, epoch=step / 10, max_steps=10)
    mirror.on_step_end(None, state, SimpleNamespace(should_save=True))
    ckpt = local_run / f'checkpoint-{step}'
    ckpt.mkdir(parents=True)
    (ckpt / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
    (ckpt / 'optim.pt').write_bytes(os.urandom(2000))
    mirror.on_save(None, state, SimpleNamespace(should_save=False))
    mirror.wait()
names = sorted(p.name for p in (drive_run / 'ckpt_mirror').iterdir())
check('only the newest mirror is kept on Drive', names == ['checkpoint-6'], names)
check('mirror has its completion marker', (drive_run / 'ckpt_mirror/checkpoint-6' / MIRROR_MARK).is_file())
check('saves.jsonl has one record per save', [r['step'] for r in read_jsonl(drive_run / 'saves.jsonl')] == [2, 4, 6])
check('mirror error is empty', mirror.error is None, mirror.error)
resume = resolve_resume(tmp / 'vm2/runs/r', drive_run)
check('resume on a new VM restores the newest mirror', resume and resume.endswith('checkpoint-6')
      and (Path(resume) / 'optim.pt').stat().st_size == 2000 and (Path(resume) / SAVED_MARK).is_file(), resume)
prune_all_mirrors(drive_run)
check('prune_all_mirrors empties the mirror folder', not any((drive_run / 'ckpt_mirror').iterdir()))

# --- JsonlAppender with a Drive mirror
local_j, drive_j = tmp / 'vm1/metrics.jsonl', mount / 'runs/project1/v93/runs/r/metrics.jsonl'
j = JsonlAppender(local_j, mirror=drive_j, mirror_seconds=0.2)
j.write({'step': 1})
check('first line stays local until the mirror interval', not drive_j.exists())
time.sleep(0.3)
j.write({'step': 2})
check('mirror updated after the interval', [r['step'] for r in read_jsonl(drive_j)] == [1, 2])
j.write({'step': 3})
j.close()
check('close updates the mirror', [r['step'] for r in read_jsonl(drive_j)] == [1, 2, 3])
j2 = JsonlAppender(tmp / 'vm3/metrics.jsonl', mirror=drive_j, mirror_seconds=0.2)
j2.write({'step': 4})
j2.close()
check('a new VM continues the Drive history', [r['step'] for r in read_jsonl(drive_j)] == [1, 2, 3, 4])

# --- Drive daily upload limit: rclone is told to stop, and the error says what happened
check('uploads stop at the Drive daily limit (--drive-stop-on-upload-limit)',
      '--drive-stop-on-upload-limit' in drive_io.UPLOAD_FLAGS
      and any(c.startswith('copy ') and '--drive-stop-on-upload-limit' in c for c in calls()))
os.environ['MOCK_RCLONE_UPLOAD_LIMIT'] = '1'
try:
    drive_io.upload_tree(src, mount / 'runs/x/limited')
    check('an upload past the daily limit raises', False, 'no exception')
except RuntimeError as error:
    check('an upload past the daily limit raises with the 750 GiB explanation',
          'Received upload limit error' in str(error) and '750 GiB' in str(error), str(error)[-300:])
del os.environ['MOCK_RCLONE_UPLOAD_LIMIT']
limited = CheckpointMirror(tmp / 'vm4/runs/l', mount / 'runs/project1/v93/runs/l', every=1, keep=1, watch_path=str(tmp))
os.environ['MOCK_RCLONE_UPLOAD_LIMIT'] = '1'
state = SimpleNamespace(global_step=2, epoch=0.2, max_steps=10)
ckpt = tmp / 'vm4/runs/l/checkpoint-2'
ckpt.mkdir(parents=True)
(ckpt / 'trainer_state.json').write_text('{}')
limited.on_save(None, state, SimpleNamespace(should_save=False))
limited.wait()
del os.environ['MOCK_RCLONE_UPLOAD_LIMIT']
check('a mirror that hits the limit fails fast as a warning; the local checkpoint stays',
      limited.error and 'upload limit' in limited.error and (ckpt / 'trainer_state.json').exists(), limited.error)

# --- sync_file when the mount refuses the rename of its temp file (seen with rclone on a G4 VM)
from experiments.qfft_v93 import common  # noqa: E402
log_src, log_dst = tmp / 'vm1/stage.log', mount / 'runs/project1/v93/logs/stage.log'
log_src.write_text('line 1\nline 2\n')
real_replace = common._replace_with_retry


def refuse(tmp_path, path):
    raise FileNotFoundError(2, 'No such file or directory', str(tmp_path))


common._replace_with_retry = refuse
common.sync_file(log_src, log_dst)
common._replace_with_retry = real_replace
check('sync_file falls back to a direct copy and leaves no temp file',
      log_dst.read_text() == 'line 1\nline 2\n' and not list(log_dst.parent.glob('stage.log.*.sync')),
      sorted(p.name for p in log_dst.parent.iterdir()))

# --- report_push helpers
cases = {
    'git@github.com:XDE10/qwen2.5-7b-math-posttrain.git': 'https://github.com/XDE10/qwen2.5-7b-math-posttrain.git',
    'https://github.com/XDE10/qwen2.5-7b-math-posttrain.git': 'https://github.com/XDE10/qwen2.5-7b-math-posttrain.git',
    'https://XDE10:ghp_abcdefghijklmnopqrstuvwxyz0123@github.com/XDE10/qwen2.5-7b-math-posttrain': 'https://github.com/XDE10/qwen2.5-7b-math-posttrain.git',
    'ssh://git@github.com/XDE10/repo.git': 'https://github.com/XDE10/repo.git',
    'https://gateway.example.com/github/XDE10/repo.git': None,
    '': None,
}
for value, want in cases.items():
    check(f'repo_url({value[:40]!r})', repo_url(value) == want, repo_url(value))
text = ('push https://XDE10:ghp_abcdefghijklmnopqrstuvwxyz0123@github.com/x.git; github_pat_11ABCDEFGHIJKLMNOPQRSTUV_xyz; '
        'Authorization: Bearer sk-abcdefghijklmnopqrstu; key sk-proj-abcdefghijklmnop; ok text 12345')
red = redact(text)
check('redact removes tokens', all(t not in red for t in ('ghp_abc', 'github_pat_11', 'sk-abc', 'sk-proj')) and 'ok text 12345' in red, red)

print(f'{sum(results)}/{len(results)} checks passed')
sys.exit(0 if all(results) else 1)
