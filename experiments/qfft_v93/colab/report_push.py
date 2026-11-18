"""Push small text reports from the Colab VM to a separate branch of the GitHub repository.

The assistant reads results from GitHub instead of the user downloading and re-uploading them. Only small
text goes out (size-capped, secrets redacted), always to the `colab-reports` branch, never to main:

    v93/latest/status.md                 stage markers, run progress, evaluation completeness
    v93/latest/smoke_summary.txt         the P1 SMOKE SUMMARY block of the latest smoke
    v93/latest/runs/<run>/...            status, identity, tokenization, saves, metrics.jsonl (tail), ...
    v93/latest/evals/...                 results.md, results.json, per-benchmark summary/generation status
    v93/sessions/<session>/README.md     what ran, on which GPU, with which code, and the result
    v93/sessions/<session>/session_log_tail.txt, logs/<stage>.log (tails), smoke_summary.txt

Authentication: a git credential-store file on the VM (default /content/gh_credentials, one line
`https://USER:TOKEN@github.com`), uploaded by launch.sh from the user's machine or created by hand in
`colab console`. The token never appears in a command line, a remote URL or a pushed file.
Standard library + the git CLI only; every failure is a warning, never a job failure.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

LOG_TAIL = 300_000      # bytes of the session log
STAGE_LOG_TAIL = 64_000  # bytes per stage log
FILE_CAP = 2_000_000     # larger JSON/JSONL/MD files are tailed to this size
RUN_FILES = ['status.json', 'run_identity.json', 'tokenization.json', 'TRAIN_COMPLETE.json', 'GRPO_COMPLETE.json',
             'saves.jsonl', 'rl_pool_report.json', 'swanlab.json', 'metrics.jsonl', 'reward_status.jsonl']
EVAL_FILES = ['summary.json', 'generation_status.json', 'identity.json']
SECRET = re.compile(r'(github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{16,}|oc_sk_[\w-]+'
                    r'|(?i:bearer)\s+[A-Za-z0-9._~+/=-]{12,}|(?<=://)[^/\s:@]+:[^@\s/]+(?=@))')


def redact(text: str) -> str:
    return SECRET.sub('[REDACTED]', text)


def _tail_text(path: Path, limit: int) -> str | None:
    try:
        size = path.stat().st_size
        with path.open('rb') as handle:
            if size > limit:
                handle.seek(size - limit)
                data = handle.read()
                data = data[data.find(b'\n') + 1:]  # start on a full line
                prefix = f'[... first {size - len(data):,} bytes omitted ...]\n'
            else:
                data, prefix = handle.read(), ''
        return prefix + data.decode('utf-8', errors='replace')
    except OSError:
        return None


def _put(dst: Path, text: str | None) -> None:
    if text is None:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(redact(text), encoding='utf-8')


class Reporter:
    def __init__(self, cfg: dict, drive: Path, local: Path, session: str, info: dict):
        self.repo = cfg.get('repo') or ''
        self.branch = cfg.get('branch') or 'colab-reports'
        self.credentials = Path(cfg.get('credentials') or '/content/gh_credentials')
        self.interval = float(cfg.get('interval_minutes') or 30) * 60
        self.drive, self.local, self.session, self.info = drive, local, session or 'unnamed', info
        self.root = drive / 'runs/project1/v93'
        self.work = local / 'report_repo'
        self.enabled = bool(self.repo) and (self.credentials.is_file() or not self.repo.startswith('https://'))
        self.last_error = None
        self.lock = threading.Lock()  # the periodic and the final push share one work tree

    # ------------------------------------------------------------------ git
    def _git(self, *args, timeout: float = 90):
        env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_ASKPASS='true', SSH_ASKPASS='true')
        cmd = ['git', '-c', 'credential.helper=', '-c', f'credential.helper=store --file={self.credentials}',
               '-c', 'user.name=p1-colab', '-c', 'user.email=p1-colab@users.noreply.github.com',
               '-c', 'init.defaultBranch=main', '-c', 'core.autocrlf=false', '-C', str(self.work), *args]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)

    def _sync_branch(self) -> None:
        """A fresh shallow copy of the reports branch (or a new orphan branch)."""
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True)
        self._git('init', '-q')
        self._git('remote', 'add', 'origin', self.repo)
        fetched = self._git('fetch', '-q', '--depth', '1', 'origin', self.branch)
        if fetched.returncode == 0:
            self._git('checkout', '-q', '-B', self.branch, 'FETCH_HEAD')
        elif 'couldn' in fetched.stderr.lower() or 'not found' in fetched.stderr.lower():
            self._git('checkout', '-q', '--orphan', self.branch)
        else:
            raise RuntimeError('git fetch failed: ' + redact(fetched.stderr.strip())[-400:])

    # ------------------------------------------------------------------ content
    def _collect(self, result: str | None) -> None:
        base = self.work / 'v93'
        latest, session = base / 'latest', base / 'sessions' / self.session
        shutil.rmtree(latest, ignore_errors=True)
        now = datetime.now().isoformat(timespec='seconds')
        info = {**self.info, 'updated': now, 'result': result or 'running'}
        lines = [f"# Colab session {self.session}", '',
                 *(f'- {k}: {v}' for k, v in info.items()), '']
        _put(session / 'README.md', '\n'.join(lines))
        _put(session / 'session_log_tail.txt', _tail_text(self.local / 'session.log', LOG_TAIL))
        try:
            from experiments.qfft_v93.colab import stages  # same kernel; already imported by remote.py
            status = '\n'.join(stages.status_lines())
        except Exception as error:
            status = f'(status unavailable: {type(error).__name__}: {error})'
        _put(latest / 'status.md', f'# v93 status ({now}, session {self.session})\n\n```\n{status}\n```\n')
        _put(latest / 'session.txt', f'{self.session}\n{result or "running"}\n')
        smoke = self.root / 'smoke' / f'summary_{self.session}.txt'
        if smoke.is_file():
            text = _tail_text(smoke, FILE_CAP)
            _put(session / 'smoke_summary.txt', text)
            _put(base / 'latest_smoke_summary.txt', text)
        for log in sorted((self.root / 'logs').glob('*.log')) if (self.root / 'logs').is_dir() else []:
            if log.stat().st_mtime >= self.info.get('started_ts', 0):
                _put(session / 'logs' / log.name, _tail_text(log, STAGE_LOG_TAIL))
        runs = self.root / 'runs'
        for run in sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []:
            for name in RUN_FILES:
                if (run / name).is_file():
                    _put(latest / 'runs' / run.name / name, _tail_text(run / name, FILE_CAP))
        evals = self.root / 'evals'
        for dev_suite in ('grpo_dev', 'r1_competition_2026', 'r1_fura50_eval'):
            for name in ('results.md', 'results.json'):
                path = self.root / dev_suite / name
                if path.is_file():
                    _put(latest / dev_suite / name, _tail_text(path, FILE_CAP))
        for suite in ('r1_mmr_dr50_eval', 'r1_mmr_dr150_eval'):
            for name in ('results.md', 'results.json', 'per_problem_pairs.jsonl'):
                path = self.root / suite / name
                if path.is_file():
                    text = _tail_text(path, FILE_CAP)
                    _put(latest / suite / name, text)
                    if path.stat().st_mtime >= self.info.get('started_ts', 0):
                        _put(session / suite / name, text)
        for name in ('results.md', 'results.json'):
            if (evals / name).is_file():
                _put(latest / 'evals' / name, _tail_text(evals / name, FILE_CAP))
        for tag in sorted(p for p in evals.iterdir() if p.is_dir()) if evals.is_dir() else []:
            for bench in sorted(p for p in tag.iterdir() if p.is_dir()):
                for name in EVAL_FILES:
                    if (bench / name).is_file():
                        _put(latest / 'evals' / tag.name / bench.name / name, _tail_text(bench / name, FILE_CAP))
        stages_dir = self.root / 'stages'
        markers = {}
        for marker in sorted(stages_dir.glob('*.json')) if stages_dir.is_dir() else []:
            try:
                markers[marker.stem] = json.loads(marker.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass
        _put(latest / 'stages.json', json.dumps(markers, ensure_ascii=False, indent=1))

    # ------------------------------------------------------------------ push
    def push(self, message: str, result: str | None = None, attempts: int = 3) -> bool:
        if not self.enabled:
            return False
        with self.lock:
            return self._push(message, result, attempts)

    def _push(self, message: str, result: str | None, attempts: int) -> bool:
        for attempt in range(1, attempts + 1):
            try:
                self._sync_branch()
                self._collect(result)
                self._git('add', '-A')
                if self._git('diff', '--cached', '--quiet').returncode == 0:
                    return True
                committed = self._git('commit', '-q', '-m', message)
                if committed.returncode != 0:
                    raise RuntimeError('git commit failed: ' + redact(committed.stderr.strip())[-400:])
                pushed = self._git('push', '-q', 'origin', f'HEAD:refs/heads/{self.branch}', timeout=120)
                if pushed.returncode == 0:
                    print(f'[report] pushed to {self.branch}: {message}', flush=True)
                    self.last_error = None
                    return True
                raise RuntimeError('git push failed: ' + redact(pushed.stderr.strip())[-400:])
            except Exception as error:  # includes a remote that moved on (another VM pushed): retry fresh
                self.last_error = f'{type(error).__name__}: {error}'
                if attempt < attempts:
                    time.sleep(5 * attempt)
        print(f'[report] WARNING: could not push to GitHub ({self.last_error}); results stay on Drive', flush=True)
        return False


def repo_url(value: str) -> str | None:
    """https://github.com/OWNER/REPO.git for any GitHub remote form, without credentials; else None."""
    value = (value or '').strip()
    match = (re.match(r'^git@github\.com:([\w.-]+)/([\w.-]+?)(?:\.git)?/?$', value) or
             re.match(r'^(?:ssh://git@|https?://(?:[^@/]+@)?)github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$', value))
    return f'https://github.com/{match.group(1)}/{match.group(2)}.git' if match else None


if __name__ == '__main__':  # used by launch.sh to normalize `git remote get-url origin`
    import sys
    print(repo_url(sys.argv[1]) or '')
