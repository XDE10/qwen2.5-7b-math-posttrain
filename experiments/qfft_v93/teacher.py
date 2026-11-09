"""Fill DeepSeek-V4.1-Flash gaps until every plan question has a usable trace (local workstation, API only).

    python -m experiments.qfft_v93.teacher status
    python -m experiments.qfft_v93.teacher fill   [--provider go|official] [--workers 4] [--max-attempts 8]
    python -m experiments.qfft_v93.teacher verify [--max-attempts 12]   # optional: resample checkably-wrong ones

* `fill` only targets questions without a usable trace (see data_build.judge_candidate) and stops per
  question at the first usable attempt, so the `ds` variant stays "first usable attempt" for all 1000.
* `verify` adds attempts for questions whose usable trace is checkably wrong; only the optional
  `ds_verified` variant may use them.
* Every attempt is a new folder artifacts/qfft_teacher_v93/attempts/<index>/aNN/ with request.json,
  stream_chunks.jsonl, response.json and candidate.json (or error.json). Re-running resumes; a request
  whose outcome is unknown is never reused, the next attempt number is taken instead.
* The per-question budget (--max-attempts) counts attempts that produced or may have produced a
  completion (candidate, broken stream, interrupted run). Requests the provider refused before generating
  anything (quota, rate limit, HTTP 5xx) take an attempt number but not budget.
* The key is loaded before any attempt folder is created. Authentication or request errors (HTTP 400/401/
  403/404/413/422) stop the whole run; so do 5 failures in a row or 20 in one run (outage protection),
  so an outage cannot use up every question's budget.
* `fill` exits with code 3 when questions are still without a usable trace (budget used up).
* Keys are read from environment variables or private files and are never written or printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path

from experiments.qfft_v93.common import exclusive_lock, log, now_iso, read_json, write_json
from experiments.qfft_v93.data_build import DS_DIRS, TokenCounter, ds_selection, judge_candidate, load_plan
from experiments.qfft_v93.grading import Grader

OUT = DS_DIRS['v93']
PROVIDERS = {
    'go': {'url': 'https://opencode.ai/inference/go/openai/v1/chat/completions', 'model': 'deepseek-v4.1-flash'},
    'official': {'url': 'https://api.deepseek.com/chat/completions', 'model': 'deepseek-flash'},
}
# Key locations, Windows first, then the same files seen from WSL2 (/mnt/<drive>), then Linux/Colab places.
# Environment variables (OPENCODE_GO_API_KEY, DEEPSEEK_API_KEY) and --go-db / --official-key-file come first.
DEFAULT_GO_DBS = ['D:/OpenCode/runtime/data/opencode/opencode.db', '/mnt/d/OpenCode/runtime/data/opencode/opencode.db',
                  str(Path.home() / '.local/share/opencode/opencode.db')]
DEFAULT_OFFICIAL_KEYS = ['C:/Users/Mr.Z/.codex/private/project1/deepseek.json',
                         '/mnt/c/Users/Mr.Z/.codex/private/project1/deepseek.json',
                         str(Path.home() / '.config/p1/deepseek.json'),
                         '/content/mydrive/.project1_credentials/deepseek.json',
                         '/content/drive/MyDrive/.project1_credentials/deepseek.json']

# Exact copy of the v66 teacher system prompt (tools/qfft/qfft_ds_common_v66.py: SYSTEM).
SYSTEM_V66 = r'''Please reason step by step, and put your final answer within \boxed{}.
Work from the original problem and give a complete, self-contained derivation that a student can follow.
Show nontrivial algebraic transformations and justify essential inferences. When substituting variables, subtracting equations, or changing notation, recompute every affected coefficient and constant from the equation being transformed; do not carry a value over from a different equation without checking it.
State relevant domain restrictions and boundary cases. For an optimum, prove the global bound and show that equality is attainable under the original conditions. For cases or enumeration, justify that all possibilities are covered. Numerical trials may check a result but must not replace an exact argument when a proof is required.
Use alternative approaches and independent checks when they help resolve a real uncertainty. If an attempted approach fails, identify the obstruction, correct the error, and resolve any resulting contradiction. Do not invent failed attempts or add unnecessary alternative methods to an already complete argument.
Before giving the final answer, recheck the intermediate equations and key calculations that will appear in the final solution, verify substitutions against the original conditions, and ensure the final derivation is internally consistent and agrees with the conclusion reached during reasoning. Resolve any mismatch rather than copying an unchecked expression into the final answer.
Use consistent notation and well-formed LaTeX. Give an unambiguous final result answering the original question, within \boxed{}.'''


def system_v91() -> str:
    """The v91 anti-redundancy variant, rebuilt exactly as tools/qfft/supplement_incomplete_high_v91.pyw."""
    lines = SYSTEM_V66.splitlines()
    assert lines[4].startswith('Use alternative approaches') and lines[5].startswith('Before giving the final answer')
    return '\n'.join(lines[:4] + [
        'Once a rigorous derivation reaches a verified result, proceed directly to the final solution without '
        'repeating full-problem recalculations or exploring redundant secondary methods. Avoid repetitive '
        'self-doubt loops.',
        'If an approach fails, identify and correct the specific error, resolve contradictions, and continue '
        'with a viable method. Keep all essential derivations and checks needed to justify the result.',
    ] + lines[6:])


PROMPTS = {'v66_original': SYSTEM_V66, 'v91_anti_redundancy': system_v91()}


class QuotaExhausted(RuntimeError):
    pass


class RateLimited(RuntimeError):
    pass


class FatalRequestError(RuntimeError):
    """The provider rejected the request itself (auth, model name, payload): every attempt would fail."""


class ServerError(RuntimeError):
    """HTTP error status before any streamed output (5xx and retryable 4xx)."""


class TooManyErrors(RuntimeError):
    pass


FATAL_HTTP = {400, 401, 403, 404, 405, 413, 415, 422}
# error.json kinds that prove no completion was generated; they do not count toward --max-attempts.
NO_OUTPUT_KINDS = {'quota', 'rate_limited', 'rejected', 'server_error', 'cancelled'}


def load_key(provider: str, args) -> str:
    if provider == 'go':
        value = os.environ.get('OPENCODE_GO_API_KEY')
        for db_path in ([args.go_db] if args.go_db else DEFAULT_GO_DBS):
            if value or not db_path or not Path(db_path).is_file():
                continue
            with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
                rows = db.execute('SELECT value FROM credential WHERE integration_id=? AND active=1',
                                  ('opencode-go',)).fetchall()
            if len(rows) == 1:
                obj = json.loads(rows[0][0])
                value = obj.get('key') if obj.get('type') in ('key', 'api') else None
    else:
        value = os.environ.get('DEEPSEEK_API_KEY')
        for path in ([args.official_key_file] if args.official_key_file else DEFAULT_OFFICIAL_KEYS):
            if value:
                break
            if path and Path(path).exists():
                value = read_json(path).get('DEEPSEEK_API_KEY')
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f'No API key for provider {provider}; set the environment variable or key file')
    return value.strip()


def request_body(question: str, effort: str, prompt: str, max_tokens: int, provider: str) -> dict:
    return {'model': PROVIDERS[provider]['model'],
            'messages': [{'role': 'system', 'content': PROMPTS[prompt]}, {'role': 'user', 'content': question}],
            'thinking': {'type': 'enabled'}, 'reasoning_effort': effort, 'max_tokens': max_tokens, 'top_p': 1.0,
            'stream': True, 'stream_options': {'include_usage': True}}


def _redact(text: str, secret: str) -> str:
    text = text.replace(secret, '[REDACTED]')
    return re.sub(r'(?i)Bearer\s+\S+|\bsk-[\w-]{16,}|oc_sk_[\w-]+', '[REDACTED]', text)[:2000]


def stream_chat(url: str, body: dict, secret: str, session: str, folder: Path, provider: str) -> dict:
    """OpenAI-style SSE; journals every chunk; returns the assembled completion or raises."""
    headers = {'Authorization': 'Bearer ' + secret, 'Content-Type': 'application/json',
               'Accept': 'text/event-stream', 'User-Agent': 'project1-qfft-data/93'}
    if provider == 'go':
        headers['x-opencode-session'] = session
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    reasoning, content, usage, finish, model, done, events = [], [], None, None, None, False, 0
    try:
        response = urllib.request.urlopen(request, timeout=1800)
    except urllib.error.HTTPError as error:
        text = _redact(error.read().decode('utf-8', 'replace'), secret)
        if error.code == 402 or (error.code == 429 and re.search(
                r'quota|allowance|usage[ _-]*limit|spending|balance', text, re.I)):
            raise QuotaExhausted(f'HTTP {error.code}: {text[:300]}') from None
        if error.code == 429:
            raise RateLimited(text[:300]) from None
        if error.code in FATAL_HTTP:
            raise FatalRequestError(f'HTTP {error.code}: {text[:300]}') from None
        raise ServerError(f'HTTP {error.code}: {text[:300]}') from None

    def handle(data: str, journal) -> bool:
        """One SSE event; True at [DONE]."""
        nonlocal events, model, usage, finish
        if data == '[DONE]':
            return True
        event = json.loads(data)
        if event.get('error'):
            raise RuntimeError('Provider SSE error event: ' + _redact(json.dumps(event['error']), secret))
        journal.write(json.dumps(event, ensure_ascii=False) + '\n')
        events += 1
        model = event.get('model') or model
        if isinstance(event.get('usage'), dict):
            usage = event['usage']
        for choice in event.get('choices', []):
            delta = choice.get('delta') or {}
            if isinstance(delta.get('reasoning_content'), str):
                reasoning.append(delta['reasoning_content'])
            if isinstance(delta.get('content'), str):
                content.append(delta['content'])
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
        return False

    with response, (folder / 'stream_chunks.jsonl').open('w', encoding='utf-8') as journal:
        if 'text/event-stream' not in response.headers.get('Content-Type', '').lower():
            raise RuntimeError('Expected text/event-stream')
        data_lines = []
        for raw in response:
            line = raw.decode('utf-8').rstrip('\r\n')
            if line.startswith('data:'):
                data_lines.append(line[5:].lstrip(' '))
                continue
            if line or not data_lines:
                continue
            data, data_lines = '\n'.join(data_lines), []
            if handle(data, journal):
                done = True
                break
        else:  # the server may close right after the last event without the terminating blank line
            if data_lines and handle('\n'.join(data_lines), journal):
                done = True
    if not done or finish is None:
        raise RuntimeError(f'Stream ended before finish_reason and [DONE] ({events} events)')
    return {'model': model, 'finish_reason': finish, 'reasoning_content': ''.join(reasoning),
            'content': ''.join(content), 'usage': usage or {}, 'events': events}


class Runner:
    def __init__(self, args):
        self.args = args
        self.provider = args.provider
        self.keys = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.counter = TokenCounter()
        self.stats = {'attempts': 0, 'usable': 0, 'failed': 0, 'transport_errors': 0, 'server_errors': 0}
        self.consecutive_errors = 0
        self.total_errors = 0

    def key(self, provider):
        with self.lock:
            if provider not in self.keys:
                self.keys[provider] = load_key(provider, self.args)
            return self.keys[provider]

    def attempt_dirs(self, index):
        base = OUT / 'attempts' / f'{index:04d}'
        return sorted(base.glob('a[0-9][0-9]')) if base.exists() else []

    def attempts(self, index):
        """[(number, sample_id, status)] with status candidate | transport | unknown | no_output."""
        out = []
        for folder in self.attempt_dirs(index):
            request = read_json(folder / 'request.json') if (folder / 'request.json').exists() else {}
            if (folder / 'candidate.json').exists():
                status = 'candidate'
            elif (folder / 'error.json').exists():
                kind = read_json(folder / 'error.json').get('kind')
                status = 'no_output' if kind in NO_OUTPUT_KINDS else 'transport'
            else:
                status = 'unknown' if request else 'no_output'  # request.json is written before sending
            out.append((int(folder.name[1:]), request.get('sample_id'), status))
        return out

    def budget_used(self, row) -> int:
        return sum(1 for _, sid, status in self.attempts(row['index'])
                   if sid == row['sample_id'] and status != 'no_output')

    def abort(self, reason: str) -> None:
        if not self.stop.is_set():
            log('STOPPING the run:', reason)
        self.stop.set()

    def _failure(self, kind: str) -> None:
        with self.lock:
            self.stats['server_errors' if kind == 'server_error' else 'transport_errors'] += 1
            self.consecutive_errors += 1
            self.total_errors += 1
            consecutive, total = self.consecutive_errors, self.total_errors
        if consecutive >= self.args.max_consecutive_errors or total >= self.args.max_total_errors:
            reason = (f'{consecutive} failed requests in a row, {total} in this run (provider outage or network '
                      'problem); finished attempts are kept, re-run later to continue')
            self.abort(reason)
            raise TooManyErrors(reason)

    def run_attempt(self, row, k, effort, prompt):
        folder = OUT / 'attempts' / f"{row['index']:04d}" / f'a{k:02d}'
        if (folder / 'candidate.json').exists():
            return read_json(folder / 'candidate.json')
        if folder.exists():
            return None  # unknown or failed outcome: never resent, caller moves to the next number
        provider = self.provider
        secret = self.key(provider)  # before anything is written: a missing key must not use up attempts
        folder.mkdir(parents=True)
        body = request_body(row['question'], effort, prompt, self.args.max_tokens, provider)
        write_json(folder / 'request.json', {'sample_id': row['sample_id'], 'index': row['index'], 'attempt': k,
                                             'provider': provider, 'effort': effort, 'prompt_variant': prompt,
                                             'started': now_iso(), 'body': body})
        started = time.monotonic()
        for wait in (0, 30, 60, 120, 300):
            if wait and self.stop.wait(wait):
                write_json(folder / 'error.json', {'kind': 'cancelled', 'at': now_iso()})
                return None
            try:
                result = stream_chat(PROVIDERS[provider]['url'], body, secret,
                                     f"project1-qfft-v93-{row['index']}-{k}", folder, provider)
                break
            except RateLimited:
                continue
            except QuotaExhausted as error:
                write_json(folder / 'error.json', {'kind': 'quota', 'message': str(error), 'at': now_iso()})
                if provider == 'go' and self.args.fallback_official:
                    with self.lock:
                        self.provider = 'official'
                    log('Go quota exhausted; switching to the official API for the remaining attempts')
                    return None
                self.abort(f'{provider} quota exhausted')
                raise
            except FatalRequestError as error:
                write_json(folder / 'error.json', {'kind': 'rejected', 'message': str(error)[:500], 'at': now_iso()})
                self.abort(f'{provider} rejected the request ({str(error)[:120]}); check the key, model name '
                           'and request format')
                raise
            except ServerError as error:
                write_json(folder / 'error.json', {'kind': 'server_error', 'message': str(error)[:500],
                                                   'at': now_iso()})
                self._failure('server_error')
                return None
            except Exception as error:
                write_json(folder / 'error.json', {'kind': 'transport', 'type': type(error).__name__,
                                                   'message': str(error)[:500], 'at': now_iso()})
                self._failure('transport')
                return None
        else:
            write_json(folder / 'error.json', {'kind': 'rate_limited', 'at': now_iso()})
            self._failure('server_error')
            return None
        with self.lock:
            self.consecutive_errors = 0
        write_json(folder / 'response.json', {'received': now_iso(), 'latency_s': round(time.monotonic() - started, 1),
                                              'response': result})
        candidate = {'sample_id': row['sample_id'], 'index': row['index'], 'attempt': k, 'provider': provider,
                     'model_requested': body['model'], 'model_returned': result['model'],
                     'reasoning_effort_requested': effort, 'prompt_variant': prompt,
                     'finish_reason': result['finish_reason'], 'reasoning_content_raw': result['reasoning_content'],
                     'final_content_raw': result['content'], 'usage': result['usage'], 'created': now_iso(),
                     'training_approved': False}
        write_json(folder / 'candidate.json', candidate)
        return candidate

    def solve(self, row, until, max_attempts):
        """Run attempts for one question until `until(candidate)` or the attempt budget is spent.

        The budget counts this question's attempts that produced (or may have produced) a completion;
        attempt numbers always increase, so no folder is ever reused.
        """
        schedule = self.args.efforts.split(',')
        while not self.stop.is_set():
            used = self.budget_used(row)
            if used >= max_attempts:
                with self.lock:
                    self.stats['failed'] += 1
                return False
            numbers = [n for n, _, _ in self.attempts(row['index'])]
            k = max(numbers, default=0) + 1
            if k > 99:
                raise RuntimeError(f"index {row['index']:04d}: attempt numbers a01-a99 are used up")
            effort = schedule[used % len(schedule)]  # a refused request retries the same effort
            candidate = self.run_attempt(row, k, effort, self.args.prompt)
            with self.lock:
                self.stats['attempts'] += 1
            if candidate is None:
                continue
            verdict = judge_candidate(candidate, row, self.counter)
            log(f"index={row['index']:04d} a{k:02d} effort={effort} finish={candidate['finish_reason']} "
                f"usable={verdict['usable']} ({verdict['reason']}) tokens={verdict.get('tokens', {}).get('sft_total')}")
            if until(candidate, verdict):
                with self.lock:
                    self.stats['usable'] += 1
                return True
        return False


def _targets_fill(plan, counter):
    return [row for row in plan['questions'] if ds_selection(row, counter)[0] is None]


def _targets_verify(plan, counter, workers):
    rows, finals = [], []
    for row in plan['questions']:
        first, judged = ds_selection(row, counter)
        if first is None or not row.get('reference_answer'):
            continue
        # Skip questions that already have a later usable attempt graded correct.
        usable = [j for j in judged if j[3]['usable']]
        rows.append((row, usable))
        finals.append([(j[3]['final'], row['reference_answer']) for j in usable])
    out = []
    with Grader(workers=workers) as grader:
        flat = [x for group in finals for x in group]
        grades = iter(grader.grade_many(flat))
        for (row, usable), group in zip(rows, finals):
            statuses = [next(grades)['status'] for _ in group]
            if statuses[0] == 'wrong' and 'correct' not in statuses:
                out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['status', 'fill', 'verify'])
    p.add_argument('--provider', choices=list(PROVIDERS), default='go')
    p.add_argument('--fallback-official', action='store_true', help='Switch to the official API on Go quota')
    p.add_argument('--go-db', help='opencode.db with the Go key (default: the DEFAULT_GO_DBS locations)')
    p.add_argument('--official-key-file')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-attempts', type=int, default=8, help='Total v93 attempts per question (fill)')
    p.add_argument('--efforts', default='high,high,max,high,max,high,high,max')
    p.add_argument('--prompt', choices=list(PROMPTS), default='v91_anti_redundancy')
    p.add_argument('--max-tokens', type=int, default=32768)
    p.add_argument('--max-consecutive-errors', type=int, default=5, help='Stop after this many failures in a row')
    p.add_argument('--max-total-errors', type=int, default=20, help='Stop after this many failures in one run')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    plan = load_plan()
    runner = Runner(a)
    found = {}
    for provider in dict.fromkeys([a.provider] + (['official'] if a.fallback_official else [])):
        try:
            runner.key(provider)
            found[provider] = True
            log(f'{provider} API key: found')
        except RuntimeError as error:
            found[provider] = False
            log(f'{provider} API key: NOT found ({error})' +
                (' - the quota fallback will not work' if provider != a.provider else ''))
    if not found[a.provider] and a.cmd != 'status' and not a.dry_run:
        if found.get('official'):  # e.g. WSL2 without the Windows opencode.db: start on the official API
            log(f'Starting with the official API because no {a.provider} key was found')
            runner.provider = 'official'
        else:  # nothing has been written yet
            raise SystemExit(f'{a.provider} API key not found. Nothing was sent. Set OPENCODE_GO_API_KEY or '
                             'DEEPSEEK_API_KEY, or pass --go-db / --official-key-file.')
    references = {r['sample_id']: r.get('reference_answer') for r in plan['questions']}
    grader, grade_lock = None, threading.Lock()
    if a.cmd == 'verify':
        targets = _targets_verify(plan, runner.counter, a.workers)

        def until(candidate, verdict):
            if not verdict['usable']:
                return False
            with grade_lock:  # one shared pool; grading is cheap next to a 10-minute API call
                status = grader.grade_many([(verdict['final'], references[candidate['sample_id']])])[0]['status']
            return status == 'correct'
    else:
        targets = _targets_fill(plan, runner.counter)

        def until(candidate, verdict):
            return verdict['usable']
    log(f'{a.cmd}: {len(targets)} questions need attempts; provider={a.provider}; prompt={a.prompt}')
    for row in targets[:50]:
        log('  target', row['index'], row['sample_id'], f'v93 budget used {runner.budget_used(row)}/{a.max_attempts}')
    if a.dry_run or a.cmd == 'status' or not targets:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    failure = None
    try:
        if a.cmd == 'verify':
            grader = Grader(workers=2)
        with exclusive_lock(OUT / '.run.lock'):
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                futures = [pool.submit(runner.solve, row, until, a.max_attempts) for row in targets]
                for future in as_completed(futures):
                    if future.cancelled():
                        continue
                    error = future.exception()
                    if error is not None and failure is None:
                        failure = error
                        runner.abort(f'{type(error).__name__}: {error}')
                        for other in futures:
                            other.cancel()  # queued questions never start; running ones stop at their next check
                    write_json(OUT / 'status.json', {'updated': now_iso(), 'cmd': a.cmd, 'targets': len(targets),
                                                     'provider': runner.provider, 'stopped': runner.stop.is_set(),
                                                     **runner.stats})
    finally:
        if grader is not None:
            grader.close()
    if failure is not None:
        log(f'Stopped early: {type(failure).__name__}: {failure}')
        log(f'Progress so far: {runner.stats}. Fix the cause and re-run the same command; finished attempts are kept.')
        raise SystemExit(2)
    remaining = _targets_fill(plan, runner.counter)
    log(f'Done: {runner.stats}; questions still without a usable trace: {len(remaining)}')
    if remaining and a.cmd == 'fill':
        for row in remaining[:50]:
            log('  still missing', f"{row['index']:04d}", row['sample_id'], f'budget used {runner.budget_used(row)}')
        log(f'Every listed question used its {a.max_attempts} attempts. Re-run with a larger budget, e.g. '
            f'`teacher fill --max-attempts {a.max_attempts + 4}` (or set P1_FILL_ARGS for v93_local_data.cmd), '
            'or switch --provider/--prompt.')
        raise SystemExit(3)


if __name__ == '__main__':
    main()
