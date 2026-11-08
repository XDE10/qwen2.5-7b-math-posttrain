r"""Final-answer grading for v93 evaluation, data reports and RL rewards.

Policy (replaces the frozen strict-v3 path for all v93 numbers):

1. Only the final answer segment is graded: text after the last ``<|im_start|>answer`` or ``</think>``;
   the whole output when neither exists (plain Instruct models). Boxes written while still thinking are
   ignored, so self-corrected traces are not punished.
2. Inside that segment the last ``\boxed{}`` is the prediction. If the segment contains several distinct
   boxes, their comma-joined set is tried as well (multi-answer questions such as "find all x").
3. Without any box, Math-Verify's own extraction runs on the tail of the final segment.
4. Every comparison runs in a worker process with a hard timeout; infrastructure failures are reported
   as ``timeout``/``error`` and never silently become wrong answers.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRADER_VERSION = 'v93-final-segment-last-box/math-verify-0.9.0'
_BOX = re.compile(r'\\(?:boxed|fbox)\s*\{')


def final_segment(text: str) -> str:
    # Local copy keeps worker start-up light; must match formats.final_segment.
    i = text.rfind('<|im_start|>answer')
    if i >= 0:
        return text[i + len('<|im_start|>answer'):].lstrip('\n')
    j = text.rfind('</think>')
    if j >= 0:
        return text[j + len('</think>'):]
    return text


def boxed_contents(text: str) -> list:
    """Contents of every balanced \\boxed{...} / \\fbox{...}, in order of appearance."""
    out = []
    for match in _BOX.finditer(text):
        i, depth = match.end(), 1
        while i < len(text) and depth:
            depth += (text[i] == '{') - (text[i] == '}')
            i += 1
        if depth == 0:
            out.append(text[match.end():i - 1])
    return out


def clean_reference(reference) -> str:
    ref = str(reference if reference is not None else '').strip()
    if len(ref) >= 2 and ref[0] == ref[-1] == '$':
        ref = ref[1:-1].strip()
    if re.fullmatch(r'-?\d+\.0+', ref):  # AMC23 stores integers as "27.0"
        ref = ref.split('.')[0]
    return ref


def normalize_answer(answer: str) -> str:
    """Cheap canonical form used only to group identical answers before Math-Verify comparisons."""
    s = answer.strip().strip('.').strip()
    s = re.sub(r'\\(?:left|right|displaystyle|!|,|;|:)', '', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac').replace('$', '')
    return re.sub(r'\s+', '', s)


def _math_verify():
    import importlib
    try:
        return importlib.import_module('math_verify')
    except ImportError:  # project-local pure-Python copy used on the Windows workstation
        sys.path.insert(0, str(ROOT / '.pilot_deps'))
        return importlib.import_module('math_verify')


def _timeout(seconds: float) -> int:
    # Math-Verify timeouts use signal.alarm, which needs an int and does not exist on Windows;
    # there the parent process enforces the bound instead.
    return 0 if os.name == 'nt' else max(1, int(round(seconds)))


def _parse_gold(mv, reference: str, t: float):
    gold = mv.parse('\\boxed{' + reference + '}', extraction_config=[mv.LatexExtractionConfig()],
                    parsing_timeout=t)
    if not gold:
        gold = mv.parse(reference, extraction_config=[mv.LatexExtractionConfig(), mv.ExprExtractionConfig()],
                        parsing_timeout=t)
    return gold


def grade_text(text: str, reference, inner_timeout: float = 5.0) -> dict:
    """Grade one output against one reference. Runs inside a worker process."""
    mv = _math_verify()
    t = _timeout(inner_timeout)
    ref = clean_reference(reference)
    if not ref:
        return {'status': 'no_reference'}
    gold = _parse_gold(mv, ref, t)
    if not gold:
        return {'status': 'ref_unparsed'}
    segment = final_segment(text or '')
    boxes = [b.strip() for b in boxed_contents(segment) if b.strip()]
    if boxes:
        candidates = [boxes[-1]]
        distinct = list(dict.fromkeys(boxes))
        if len(distinct) > 1:
            candidates.append(', '.join(distinct))
        predictions = [mv.parse('\\boxed{' + c + '}', extraction_config=[mv.LatexExtractionConfig()],
                                parsing_timeout=t) for c in candidates]
        how, shown = 'last_box', boxes[-1]
    else:
        tail = segment[-3000:]
        predictions = [mv.parse(tail, extraction_config=[mv.LatexExtractionConfig(boxed_match_priority=0),
                                                         mv.ExprExtractionConfig()], parsing_timeout=t)]
        how, shown = 'no_box_fallback', None
    predictions = [p for p in predictions if p]
    if not predictions:
        return {'status': 'no_answer', 'how': how, 'pred': shown}
    if shown is None:
        shown = str(predictions[0][-1])[:200]
    ok = any(mv.verify(gold, p, timeout_seconds=t) for p in predictions)
    return {'status': 'correct' if ok else 'wrong', 'how': how, 'pred': shown[:500]}


def equivalent(a: str, b: str, inner_timeout: float = 5.0) -> bool:
    if normalize_answer(a) == normalize_answer(b):
        return True
    mv = _math_verify()
    t = _timeout(inner_timeout)
    pa = mv.parse('\\boxed{' + a + '}', extraction_config=[mv.LatexExtractionConfig()], parsing_timeout=t)
    pb = mv.parse('\\boxed{' + b + '}', extraction_config=[mv.LatexExtractionConfig()], parsing_timeout=t)
    return bool(pa and pb and mv.verify(pa, pb, timeout_seconds=t))


def _worker_init() -> None:
    try:
        _math_verify()
    except Exception:
        pass


def _ping():
    _math_verify()
    return 'pong'


def _grade_task(args):
    text, reference, inner = args
    try:
        return grade_text(text, reference, inner)
    except Exception as error:  # surfaced to the caller as an infrastructure status
        return {'status': 'error', 'error': f'{type(error).__name__}: {error}'[:300]}


def _equiv_task(args):
    a, b, inner = args
    try:
        return equivalent(a, b, inner)
    except Exception:
        return False


class Grader:
    """Process pool with per-item hard timeouts. Use as a context manager."""

    def __init__(self, workers: int | None = None, timeout: float = 30.0, inner_timeout: float = 5.0):
        self.workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
        self.timeout, self.inner_timeout = timeout, inner_timeout
        self.pool = None
        self._start()

    def _start(self) -> None:
        self.pool = mp.get_context('spawn').Pool(self.workers, initializer=_worker_init)
        try:  # fail fast if workers cannot even start (otherwise every item would just time out)
            if self.pool.apply_async(_ping).get(timeout=120) != 'pong':
                raise RuntimeError
        except Exception:
            self.pool.terminate()
            raise RuntimeError('Grader worker processes failed to start; run as `python -m ...` '
                               'from the repository root') from None

    def restart(self) -> None:
        self.pool.terminate()
        self.pool.join()
        self._start()

    def close(self) -> None:
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _map(self, func, payloads, on_timeout):
        results = [None] * len(payloads)
        tries = Counter()
        pending = list(range(len(payloads)))
        while pending:
            submitted = [(i, self.pool.apply_async(func, (payloads[i],))) for i in pending]
            pending, hung = [], False
            for i, future in submitted:
                if hung:
                    if future.ready():
                        results[i] = future.get()
                    else:
                        pending.append(i)
                    continue
                try:
                    results[i] = future.get(timeout=self.timeout)
                except mp.TimeoutError:
                    # A worker is stuck in a pathological expression: give this item one retry on
                    # a fresh pool, collect whatever already finished, resubmit the rest.
                    tries[i] += 1
                    if tries[i] >= 2:
                        results[i] = on_timeout
                    else:
                        pending.append(i)
                    hung = True
            if hung:
                self.restart()
        return results

    def grade_many(self, items) -> list:
        """items: iterable of (text, reference) -> list of grade dicts, same order."""
        payloads = [(text, ref, self.inner_timeout) for text, ref in items]
        return self._map(_grade_task, payloads, {'status': 'timeout'})

    def equivalent_many(self, pairs) -> list:
        payloads = [(a, b, self.inner_timeout) for a, b in pairs]
        return self._map(_equiv_task, payloads, False)

    def majority_cluster(self, predictions: list):
        """Index list of the plurality answer cluster (answers merged by Math-Verify equivalence)."""
        groups = {}
        for i, p in enumerate(predictions):
            if p:
                groups.setdefault(normalize_answer(p), []).append(i)
        if not groups:
            return []
        keys = sorted(groups, key=lambda k: -len(groups[k]))[:8]
        reps = {k: predictions[groups[k][0]] for k in keys}
        pairs = [(a, b) for x, a in enumerate(keys) for b in keys[x + 1:]]
        same = self.equivalent_many([(reps[a], reps[b]) for a, b in pairs]) if pairs else []
        parent = {k: k for k in keys}

        def find(k):
            while parent[k] != k:
                k = parent[k]
            return k
        for (a, b), eq in zip(pairs, same):
            if eq:
                parent[find(b)] = find(a)
        merged = {}
        for k in keys:
            merged.setdefault(find(k), []).extend(groups[k])
        return max(merged.values(), key=len)


def reference_for_s1(metadata: dict | None, solution: str | None):
    """Reference answer for an s1K-style row: metadata answer, last box in the solution, or a short solution."""
    if isinstance(metadata, dict):
        answer = metadata.get('answer')
        if answer is not None and str(answer).strip():
            return str(answer).strip(), 'metadata_answer'
    boxes = boxed_contents(str(solution or ''))
    if boxes:
        return boxes[-1].strip(), 'solution_last_box'
    text = str(solution or '').strip()
    if text and len(text) <= 80:
        return text, 'short_solution'
    return None, 'unverifiable'
