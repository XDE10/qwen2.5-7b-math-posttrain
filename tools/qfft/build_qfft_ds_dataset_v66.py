"""Prepare by default. --run sends paid/subscription requests only during off-peak.

Each HTTP call contains ONE blind question. Logical groups of 50 are for human
review, never a multi-question completion. No code here starts GPU training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import http.client
import json
from pathlib import Path
import re
import ssl
import threading
import time
import urllib.error
import urllib.request

from qfft_ds_common_v66 import (ROOT, REPORT, OUT, PROVIDERS, SYSTEM, MAX_OUTPUT, WINDOW, SEPARATOR,
                                 read, save, freeze, digest, payload, prepare, clean_meta, check_time,
                                 quota_error, key_for, TZ, sha_file)

_HTTP_LOCAL = threading.local()


def wire_payload(row, provider):
    body = payload(row, provider)
    body.update({'stream': True, 'stream_options': {'include_usage': True}})
    return body


def same_generation_contract(a, z):
    ignored = {'stream', 'stream_options'}
    return {k:v for k,v in a.items() if k not in ignored} == {k:v for k,v in z.items() if k not in ignored}


def unreviewed_transient_total(policy):
    """A human-authorized review starts a new circuit window; lifetime count stays."""
    total = policy['total_go_transient']
    baseline = policy.get('reviewed_go_transient_total', 0)
    if type(total) is not int or type(baseline) is not int or not 0 <= baseline <= total:
        raise RuntimeError('Invalid reviewed circuit baseline; no new request sent')
    return total - baseline


def reviewed_unknowns(out, identity, allowed_indices=None):
    """Explicitly reviewed Go-only unknowns may be deferred, never replayed.

    This does not authorize ignoring an unknown official bill, a changed request,
    or any new unknown request not listed in the dated local review manifest.
    """
    out = Path(out)
    path = out / 'reviewed_unresolved_v73.json'
    if not path.exists():
        return {}
    manifest = read(path)
    if manifest.get('identity') != identity or manifest.get('action') != 'defer_without_retry':
        raise RuntimeError('Reviewed unknown-request manifest identity/action mismatch')
    result = {}
    allowed_indices = range(1000) if allowed_indices is None else allowed_indices
    for item in manifest.get('records', []):
        index = item['index']
        if type(index) is not int or index not in allowed_indices or index in result or item['provider'] != 'go':
            raise RuntimeError('Only unique reviewed Go request indices may be deferred')
        folder = out / 'calls' / f'{index:04d}' / 'go'
        request = folder / 'request.json'
        if not request.exists() or sha_file(request) != item['request_sha256']:
            raise RuntimeError('Reviewed unknown request bytes changed')
        dispatch = read(request)
        if dispatch.get('identity') != identity or dispatch.get('sample_id') != item['sample_id']:
            raise RuntimeError('Reviewed unknown request identity changed')
        if (folder / 'response.json').exists() or (out / 'candidates' / f'{index:04d}.json').exists():
            raise RuntimeError('Reviewed deferred request now has a response; reconcile before resuming')
        if item.get('error_sha256') and sha_file(folder / 'error.json') != item['error_sha256']:
            raise RuntimeError('Deferred server error evidence changed')
        result[index] = item
    return result


@contextmanager
def run_lock(out):
    """OS lock automatically releases on exit/crash; never kill a process to unlock."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if __import__('os').name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('Another v66 operation holds the run lock. Do not start twice.') from None
        try:
            yield
        finally:
            handle.seek(0)
            if __import__('os').name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)


class Rejected(Exception):
    def __init__(self, status, body):
        self.status, self.body = status, body
        super().__init__('HTTP request rejected: ' + str(status))


class DeferredGoServerFailure(RuntimeError):
    def __init__(self, status, circuit_open, kind='http_5xx'):
        self.status, self.circuit_open, self.kind = status, circuit_open, kind
        super().__init__('Go ' + kind + ' ' + str(status) + ' deferred without retry' +
                         ('; server-error circuit open' if circuit_open else ''))


class GoTransportOutcomeUnknown(RuntimeError):
    pass


def safe_failure_details(error, secret, phase, elapsed):
    reason = getattr(error, 'reason', None)
    text = str(reason if reason is not None else error).replace(secret, '[REDACTED]')
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'(?i)Bearer\s+\S+|oc_sk_[\w-]+|\bsk-[\w-]{16,}', '[REDACTED]', text)
    return {'error_type': type(error).__name__, 'reason_type': type(reason).__name__ if reason is not None else None,
            'reason': text[:600], 'phase': phase, 'elapsed_seconds': round(elapsed, 3),
            'outcome': 'unknown_may_have_been_billed', 'automatic_retry': False}


def transient_go_url_error(error):
    if isinstance(error, (ConnectionError, TimeoutError, http.client.IncompleteRead, ssl.SSLEOFError)):
        return not isinstance(error, ssl.SSLCertVerificationError)
    if not isinstance(error, urllib.error.URLError):
        return False
    reason = getattr(error, 'reason', None)
    if isinstance(reason, (ssl.SSLCertVerificationError, ValueError, PermissionError)):
        return False
    text = str(reason).lower()
    return not any(s in text for s in ('certificate verify failed', 'proxy authentication', '407',
                                     'unknown url type', 'unsupported protocol'))


def http_post(provider, body, secret, session):
    headers = {'Authorization': 'Bearer ' + secret, 'Content-Type': 'application/json',
               'User-Agent': 'project1-qfft-data/66'}
    if provider == 'go':
        headers['x-opencode-session'] = session
    request = urllib.request.Request(PROVIDERS[provider]['url'], data=json.dumps(body).encode(), headers=headers)
    try:
        if body.get('stream'):
            from qfft_stream_transport_v79 import post
            folder = getattr(_HTTP_LOCAL, 'folder', None)
            if folder is None:
                raise RuntimeError('Streaming requires a durable original-dispatch folder')
            return post(PROVIDERS[provider]['url'], provider, body, secret, session, folder)
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        # Never write the credential or request headers, including on error.
        body = error.read().decode('utf-8', errors='replace').replace(secret, '[REDACTED]')
        raise Rejected(error.code, body) from None


def estimated_usd(usage):
    """Public Flash off-peak rates, USD; not an account-bill readback."""
    if not all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
        return None
    hit = usage.get('prompt_cache_hit_tokens', usage.get('prompt_tokens_details', {}).get('cached_tokens', 0))
    hit = min(max(0, int(hit or 0)), usage['prompt_tokens'])
    return (hit * .003 + (usage['prompt_tokens'] - hit) * .15 + usage['completion_tokens'] * .6) / 1e6


class Router:
    def __init__(self, out, identity, official_cap=5.0, transport=http_post, key_loader=key_for, *, allowed_deferred_indices=None):
        self.out, self.identity, self.cap = Path(out), identity, official_cap
        self.transport, self.key_loader = transport, key_loader
        self.lock = threading.RLock()
        self.reserved = 0.0
        self.spent = 0.0
        self.deferred = reviewed_unknowns(self.out, identity, allowed_deferred_indices)
        self.server_policy_path = self.out / 'server_error_policy_v74.json'
        self.server_policy = read(self.server_policy_path) if self.server_policy_path.exists() else {
            'identity': identity, 'consecutive_go_5xx': 0, 'total_go_5xx': 0}
        if self.server_policy['identity'] != identity:
            raise RuntimeError('Server-error ledger identity changed')
        self.server_policy.setdefault('consecutive_go_transient', self.server_policy['consecutive_go_5xx'])
        self.server_policy.setdefault('total_go_transient', self.server_policy['total_go_5xx'])
        if (self.server_policy['consecutive_go_transient'] >= 3
                or unreviewed_transient_total(self.server_policy) >= 10):
            raise RuntimeError('Go server-error circuit requires review; no new request sent')
        for dispatch in (self.out / 'calls').glob('*/*/request.json'):
            parent = dispatch.parent
            error_path = parent / 'error.json'
            if not (parent / 'response.json').exists() and (
                not error_path.exists() or read(error_path).get('outcome') == 'unknown_may_have_been_billed'
            ):
                if parent.name == 'go' and int(parent.parent.name) in self.deferred:
                    continue  # Keep request untouched; worker will skip this exact reviewed index.
                raise RuntimeError('An earlier request has an unknown paid outcome; inspect it before any new request.')
        path = self.out / 'routing.json'
        self.fallback = read(path)['official_after_quota'] if path.exists() else False
        if path.exists() and read(path)['identity'] != identity:
            raise RuntimeError('Routing identity changed')
        for path in (self.out / 'calls').glob('*/official/response.json'):
            record = read(path)
            value = estimated_usd(record.get('response', {}).get('usage', {}))
            if value is None:
                raise RuntimeError('Existing official response has unknown cost; inspect before more spending.')
            self.spent += value
        self.persist()

    def persist(self):
        save(self.out / 'routing.json', {'identity': self.identity, 'official_after_quota': self.fallback,
             'official_estimated_usd': self.spent, 'official_inflight_reserved_usd': self.reserved,
             'official_cap_usd': self.cap, 'pricing': 'Flash off-peak public rates; estimate, not bill',
             'updated_utc': datetime.now(timezone.utc).isoformat()})

    def defer_go_server_error(self, row, error):
        """Skip this exact Go 5xx once; no retry, no paid fallback, bounded circuit."""
        folder = self.out / 'calls' / f'{row["index"]:04d}' / 'go'
        with self.lock:
            path = self.out / 'reviewed_unresolved_v73.json'
            manifest = read(path) if path.exists() else {
                'identity': self.identity, 'action': 'defer_without_retry', 'records': []}
            if manifest['identity'] != self.identity or manifest['action'] != 'defer_without_retry':
                raise RuntimeError('Deferred ledger changed')
            is_http = isinstance(error, Rejected)
            kind = 'http_5xx' if is_http else 'url_transport'
            item = {'index': row['index'], 'sample_id': row['sample_id'], 'provider': 'go',
                    'request_sha256': sha_file(folder / 'request.json'),
                    'error_sha256': sha_file(folder / 'error.json'), 'http_status': error.status if is_http else None,
                    'failure_kind': kind, 'outcome': 'unknown_after_go_' + kind, 'automatic_retry': False,
                    'followup_status': 'pending_separate_review',
                    'recorded_utc': datetime.now(timezone.utc).isoformat()}
            if row['index'] in self.deferred:
                raise RuntimeError('Already deferred; refuse repeated dispatch')
            manifest['records'].append(item)
            save(path, manifest)
            self.deferred[row['index']] = item
            if is_http:
                self.server_policy['consecutive_go_5xx'] += 1
                self.server_policy['total_go_5xx'] += 1
            self.server_policy['consecutive_go_transient'] += 1
            self.server_policy['total_go_transient'] += 1
            self.server_policy['last_http_status'] = error.status if is_http else None
            self.server_policy['last_failure_kind'] = kind
            self.server_policy['last_index'] = row['index']
            save(self.server_policy_path, self.server_policy)
            return self.server_policy['consecutive_go_transient'] >= 3 or unreviewed_transient_total(self.server_policy) >= 10

    def reserve(self, body):
        # UTF-8 byte count is a conservative input-token bound for this tokenizer family.
        bound = (len(json.dumps(body['messages'], ensure_ascii=False).encode()) * .15 + MAX_OUTPUT * .6) / 1e6
        with self.lock:
            if self.spent + self.reserved + bound > self.cap:
                raise RuntimeError('Official API spending ceiling reached; saved results remain resumable.')
            self.reserved += bound
            self.persist()
        return bound

    def invoke(self, row, provider):
        body = wire_payload(row, provider)
        path = self.out / 'calls' / f'{row["index"]:04d}' / provider
        request_path, response_path = path / 'request.json', path / 'response.json'
        if response_path.exists():
            record = read(response_path)
            original = read(request_path)
            if (record['identity'] != self.identity or original.get('identity') != self.identity
                    or original.get('sample_id') != row['sample_id']
                    or original['request_sha256'] != record['request_sha256']
                    or record['request_sha256'] != digest(original['body'])
                    or not same_generation_contract(original['body'], body)):
                raise RuntimeError('Saved response identity mismatch')
            return record
        if request_path.exists():
            old = read(request_path)
            if (old['identity'] != self.identity or old['request_sha256'] != digest(old['body'])
                    or not same_generation_contract(old['body'], body)):
                raise RuntimeError('Saved dispatch identity mismatch')
            error_path = path / 'error.json'
            if error_path.exists():
                error = read(error_path)
                if provider == 'go' and error.get('explicit_go_quota'):
                    raise Rejected(error['status'], error['body'])
            raise RuntimeError('Prior dispatch has no usable response; no automatic paid retry: ' + row['sample_id'])
        check_time()
        secret = self.key_loader(provider)
        reservation = self.reserve(body) if provider == 'official' else 0.0
        # Marker saved BEFORE network call: crash/timeouts cannot silently duplicate paid work.
        dispatch = {'identity': self.identity, 'sample_id': row['sample_id'], 'provider': provider,
                    'request_sha256': digest(body), 'body': body, 'started_utc': datetime.now(timezone.utc).isoformat()}
        save(request_path, dispatch)
        start = time.perf_counter()
        phase = 'transport_or_response_decode'
        try:
            _HTTP_LOCAL.folder = path
            response = self.transport(provider, body, secret, 'project1-qfft-v66-' + str(row['index']))
            record = {'identity': self.identity, 'sample_id': row['sample_id'], 'index': row['index'],
                      'provider': provider, 'endpoint': PROVIDERS[provider]['url'],
                      'request_sha256': digest(body), 'model_requested': body['model'],
                      'reasoning_effort_requested': body['reasoning_effort'],
                      'latency_seconds': time.perf_counter() - start,
                      'completed_utc': datetime.now(timezone.utc).isoformat(),
                      'response': response, 'training_approved': False}
            phase = 'response_save'
            save(response_path, record)  # Preserve the paid response before parsing/scoring.
        except Rejected as error:
            save(path / 'error.json', {'status': error.status, 'body': error.body,
                 'explicit_go_quota': provider == 'go' and quota_error(error.status, error.body),
                 'outcome': 'unknown_may_have_been_billed' if error.status >= 500 else 'http_rejection',
                 'automatic_retry': False})
            if reservation and error.status < 500:
                with self.lock:
                    self.reserved -= reservation
                    # 4xx rejection retained; no generation usage was returned.
                    self.persist()
            raise
        except Exception as error:
            save(path / 'error.json', safe_failure_details(error, secret, phase, time.perf_counter()-start))
            # Keep reservation while outcome is unknown. Stop this run.
            if provider == 'go' and phase == 'transport_or_response_decode' and transient_go_url_error(error):
                raise GoTransportOutcomeUnknown('Go URL transport failed; saved outcome remains unknown.') from None
            raise RuntimeError('Transport/output-save failure; response outcome unknown; no automatic retry.') from None
        if provider == 'go':
            with self.lock:
                self.server_policy['consecutive_go_5xx'] = 0
                self.server_policy['consecutive_go_transient'] = 0
                save(self.server_policy_path, self.server_policy)
        if reservation:
            cost = estimated_usd(response.get('usage', {}))
            with self.lock:
                if cost is None:
                    self.persist()
                    raise RuntimeError('Official usage missing; saved response; inspect before further spending.')
                self.reserved -= reservation
                self.spent += cost
                self.persist()
        return record

    def generate(self, row):
        # Replay a received response even if the global route changed after it was saved.
        for old_provider in PROVIDERS:
            if (self.out / 'calls' / f'{row["index"]:04d}' / old_provider / 'response.json').exists():
                return self.invoke(row, old_provider)
        for old_provider in PROVIDERS:
            old_path = self.out / 'calls' / f'{row["index"]:04d}' / old_provider
            if (old_path / 'request.json').exists():
                if not (old_path / 'error.json').exists() or not read(old_path / 'error.json').get('explicit_go_quota'):
                    raise RuntimeError('Prior dispatch unresolved/rejected; no cross-provider duplicate: ' + row['sample_id'])
        with self.lock:
            provider = 'official' if self.fallback else 'go'
        try:
            return self.invoke(row, provider)
        except GoTransportOutcomeUnknown as error:
            opened = self.defer_go_server_error(row, error)
            raise DeferredGoServerFailure(None, opened, 'url_transport') from None
        except Rejected as error:
            if provider == 'go' and 500 <= error.status < 600:
                opened = self.defer_go_server_error(row, error)
                raise DeferredGoServerFailure(error.status, opened) from None
            if provider != 'go' or not quota_error(error.status, error.body):
                raise RuntimeError('Non-quota API rejection; stopping without changing providers. HTTP ' + str(error.status)) from None
            with self.lock:
                self.fallback = True
                self.persist()
            return self.invoke(row, 'official')


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(ROOT / 'data/tokenizers/Qwen2.5-7B-Instruct/tokenizer.json'))


_JUDGE_LOCAL = threading.local()


def grade(final, reference):
    if not isinstance(reference, str) or not reference.strip():
        return {'correct': None, 'status': 'reference_not_a_single_string_requires_review'}
    module = getattr(_JUDGE_LOCAL, 'module', None)
    if module is None:
        spec = importlib.util.spec_from_file_location('judge_v66', ROOT / 'tools/evaluation/evaluate_sft_ds_v29.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _JUDGE_LOCAL.module = module
    return module.grade_bounded(final, reference, ROOT)


def candidate(row, record, tok):
    choice = record['response'].get('choices', [{}])[0]
    message = choice.get('message', {})
    reasoning, final = message.get('reasoning_content') or '', message.get('content') or ''
    if not isinstance(reasoning, str) or not isinstance(final, str):
        raise RuntimeError('Non-text response; saved raw result requires inspection.')
    cleaned, edits = clean_meta(reasoning)
    raw_tokens = len(tok.encode(reasoning + SEPARATOR + final, add_special_tokens=False).ids)
    clean_tokens = len(tok.encode(cleaned + SEPARATOR + final, add_special_tokens=False).ids)
    flags = []
    if choice.get('finish_reason') != 'stop': flags.append('not_normal_stop')
    if not reasoning.strip(): flags.append('missing_reasoning')
    if not final.strip(): flags.append('missing_final')
    if clean_tokens > WINDOW: flags.append('exceeds_student_16k_no_truncation')
    structurally_complete = choice.get('finish_reason') == 'stop' and bool(reasoning.strip()) and bool(final.strip())
    if not structurally_complete:
        length_class = 'incomplete'
    elif clean_tokens <= WINDOW:
        length_class = 'complete_within_16k'
    elif clean_tokens <= 32768:
        length_class = 'complete_16k_to_32k'
    else:
        length_class = 'complete_over_32k_student_tokens'
    auto = grade(final, row['reference_answer'])
    return {'identity': record['identity'], 'sample_id': row['sample_id'], 'index': row['index'], 'batch': row['batch'],
            'provider': record['provider'], 'model_requested': record['model_requested'],
            'model_returned': record['response'].get('model'),
            'reasoning_effort_requested': record['reasoning_effort_requested'],
            'finish_reason': choice.get('finish_reason'), 'reasoning_content_raw': reasoning,
            'final_content_raw': final, 'reasoning_content_cleaned': cleaned, 'cleaning_edits': edits,
            'raw_student_tokens': raw_tokens, 'cleaned_student_tokens': clean_tokens,
            'structurally_complete': structurally_complete, 'length_class': length_class,
            'fits_current_training_window': clean_tokens <= WINDOW,
            'automatic_resampling_after_completion': 0,
            'usage': record['response'].get('usage', {}), 'structural_flags': flags,
            'native_answer_check': auto, 'proof_check': 'pending_independent_review',
            'training_approved': False}


def review_prompt(count):
    return f'''请独立审查附件的全部{count}份新教师完整轨迹，逐题覆盖sample_id，不要只比较最终答案。
QFFT在此指训练时去掉题目，但监督序列仍包括完整思考、answer分隔符和最终回答；不是仅训练最终content。
原题和参考答案是事后审查信息，生成教师未见参考答案。参考答案来自公开数据，并非新人工证明金标；可质疑但须给独立依据。
逐题检查：答案是否明确正确；完整思考和最终解答有无未纠正计算/逻辑错误；关键证明、定义域、边界、等号可达性、分类完备性是否充分；区分已自纠探索、合法非最简方法、实际未纠正错误和可读性改进。不得仅因长或短判错，也不要因未使用巧解就剔除正确坐标法。对式子逐项核算，不能用自己的正确推导代替核查原文错误。
报告保留/修复/剔除/不确定建议。对每个数学问题给source_field（reasoning_content_raw或final_content_raw）、逐字quote、解释及可核验推导；只有格式和已列元话语可作局部清理。压缩/重写CoT属于新数据版本，不能伪称原教师输出。检查元话语时只建议精确删除，不笼统删除整段。
特别核查：最终回答中的方程常数、系数、变量替换、等号条件是否逐步成立，是否与思考中使用的方程属于同一变换；不得因答案正确就放过中间算式抄串行。证明正确的直接解法不因没有失败分支判低质；有明确诊断和解决过程的自纠也不因中间出现错误直接判无效。
finish_reason=length、最终回答为空不得补写成原模型成功。教师API输出上限32768，学生Qwen窗口16384，两个分词器不作固定比例换算；完整超16K样本仍要评数学质量，与当前训练窗口可用性分开，不能仅因长直接判数学错误，也不自动压缩或裁剪。length_class里的complete只指正常结束且思考/最终回答非空，不是证明已通过。没有Math Level的样本不得编造等级。已设置max也不等于推导正确。
请逐题输出JSONL：sample_id、answer_verdict(correct/incorrect/uncertain)、reasoning_verdict(valid/needs_repair/invalid/uncertain)、issues[{{source_field,quote,explanation,severity}}]、recommendation(keep/repair/reject/uncertain)。最后列出已审查ID和未审查ID。若容量不足，明确未读部分，不得声明全量逐字审查。审查不是训练自动批准。'''


def export_review(rows, out=OUT, report=REPORT, tok=None, only_batches=None):
    tok = tok or tokenizer()
    completed = []
    for batch in range(1, 21):
        if only_batches is not None and batch not in only_batches: continue
        selected = [r for r in rows if r['batch'] == batch]
        items = []
        for row in selected:
            path = out / 'candidates' / f'{row["index"]:04d}.json'
            if not path.exists(): continue
            result = read(path)
            # Deliberately omit native verdict/prior reviewer conclusions from blind review material.
            items.append({**{k: row[k] for k in ('sample_id', 'problem', 'reference_answer', 'reference_provenance',
                                                 'source', 'category', 'difficulty')},
                          **{k: result[k] for k in ('provider', 'model_requested', 'model_returned', 'finish_reason',
                              'reasoning_content_raw', 'final_content_raw', 'cleaning_edits',
                              'raw_student_tokens', 'cleaned_student_tokens',
                              'length_class', 'structurally_complete', 'fits_current_training_window')}})
        folder = out / 'gemini_batches' / f'batch_{batch:02d}'
        if not items:
            continue  # Never create fake generated/review data before generation.
        completed.extend(items)
        envelope = {'logical_batch': batch, 'expected_questions': len(selected), 'available_questions': len(items),
                    'missing_ids': [r['sample_id'] for r in selected if r['sample_id'] not in {x['sample_id'] for x in items}],
                    'teacher_system_prompt': SYSTEM, 'scope': 'Full raw CoT + final, no truncation; cleaning diff separate',
                    'training_approved': False, 'records': items}
        save(folder / '完整50题审查数据.json', envelope)
        folder.joinpath('本批审查提示词.txt').write_text(review_prompt(len(items)), encoding='utf-8')
        parts, current, count = [], [], 0
        for item in items:
            weight = len(tok.encode(json.dumps(item, ensure_ascii=False), add_special_tokens=False).ids)
            if current and (len(current) >= 10 or count + weight > 60000):
                parts.append(current); current, count = [], 0
            current.append(item); count += weight
        if current: parts.append(current)
        for n, part in enumerate(parts, 1):
            save(folder / f'part_{n:02d}.json', {'logical_batch': batch, 'part': n, 'part_count': len(parts),
                 'expected_in_this_part': len(part), 'teacher_system_prompt': SYSTEM, 'records': part})
            folder.joinpath(f'part_{n:02d}_审查提示词.txt').write_text(review_prompt(len(part)), encoding='utf-8')
        save(folder / 'packet_manifest.json', {'expected_ids': [r['sample_id'] for r in selected],
             'available_ids': [r['sample_id'] for r in items], 'complete_50': len(items) == len(selected),
             'parts': [[r['sample_id'] for r in part] for part in parts],
             'part_limit': '10 questions or 60000 Qwen-token proxy per part; not a Gemini tokenizer guarantee'})
    total_saved = len(list((out / 'candidates').glob('*.json')))
    saved = [read(path) for path in (out / 'candidates').glob('*.json')]
    length_counts = {}
    long_complete = []
    for item in saved:
        kind = item.get('length_class', 'unclassified')
        length_counts[kind] = length_counts.get(kind, 0) + 1
        if kind in ('complete_16k_to_32k', 'complete_over_32k_student_tokens'):
            long_complete.append({'sample_id': item['sample_id'], 'index': item['index'],
                 'cleaned_student_tokens': item['cleaned_student_tokens'],
                 'candidate_path': str(out / 'candidates' / f'{item["index"]:04d}.json'),
                 'quality_status': 'pending_review', 'training_approved': False})
    save(out / 'long_complete_candidates' / 'index.json', {
        'note': 'All full text remains in candidates/ and calls/. Complete is structural, not proof correctness.',
        'count': len(long_complete), 'records': long_complete})
    save(report / 'generation_progress.json', {'candidate_responses': total_saved, 'target': 1000,
         'all_generated': total_saved == 1000, 'training_approved': False,
         'length_class_counts': length_counts,
         'review_packages_root': str(out / 'gemini_batches')})
    return len(completed)


def run(rows, protocol, batches, workers, cap):
    check_time()  # No key lookup, dispatch or sleeping before the user-specified window.
    OUT.mkdir(parents=True, exist_ok=True)
    identity = digest(protocol)
    freeze(OUT / 'identity.json', {'identity': identity, 'protocol': protocol})
    router = Router(OUT, identity, cap)
    tok = tokenizer()
    pending = [r for r in rows if r['batch'] in batches and r['index'] not in router.deferred]
    if router.deferred:
        save(OUT / 'deferred_progress_v73.json', {'identity': identity,
             'action': 'defer_without_retry', 'records': list(router.deferred.values()),
             'not_generated_successfully': len(router.deferred), 'training_approved': False})
        print('DEFERRED_UNKNOWN_REQUESTS_NO_RETRY indices=' + ','.join(map(str, sorted(router.deferred))), flush=True)
    for row in pending:
        existing = OUT / 'candidates' / f'{row["index"]:04d}.json'
        if existing.exists() and 'missing_reasoning' in read(existing).get('structural_flags', []):
            raise RuntimeError('Previously missing reasoning_content must be resolved before resuming this batch.')
    stop = threading.Event()
    next_lock = threading.Lock()
    export_lock = threading.Lock()
    exported = set()
    iterator = iter(pending)
    failures = []

    def worker():
        while not stop.is_set():
            with next_lock:
                if stop.is_set(): return
                row = next(iterator, None)
            if row is None: return
            path = OUT / 'candidates' / f'{row["index"]:04d}.json'
            if path.exists():
                if read(path)['identity'] != identity:
                    stop.set(); raise RuntimeError('Candidate identity changed')
                continue
            try:
                record = router.generate(row)
                result = candidate(row, record, tok)
                save(path, result)
                print(f'SAVED batch={row["batch"]:02d} index={row["index"]:04d} provider={record["provider"]} effort={row["difficulty"]["reasoning_effort"]} flags={result["structural_flags"]}', flush=True)
                if 'missing_reasoning' in result['structural_flags']:
                    raise RuntimeError('Provider did not return reasoning_content; raw response saved; pause before bulk generation.')
                members = [r for r in rows if r['batch'] == row['batch']]
                if all((OUT / 'candidates' / f'{r["index"]:04d}.json').exists() for r in members):
                    with export_lock:
                        if row['batch'] not in exported:
                            export_review(rows, tok=tok, only_batches={row['batch']})
                            exported.add(row['batch'])
                            print(f'GEMINI_REVIEW_BATCH_{row["batch"]:02d}_READY_50_QUESTIONS', flush=True)
            except DeferredGoServerFailure as error:
                print(f'DEFERRED_GO_TRANSIENT index={row["index"]:04d} kind={error.kind} status={error.status}; no retry/fallback', flush=True)
                if error.circuit_open:
                    failures.append({'sample_id': row['sample_id'], 'error_type': type(error).__name__, 'message': str(error)})
                    stop.set()
                    return
                continue
            except Exception as error:
                failures.append({'sample_id': row['sample_id'], 'error_type': type(error).__name__,
                                 'message': str(error)[:500]})
                stop.set()
                return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker) for _ in range(workers)]
        for future in as_completed(futures): future.result()
    # Always export completed work even if a later request hit a quota, clock or network error.
    total = export_review(rows, tok=tok)
    if failures:
        save(OUT / 'last_pause.json', {'failures': failures, 'saved_candidates': total,
             'already_inflight_requests_may_have_completed': True, 'automatic_retry': False})
        raise RuntimeError('Paused; all received outputs saved. See last_pause.json; no automatic retry.')
    if router.deferred:
        print('QFFT_DS_V73_SELECTED_BATCHES_FINISHED_WITH_DEFERRED_REQUESTS; candidate responses=' + str(total), flush=True)
    else:
        print('QFFT_DS_V66_SELECTED_BATCHES_GENERATION_COMPLETE; candidate responses=' + str(total), flush=True)
    if total == 1000: print('QFFT_DS_V66_ALL_1000_GENERATION_COMPLETE_NOT_TRAINING_APPROVAL', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true')
    action.add_argument('--export', action='store_true')
    parser.add_argument('--batch', type=int, action='append', help='One or more review batches, 1..20; default 1')
    parser.add_argument('--all', action='store_true', help='All 20 frozen batches, never additional questions')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--official-cap-usd', type=float, default=5.0, help='Estimated official fallback spending ceiling')
    args = parser.parse_args()
    if args.all and args.batch: parser.error('Use --all or --batch, not both')
    batches = set(range(1, 21)) if args.all else set(args.batch or [1])
    if not batches <= set(range(1, 21)): parser.error('Batch must be 1..20')
    if not (0 < args.official_cap_usd <= 20): parser.error('Official ceiling must be >0 and <=20 USD')
    with run_lock(OUT):
        rows, protocol = prepare()
        (REPORT / 'gemini_review_prompt_50.txt').write_text(review_prompt(50) + '\n', encoding='utf-8')
        if args.run: run(rows, protocol, batches, args.workers, args.official_cap_usd)
        elif args.export: print('Exported responses=' + str(export_review(rows)))
        else: print('PREPARED ONLY: 1000 frozen questions, 20 x 50; zero API calls. Use --run after 12:00.')


if __name__ == '__main__':
    main()
