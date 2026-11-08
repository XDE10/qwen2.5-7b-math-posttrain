"""Frozen questions and blind teacher contract; importing this module never calls an API."""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/qfft_ds_dataset_20261029_v66'
OUT = ROOT / 'artifacts/qfft_ds_dataset_v66'
PREPARED = ROOT / 'reports/romescape_reproduction_20261028_v57/train1000_prepared.jsonl'
PREPARED_SHA = '0e8c6c83c5916e64d57247caac68bef70cb0ae41c20e07ed96cf4015f0e26781'
GO_DB = Path('D:/OpenCode/runtime/data/opencode/opencode.db')
OFFICIAL_KEY = Path('C:/Users/Mr.Z/.codex/private/project1/deepseek.json')
TZ = timezone(timedelta(hours=8))
NOT_BEFORE = datetime(2026, 10, 29, 12, tzinfo=TZ)
MAX_OUTPUT = 32768
WINDOW = 16384
SEPARATOR = '<|im_start|>answer\n'
PROVIDERS = {
    'go': {'url': 'https://opencode.ai/inference/go/openai/v1/chat/completions',
           'model': 'deepseek-v4.1-flash'},
    'official': {'url': 'https://api.deepseek.com/chat/completions', 'model': 'deepseek-flash'},
}
SYSTEM = r'''Please reason step by step, and put your final answer within \boxed{}.
Work from the original problem and give a complete, self-contained derivation that a student can follow.
Show nontrivial algebraic transformations and justify essential inferences. When substituting variables, subtracting equations, or changing notation, recompute every affected coefficient and constant from the equation being transformed; do not carry a value over from a different equation without checking it.
State relevant domain restrictions and boundary cases. For an optimum, prove the global bound and show that equality is attainable under the original conditions. For cases or enumeration, justify that all possibilities are covered. Numerical trials may check a result but must not replace an exact argument when a proof is required.
Use alternative approaches and independent checks when they help resolve a real uncertainty. If an attempted approach fails, identify the obstruction, correct the error, and resolve any resulting contradiction. Do not invent failed attempts or add unnecessary alternative methods to an already complete argument.
Before giving the final answer, recheck the intermediate equations and key calculations that will appear in the final solution, verify substitutions against the original conditions, and ensure the final derivation is internally consistent and agrees with the conclusion reached during reasoning. Resolve any mismatch rather than copying an unchecked expression into the final answer.
Use consistent notation and well-formed LaTeX. Give an unambiguous final result answering the original question, within \boxed{}.'''

# Only exact, previously observed meta sentences; never delete a whole paragraph.
META = re.compile(r'(?:Since desired oververbosity 5, (?:step-by-step|but final can explain)\.|'
                  r'Since desired oververbosity 5\.|Desired oververbosity 5, so moderate\.|'
                  r'Desired oververbosity 5\.|Desired oververbose 5\.|desired 5\.)')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    import os
    import time
    import uuid
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.writing')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    # Windows readers/scanners can briefly deny rename of an otherwise valid file.
    # Retry only the local atomic commit, never rewrite the old file or resend API calls.
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                raise  # Keep the old destination and the complete .writing file for recovery.
            time.sleep(.05 * (attempt + 1))


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError('Frozen protocol differs; preserve old outputs: ' + str(path))
    else:
        save(path, value)


def clean_meta(text):
    edits = [{'start': m.start(), 'end': m.end(), 'removed': m.group()} for m in META.finditer(text)]
    return META.sub('', text), edits


def difficulty(meta, source):
    original = {k: meta[k] for k in ('level', 'difficulty', "Writer's Difficulty Estimate") if k in meta}
    reason = 'No supported high-difficulty rule; high does not mean an easy problem.'
    effort = 'high'
    if source.startswith('qfq/openaimath/') and meta.get('level') == 5:
        effort, reason = 'max', 'Original MATH Level 5'
    elif source.startswith('KbsdJames/Omni-MATH'):
        try:
            if float(meta.get('difficulty', -1)) >= 7:
                effort, reason = 'max', 'Original Omni-MATH rating >=7; not a MATH Level'
        except (TypeError, ValueError):
            pass
    elif source.startswith('Hothan/OlympiadBench/') and meta.get('difficulty') == 'Competition':
        effort, reason = 'max', 'Original competition category; precautionary routing, not a common numeric level'
    return {'original': original, 'source_scale': source, 'reasoning_effort': effort, 'routing_reason': reason}


def payload(row, provider):
    return {'model': PROVIDERS[provider]['model'],
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row['problem']}],
            'thinking': {'type': 'enabled'}, 'reasoning_effort': row['difficulty']['reasoning_effort'],
            'max_tokens': MAX_OUTPUT, 'top_p': 1.0, 'stream': False}


def is_offpeak(now):
    utc = now.astimezone(timezone.utc)
    # Conservative weekday rule. No holiday API or invented holiday calendar.
    return utc.weekday() >= 5 or not (1 <= utc.hour < 4 or 6 <= utc.hour < 10)


def check_time(now=None):
    now = now or datetime.now(TZ)
    if now < NOT_BEFORE:
        raise RuntimeError('No generation before 2026-10-29 12:00 Asia/Shanghai. This command does not wait or schedule.')
    if not is_offpeak(now):
        raise RuntimeError('Peak window: pause before new requests. Resume at 12:00-14:00 or 18:00-next09:00 Beijing time.')


def quota_error(status, body):
    if status not in (402, 429):
        return False
    lower = body.lower()
    return bool(re.search(r'(?:quota|allowance|usage[ _-]*limit|monthly[ _-]*limit|weekly[ _-]*limit|'
                          r'5[ -]*hour[ _-]*limit|spending[ _-]*limit)', lower))


def key_for(provider):
    if provider == 'official':
        value = read(OFFICIAL_KEY).get('DEEPSEEK_API_KEY')
    else:
        with closing(sqlite3.connect(GO_DB.as_uri() + '?mode=ro', uri=True)) as db:
            rows = db.execute('SELECT value FROM credential WHERE integration_id=? AND active=1',
                              ('opencode-go',)).fetchall()
        if len(rows) != 1:
            raise RuntimeError('Expected exactly one active saved Go credential; use the existing local login window.')
        obj = json.loads(rows[0][0])
        value = obj.get('key') if obj.get('type') in ('key', 'api') else None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError('Saved credential unavailable; no key will be logged.')
    return value


def prepare():
    import pandas as pd
    if sha_file(PREPARED) != PREPARED_SHA:
        raise RuntimeError('The existing 1000-question source changed; no new selection is permitted here.')
    frame = pd.read_parquet(ROOT / 'artifacts/research_s1_7b/s1K-1.1.parquet')
    replacements = {r['new_sample_id']: r for r in read(ROOT / 'reports/romescape_reproduction_20261028_v57/replacement6_checks.json')}
    selected = []
    with PREPARED.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            old = json.loads(line)
            sid = old['sample_id']
            if sid.startswith('s1k11/'):
                pos = int(old['original_s1_row'])
                source_row = frame.iloc[pos]
                meta = ast.literal_eval(source_row['metadata']) if isinstance(source_row['metadata'], str) else source_row['metadata']
                source = str(source_row['source_type'])
                reference = meta.get('answer')
                if reference is None:
                    reference = str(source_row['solution'])
                row = {'sample_id': sid, 'problem': old['question'], 'reference_answer': reference,
                       'reference_provenance': 'Published s1K-1.1 answer/solution; not a new human proof gold',
                       'source': source, 'source_revision': '96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d',
                       'source_position': pos, 'category': str(meta.get('subject', source_row['cot_type'])),
                       'task_type': str(source_row['cot_type']), 'difficulty': difficulty(meta, source)}
            else:
                rep = replacements[sid]
                row = {'sample_id': sid, 'problem': old['question'], 'reference_answer': rep['reference_answer'],
                       'reference_provenance': 'Existing six-row replacement checks; not new teacher input',
                       'source': 'previously-approved-openr1-raw', 'source_revision': rep['source_revision'],
                       'source_position': rep['source_row'], 'category': rep['category'], 'task_type': 'math',
                       'difficulty': difficulty({}, 'previously-approved-openr1-raw')}
            row.update({'index': i, 'batch': i // 50 + 1, 'question_sha256': digest(row['problem']),
                        'training_approved': False})
            selected.append(row)
    assert len(selected) == len({r['sample_id'] for r in selected}) == 1000
    assert len({r['question_sha256'] for r in selected}) == 1000
    assert Counter(r['batch'] for r in selected) == {i: 50 for i in range(1, 21)}
    protocol = {'version': 66, 'source_prepared_sha256': PREPARED_SHA, 'questions_sha256': digest(selected),
                'scope': 'Same frozen 994 s1 + 6 replacement questions; 1000 single completions; no expansion',
                'system_prompt': SYSTEM, 'providers': PROVIDERS, 'max_output_tokens_api': MAX_OUTPUT,
                'student_window_qwen': WINDOW, 'batch_size_for_review_only': 50, 'logical_batches': 20,
                'automatic_resampling_after_completion': 0,
                'long_complete_policy': 'Preserve full trajectory and classify separately; quality review pending; never truncate or auto-compress',
                'teacher_budget_note': '32768 is this project choice; historical s1K-1.1 public script used 32000',
                'http_timeout_seconds': 1800,
                'fallback': 'Go first; sticky official fallback only after explicit quota/allowance limit',
                'not_before': NOT_BEFORE.isoformat(), 'offpeak_only': True,
                'requests': 'One independent question per HTTP request; no gold or old reasoning in request',
                'go_client_identity': 'project1-qfft-data/66, truthful custom client; no coding-agent impersonation',
                'training_approved': False}
    freeze(REPORT / 'protocol.json', protocol)
    freeze(REPORT / 'questions1000.json', selected)
    for n in range(1, 21):
        freeze(REPORT / 'question_batches' / f'batch_{n:02d}.json', [r for r in selected if r['batch'] == n])
    summary = {'questions': len(selected), 'logical_batches': 20, 'per_batch': 50,
               'task_types': dict(Counter(r['task_type'] for r in selected)),
               'effort_counts': dict(Counter(r['difficulty']['reasoning_effort'] for r in selected)),
               'routing_reasons': dict(Counter(r['difficulty']['routing_reason'] for r in selected)),
               'original_math_levels': dict(Counter(str(r['difficulty']['original']['level']) for r in selected
                                                   if 'level' in r['difficulty']['original'])),
               'raw_responses_generated_by_prepare': 0, 'training_approved': False}
    save(REPORT / 'preparation_summary.json', summary)
    (REPORT / 'teacher_system_prompt.txt').write_text(SYSTEM + '\n', encoding='utf-8')
    return selected, protocol
