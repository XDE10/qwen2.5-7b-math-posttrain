"""Durable OpenAI-style SSE transport. No retries, no credentials saved."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import urllib.request

from qfft_ds_common_v66 import save


class SSETransportError(ConnectionError):
    pass


def read_sse(response, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    journal_path = folder / 'stream_chunks.jsonl'
    partial_path = folder / 'stream_partial.json'
    reasoning, content = [], []
    usage, finish, model, response_id, created = None, None, None, None, None
    events, done, last_save, last_sync = 0, False, time.monotonic(), time.monotonic()

    def assembled(complete):
        return {'id': response_id, 'model': model, 'created': created, 'object': 'chat.completion',
                'choices': [{'index': 0, 'finish_reason': finish,
                             'message': {'role': 'assistant', 'reasoning_content': ''.join(reasoning),
                                         'content': ''.join(content)}}],
                'usage': usage or {}, 'stream_complete': complete,
                'stream_events_received': events, 'training_approved': False}

    def snapshot(state):
        save(partial_path, {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                           'response': assembled(False), 'note': 'Partial transport record; not a completed training trajectory.'})

    # x prevents overwriting a prior dispatch journal even if invoked incorrectly.
    with journal_path.open('x', encoding='utf-8', newline='\n') as journal:
        data_lines = []

        def handle_event(lines):
            nonlocal usage, finish, model, response_id, created, events, done, last_save, last_sync
            if not lines:
                return
            data = '\n'.join(lines)
            if data == '[DONE]':
                done = True
                return
            event = json.loads(data)
            if not isinstance(event, dict) or event.get('error'):
                raise SSETransportError('Provider returned an SSE error event; inspect durable chunks.')
            journal.write(json.dumps({'received_utc': datetime.now(timezone.utc).isoformat(),
                                      'chunk': event}, ensure_ascii=False) + '\n')
            journal.flush()
            events += 1
            if event.get('id') is not None: response_id = event['id']
            if event.get('model') is not None: model = event['model']
            if event.get('created') is not None: created = event['created']
            if isinstance(event.get('usage'), dict): usage = event['usage']
            for choice in event.get('choices', []):
                if choice.get('index', 0) != 0:
                    raise ValueError('Expected one completion per original question')
                delta = choice.get('delta') or {}
                for key, parts in [('reasoning_content', reasoning), ('content', content)]:
                    value = delta.get(key)
                    if value is not None:
                        if not isinstance(value, str): raise ValueError('Non-text SSE response field')
                        parts.append(value)
                if choice.get('finish_reason') is not None: finish = choice['finish_reason']
            current = time.monotonic()
            if current-last_sync >= 2:
                os.fsync(journal.fileno()); last_sync = current
            if current-last_save >= 5 or finish is not None:
                snapshot('receiving'); last_save = current

        try:
            for raw in response:
                line = raw.decode('utf-8').rstrip('\r\n')
                if not line:
                    handle_event(data_lines); data_lines = []
                    if done: break
                elif line.startswith('data:'):
                    data_lines.append(line[5:].lstrip(' '))
                # SSE comments/keepalives and non-data fields do not alter math text.
            if data_lines and not done: handle_event(data_lines)
            if not done or finish is None:
                raise SSETransportError('SSE connection ended before complete finish and DONE markers')
            journal.flush(); os.fsync(journal.fileno())
            result = assembled(True)
            save(folder / 'stream_complete.json', result)
            snapshot('received_complete')
            return result
        except Exception:
            try:
                journal.flush(); os.fsync(journal.fileno()); snapshot('interrupted')
            except OSError:
                pass
            raise


def post(url, provider, body, secret, session, folder):
    headers = {'Authorization': 'Bearer '+secret, 'Content-Type': 'application/json',
               'User-Agent': 'project1-qfft-data/79', 'Accept': 'text/event-stream'}
    if provider == 'go': headers['x-opencode-session'] = session
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=1800) as response:
        content_type = response.headers.get('Content-Type', '')
        if 'text/event-stream' not in content_type.lower():
            raise ValueError('Streaming request did not receive text/event-stream; do not silently reinterpret')
        return read_sse(response, folder)
