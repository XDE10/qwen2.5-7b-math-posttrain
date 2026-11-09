"""Mock DeepSeek/OpenAI-style SSE server + driver for experiments/qfft_v93/teacher.py (test only).

    python teacher_mock.py FAKE_ROOT SCENARIO_JSON -- teacher args...

SCENARIO_JSON is a list of response kinds served in order (the last one repeats):
ok, ok_noblank, length, drop, e401, e500, e429, e402
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

fake_root, scenario_path, _, *teacher_args = sys.argv[1:]
plan = json.load(open(scenario_path))
lock = threading.Lock()
served = []


def next_kind():
    with lock:
        kind = plan[min(len(served), len(plan) - 1)]
        served.append(kind)
        return kind


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert self.headers['Authorization'] == 'Bearer test-key', 'wrong key header'
        kind = next_kind()
        if kind.startswith('e'):
            code = int(kind[1:])
            text = {'error': {'message': 'quota exceeded' if code == 402 else f'mock {code}'}}
            data = json.dumps(text).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()
        question = body['messages'][1]['content']
        events = [{'model': 'deepseek-v4.1-flash', 'choices': [{'delta': {'reasoning_content': f'Think about {question}. '}}]},
                  {'choices': [{'delta': {'reasoning_content': 'Check it twice.'}}]},
                  {'choices': [{'delta': {'content': 'So the answer is \\boxed{5}.'}}]},
                  {'choices': [{'delta': {}, 'finish_reason': 'length' if kind == 'length' else 'stop'}]},
                  {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}]
        if kind == 'drop':
            events = events[:2]
        for event in events:
            self.wfile.write(b'data: ' + json.dumps(event).encode() + b'\n\n')
            self.wfile.flush()
        if kind != 'drop':
            self.wfile.write(b'data: [DONE]' + (b'' if kind == 'ok_noblank' else b'\n\n'))
        self.wfile.flush()
        self.close_connection = True


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
sys.path.insert(0, fake_root)
from experiments.qfft_v93 import teacher  # noqa: E402

for provider in teacher.PROVIDERS.values():
    provider['url'] = f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions'
sys.argv = ['teacher', *teacher_args]
code = 0
try:
    teacher.main()
except SystemExit as error:
    code = error.code if isinstance(error.code, int) else 1
    if not isinstance(error.code, int):
        print('SystemExit message:', error.code)
print('EXIT', code, '| served', served)
