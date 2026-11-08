"""Persistent, bounded subprocess transport for the unchanged v2 grader core."""
import hashlib
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from src.training.math_reward_v2 import category

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def grader_identity():
    from importlib.metadata import version
    names=['scripts/math_verify_v3_strict_worker.py','scripts/math_verify_pilot_worker.py','src/training/math_reward_v2.py','scripts/verifier_stream_worker_v4.py','src/training/frozen_verifier_v4.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'.pilot_deps').rglob('*.py'))]
    return {'core':'unchanged math_reward_v2 category + strict-v3 check','transport':'persistent-isolated-v4','per_request_timeout_seconds':20,
            'sources':{n:sha(ROOT/n) for n in names},'versions':{n:version(n) for n in ['sympy','antlr4-python3-runtime'] if _installed(n)}}


def _installed(name):
    from importlib.metadata import version,PackageNotFoundError
    try:version(name);return True
    except PackageNotFoundError:return False


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


class Worker:
    def __init__(self,timeout=20):
        self.timeout=timeout;self.proc=None;self.counter=0

    def start(self):
        self.close();self.q=queue.Queue()
        self.proc=subprocess.Popen([sys.executable,'-u','-m','scripts.verifier_stream_worker_v4'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',bufsize=1)
        proc=self.proc;q=self.q
        def reader():
            for line in proc.stdout:
                try:q.put(json.loads(line))
                except json.JSONDecodeError:pass
            q.put({'transport_closed':True})
        threading.Thread(target=reader,daemon=True).start()
        ready=self.q.get(timeout=45)
        if ready!={'ready':True}:raise RuntimeError('Verifier did not become ready')

    def check(self,output,reference):
        try:
            if self.proc is None or self.proc.poll() is not None:self.start()
            self.counter+=1
            self.proc.stdin.write(json.dumps({'id':self.counter,'output':output,'reference':str(reference)})+'\n');self.proc.stdin.flush()
            response=self.q.get(timeout=self.timeout)
            if response.get('id')!=self.counter:raise RuntimeError('Verifier protocol mismatch')
            return response['result']
        except queue.Empty:
            self.close();return {'status':'timeout','correct':False}
        except Exception as e:
            self.close();return {'status':'verifier_error','correct':False,'error_type':type(e).__name__}

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:self.proc.kill()
            self.proc.wait()
            for f in (self.proc.stdin,self.proc.stdout):
                if f:f.close()
            self.proc=None


def selfcheck(row,worker):
    answer=worker.check(r'\boxed{'+str(row['reference_answer'])+'}',row['reference_answer'])
    solution=worker.check(row['solution'],row['reference_answer'])
    return {'sample_id':row['sample_id'],'answer':answer,'solution':solution,
            'answer_category':category(answer),'solution_category':category(solution),
            'supported_and_consistent':category(answer)=='equivalent' and category(solution)=='equivalent'}
