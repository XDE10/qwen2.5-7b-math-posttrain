"""Mock-only asynchronous synthesis skeleton with durable cache and budgets.

SQLite is authoritative; a JSONL snapshot can restore a missing SQLite database.
Reserve worst-case request units BEFORE dispatch, including failed transports.
Default dry_run performs no dispatch. Units in tests are simulated, not money.
No credential lookup, network library or paid provider implementation exists here.
"""
import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable


class BudgetExceeded(RuntimeError):
    pass


class TransientError(RuntimeError):
    pass


@dataclass(frozen=True)
class BlindRequest:
    """The entire allowlisted generator input; references belong to verifier only.

    Conditions must already be audited initial facts, not solved target values.
    Typed fields prevent accidental reference-field serialization. Arbitrary text
    cannot be proven answer-free by a regex: upstream provenance audit is required.
    """
    sample_id: str
    problem: str
    source_revision: str
    split: str = 'train'
    initial_conditions: tuple[str,...] = ()

    def __post_init__(self):
        if self.split != 'train': raise ValueError('Generation seeds must be train')
        if not all(isinstance(x,str) and x for x in (self.sample_id,self.problem,self.source_revision)):
            raise ValueError('Missing provenance/problem')
        if not isinstance(self.initial_conditions,tuple) or not all(isinstance(v,str) for v in self.initial_conditions):
            raise ValueError('Conditions must be an audited tuple of strings')

    @classmethod
    def from_record(cls, record):
        allowed = {'sample_id','problem','source_revision','split','initial_conditions'}
        if set(record)-allowed: raise ValueError('Non-allowlisted generation fields')
        return cls(**record)


class MockTransport:
    """Local deterministic transport; scripts may include exceptions for tests."""
    def __init__(self, responses=None):
        self.responses = list(responses or ['#### 2'])
        self.calls = []

    async def generate(self, payload):
        self.calls.append(payload)
        result = self.responses[min(len(self.calls)-1,len(self.responses)-1)]
        if isinstance(result,Exception): raise result
        return {'text':result,'usage':{'input_tokens':10,'output_tokens':5},'cost_units':1,
                'provider':'local_mock','request_id':f'mock-{len(self.calls)}'}


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.snapshot = self.path.with_suffix('.jsonl')
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, body TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, key TEXT, reserve INTEGER, body TEXT)')
        self.db.commit()
        if not self.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] and not self.db.execute('SELECT COUNT(*) FROM calls').fetchone()[0] and self.snapshot.exists():
            with self.snapshot.open(encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if item['table']=='jobs':
                        self.db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?)',(item['key'],json.dumps(item['body'],ensure_ascii=False)))
                    elif item['table']=='calls':
                        self.db.execute('INSERT OR REPLACE INTO calls VALUES (?,?,?,?)',(item['id'],item['key'],item['reserve'],json.dumps(item['body'],ensure_ascii=False)))
                    else: raise ValueError('Unknown snapshot table')
            self.db.commit()

    def get(self,key):
        row = self.db.execute('SELECT body FROM jobs WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self,key,body):
        self.db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?)',(key,json.dumps(body,ensure_ascii=False)))
        self.db.commit()
        self.export()

    def reserve(self,key,units,max_requests,max_units):
        # Transaction also protects budgets across distinct client instances.
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            count,total = self.db.execute('SELECT COUNT(*),COALESCE(SUM(reserve),0) FROM calls').fetchone()
            if count>=max_requests or total+units>max_units:
                raise BudgetExceeded('Request or reserved-cost ceiling reached')
            cursor = self.db.execute('INSERT INTO calls(key,reserve,body) VALUES (?,?,?)',
                                    (key,units,json.dumps({'state':'reserved'})))
            call_id = cursor.lastrowid
        self.export()
        return call_id

    def finish(self,call_id,body):
        self.db.execute('UPDATE calls SET body=? WHERE id=?',(json.dumps(body,ensure_ascii=False),call_id))
        self.db.commit()
        self.export()

    def export(self):
        temp = self.snapshot.with_suffix('.jsonl.tmp')
        with temp.open('w',encoding='utf-8',newline='\n') as f:
            for key,body in self.db.execute('SELECT key,body FROM jobs ORDER BY key'):
                f.write(json.dumps({'table':'jobs','key':key,'body':json.loads(body)},ensure_ascii=False)+'\n')
            for i,key,reserve,body in self.db.execute('SELECT id,key,reserve,body FROM calls ORDER BY id'):
                f.write(json.dumps({'table':'calls','id':i,'key':key,'reserve':reserve,'body':json.loads(body)},ensure_ascii=False)+'\n')
        temp.replace(self.snapshot)

    def close(self): self.db.close()


class AsyncSynthesisClient:
    def __init__(self,cache,transport=None,*,dry_run=True,model_id='local-mock-v1',
                 prompt_version='blind-v1',verification_version='verify-v1',sampling=None,
                 max_generation_attempts=3,max_transport_retries=2,max_requests=100,
                 max_cost_units=1000,reserve_per_request=10,requests_per_second=5,
                 backoff_base=.1,sleep=asyncio.sleep,clock=time.monotonic):
        if not dry_run and not isinstance(transport,MockTransport):
            raise ValueError('Only offline MockTransport is enabled in this CPU delivery')
        if min(max_generation_attempts,max_requests,reserve_per_request,requests_per_second)<=0 or max_transport_retries<0 or max_cost_units<0:
            raise ValueError('Invalid limits')
        self.cache = Cache(cache)
        self.transport,self.dry_run = transport,dry_run
        self.model_id,self.prompt_version,self.verification_version = model_id,prompt_version,verification_version
        self.sampling = sampling or {'temperature':.8,'max_output_tokens':1024}
        self.max_generation_attempts,self.max_transport_retries = max_generation_attempts,max_transport_retries
        self.max_requests,self.max_cost_units,self.reserve_per_request = max_requests,max_cost_units,reserve_per_request
        self.interval,self.backoff_base = 1/requests_per_second,backoff_base
        self.sleep,self.clock = sleep,clock
        self.next_call = 0.
        self.rate_lock = asyncio.Lock()
        self.job_locks = {}

    def key(self,request):
        body = {'input':asdict(request),'model':self.model_id,'prompt_version':self.prompt_version,
                'sampling':self.sampling,'verification_version':self.verification_version,
                'prompt_hash':hashlib.sha256(json.dumps(self.payload(request,1)['messages'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
        return hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def payload(self,request,attempt):
        return {'model':self.model_id,'sampling':dict(self.sampling),'generation_attempt':attempt,
                'messages':[{'role':'system','content':'Solve independently. Return a complete solution and final answer. Do not invent missing conditions.'},
                            {'role':'user','content':request.problem+('\nVerified initial conditions:\n'+'\n'.join(request.initial_conditions) if request.initial_conditions else '')}]}

    async def _rate_limit(self):
        async with self.rate_lock:
            now = self.clock()
            delay = max(0.,self.next_call-now)
            if delay: await self.sleep(delay)
            self.next_call = max(self.clock(),self.next_call)+self.interval

    def _verify_saved_response(self,key,state,verifier):
        pending = state['pending_response']
        response,attempt = pending['response'],pending['attempt']
        evidence = verifier(response['text'])
        if not isinstance(evidence,dict) or not isinstance(evidence.get('accepted'),bool):
            raise ValueError('Verifier must return accepted bool and independent evidence')
        state['attempts'].append({'generation_attempt':attempt,'raw_output':response['text'],
            'verification_evidence':evidence,'response_metadata':{k:v for k,v in response.items() if k!='text'},
            'latency':pending['latency']})
        del state['pending_response']
        state.update(generation_attempt=attempt+1,transport_retry=0)
        if evidence['accepted']: state['state']='accepted'
        elif attempt==self.max_generation_attempts: state['state']='rejected'
        self.cache.put(key,state)
        return state

    async def generate(self,request,verifier):
        if not isinstance(request,BlindRequest): raise TypeError('Use allowlisted BlindRequest')
        key = self.key(request)
        async with self.job_locks.setdefault(key,asyncio.Lock()):
            cached = self.cache.get(key)
            if cached and cached['state'] in {'accepted','rejected','transport_failed'}: return cached
            if self.dry_run:
                return {'state':'dry_run','key':key,'payload':self.payload(request,1),'max_requests':self.max_requests,'max_cost_units':self.max_cost_units}
            state = cached or {'state':'pending','key':key,'sample_id':request.sample_id,'attempts':[],
                               'generation_attempt':1,'transport_retry':0}
            # A verifier failure/restart must not trigger a second generation.
            if 'pending_response' in state:
                state = self._verify_saved_response(key,state,verifier)
                if state['state'] in {'accepted','rejected'}: return state
            for attempt in range(state['generation_attempt'],self.max_generation_attempts+1):
                for retry in range(state['transport_retry'],self.max_transport_retries+1):
                    await self._rate_limit()
                    call_id = self.cache.reserve(key,self.reserve_per_request,self.max_requests,self.max_cost_units)
                    started = self.clock()
                    try:
                        response = await self.transport.generate(self.payload(request,attempt))
                    except TransientError as e:
                        self.cache.finish(call_id,{'state':'transient_error','error':str(e),'generation_attempt':attempt,'retry':retry})
                        state.update(generation_attempt=attempt,transport_retry=retry+1)
                        if retry==self.max_transport_retries: state['state']='transport_failed'
                        self.cache.put(key,state)
                        if state['state']=='transport_failed': return state
                        await self.sleep(self.backoff_base*2**retry)
                        continue
                    except Exception as e:
                        self.cache.finish(call_id,{'state':'permanent_error','error':type(e).__name__})
                        state['state']='transport_failed'
                        self.cache.put(key,state)
                        raise
                    self.cache.finish(call_id,{'state':'response','response':response,'generation_attempt':attempt,'retry':retry})
                    if response.get('cost_units',0)>self.reserve_per_request:
                        raise BudgetExceeded('Transport exceeded declared worst-case cost')
                    state['pending_response'] = {'response':response,'attempt':attempt,'latency':self.clock()-started}
                    self.cache.put(key,state)
                    state = self._verify_saved_response(key,state,verifier)
                    if state['state'] in {'accepted','rejected'}: return state
                    break
            state['state']='rejected'
            self.cache.put(key,state)
            return state
