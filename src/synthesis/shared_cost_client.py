"""Text-only shared budget: completed usage + in-flight worst-case reservations.

Unknown/error dispatches retain their entire reservation. No automatic retries.
"""
import json
import threading
import time
import urllib.request
from datetime import datetime,timezone
from pathlib import Path
from src.synthesis.deepseek_pilot import BASE,PRICES,dump,sha


class BudgetExhausted(RuntimeError):pass


class SharedCostClient:
    def __init__(self,path,key,budget=5.0,max_requests=6500):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        from filelock import FileLock
        self.process_lock=FileLock(str(self.path/'session.lock'))
        self.process_lock.acquire(timeout=0)  # one transport owner across processes
        self.key=key;self.budget=budget;self.max_requests=max_requests;self.lock=threading.Lock();self.fatal=False
        self.file=self.path/'ledger.json';self.rows=json.loads(self.file.read_text(encoding='utf-8')) if self.file.exists() else []
        policy=self.path/'budget_policy.json'
        identity={'budget_usd':budget,'max_requests':max_requests,'prices':PRICES,'date':'2026-10-24','model':'deepseek-flash','rule':'sum complete peak usage or unresolved worst reserve + inflight worst reserve <= cap; no retries'}
        # JSON round trip normalizes price tuples.
        identity=json.loads(json.dumps(identity))
        if policy.exists():assert json.loads(policy.read_text(encoding='utf-8'))==identity
        else:dump(policy,identity)

    def committed(self):
        return sum(r.get('charged_upper_usd',r['reserved_usd']) for r in self.rows)

    def call(self,payload,task_id):
        assert payload['model']=='deepseek-flash' and payload.get('thinking')=={'type':'disabled'}
        key=sha({'task':task_id,'payload':payload});path=self.path/(key+'.json')
        with self.lock:
            if self.fatal:raise RuntimeError('Provider authorization/billing failure; dispatch halted')
            prior=[r for r in self.rows if r['cache_key']==key]
            if path.exists():return json.loads(path.read_text(encoding='utf-8'))
            if prior:raise RuntimeError('Previous uncertain/error dispatch; do not auto bill again')
            ip,hp,op=PRICES[payload['model']]
            reserve=((len(json.dumps(payload['messages'],ensure_ascii=False).encode())+1024)*ip+payload['max_tokens']*op)/1e6
            if len(self.rows)>=self.max_requests or self.committed()+reserve>self.budget:raise BudgetExhausted('Shared cost cap reached before dispatch')
            row={'task_id':task_id,'cache_key':key,'reserved_usd':reserve,'state':'reserved','started_utc':datetime.now(timezone.utc).isoformat()}
            self.rows.append(row);dump(self.file,self.rows)
        req=urllib.request.Request(BASE+'/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        start=time.monotonic()
        try:
            with urllib.request.urlopen(req,timeout=180) as r:body=json.load(r)
            assert body.get('choices')
            usage=body.get('usage',{})
            hit=usage.get('prompt_cache_hit_tokens',usage.get('prompt_tokens_details',{}).get('cached_tokens',0))
            miss=usage.get('prompt_cache_miss_tokens',max(0,usage.get('prompt_tokens',0)-hit))
            cost=(miss*ip+hit*hp+usage.get('completion_tokens',0)*op)/1e6
            complete_usage=all(k in usage for k in ['prompt_tokens','completion_tokens'])
            charged=cost if complete_usage else reserve
            if charged>reserve+1e-10:raise RuntimeError('Provider usage exceeded preflight envelope; retain reserve and halt')
            result={'task_id':task_id,'payload':payload,'response':body,'usage_peak_usd':charged,'wall_seconds':time.monotonic()-start}
            with self.lock:
                dump(path,result);row.update(state='complete',charged_upper_usd=charged,usage=usage,request_id=body.get('id'));dump(self.file,self.rows)
            return result
        except Exception as e:
            with self.lock:
                if getattr(e,'code',None) in {401,402,403}:self.fatal=True
                row.update(state='error_reserved',error_type=type(e).__name__,http_status=getattr(e,'code',None));dump(self.file,self.rows)
            raise RuntimeError('Provider call failed; original reserve retained; '+type(e).__name__) from None

    def summary(self):
        with self.lock:return {'requests':len(self.rows),'committed_upper_usd':self.committed(),'hard_cap_usd':self.budget,'unresolved_requests':sum(r['state']!='complete' for r in self.rows)}

    def close(self):self.process_lock.release()
