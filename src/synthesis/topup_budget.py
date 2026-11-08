"""Authorized $8 amendment on the existing ledger; preserve the $5 policy."""
import json,threading
from pathlib import Path
from filelock import FileLock
from src.synthesis.shared_cost_client import SharedCostClient
from src.synthesis.deepseek_pilot import dump,PRICES

class TopupClient(SharedCostClient):
    def __init__(self,path,key):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.process_lock=FileLock(str(self.path/'session.lock'));self.process_lock.acquire(timeout=0)
        self.key=key;self.budget=8.;self.max_requests=22000;self.lock=threading.Lock();self.fatal=False
        self.file=self.path/'ledger.json';self.rows=json.loads(self.file.read_text(encoding='utf-8'))
        policy={'total_hard_cap_usd':8,'previous_cap_usd':5,'maximum_dispatches':22000,'authorized_by':'user explicit 2026-10-24 top-up request','existing_ledger_included':True,'prices':PRICES,'unknown_error_reservations_retained':True,'retry_policy':'No semantic resampling; at most one separately reserved URLError transport retry'}
        p=self.path/'topup_budget_policy.json';policy=json.loads(json.dumps(policy))
        if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==policy
        else:dump(p,policy)
