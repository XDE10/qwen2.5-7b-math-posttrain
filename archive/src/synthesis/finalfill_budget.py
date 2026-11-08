"""User-authorized USD11 total cap; all earlier ledger rows retained."""
import json,threading
from pathlib import Path
from filelock import FileLock
from src.synthesis.shared_cost_client import SharedCostClient
from src.synthesis.deepseek_pilot import dump,PRICES
class FinalFillClient(SharedCostClient):
    def __init__(self,path,key):
        self.path=Path(path);self.process_lock=FileLock(str(self.path/'session.lock'));self.process_lock.acquire(timeout=0)
        self.key=key;self.budget=11.;self.max_requests=24000;self.lock=threading.Lock();self.fatal=False
        self.file=self.path/'ledger.json';self.rows=json.loads(self.file.read_text(encoding='utf-8'))
        policy=json.loads(json.dumps({'total_hard_cap_usd':11,'previous_cap_usd':8,'max_requests':24000,'authorized_by':'user supplied Gemini attachment with explicit USD11 amendment; 2026-10-24','prices':PRICES,'old_unknown_error_reservations_retained':True,'semantic_retry':False,'URLError_transport_retry_max':1}))
        p=self.path/'finalfill_budget_policy.json'
        if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==policy
        else:dump(p,policy)
