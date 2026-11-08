import json
import pytest
from src.synthesis.finalfill_budget import FinalFillClient
from src.synthesis.shared_cost_client import BudgetExhausted

def test_eleven_dollar_cap_retains_previous_unknown_charges(tmp_path):
    prior=[{'task_id':'old','cache_key':'old','state':'error_reserved','reserved_usd':10.99999}]
    (tmp_path/'ledger.json').write_text(json.dumps(prior),encoding='utf-8')
    client=FinalFillClient(tmp_path,'unused-test-key')
    try:
        with pytest.raises(BudgetExhausted):
            client.call({'model':'deepseek-flash','thinking':{'type':'disabled'},'messages':[{'role':'user','content':'question'}],'max_tokens':1500},'new')
        assert client.rows==prior
        assert client.summary()['hard_cap_usd']==11
    finally:client.close()
