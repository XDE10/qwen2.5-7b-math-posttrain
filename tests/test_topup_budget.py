import json,pytest
from src.synthesis.topup_budget import TopupClient
from src.synthesis.shared_cost_client import BudgetExhausted

def test_eight_dollar_cap_includes_prior_unknown_reservations(tmp_path):
    prior=[{'task_id':'old','cache_key':'old','state':'error_reserved','reserved_usd':7.9999}]
    (tmp_path/'ledger.json').write_text(json.dumps(prior),encoding='utf-8')
    client=TopupClient(tmp_path,'unused-test-credential')
    try:
        with pytest.raises(BudgetExhausted):client.call({'model':'deepseek-flash','thinking':{'type':'disabled'},'messages':[{'role':'user','content':'question'}],'max_tokens':1500},'new')
        assert len(client.rows)==1 and client.summary()['hard_cap_usd']==8
    finally:client.close()
