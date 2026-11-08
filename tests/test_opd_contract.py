import torch
import pytest
import json
from pathlib import Path
from src.data.screening_io import sha
from src.teachers.opd_contract import teacher_batch_from_student, load_contract


def test_contract_keeps_official_math_and_existing_code_protocol():
    c = load_contract()
    assert c['math_system']==r'Please reason step by step, and put your final answer within \boxed{}.'
    assert c['code_system'] is None and c['math_max_new_tokens']==1024


def test_teacher_receives_exact_tensor_and_mask_handles_left_and_right_padding():
    ids=torch.tensor([[0,4,5,6,7],[4,5,6,7,0]])
    mask=torch.tensor([[0,1,1,1,1],[1,1,1,1,0]])
    inputs,response=teacher_batch_from_student(ids,mask,[2,2],[[4,5],[4,5]],8)
    assert inputs['input_ids'] is ids and inputs['attention_mask'] is mask
    assert response.tolist()==[[False,False,False,True,True],[False,False,True,True,False]]
    with pytest.raises(ValueError,match='exact student prefix'):
        teacher_batch_from_student(ids,mask,[2,2],[[4,6],[4,5]],8)
    with pytest.raises(ValueError,match='outside audited'):
        teacher_batch_from_student(ids,mask,[2,2],[[4,5],[4,5]],7)
    with pytest.raises(ValueError,match='discontinuous'):
        teacher_batch_from_student(ids,torch.tensor([[1,0,1,1,1],[1,1,1,1,0]]),[2,2],[[4,5],[4,5]],8)


def test_current_prompt_report_matches_contract_and_real_sealed_math_pools():
    report=json.loads(Path('reports/project2_prompt_contract_audit.json').read_text(encoding='utf-8'))
    assert report['case_count']==64 and report['all_explicit_math_prefixes_identical']
    assert report['contract_sha256']==sha('project_blueprints/configs/project2_prompt_contract.json')
    assert report['adapter_sha256']==sha('src/teachers/opd_contract.py')
    for path,detail in report['math_input_budget'].items():
        assert sha(path)==detail['sha256']
        assert detail['max_prompt_tokens']+detail['generation_reserved']<=4096
