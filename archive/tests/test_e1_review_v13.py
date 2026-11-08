import pytest
from src.training.eval_helpers_v13 import hard_summary


def records(sid,categories,finish='stop'):
    return [{'sample_id':sid,'trial':t,'category':c,'finish_reason':finish} for t,c in enumerate(categories)]


def test_valid_groups_distinguish_learnable_and_all_correct():
    data=records('a',['equivalent','mismatch','mismatch','mismatch'])+records('b',['equivalent']*4)
    result=hard_summary(data,['a','b'])
    assert result['questions_with_any_verified_correct']==2
    assert result['potential_curriculum_count_1to3']==1
    assert result['valid_group_correct_count_distribution']['4']==1
    assert result['training_approved'] is False


def test_unscorable_not_counted_as_wrong_or_eligible():
    result=hard_summary(records('a',['equivalent','mismatch','parse_failure','mismatch']),['a'])
    assert result['questions_with_any_verified_correct']==1
    assert result['group_rejections']=={'unscorable_group':1}
    assert sum(result['valid_group_correct_count_distribution'].values())==0


def test_truncation_ineligible_and_missing_or_duplicate_rejected():
    rs=records('a',['mismatch']*4,'length')
    assert hard_summary(rs,['a'])['group_rejections']=={'truncated_or_aborted_group':1}
    with pytest.raises(ValueError):hard_summary(rs[:3],['a'])
    with pytest.raises(ValueError):hard_summary(rs+[rs[0]],['a'])
