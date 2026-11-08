from src.data.code_v2_dedup import function_body_hash,Protected


def test_renamed_equivalent_body_and_changed_algorithm():
    a='def original(x):\n y=x+2\n return y'
    b='def renamed(value):\n result=value+2\n return result'
    assert function_body_hash(a,'original')==function_body_hash(b,'renamed')
    assert function_body_hash(a,'original')!=function_body_hash(b.replace('+2','*2'),'renamed')


def test_current_final_tests_and_all_legacy_dev_are_protected():
    protected=Protected()
    hit=protected.match({'problem':'Unrelated text','entry_point':'has_close_elements'})
    assert hit['reason']=='protected_entry_name_review'
    assert len(protected.files)==3
