from scripts.quarantine_code_literal_conflicts import literal_iteration_conflicts


def record(code, value):
    return {'gold_standard_solution':code,'verification_info':{'test_cases':[
        {'fn_name':'isBinary','input':'text = '+value,'output':'1'}]}}


def test_source_style_unquoted_numeric_string_conflict_is_not_silently_coerced():
    code='def isBinary(text):\n for character in text:\n  pass\n return 1'
    assert literal_iteration_conflicts(record(code,'101'))[0]['literal_type']=='int'
    assert not literal_iteration_conflicts(record(code,"'101'"))
    # A conversion before iteration makes the simple diagnostic inconclusive.
    code='def isBinary(text):\n text=str(text)\n for character in text:\n  pass\n return 1'
    assert not literal_iteration_conflicts(record(code,'101'))
