import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('project1_auth_v28',Path(__file__).resolve().parents[1]/'tools/start_aimo_sft_v28.py')
auth=importlib.util.module_from_spec(spec);spec.loader.exec_module(auth)


def test_private_credential_sets_child_environment_without_mutating_parent(tmp_path):
    path=tmp_path/'credential.json'
    path.write_text(json.dumps({'SWANLAB_API_KEY':'FAKE_TEST_KEY'}),encoding='utf-8-sig')
    original={'PATH':'kept','SWANLAB_API_KEY':'old-value'}
    env=auth.authenticated_environment(original,path)
    assert env['SWANLAB_API_KEY']=='FAKE_TEST_KEY' and env['PATH']=='kept'
    assert original['SWANLAB_API_KEY']=='old-value'


def test_bad_credential_does_not_prompt_or_fall_back_to_offline(tmp_path):
    path=tmp_path/'credential.json';path.write_text('{}')
    with pytest.raises(RuntimeError,match='contains no API key'):
        auth.authenticated_environment({},path)
