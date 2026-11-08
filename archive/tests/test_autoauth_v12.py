import json
import pytest
from scripts.run_e1_autoauth_v12 import authenticated_environment


def test_load_private_file_without_mutating_source_environment(tmp_path):
    path = tmp_path / 'credential.json'
    path.write_text(json.dumps({'SWANLAB_API_KEY':'fixture-only'}))
    source = {'SWANLAB_CREDENTIALS_FILE':str(path)}
    assert authenticated_environment(source)['SWANLAB_API_KEY'] == 'fixture-only'
    assert 'SWANLAB_API_KEY' not in source


def test_existing_environment_does_not_require_drive():
    assert authenticated_environment({'SWANLAB_API_KEY':'fixture-only'})['SWANLAB_API_KEY'] == 'fixture-only'


def test_malformed_file_does_not_echo_contents(tmp_path):
    path = tmp_path / 'credential.json'
    path.write_text('fixture-secret-content-not-json')
    with pytest.raises(RuntimeError) as error:
        authenticated_environment({'SWANLAB_CREDENTIALS_FILE':str(path)})
    assert 'fixture-secret-content' not in str(error.value)
