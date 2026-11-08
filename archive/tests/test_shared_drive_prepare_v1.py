import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'reports/shared_drive_20261025_v1/prepare_shared_models_v1.py'
SPEC = importlib.util.spec_from_file_location('shared_drive_prepare_v1', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_actual_inventory_is_pinned():
    data = MODULE.load_inventory(SCRIPT.parent)
    assert len(data['models']) == 5
    assert len(MODULE.select_models(data, 'project2')) == 3
    assert len(MODULE.select_models(data, 'project1')) == 1
    assert len(MODULE.select_models(data, 'project3')) == 1


@pytest.mark.parametrize('name', ['../other', '/absolute', 'sub/../../other', r'sub\other'])
def test_model_path_traversal_refused(tmp_path, name):
    with pytest.raises(ValueError):
        MODULE.safe_file(tmp_path, name)


@pytest.mark.parametrize('free', [None, -1, 109])
def test_quota_missing_or_insufficient_refused(free):
    with pytest.raises(ValueError):
        MODULE.require_capacity(100, free, 10)


def test_quota_exact_boundary():
    MODULE.require_capacity(100, 110, 10)


def test_weight_and_git_blob_hash_verification(tmp_path):
    raw = b'fake tiny weights'
    config = b'{"model_type":"fixture"}'
    (tmp_path / 'model.safetensors').write_bytes(raw)
    (tmp_path / 'config.json').write_bytes(config)
    model = {'role': 'fixture', 'revision': '0' * 40, 'files': [
        {'path': 'model.safetensors', 'size': len(raw),
         'lfs_sha256': hashlib.sha256(raw).hexdigest()},
        {'path': 'config.json', 'size': len(config),
         'git_blob_id': hashlib.sha1(f'blob {len(config)}\0'.encode() + config).hexdigest()}]}
    assert MODULE.inspect_model(model, tmp_path)['complete']
    (tmp_path / 'model.safetensors').write_bytes(b'X' * len(raw))
    state = MODULE.inspect_model(model, tmp_path)
    assert not state['complete'] and state['corrupt'] == ['model.safetensors']
    (tmp_path / 'model.safetensors').unlink()
    assert MODULE.inspect_model(model, tmp_path)['missing_bytes'] == len(raw)


def test_altered_inventory_refused(tmp_path):
    (tmp_path / 'model_inventory.json').write_text('{}')
    with pytest.raises(ValueError, match='inventory changed'):
        MODULE.load_inventory(tmp_path)


def test_default_check_does_not_download_weights(tmp_path, monkeypatch):
    source = SCRIPT.parent / 'model_inventory.json'
    (tmp_path / source.name).write_bytes(source.read_bytes())
    import huggingface_hub
    def forbid(**kw):
        raise AssertionError('A default check must not download weights')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', forbid)
    state = MODULE.prepare(tmp_path, 'project1')
    assert not state['all_selected_complete']
    assert not (tmp_path / 'models').exists()


def test_download_then_rerun_skips_complete_model(tmp_path, monkeypatch):
    content = b'tiny fixture'
    model = {'role': 'project1', 'repo_id': 'fixture/model', 'revision': 'a' * 40,
             'folder': 'fixture--' + 'a' * 40,
             'files': [{'path': 'model.safetensors', 'size': len(content),
                        'lfs_sha256': hashlib.sha256(content).hexdigest()}]}
    monkeypatch.setattr(MODULE, 'load_inventory', lambda root: {'models': [model]})
    import huggingface_hub
    calls = []
    def fake_download(**kwargs):
        calls.append(kwargs)
        folder = Path(kwargs['local_dir']); folder.mkdir(parents=True)
        (folder / 'model.safetensors').write_bytes(content)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', fake_download)
    assert MODULE.prepare(tmp_path, download=True, available_bytes=100, reserve_bytes=0)['all_selected_complete']
    assert MODULE.prepare(tmp_path, download=True)['all_selected_complete']
    assert len(calls) == 1
    assert calls[0]['revision'] == 'a' * 40
