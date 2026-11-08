import hashlib
from pathlib import Path
import threading

import pytest

CELL = (Path(__file__).resolve().parents[1] /
        'reports/shared_drive_progress_20261025_v3/notebook_second_cell.py')


@pytest.mark.parametrize('git_blob', [False, True])
def test_progress_hash_preserved_and_heartbeat_stops(tmp_path, monkeypatch, capsys, git_blob):
    import huggingface_hub
    raw = tmp_path / 'fixture.bin'
    with raw.open('wb') as stream:
        for _ in range(32):
            stream.write(b'x' * 1024 * 1024)
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f'blob {raw.stat().st_size}\0'.encode())
    with raw.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    script = '''
def file_digest(path, git_blob=False):
    raise AssertionError('The large-file progress implementation must be exercised')
def prepare(root, group, download, available_bytes):
    import huggingface_hub, time
    huggingface_hub.snapshot_download(repo_id='fixture/model')
    time.sleep(.04)
    assert file_digest(root / 'fixture.bin', git_blob=GIT_BLOB) == EXPECTED
    return {'all_selected_complete': True}
'''.replace('GIT_BLOB', repr(git_blob)).replace('EXPECTED', repr(digest.hexdigest()))
    (tmp_path / 'prepare_shared_models_v1.py').write_text(script)
    calls = []
    def fake_download(*args, **kwargs):
        calls.append(kwargs)
        return 'fixture-path'
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', fake_download)
    code = CELL.read_text(encoding='utf-8').replace('HEARTBEAT_SECONDS = 15', 'HEARTBEAT_SECONDS = 0.01')
    scope = {'SHARED_ROOT': tmp_path, 'DRIVE_AVAILABLE_BYTES': 100}
    exec(compile(code, str(CELL), 'exec'), scope)
    assert scope['report']['all_selected_complete']
    assert calls[0]['repo_id'] == 'fixture/model' and 'tqdm_class' in calls[0]
    assert huggingface_hub.snapshot_download is fake_download
    assert not scope['worker'].is_alive()
    text = capsys.readouterr().out
    assert '心跳' in text and '完成' in text


def test_error_restores_functions_and_stops_heartbeat(tmp_path, monkeypatch):
    import huggingface_hub
    (tmp_path / 'prepare_shared_models_v1.py').write_text('''
def file_digest(path, git_blob=False): return 'fixture'
def prepare(*args, **kwargs): raise ValueError('fixture quota refusal')
''')
    original = huggingface_hub.snapshot_download
    scope = {'SHARED_ROOT': tmp_path, 'DRIVE_AVAILABLE_BYTES': 0}
    with pytest.raises(ValueError, match='quota refusal'):
        exec(compile(CELL.read_text(encoding='utf-8'), str(CELL), 'exec'), scope)
    assert huggingface_hub.snapshot_download is original
    assert scope['runner'].file_digest is scope['original_digest']
    assert not scope['worker'].is_alive()
    assert not any(t.name == 'drive-progress-heartbeat-v3' and t.is_alive()
                   for t in threading.enumerate())
