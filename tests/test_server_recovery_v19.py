import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from scripts.recover_compact_server_v19 import dependency_plan,extract_preserving,copy_snapshot


def test_dependency_plan_only_missing_and_user_pyro_constraint():
    assert dependency_plan({'trl':None,'swanlab':None,'pyroaring':'1.2.0','sympy':'1.14.0'})==['trl==1.16.0','swanlab==0.10.2','pyroaring<1.0.4']
    assert dependency_plan({'trl':'1.16.0','swanlab':'0.10.2','pyroaring':'1.0.3','sympy':'1.14.0'})==[]


def test_recovery_extraction_preserves_changes_and_rejects_traversal(tmp_path):
    archive=tmp_path/'good.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('project1/file.txt','old')
    root=tmp_path/'workspace';extract_preserving(archive,root);extract_preserving(archive,root)
    (root/'project1/file.txt').write_text('user change')
    with pytest.raises(ValueError,match='Existing file differs'):extract_preserving(archive,root)
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:z.writestr('project1/../../escape.txt','bad')
    with pytest.raises(ValueError):extract_preserving(bad,root)


def test_copy_stream_verifies_supplied_snapshot_without_printing_credentials(tmp_path):
    src=tmp_path/'drive';src.mkdir();body=b'tiny-copy-fixture';(src/'model.safetensors').write_bytes(body)
    class Progress:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def update(self,n):pass
    identity={'files':{'model.safetensors':hashlib.sha256(body).hexdigest()}}
    dst=tmp_path/'local';copy_snapshot(src,dst,identity,Progress)
    assert (dst/'model.safetensors').read_bytes()==body
    copy_snapshot(src,dst,identity,Progress)
    (dst/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Existing local model file differs'):copy_snapshot(src,dst,identity,Progress)
