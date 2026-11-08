"""Verify already-local 7B files against pinned Hub metadata; no weight download."""
import argparse
import hashlib
from pathlib import Path,PurePosixPath
from src.training.rollout_pipeline_v4 import BASE,REV,write_new
from src.training.frozen_verifier_v4 import sha


def verify_local(root,metadata):
    if metadata['repo']!=BASE or metadata['revision']!=REV:raise ValueError('Wrong official metadata identity')
    files={};weights=0
    for item in metadata['files']:
        name=item['name'];logical=PurePosixPath(name)
        if logical.is_absolute() or '..' in logical.parts or '\\' in name:raise ValueError('Escaping official E0 name')
        # Official Hub snapshots use symlinks into ../../blobs. Validate the
        # logical relative filename and the expected content digest, not the
        # physical symlink destination's containment in the snapshot folder.
        path=root.joinpath(*logical.parts)
        if not path.is_file():raise ValueError('Missing official E0 file: '+name)
        if item['lfs_sha256']:
            if sha(path)!=item['lfs_sha256']:raise ValueError('Official E0 LFS SHA mismatch: '+name)
            if name.endswith('.safetensors'):weights+=1
        else:
            data=path.read_bytes()
            git_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if git_sha!=item['git_blob_id']:raise ValueError('Official E0 metadata mismatch: '+name)
        files[name]=sha(path)
    if not weights:raise ValueError('Official E0 weight set empty')
    return {'repo':BASE,'revision':REV,'verified_against_hf_lfs':True,'files':files}


def main():
    import json
    p=argparse.ArgumentParser();p.add_argument('--model-path',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--expected-metadata',default='reports/local_cpu_ready_20261025_v4/e0_expected_metadata.json')
    a=p.parse_args();root=Path(a.model_path).resolve();out=Path(a.output_root).resolve();out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(Path(a.expected_metadata).read_text(encoding='utf-8'))
    identity=verify_local(root,metadata);identity['official_metadata_sha256']=sha(a.expected_metadata)
    write_new(out/'e0_model_identity.json',identity)


if __name__=='__main__':main()
