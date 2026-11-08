"""Release this Colab VM only after a complete final-export receipt verifies.

Never called on import. --execute is for the future training launcher's final
success branch, after its GPU worker exits. Current GPU work is never stopped.
The POST is the same runtime-management operation used by google.colab.runtime:
https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py
"""
import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    if not relative or Path(relative).is_absolute():
        raise ValueError('Expected project-relative receipt path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Receipt path is missing or outside the run root')
    return path


def verify_saved_run(root, receipt_path):
    root = Path(root).resolve()
    if not root.is_relative_to(Path('/content/drive/MyDrive').resolve()):
        raise ValueError('Release requires persistent MyDrive output, not /content temporary storage')
    return verify_receipt_files(root, receipt_path)


def verify_receipt_files(root, receipt_path):
    """Pure file verification, separately testable without a Colab environment."""
    root = Path(root).resolve()
    receipt_path = inside(root, receipt_path)
    value = json.loads(receipt_path.read_text(encoding='utf-8'))
    if value.get('status') != 'success' or value.get('epoch') != 6 or value.get('global_step') != 750:
        raise ValueError('Final six-epoch/750-update success receipt is required')
    files = value.get('files', {})
    required = {'final_model/config.json', 'final_model/model.safetensors.index.json',
                'final_model/tokenizer.json', 'final_model/tokenizer_config.json',
                'final_model/chat_template.jinja', 'identity.json', 'training_manifest.json', 'train_metrics.json'}
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError('Receipt does not cover all required model and training artifacts')
    for name, digest in files.items():
        path = inside(root, name)
        if path.stat().st_size == 0 or file_sha(path) != digest:
            raise ValueError('Saved artifact is empty or differs: ' + name)
    index = json.loads(inside(root, 'final_model/model.safetensors.index.json').read_text(encoding='utf-8'))
    shards = set(index.get('weight_map', {}).values())
    if not shards or any('final_model/' + shard not in files for shard in shards):
        raise ValueError('Receipt is missing an indexed model shard')
    total_size = index.get('metadata', {}).get('total_size')
    if not isinstance(total_size, int) or total_size <= 0:
        raise ValueError('Missing weight tensor byte count')
    if sum(inside(root, 'final_model/' + s).stat().st_size for s in shards) < total_size:
        raise ValueError('Model shard byte total is incomplete')
    return {'verified_files': len(files), 'model_shards': len(shards), 'receipt_sha256': file_sha(receipt_path)}


def assert_gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError('GPU compute process remains; do not unassign a runtime used by another task')


def release_request(address, opener=urlopen):
    # TBE_RUNTIME_ADDR is supplied by Colab. Never send this operation to a
    # public hostname or an arbitrary URL from a receipt/model/data file.
    parsed = urlsplit('http://' + address)
    try:
        host = parsed.hostname
        local = host == 'localhost' or (host is not None and ipaddress.ip_address(host).is_private)
        valid_port = parsed.port is not None and 1 <= parsed.port <= 65535
    except ValueError:
        raise RuntimeError('Invalid local Colab runtime management address') from None
    if not local or not valid_port or parsed.path or parsed.query or parsed.fragment or parsed.username:
        raise RuntimeError('Expected local Colab runtime management host:port')
    request = Request('http://' + address + '/unassign', data=b'', method='POST')
    with opener(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError('Colab did not accept VM unassignment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--receipt', default='FINAL_EXPORT_COMPLETE.json')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    checked = verify_saved_run(args.run_root, args.receipt)
    print('FINAL_DRIVE_ARTIFACTS_VERIFIED ' + json.dumps(checked), flush=True)
    if not args.execute:
        print('CHECK_ONLY: no runtime release requested', flush=True)
        return
    assert_gpu_idle()
    address = os.environ.get('TBE_RUNTIME_ADDR', '').strip()
    if not address:
        raise RuntimeError('Colab runtime management address unavailable; no release request was sent')
    status_path = args.run_root / 'AUTO_RELEASE_STATUS.json'
    status_path.write_text(json.dumps({'status': 'requesting_unassign', **checked})+'\n', encoding='utf-8')
    if hasattr(os, 'sync'):
        os.sync()
    try:
        release_request(address)
    except Exception as error:
        # Persist a truthful state when still alive; never label an error as a
        # confirmed resource release. Do not emit URLs or credentials.
        status_path.write_text(json.dumps({'status': 'request_failed', 'error_type': type(error).__name__, **checked})+'\n', encoding='utf-8')
        raise
    print('COLAB_UNASSIGN_REQUEST_ACCEPTED: cloud resource disappearance must be observed externally', flush=True)


if __name__ == '__main__':
    main()
