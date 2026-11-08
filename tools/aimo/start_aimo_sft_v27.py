"""Apply the small v27 Colab compatibility overlay, then train the v26 recipe."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import zipfile

PATCH_SHA256='3111b23ccb6d5a5b267f8a91cb9404692afea82ab4c17ca8e60a9f777df174de'


def apply_overlay(source,root):
    patch=source/'project1_colab_compat_v27.zip'
    if hashlib.sha256(patch.read_bytes()).hexdigest()!=PATCH_SHA256:
        raise ValueError('Compatibility patch checksum differs; wait for Drive synchronization')
    with zipfile.ZipFile(patch) as archive:archive.extractall(root)
    import json
    candidates=[root/'reports/runs/colab_compat_20261026_v27/patch_manifest.json',
                root/'reports/colab_compat_20261026_v27/patch_manifest.json']
    manifest_path=next((p for p in candidates if p.is_file()),None)
    if manifest_path is None:
        raise ValueError('Patch manifest missing; the restored v26 bundle is a pre-refactor build, rebuild it with tools/aimo/package_aimo_sft_v26.py')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for name,expected in manifest['patched_files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Patched file checksum differs: '+name)
    print('v27 compatibility overlay verified: integer warmup, sampler API, torchao repair.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--prepare-only',action='store_true',help='Apply patch without launching training')
    a,training_args=p.parse_known_args()
    source=Path(__file__).resolve().parent
    if not a.worker:
        notifier=source/'start_aimo_sft_v26_notify.py'
        command=[sys.executable,str(notifier),'--title','v26 数学 SFT（兼容修复 v27）',
            '--command',sys.executable,str(Path(__file__).resolve()),'--worker',*sys.argv[1:]]
        return subprocess.run(command).returncode
    subprocess.run([sys.executable,str(source/'start_aimo_sft_v26.py'),'--prepare-only'],check=True)
    root=Path('/content/workspaces/aimo-sft-v26')
    apply_overlay(source,root)
    if not a.prepare_only:
        subprocess.run(['bash',str(root/'experiments/aimo_sft_v26/start_colab.sh'),*training_args],cwd=root,check=True)
    return 0


if __name__=='__main__':sys.exit(main())
