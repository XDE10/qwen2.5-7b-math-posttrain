"""Colab terminal: reconstruct the v26 bundle and start training automatically."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import zipfile

BUNDLE_SHA256='8b105d8ef0c045d795cf4878e8e4bc73e6f8697ae5d4e2dbd80ae7fcc22c7676'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only',action='store_true',help='Reconstruct and extract without starting GPU training')
    a,training_args=p.parse_known_args()
    drive=Path('/content/drive/MyDrive')
    if not drive.is_dir():raise RuntimeError('First mount Drive in the Colab notebook: drive.mount("/content/drive")')
    source=drive/'projects/project1';work=Path('/content/workspaces');work.mkdir(parents=True,exist_ok=True)
    bundle=work/'project1_aimo_sft_v26.zip'
    if not bundle.exists() or hashlib.sha256(bundle.read_bytes()).hexdigest()!=BUNDLE_SHA256:
        temporary=bundle.with_suffix('.assembling')
        with temporary.open('wb') as output:
            for number in range(1,4):
                part=source/f'project1_aimo_sft_v26.zip.part{number:02d}'
                if not part.is_file():raise FileNotFoundError(f'Drive synchronization incomplete: {part}')
                output.write(part.read_bytes())
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=BUNDLE_SHA256:
            raise ValueError('Bundle checksum differs; wait for Drive files to finish synchronizing')
        temporary.replace(bundle)
    with zipfile.ZipFile(bundle) as archive:archive.extractall(work)
    root=work/'aimo-sft-v26'
    print('Bundle reconstructed, verified and extracted: '+str(root),flush=True)
    if not a.prepare_only:
        subprocess.run(['bash',str(root/'experiments/aimo_sft_v26/start_colab.sh'),*training_args],cwd=root,check=True)

if __name__=='__main__':main()
