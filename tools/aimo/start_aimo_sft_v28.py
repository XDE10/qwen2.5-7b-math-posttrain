"""Read the private SwanLab key automatically, then launch the fixed v26 recipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DEFAULT_CREDENTIAL='/content/drive/MyDrive/.project1_credentials/swanlab.json'


def authenticated_environment(environment,credential_path=None):
    env=dict(environment)
    path=Path(credential_path or env.get('SWANLAB_CREDENTIALS_FILE',DEFAULT_CREDENTIAL))
    try:credential=json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):
        raise RuntimeError('Cannot read private SwanLab credential; check the mounted Drive account') from None
    key=credential.get('SWANLAB_API_KEY') if isinstance(credential,dict) else None
    if not isinstance(key,str) or not key.strip():
        raise RuntimeError('Private SwanLab credential contains no API key')
    env['SWANLAB_API_KEY']=key.strip()
    return env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only',action='store_true',help='Restore and patch project without training')
    a,training_args=p.parse_known_args()
    env=authenticated_environment(os.environ)
    launcher=Path(__file__).with_name('start_aimo_sft_v27.py')
    if not launcher.exists():raise FileNotFoundError('Missing compatibility launcher: '+str(launcher))
    print('Private SwanLab credential loaded automatically; online logging enabled. No manual key input.',flush=True)
    command=[sys.executable,str(launcher),'--worker',*training_args]
    if a.prepare_only:command.append('--prepare-only')
    # Skip the optional notification wrapper: user explicitly paused popup work.
    return subprocess.run(command,env=env,check=False).returncode


if __name__=='__main__':sys.exit(main())
