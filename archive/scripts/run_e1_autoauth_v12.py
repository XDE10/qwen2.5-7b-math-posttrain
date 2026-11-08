"""Load the user-authorized private Drive credential, then launch online E1."""
import json
import os
from pathlib import Path
import sys

DEFAULT_CREDENTIAL = '/content/drive/MyDrive/.project1_credentials/swanlab.json'


def authenticated_environment(environment):
    env = dict(environment)
    if not env.get('SWANLAB_API_KEY', '').strip():
        path = Path(env.get('SWANLAB_CREDENTIALS_FILE', DEFAULT_CREDENTIAL))
        try:
            credential = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            raise RuntimeError('Cannot read the independent SwanLab credential file; check the mounted Drive account') from None
        key = credential.get('SWANLAB_API_KEY') if isinstance(credential, dict) else None
        if not isinstance(key, str) or not key.strip():
            raise RuntimeError('SwanLab credential file contains no API key')
        env['SWANLAB_API_KEY'] = key.strip()
    return env


def main():
    env = authenticated_environment(os.environ)
    launcher = Path(__file__).with_name('run_e1_online_v11.sh')
    print('SwanLab credential loaded automatically; launching online E1. No manual key entry.', flush=True)
    os.execvpe('bash', ['bash', str(launcher), *sys.argv[1:]], env)


if __name__ == '__main__':
    main()
