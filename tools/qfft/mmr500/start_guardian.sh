#!/usr/bin/env bash
set -eo pipefail
source "$HOME/.proxy_env.sh"
set -u
python3 - <<'PY'
import json,os,pathlib,subprocess
root=pathlib.Path('/mnt/d/CodexArtifacts/project1-mmr-step500-20261118')
assert (root/'READY.json').is_file(),'The frozen continuation package is not ready'
state=root/'guardian_state.json'
if state.exists():
 saved=json.loads(state.read_text());pid=saved.get('guardian_pid')
 if pid and pathlib.Path(f'/proc/{pid}/cmdline').exists() and str(root/'guardian500.py').encode() in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes():
  print('Guardian is already running:',pid);print(state.read_text());raise SystemExit(0)
log=root/'guardian_console.log'
with log.open('a') as f:
 process=subprocess.Popen(['python3','-u',str(root/'guardian500.py')],stdin=subprocess.DEVNULL,
                           stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
(root/'daemon_process.json').write_text(json.dumps({'pid':process.pid,'script':str(root/'guardian500.py'),'start_new_session':True,'log':str(log)},indent=2)+'\n')
print('Guardian started:',process.pid)
PY
