#!/usr/bin/env bash
set -eo pipefail
source "$HOME/.proxy_env.sh"
python3 - <<'PY'
import json,os,pathlib,subprocess,time
r=pathlib.Path('/mnt/d/CodexArtifacts/project1-mmr-step500-20261118/validation25');state=r/'controller_state.json'
if state.exists():
 old=json.loads(state.read_text());pid=old.get('pid');p=pathlib.Path(f'/proc/{pid}/cmdline')
 if pid and p.exists() and str(r/'controller.py').encode() in p.read_bytes():print('Validation controller already running:',pid);raise SystemExit(0)
else:
 initial=json.loads((r/'initial_launch.json').read_text());state.write_text(json.dumps({'state':'evaluating','session':initial['session'],'step':100,'started':time.time(),'completed':[],'missing':0},indent=2)+'\n')
with (r/'controller.log').open('a') as f:
 p=subprocess.Popen(['python3','-u',str(r/'controller.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
(r/'controller_process.json').write_text(json.dumps({'pid':p.pid,'script':str(r/'controller.py')},indent=2)+'\n');print('Validation controller started:',p.pid)
PY
