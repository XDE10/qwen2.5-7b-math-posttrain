"""Wait for durable step 150, then supervise only the authorized account-C 150->500 continuation."""
from datetime import datetime
import fcntl,json,os,pathlib,re,subprocess,tempfile,time
ROOT=pathlib.Path('/mnt/d/CodexArtifacts/project1-mmr-step500-20261118')
PRESERVED=pathlib.Path('/mnt/d/CodexArtifacts/project1-mmr-step150-preserve-20261117/RESTORE_READY.json')
CONF=pathlib.Path.home()/'.config/p1/rclone_project1.conf'
CLI=pathlib.Path.home()/'.local/bin/colab-c'
RUN='p1drive:runs/project1/v93/runs/grpo_qfft_r1_mmr_dr500_v93'
STAGE='p1drive:runs/project1/v93/stages/grpo__grpo_qfft_r1_mmr_dr500.json'
PARENT_STAGE='p1drive:runs/project1/v93/stages/grpo__grpo_qfft_r1_mmr_dr150.json'
STEPS=list(range(150,501,25))
def command(args,timeout=20):return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
def remote(path,optional=False):
 r=command(['rclone','--config',str(CONF),'cat',path])
 if r.returncode:
  if optional and any(x in r.stderr.lower() for x in ['not found','no such file']):return None
  raise OSError(r.stderr.strip()[-500:])
 return json.loads(r.stdout)
def emit(event,**fields):
 row={'time':datetime.now().astimezone().isoformat(timespec='seconds'),'event':event,**fields}
 with (ROOT/'guardian_events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(json.dumps(row,ensure_ascii=False),flush=True)
def save(state):
 state['checked']=datetime.now().astimezone().isoformat(timespec='seconds')
 p=ROOT/'guardian_state.tmp';p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');p.replace(ROOT/'guardian_state.json')
def tail(session):
 with tempfile.TemporaryDirectory(dir=ROOT) as temp:
  p=pathlib.Path(temp)/'tail.json';r=command([str(CLI),'download','-s',session,'/content/p1/session_tail.json',str(p)],15)
  return json.loads(p.read_text()) if r.returncode==0 else None
def missing(session):
 r=command([str(CLI),'status','-s',session],15)
 if r.returncode:raise OSError(r.stderr.strip()[-400:])
 if f"Session '{session}' not found." in r.stdout:return True
 if f'[{session}]' in r.stdout:return False
 raise OSError('Unexpected scoped session status')
def parent_active():
 stage=remote(PARENT_STAGE,True)
 if stage:
  owner=stage.get('host','').split('@',1)[0]
  if not owner.startswith('p1-mmr150') and not owner.startswith('p1-mmr-dr150'):raise RuntimeError('Unexpected parent training owner')
  if not missing(owner):return True
 sessions=command([str(CLI),'sessions'],15)
 if sessions.returncode:raise OSError('Could not check parent training sessions')
 return bool(re.search(r'\[p1-mmr(?:150|-dr150)[^\]]*\]',sessions.stdout))
def launch(state,recovery=False):
 session='p1-mmr500-c-'+datetime.now().strftime('%m%d-%H%M%S');log=ROOT/(session+'.log')
 env={**os.environ,'P1_500_SESSION':session,'P1_500_RECOVERY_CONFIRMED':str(int(recovery))}
 with log.open('w') as f:
  p=subprocess.Popen(['bash',str(ROOT/'resume500.sh')],stdout=f,stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL,env=env,start_new_session=True)
 state.update(session=session,launcher_pid=p.pid,launcher_log=str(log),missing=0,first_missing=None,state='allocating',launched_ts=time.time())
 save(state);emit('continuation_launch_started',session=session,recovery=recovery,log=str(log))
def latest_full():
 r=command(['rclone','--config',str(CONF),'lsf',RUN+'/ckpt_mirror','--dirs-only'],30)
 if r.returncode:raise OSError(r.stderr.strip()[-400:])
 for step in sorted([int(x) for x in re.findall(r'checkpoint-(\d+)/',r.stdout)],reverse=True):
  m=remote(RUN+f'/ckpt_mirror/checkpoint-{step}/MIRROR_COMPLETE.json',True)
  if not m or m.get('artifact_type')!='full_state':continue
  files=m.get('file_manifest',{})
  if not all(x in files for x in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']):continue
  s=remote(RUN+f'/ckpt_mirror/checkpoint-{step}/trainer_state.json')
  if s['global_step']==step:return step
 return None
def main():
 lock=(ROOT/'guardian.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ready=json.loads((ROOT/'READY.json').read_text());state={'state':'waiting_for_durable_step150','session':None,'missing':0}
 if (ROOT/'guardian_state.json').exists():state.update(json.loads((ROOT/'guardian_state.json').read_text()))
 state['guardian_pid']=os.getpid();save(state);emit('guardian_started',session=state.get('session'))
 while True:
  try:
   if state.get('session') is None:
    if not PRESERVED.exists():state['state']='waiting_for_durable_step150';save(state);time.sleep(60);continue
    if parent_active():state['state']='waiting_for_parent_runtime_release';save(state);time.sleep(60);continue
    # A fresh stage or an existing run is evidence of another owner, never authorization to duplicate it.
    stage=remote(STAGE,True)
    if stage and stage.get('state')=='running' and time.time()-stage.get('heartbeat_ts',0)<900:
     host=stage['host'].split('@',1)[0]
     if not host.startswith('p1-mmr500-c-'):raise RuntimeError('The continuation stage has an unexpected session owner')
     state.update(session=host,state='monitoring',missing=0);save(state)
    else:launch(state)
    time.sleep(30);continue
   session=state['session']
   if not session.startswith('p1-mmr500-c-'):raise RuntimeError('Refusing an out-of-scope runtime')
   stage=remote(STAGE,True)
   absent=missing(session)
   if absent and state.get('state')=='allocating' and time.time()-state.get('launched_ts',0)<180:
    save(state);time.sleep(30);continue
   data=None if absent else tail(session)
   if stage and stage.get('state')=='failed' and stage.get('host','').split('@',1)[0]==session:
    raise RuntimeError('Continuation stage failed; inspect saved full checkpoints and '+state.get('launcher_log','')+': '+str(stage.get('error')))
   if data:
    lines=[x for x in data.get('lines',[]) if len(x)<1200 and re.search(r'step \d+/500',x)]
    if lines and state.get('progress')!=lines[-1]:state['progress']=lines[-1];emit('training_progress',message=lines[-1])
    if str(data.get('result','')).startswith('P1_RESULT=FAILED'):raise RuntimeError('Continuation failed: '+str(data['result']))
   complete=remote(RUN+'/GRPO_COMPLETE.json',True)
   if complete:
    milestones=remote(RUN+'/MILESTONES_COMPLETE.json')
    if complete.get('steps')!=500 or milestones.get('steps')!=STEPS:raise RuntimeError('Completion receipt has wrong final or milestone steps')
    identity=remote(RUN+'/run_identity.json')['identity']
    if identity.get('code_bundle')!=ready['code_bundle'] or identity.get('approval_sha256')!=ready['approval_sha256']:raise RuntimeError('Completed run identity differs from the frozen plan')
    final=remote(RUN+'/ckpt_mirror/checkpoint-500/MIRROR_COMPLETE.json')
    if final.get('artifact_type')!='full_state':raise RuntimeError('Final checkpoint is not resumable')
    if stage and stage.get('state')=='complete' and (absent or data and str(data.get('result','')).startswith('P1_RESULT=OK')):
     if not absent:command([str(CLI),'stop','-s',session],15)
     if not missing(session):raise OSError('Final results are durable; runtime release not yet confirmed')
     for name,value in [('GRPO_COMPLETE.json',complete),('MILESTONES_COMPLETE.json',milestones)]:
      (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
     state.update(state='complete',final_step=500,milestone_steps=STEPS);save(state);emit('complete_and_runtime_released',session=session);return
   if absent:
    state['first_missing']=state.get('first_missing') or time.time()
    state['missing']=state.get('missing',0)+1
    if stage and stage.get('host','').split('@',1)[0]==session and stage.get('heartbeat_ts',0)>state['first_missing']:
     state.update(missing=0,first_missing=None,state='monitoring')
     save(state);time.sleep(60);continue
    if state['missing']>=3:
     sessions=command([str(CLI),'sessions'],15)
     if sessions.returncode==0 and f'[{session}]' not in sessions.stdout:
      if state.get('state')=='allocating':
       state['allocation_failures']=state.get('allocation_failures',0)+1
       if state['allocation_failures']>=3:raise RuntimeError('Account C could not allocate the continuation runtime in three attempts; checkpoints remain preserved')
      last=latest_full()
      if last is None:
       if state.get('state')=='allocating':raise RuntimeError('Allocation failed; no continuation checkpoint was produced')
       last=150
      emit('runtime_loss_confirmed',session=session,latest_full_step=last);launch(state,True)
   else:state.update(state='monitoring',missing=0,first_missing=None,allocation_failures=0)
   save(state)
  except (OSError,ValueError,subprocess.TimeoutExpired) as error:
   state['last_read_error']=str(error)[:500];save(state);emit('read_retry',error=str(error)[:500])
  except RuntimeError as error:
   state.update(state='needs_attention',error=str(error));save(state);emit('attention_required',error=str(error));raise
  time.sleep(60)
if __name__=='__main__':main()
