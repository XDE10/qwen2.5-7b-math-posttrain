"""FIFO independent development jobs; keep training running, release each evaluation GPU."""
from datetime import datetime
import fcntl,json,os,pathlib,re,subprocess,time
ROOT=pathlib.Path(__file__).parent
CLI=str(pathlib.Path.home()/'.local/bin/colab-a')
INPUT_CONF=str(pathlib.Path.home()/'.config/p1/rclone_project1.conf')
RESULT_CONF=str(pathlib.Path.home()/'.config/p1/rclone_validation.conf')
INPUT='p1drive:runs/project1/v93/runs/'
RESULT='mydrive:runs/project1/v93/mmr_validation25'
STEPS=[100]+list(range(150,501,25))
def cmd(args,timeout=25):return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
def remote(path,result=False,optional=False):
 r=cmd(['rclone','--config',RESULT_CONF if result else INPUT_CONF,'cat',path],20)
 if r.returncode:
  if optional and any(x in r.stderr.lower() for x in ['not found','no such file']):return None
  raise OSError(r.stderr[-400:])
 return json.loads(r.stdout)
def emit(event,**kw):
 row={'time':datetime.now().astimezone().isoformat(), 'event':event,**kw}
 with (ROOT/'controller_events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(json.dumps(row,ensure_ascii=False),flush=True)
def save(state):
 state.update(pid=os.getpid(),checked=datetime.now().astimezone().isoformat());temp=ROOT/'controller_state.tmp';temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');temp.replace(ROOT/'controller_state.json')
def absent(session):
 assert session.startswith('p1-mmrval-a-')
 r=cmd([CLI,'status','-s',session],15)
 if r.returncode:raise OSError('Could not confirm evaluation runtime status')
 if f"Session '{session}' not found." in r.stdout:return True
 if '['+session+']' in r.stdout:return False
 raise OSError('Unexpected evaluation runtime status')
def read_status(session):
 p=ROOT/'current_remote_status.json';r=cmd([CLI,'download','-s',session,'/content/p1/validation_status.json',str(p)],15)
 return json.loads(p.read_text()) if r.returncode==0 else None
def ready(step):
 if step==100:return True
 run='grpo_qfft_r1_mmr_dr150_v93' if step==150 else 'grpo_qfft_r1_mmr_dr500_v93'
 suffix='/GRPO_COMPLETE.json' if step==150 else f'/milestones/step-{step}/EXPORT_COMPLETE.json'
 return remote(INPUT+run+suffix,optional=True) is not None
def launch(state,step):
 session='p1-mmrval-a-'+str(step)+'-'+datetime.now().strftime('%m%d-%H%M%S');log=ROOT/(session+'.log')
 with log.open('w') as f:
  p=subprocess.Popen(['bash',str(ROOT/'run_validation.sh'),str(step)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'P1_VAL_SESSION':session})
 state.update(state='allocating',step=step,session=session,launcher_pid=p.pid,started=time.time(),missing=0,have_worker=False)
 emit('validation_launch',step=step,session=session,log=str(log));save(state)
def collect(step):
 dest=ROOT/'results'/f'step-{step}';dest.mkdir(parents=True,exist_ok=True)
 r=cmd(['rclone','--config',RESULT_CONF,'copy',RESULT+f'/reports/step-{step}',str(dest),'--retries','1','--low-level-retries','2'],45)
 if r.returncode:raise OSError('Could not collect durable validation results')
 row=json.loads((dest/'VALIDATION_COMPLETE.json').read_text())
 (dest/'报告.md').write_text(f"# 第 {step} 步验证集结果\n\n249 题，每题 2 次；相同 prompt、采样、32K 上下文和最终答案框判分。\n\nR1 QFFT：{row['baseline_mean_at_n']*100:.4f}%；第 {step} 步：{row['mean_at_n']*100:.4f}%。\n\n按题配对差值 {row['diff']*100:+.4f} 个百分点，95% CI [{row['ci95'][0]*100:+.4f}, {row['ci95'][1]*100:+.4f}]。\n\n逐题配对见 per_problem_pairs.jsonl。验证成绩用于选择 checkpoint，不能替代独立最终测试。\n",encoding='utf8')
 return row
def main():
 lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads((ROOT/'controller_state.json').read_text()) if (ROOT/'controller_state.json').exists() else {'state':'waiting','session':None,'completed':[]}
 state['pid']=os.getpid();save(state)
 while True:
  try:
   if not state.get('session'):
    next_step=next((s for s in STEPS if s not in state.get('completed',[])),None)
    if next_step is None:state.update(state='complete');save(state);emit('all_validation_complete');return
    receipt=remote(RESULT+f'/reports/step-{next_step}/VALIDATION_COMPLETE.json',result=True,optional=True)
    if receipt:
     collect(next_step);state.setdefault('completed',[]).append(next_step);save(state);continue
    if ready(next_step):launch(state,next_step)
    else:state.update(state='waiting_for_export',next_step=next_step);save(state)
   else:
    session,step=state['session'],state['step'];missing=absent(session)
    if missing:
     if state['state']=='allocating' and time.time()-state['started']<600:time.sleep(30);continue
     state['missing']=state.get('missing',0)+1
     if state['missing']>=3:
      sessions=cmd([CLI,'sessions'],15)
      if sessions.returncode==0 and '['+session+']' not in sessions.stdout:
       state['restarts']=state.get('restarts',0)+1
       if state['restarts']>3:raise RuntimeError('Validation runtime allocation/recovery failed three times; checkpoints remain retained')
       emit('runtime_loss_confirmed',step=step,session=session);state['session']=None
     save(state)
    else:
     state['missing']=0;status=read_status(session)
     if status:
      if status['step']!=step:raise RuntimeError('Evaluation worker belongs to another step')
      state['have_worker']=True
     if status and status['state']=='failed':raise RuntimeError('Validation worker failed: '+status.get('error','unknown'))
     receipt=remote(RESULT+f'/reports/step-{step}/VALIDATION_COMPLETE.json',result=True,optional=True)
     if receipt and status and status['state']=='complete':
      row=collect(step)
      if receipt['step']!=step or receipt['paired_problems']!=249:raise RuntimeError('Validation receipt identity differs')
      released=cmd([CLI,'stop','-s',session],15)
      if not absent(session):raise OSError('Evaluation finished; runtime release remains unconfirmed')
      state.setdefault('completed',[]).append(step);state.update(state='waiting',session=None,restarts=0)
      emit('validation_complete_and_released',step=step,accuracy=row['mean_at_n'],diff=row['diff'],ci95=row['ci95'])
     else:
      cmd([CLI,'download','-s',session,'/content/p1/validation.log',str(ROOT/'current_validation.log')],15)
      state['state']='evaluating' if status or state.get('have_worker') else 'allocating'
      if not status and not state.get('have_worker') and time.time()-state['started']>900:raise RuntimeError('Evaluation setup has not produced a worker status in 15 minutes')
     save(state)
  except (OSError,ValueError,subprocess.TimeoutExpired) as e:
   state['last_read_error']=str(e)[:500];save(state);emit('read_retry',error=str(e)[:500])
  except RuntimeError as e:
   state.update(state='needs_attention',error=str(e));save(state);emit('attention_required',error=str(e));raise
  time.sleep(60)
if __name__=='__main__':main()
