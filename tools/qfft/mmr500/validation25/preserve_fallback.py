import pathlib,os,json,shutil,hashlib,subprocess,time,datetime
ROOT=pathlib.Path('/content/p1_protected');ROOT.mkdir(exist_ok=True)
RUN=pathlib.Path('/content/p1/runs/grpo_qfft_r1_mmr_dr150_v93')
REMOTE='mydrive:runs/project1/v93/preserved_upload_fallback/grpo_qfft_r1_mmr_dr150_v93'
CONF='/content/validation_rclone.conf'
REQUIRED=['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
def cmd(args,seconds=300):
 r=subprocess.run(['rclone','--config',CONF,*args,'--retries','1','--low-level-retries','2'],capture_output=True,text=True,timeout=seconds)
 if r.returncode:raise OSError(r.stderr[-500:])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save(row):
 row['checked']=datetime.datetime.now().isoformat();(ROOT/'fallback_status.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
while True:
 try:
  cps=sorted(RUN.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
  for src in cps:
   step=int(src.name.split('-')[-1]);dest=ROOT/src.name
   if not all((src/n).is_file() for n in REQUIRED):continue
   if json.loads((src/'trainer_state.json').read_text())['global_step']!=step:continue
   if not dest.exists():
    temp=ROOT/(src.name+'.tmp');shutil.copytree(src,temp,copy_function=os.link);temp.rename(dest)
  for dest in sorted(ROOT.glob('checkpoint-*')):
   if not dest.is_dir() or dest.name.endswith('.tmp') or (dest/'FALLBACK_UPLOADED.json').exists():continue
   manifest={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in dest.iterdir() if p.is_file() and p.name not in ['MIRROR_COMPLETE.json','FALLBACK_UPLOADED.json']}
   cmd(['copy',str(dest),REMOTE+'/'+dest.name,'--exclude','FALLBACK_UPLOADED.json','--exclude','MIRROR_COMPLETE.json','--transfers','2','--drive-stop-on-upload-limit'])
   cmd(['check',str(dest),REMOTE+'/'+dest.name,'--exclude','FALLBACK_UPLOADED.json','--exclude','MIRROR_COMPLETE.json','--one-way'])
   marker={'artifact_type':'full_state','file_manifest':manifest,'files':len(manifest),'bytes':sum(r['bytes'] for r in manifest.values()),'at':datetime.datetime.now().isoformat(),'reason':'New input drive rejects uploads; independent fallback copy while training continues'}
   (dest/'MIRROR_COMPLETE.json').write_text(json.dumps(marker,indent=2)+'\n');cmd(['copyto',str(dest/'MIRROR_COMPLETE.json'),REMOTE+'/'+dest.name+'/MIRROR_COMPLETE.json'])
   (dest/'FALLBACK_UPLOADED.json').write_text(json.dumps({'remote':REMOTE+'/'+dest.name,'step':int(dest.name.split('-')[-1])})+'\n')
   save({'state':'full_checkpoint_durable','step':int(dest.name.split('-')[-1]),'remote':REMOTE+'/'+dest.name})
  # Preserve small reconstruction files without relying on the throttled FUSE upload queue.
  for name in ['run_identity.json','recipe.json','pool_manifest.json']:
   if (RUN/name).exists():cmd(['copyto',str(RUN/name),REMOTE+'/'+name],45)
  if (ROOT/'checkpoint-150/FALLBACK_UPLOADED.json').exists():
   save({'state':'step150_full_state_durable','step':150,'remote':REMOTE+'/checkpoint-150'});break
 except Exception as e:save({'state':'retrying','error':str(e)[:500]})
 time.sleep(60)
