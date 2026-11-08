"""Fresh-server recovery from existing Drive artifacts; no model downloads."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

MODEL='Qwen2.5-7B-Instruct--a09a35458c702b33eeacc393d103063234e8bc28'
REV='a09a35458c702b33eeacc393d103063234e8bc28'
PACKAGES={
    'project1_runtime_20261025_v7.zip':'ec970aaec88308ae39d460e1ba5ee4a3b6d0c03351214136366a92bce3d577ed',
    'project1_compact_A10080_v18.zip':'c707e2ab302a1e5057183e66af985b2572dee373a05f2d04fd525dd7cfa4f483',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


def extract_preserving(archive,workspace):
    workspace=Path(workspace).resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            name=Path(item.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in item.filename or not name.parts or name.parts[0]!='project1':
                raise ValueError('Unexpected archive path')
            target=(workspace/name).resolve()
            if not target.is_relative_to(workspace/'project1'):raise ValueError('Archive path escapes project')
            if item.is_dir():continue
            body=z.read(item)
            if target.exists():
                if target.read_bytes()!=body:raise ValueError('Existing file differs; preserve it: '+str(target))
                continue
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)


def installed_versions(names):
    result={}
    for name in names:
        try:result[name]=metadata.version(name)
        except metadata.PackageNotFoundError:result[name]=None
    return result


def dependency_plan(installed):
    targets=[]
    if not installed.get('trl'):targets.append('trl==1.16.0')
    if not installed.get('swanlab'):targets.append('swanlab==0.10.2')
    pyro=installed.get('pyroaring')
    if not pyro or tuple(int(x) for x in re.findall(r'\d+',pyro)[:3])>=(1,0,4):targets.append('pyroaring<1.0.4')
    if installed.get('sympy')!='1.14.0':targets.append('sympy==1.14.0')
    return targets


def copy_snapshot(source,target,identity,progress):
    source=Path(source);target=Path(target);target.mkdir(parents=True,exist_ok=True)
    total=0;missing=0
    for name,digest in identity['files'].items():
        logical=Path(name)
        if logical.is_absolute() or '..' in logical.parts or '\\' in name:raise ValueError('Model path escapes snapshot')
        size=(source/logical).stat().st_size;total+=size
        dst=target/logical
        if dst.exists():
            if sha(dst)!=digest:raise ValueError('Existing local model file differs: '+name)
        else:missing+=size
    if shutil.disk_usage(target).free<missing+2_000_000_000:raise ValueError('Local SSD needs model bytes plus 2GB spare')
    with progress(total=total,unit='B',unit_scale=True,desc='Drive -> local E0 SSD') as bar:
        for name,expected in identity['files'].items():
            src=source/name;dst=target/name
            if dst.exists():bar.update(dst.stat().st_size);continue
            dst.parent.mkdir(parents=True,exist_ok=True);temporary=dst.with_name(dst.name+'.copying');h=hashlib.sha256()
            with src.open('rb') as reader,temporary.open('wb') as writer:
                for block in iter(lambda:reader.read(16*1024*1024),b''):
                    writer.write(block);h.update(block);bar.update(len(block))
            if h.hexdigest()!=expected:raise ValueError('Drive bytes differ from prior verified E0 identity: '+name)
            temporary.replace(dst)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--drive-root',default='/content/drive/MyDrive')
    p.add_argument('--workspace-root',default='/content/workspaces')
    p.add_argument('--model-cache-root',default='/content/models')
    p.add_argument('--evidence-root',default='/content/project1-env-evidence')
    p.add_argument('--start-training',action='store_true')
    a=p.parse_args();drive=Path(a.drive_root);project=Path(a.workspace_root)/'project1';evidence=Path(a.evidence_root)
    print('[1/5] Read the mounted Drive account and existing artifacts',flush=True)
    if not drive.is_dir():raise RuntimeError('Mount Google Drive in a Colab notebook Python cell first')
    binding=json.loads((drive/'drive_account_binding_v2.json').read_text(encoding='utf-8'))
    if binding['email'].lower()!='padaxingbakamaka@gmail.com':raise ValueError('Mount the selected new Drive account')
    if not (drive/'.project1_credentials/swanlab.json').is_file():raise ValueError('Independent automatic SwanLab credential missing')
    packages=drive/'projects/project1'
    print('[2/5] Restore the base runtime and the A10080 compact patch',flush=True)
    base_present=(project/'src/training/guards_v4.py').is_file() and (project/'data/portable_release/manifest.json').is_file()
    for name,expected in PACKAGES.items():
        if name.startswith('project1_runtime') and base_present:continue
        archive=packages/name
        if sha(archive)!=expected:raise ValueError('Uploaded package differs: '+name)
        extract_preserving(archive,a.workspace_root)
    print('[3/5] Preserve preinstalled core packages; install only required gaps',flush=True)
    evidence.mkdir(parents=True,exist_ok=True)
    protected=['torch','transformers','peft','accelerate','datasets','tokenizers','requests','numpy']
    before=installed_versions(protected+['trl','swanlab','pyroaring','sympy'])
    if any(not before.get(k) for k in ['torch','transformers','peft','accelerate','datasets']):
        raise RuntimeError('A core preinstalled package is missing; this recovery does not replace the GPU stack')
    constraints=evidence/'core_constraints_v19.txt'
    constraints.write_text(''.join(k+'=='+before[k]+'\n' for k in protected if before.get(k)),encoding='utf-8')
    requirements=dependency_plan(before)
    if requirements:subprocess.run([sys.executable,'-m','pip','install','-c',str(constraints),*requirements],check=True)
    after=installed_versions(list(before))
    if any(after[k]!=before[k] for k in protected if before.get(k)):raise RuntimeError('Protected core version changed')
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (evidence/'pip-freeze-recovered-v19.txt').write_text(freeze.stdout,encoding='utf-8')
    print('[4/5] Restore the original E0 identity and copy its existing Drive weights',flush=True)
    baselines=sorted(folder for folder in (drive/'runs/project1').glob('E0_dev_v9_*')
                     if folder.is_dir() and (folder/'summary.json').is_file() and (folder/'evaluation_identity.json').is_file())
    if not baselines:raise ValueError('Prior verified E0 evaluation identity was not found')
    original=json.loads((baselines[-1]/'evaluation_identity.json').read_text(encoding='utf-8'))['model_identity']
    if original['repo']!='Qwen/Qwen2.5-7B-Instruct' or original['revision']!=REV or original.get('verified_against_hf_lfs') is not True:
        raise ValueError('Require the previously verified original E0 identity')
    from tqdm import tqdm
    model=Path(a.model_cache_root)/MODEL
    copy_snapshot(drive/'models'/MODEL,model,original,tqdm)
    identity_path=evidence/'e0-identity/e0_model_identity.json';identity_path.parent.mkdir(parents=True,exist_ok=True)
    if identity_path.exists() and json.loads(identity_path.read_text())!=original:raise ValueError('Existing local identity differs')
    identity_path.write_text(json.dumps(original,ensure_ascii=False,indent=2),encoding='utf-8')
    receipt={'python':sys.version,'before':before,'after':after,'installed_requirements':requirements,
             'project':str(project),'model':str(model),'identity':str(identity_path),
             'identity_source':str(baselines[-1]/'evaluation_identity.json'),'network_model_downloads':False}
    (evidence/'recovery_v19.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[5/5] Recovery complete; compact-only online training entry is ready',flush=True)
    command=[sys.executable,'-u','-m','scripts.run_compact_a10080_v18','--e0-model-path',str(model),'--e0-model-identity',str(identity_path)]
    if a.start_training:subprocess.run(command,cwd=project,check=True)
    else:print('From '+str(project)+': '+' '.join(command),flush=True)


if __name__=='__main__':main()
