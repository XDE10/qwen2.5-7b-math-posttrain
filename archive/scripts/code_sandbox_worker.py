"""Container-only worker. Never import or execute this file on the Windows host."""
import json
import os
import resource
import subprocess
import sys
import tempfile
import time


def limits():
    resource.setrlimit(resource.RLIMIT_CPU,(3,3))
    resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
    resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_NPROC,(8,8))


def execute(program,stdin=''):
    with tempfile.TemporaryDirectory(dir='/tmp') as directory:
        path=os.path.join(directory,'candidate.py')
        with open(path,'w',encoding='utf-8') as file:file.write(program)
        with tempfile.TemporaryFile() as output,tempfile.TemporaryFile() as error:
            try:
                result=subprocess.run([sys.executable,'-I',path],input=stdin.encode(),stdout=output,
                    stderr=error,timeout=5,cwd=directory,env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'},preexec_fn=limits)
                output.seek(0);error.seek(0)
                return {'passed':result.returncode==0,'exit_code':result.returncode,
                        'stdout':output.read(32768).decode('utf-8','replace'),
                        'error':error.read(1500).decode('utf-8','replace')}
            except subprocess.TimeoutExpired:return {'passed':False,'error':'timeout'}


def main():
    jobs=json.load(sys.stdin);results=[]
    for job in jobs:
        if job['kind']=='function':
            tests='\n'.join(job['assertions'])
            prefix='from __future__ import annotations\n'
            result=execute(prefix+job['reference_code']+'\n'+tests+'\n')
            if result['passed']:
                # Weak-test diagnostics, not a proof of comprehensive coverage.
                mutants=[]
                for value in ['None',job['constant_expected']]:
                    mutant='def '+job['entry_point']+'(*args, **kwargs):\n    return '+value+'\n'
                    mutants.append(execute(prefix+mutant+tests+'\n')['passed'])
                result['constant_mutant_passes']=mutants
        elif job['kind']=='stdio':
            result={'passed':True,'cases_checked':0}
            started=time.monotonic()
            for case in job['cases']:
                if time.monotonic()-started>20:
                    result={'passed':False,'cases_checked':result['cases_checked'],'error':'per_problem_time_budget'};break
                case_result=execute(job['reference_code'],case['input'])
                if not case_result['passed'] or case_result.get('stdout','').split()!=case['output'].split():
                    result={'passed':False,'cases_checked':result['cases_checked'],
                            'error':case_result.get('error') or 'stdout_mismatch'};break
                result['cases_checked']+=1
        else:result={'passed':False,'error':'unknown_job_kind'}
        results.append({'sample_id':job['sample_id'],**result})
    json.dump({'python':sys.version,'results':results},sys.stdout)


if __name__=='__main__':main()
