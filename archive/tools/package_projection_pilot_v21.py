"""Package only the independent inference pilot and its existing grader deps."""
from pathlib import Path
import json
import zipfile
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.projection_pilot_v21.pilot import sha,save


def main():
    report=ROOT/'reports/projection_pilot_20261026_v21'
    names=['configs/project1_projection_pilot_v21.json',
        'experiments/projection_pilot_v21/pilot.py','experiments/projection_pilot_v21/launch.py',
        'experiments/projection_pilot_v21/install_missing.sh',
        'src/training/frozen_verifier_v4.py','src/training/math_reward_v2.py',
        'scripts/math_verify_v3_strict_worker.py','scripts/math_verify_pilot_worker.py',
        'scripts/verifier_stream_worker_v4.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'.pilot_deps').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    names += [p.relative_to(ROOT).as_posix() for p in sorted(report.rglob('*')) if p.is_file()
              and p.suffix not in ('.zip','.png') and p.name not in ('bundle_checksums.json','bundle_info.json')]
    checks={n:sha(ROOT/n) for n in names}
    save(report/'bundle_checksums.json',checks)
    names.append('reports/projection_pilot_20261026_v21/bundle_checksums.json')
    out=report/'project1_projection_pilot_v21.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,'projection-pilot-v21/'+name)
    with zipfile.ZipFile(out) as z:
        bad=z.testzip()
        if bad:raise ValueError('Corrupt ZIP entry '+bad)
        for name,value in checks.items():
            import hashlib
            if hashlib.sha256(z.read('projection-pilot-v21/'+name)).hexdigest()!=value:
                raise ValueError('ZIP contents differ '+name)
    info={'path':str(out),'bytes':out.stat().st_size,'sha256':sha(out),'files':len(names)}
    save(report/'bundle_info.json',info)
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
