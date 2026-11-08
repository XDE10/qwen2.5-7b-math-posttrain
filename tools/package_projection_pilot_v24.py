"""Build a self-contained v24 ZIP without changing previous releases."""
from pathlib import Path
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.projection_pilot_v21.pilot import save,sha


def main():
    report = ROOT/'reports/projection_pilot_20261026_v24'
    names = ['configs/project1_projection_pilot_v24.json',
        'experiments/projection_pilot_v24/pilot.py','experiments/projection_pilot_v24/launch.py',
        'experiments/projection_pilot_v21/pilot.py','experiments/projection_pilot_v21/parallel_bench.py',
        'experiments/projection_pilot_v21/install_missing.sh',
        'reports/projection_pilot_20261026_v21/selected32.jsonl',
        'src/training/frozen_verifier_v4.py','src/training/math_reward_v2.py',
        'scripts/math_verify_v3_strict_worker.py','scripts/math_verify_pilot_worker.py',
        'scripts/verifier_stream_worker_v4.py','tools/render_projection_prompt_v24.py',
        'tests/test_projection_pilot_v24.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'.pilot_deps').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    names += [p.relative_to(ROOT).as_posix() for p in sorted(report.glob('*'))
              if p.is_file() and p.suffix!='.zip' and p.name not in ('bundle_checksums.json','bundle_info.json')]
    # Include the small frozen Qwen tokenizer for independent CPU prompt inspection.
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'data/tokenizers/Qwen2.5-7B-Instruct').glob('*'))
              if p.is_file() and p.name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json']]
    if len(names)!=len(set(names)):
        raise ValueError('Duplicate package paths')
    checks = {n:sha(ROOT/n) for n in names}
    save(report/'bundle_checksums.json',checks)
    names.append('reports/projection_pilot_20261026_v24/bundle_checksums.json')
    output = report/'project1_projection_pilot_v24.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(ROOT/name,'projection-pilot-v24/'+name)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP integrity failure')
        import hashlib
        for name,expected in checks.items():
            if hashlib.sha256(z.read('projection-pilot-v24/'+name)).hexdigest()!=expected:
                raise ValueError('ZIP entry mismatch')
    info = {'path':str(output),'files':len(names),'bytes':output.stat().st_size,'sha256':sha(output)}
    save(report/'bundle_info.json',info)
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
