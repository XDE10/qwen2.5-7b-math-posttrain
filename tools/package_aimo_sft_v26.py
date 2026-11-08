"""Package all eligible original traces and independent train/eval entrypoints."""
from pathlib import Path
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.aimo_sft_v26.common import save,sha


def main():
    report=ROOT/'reports/aimo_sft_20261026_v26'
    names=['configs/project1_aimo_sft_v26.json','tests/test_aimo_sft_v26.py',
        'tools/prepare_aimo_sft_v26.py',
        'src/training/frozen_verifier_v4.py','src/training/math_reward_v2.py',
        'scripts/math_verify_v3_strict_worker.py','scripts/math_verify_pilot_worker.py','scripts/verifier_stream_worker_v4.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'experiments/aimo_sft_v26').glob('*.py'))]
    names += ['experiments/aimo_sft_v26/start_colab.sh']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'.pilot_deps').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    names += [p.relative_to(ROOT).as_posix() for p in sorted(report.glob('*'))
              if p.is_file() and p.suffix!='.zip' and p.name not in ['bundle_checksums.json','bundle_info.json']]
    names += ['data/tokenizers/Qwen2.5-7B-Instruct/'+f for f in ['tokenizer.json','tokenizer_config.json']]
    checks={name:sha(ROOT/name) for name in names}
    save(report/'bundle_checksums.json',checks)
    names.append('reports/aimo_sft_20261026_v26/bundle_checksums.json')
    output=report/'project1_aimo_sft_v26.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,'aimo-sft-v26/'+name)
    with zipfile.ZipFile(output) as z:
        import hashlib
        if z.testzip():raise ValueError('Corrupt ZIP')
        for name,value in checks.items():
            if hashlib.sha256(z.read('aimo-sft-v26/'+name)).hexdigest()!=value:raise ValueError('ZIP content mismatch')
    info={'path':str(output),'bytes':output.stat().st_size,'sha256':sha(output),'files':len(names)}
    save(report/'bundle_info.json',info);print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
