"""Add the parallel benchmark to an unchanged v21 ZIP, verifying every entry."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT/'reports/projection_pilot_20261026_v21/project1_projection_pilot_v21.zip'
    report = ROOT/'reports/parallel_bench_20261026_v22'
    additions = ['experiments/projection_pilot_v21/parallel_bench.py',
                 'reports/parallel_bench_20261026_v22/README.md',
                 'reports/parallel_bench_20261026_v22/validation.json']
    output = report/'project1_parallel_bench_v22.zip'
    with zipfile.ZipFile(base) as original, zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for entry in original.infolist():
            bundle.writestr(entry.filename, original.read(entry.filename))
        for name in additions:
            bundle.write(ROOT/name, 'projection-pilot-v21/'+name)
    with zipfile.ZipFile(base) as original, zipfile.ZipFile(output) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('ZIP integrity failure')
        if len(bundle.namelist()) != len(set(bundle.namelist())):
            raise ValueError('Duplicate ZIP paths')
        for name in original.namelist():
            if bundle.read(name) != original.read(name):
                raise ValueError('Original v21 entry modified: '+name)
        for name in additions:
            if bundle.read('projection-pilot-v21/'+name) != (ROOT/name).read_bytes():
                raise ValueError('Addition mismatch: '+name)
    info = {'path':str(output), 'bytes':output.stat().st_size,
            'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'original_v21_entries_unchanged':True,
            'gpu_tested':False}
    (report/'bundle_info.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
