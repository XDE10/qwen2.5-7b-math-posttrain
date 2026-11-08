"""Build a small, explicit source overlay for the already-delivered v26 bundle."""
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.aimo_sft_v26.common import read,save,sha

def main():
    report=ROOT/'reports/colab_compat_20261026_v27'
    path=ROOT/'reports/aimo_sft_20261026_v26/bundle_checksums.json'
    checks=read(path)
    names=['experiments/aimo_sft_v26/train.py','experiments/aimo_sft_v26/install_missing.py']
    original={}
    for name in names:
        old=sha(report/'originals'/Path(name).name)
        if checks[name] not in [old,sha(ROOT/name)]:raise ValueError('Unexpected base source: '+name)
        original[name]=old;checks[name]=sha(ROOT/name)
    save(path,checks)
    names+=['reports/aimo_sft_20261026_v26/bundle_checksums.json','tests/test_colab_compat_v27.py',
        'reports/colab_compat_20261026_v27/README.md','reports/colab_compat_20261026_v27/validation.json']
    save(report/'patch_manifest.json',{'original_source':original,'patched_files':{name:sha(ROOT/name) for name in names}})
    names.append('reports/colab_compat_20261026_v27/patch_manifest.json')
    output=report/'project1_colab_compat_v27.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,name)
    save(report/'bundle_info.json',{'path':str(output),'bytes':output.stat().st_size,'sha256':sha(output)})
    print(read(report/'bundle_info.json'))

if __name__=='__main__':main()
