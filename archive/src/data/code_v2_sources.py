"""Pinned AceCode/Magicoder sources; pilot downloads only one AceCode shard."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from src.data.screening_io import json_write, sha, rows

ROOT=Path('data/code_v2_sources')
REPOS={'ace':'TIGER-Lab/AceCode-87K','magicoder':'ise-uiuc/Magicoder-OSS-Instruct-75K'}


def metadata(key):
    path=ROOT/f'{key}_manifest.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    info=HfApi().dataset_info(REPOS[key],files_metadata=True)
    files=[{'name':f.rfilename,'bytes':f.size} for f in info.siblings
           if (key=='ace' and f.rfilename.endswith('.parquet')) or
           (key=='magicoder' and f.rfilename=='data-oss_instruct-decontaminated.jsonl')]
    result={'repo':REPOS[key],'revision':info.sha,'files':files,'license_tag':'mit'}
    json_write(path,result)
    return result


def download(key,full=False,shards=1):
    meta=metadata(key)
    wanted=meta['files'] if key=='magicoder' or full else meta['files'][:shards]
    for item in wanted:
        if item.get('path') and Path(item['path']).exists() and sha(item['path'])==item['sha256']:continue
        for attempt in range(6):
            try:
                path=Path(hf_hub_download(meta['repo'],item['name'],repo_type='dataset',
                    revision=meta['revision'],local_dir=ROOT/key))
                if path.stat().st_size!=item['bytes']:raise ValueError('Source size mismatch')
                item.update(path=path.as_posix(),sha256=sha(path));break
            except Exception:
                if attempt==5:raise
                time.sleep(min(2**attempt,16))
        json_write(ROOT/f'{key}_manifest.json',meta)
    print(key, 'downloaded',len(wanted),'files',flush=True)
    return meta


def source_rows(key,full=False,shards=1):
    meta=metadata(key)
    files=meta['files'] if full or key=='magicoder' else meta['files'][:shards]
    for item in files:
        if not item.get('path'):raise ValueError('Download requested scope first')
        if key=='magicoder':yield from rows(item['path'])
        else:
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(item['path']).iter_batches(batch_size=32):
                yield from batch.to_pylist()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');parser.add_argument('--ace-shards',type=int,default=1);args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda key:download(key,args.full,args.ace_shards),REPOS))
