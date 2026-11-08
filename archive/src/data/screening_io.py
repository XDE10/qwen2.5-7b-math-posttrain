"""Pinned, parquet-only public dataset download helpers (never model weights)."""
import hashlib
import json
import time
import math
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from src.data.prepare_math_data import SimilarityIndex, fingerprint, shingles

SOURCES = {'openr1':('open-r1/OpenR1-Math-220k','data/'),
           'code':('open-r1/verifiable-coding-problems-python','data/')}


class ScreeningSimilarityIndex(SimilarityIndex):
    """Exact Jaccard threshold search using a safe rare-prefix candidate filter.

    If J(A,B)>=t, A can miss at most (1-t)*|A| tokens from B. Therefore any
    prefix of A longer than that must share a token with B. Selecting rare
    tokens reduces common-boilerplate postings without losing qualifying pairs.
    Final intersection/Jaccard and tie ordering are identical to the reference.
    """
    def match(self,text):
        raw=hashlib.sha256(text.encode()).hexdigest()
        if raw in self.raw_exact:return {'matched_id':self.raw_exact[raw],'similarity':1.,'kind':'raw_exact'}
        key=fingerprint(text)
        if key in self.exact:return {'matched_id':self.exact[key],'similarity':1.,'kind':'normalized_exact'}
        grams=shingles(text)
        if not grams:return None
        k=min(len(grams),math.ceil((1-self.threshold)*len(grams))+1)
        prefix=sorted(grams,key=lambda g:(len(self.postings.get(g,())),g))[:k]
        candidates={i for g in prefix for i in self.postings.get(g,())}
        passing=[]
        for i in candidates:
            name,other=self.entries[i]
            if min(len(grams),len(other))+1e-12<self.threshold*max(len(grams),len(other)):continue
            common=len(grams&other)
            score=common/len(grams|other)
            if score+1e-12>=self.threshold:passing.append((common,i,name,score))
        if not passing:return None
        _,_,name,score=min(passing,key=lambda x:(-x[0],x[1]))
        return {'matched_id':name,'similarity':score,'kind':'jaccard_3gram'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def source_files(key, root=Path('.')):
    repo,prefix = SOURCES[key]
    path = root/'data/screening_sources'/f'{key}_source.json'
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    info = HfApi().dataset_info(repo,files_metadata=True)
    files = [{'name':f.rfilename,'bytes':f.size} for f in info.siblings
             if f.rfilename.startswith(prefix) and f.rfilename.endswith('.parquet')]
    if not files: raise ValueError('No allowlisted parquet files')
    result = {'repo':repo,'revision':info.sha,'config':'default','split':'train','files':files}
    json_write(path,result)
    return result


def download_source(key,root=Path('.')):
    from concurrent.futures import ThreadPoolExecutor
    meta = source_files(key,root)
    def one(f):
        name = f['name']
        if not name.startswith('data/') or not name.endswith('.parquet'):
            raise ValueError('Only public dataset parquet is allowed')
        for attempt in range(6):
            try:
                p = Path(hf_hub_download(meta['repo'],name,repo_type='dataset',revision=meta['revision'],
                                       local_dir=root/'data/screening_sources'/key))
                break
            except Exception as exc:
                if attempt == 5: raise
                print(f'{key}: retry {attempt+1} for {name} ({type(exc).__name__})',flush=True)
                time.sleep(min(2**attempt,16))
        if f['bytes'] is not None and p.stat().st_size != f['bytes']:
            raise ValueError('Dataset file size mismatch')
        return {**f,'path':p.relative_to(root).as_posix(),'sha256':sha(p)}
    with ThreadPoolExecutor(max_workers=2) as pool: meta['files'] = list(pool.map(one,meta['files']))
    json_write(root/'data/screening_sources'/f'{key}_source.json',meta)
    print(f'{key}: {len(meta["files"])} dataset shards downloaded and hashed',flush=True)
    return meta


def parquet_rows(meta,root=Path('.')):
    import pyarrow.parquet as pq
    for f in meta['files']:
        if 'path' not in f: raise ValueError('Source download is not complete')
        path = root/f['path']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            yield from batch.to_pylist()
