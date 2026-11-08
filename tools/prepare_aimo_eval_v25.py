"""Freeze public AIME 2024/2025 inputs and reuse the existing MATH500 file."""
import io
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.aimo_sft_v25.common import rows,save,sha


def main():
    import pyarrow.parquet as pq
    report=ROOT/'reports/aimo_sft_20261026_v25';sources={}
    specs=[('aime2024','HuggingFaceH4/aime_2024','2fe88a2f1091d5048c0f36abc874fb997b3dd99a','data/train-00000-of-00001.parquet'),
           ('aime2025','yentinglin/aime_2025','6f71d77b0b89b9dabe07ab466c51df33f514df7f','data/train-00000-of-00001-243207c6c994e1bd.parquet')]
    for label,repo,revision,file in specs:
        url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}'
        with urllib.request.urlopen(url,timeout=90) as response:raw=response.read()
        data=pq.read_table(io.BytesIO(raw)).to_pylist()
        if len(data)!=30:raise ValueError('Expected 30 AIME problems')
        normalized=[{'sample_id':f'{label}-{r["id"]}','problem':r['problem'],'answer':str(int(r['answer']))} for r in data]
        path=report/(label+'.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in normalized),encoding='utf-8')
        sources[label]={'repo':repo,'revision':revision,'source_file':file,'file':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'count':30}
    data=rows(ROOT/'data/math/math500_test.jsonl')
    if len(data)!=500:raise ValueError('Expected existing MATH500 500 rows')
    normalized=[{'sample_id':r['sample_id'],'problem':r['problem'],'answer':str(r['reference_answer'])} for r in data]
    path=report/'math500.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in normalized),encoding='utf-8')
    sources['math500']={'source':'existing project data/math/math500_test.jsonl','file':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'count':500}
    save(report/'evaluation_sources.json',sources)
    print(json.dumps(sources,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
