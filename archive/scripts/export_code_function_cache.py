"""Reproducible small function-row cache from the pinned full parquet source."""
from pathlib import Path
from src.data.screening_io import source_files,parquet_rows,json_write,sha
from src.data.prepare_math_data import write_jsonl


def run(root=Path('.')):
    source=source_files('code',root);selected=[];scanned=0
    for row in parquet_rows(source,root):
        scanned+=1
        cases=(row.get('verification_info') or {}).get('test_cases') or []
        if any(c.get('type')=='function_call' for c in cases if isinstance(c,dict)):
            selected.append(row)
    output=root/'data/screening_sources/code_function_rows.jsonl'
    temp=output.with_suffix('.jsonl.tmp')
    write_jsonl(temp,selected)
    if output.exists() and sha(output)!=sha(temp):
        raise ValueError('Existing function cache differs from pinned source; preserve and investigate')
    temp.replace(output)
    json_write(output.with_suffix('.manifest.json'),{'source_revision':source['revision'],
        'source_rows_scanned':scanned,'function_rows':len(selected),'sha256':sha(output),
        'script_sha256':sha(__file__)})
    print('Function cache verified:',scanned,'source rows;',len(selected),'function rows')


if __name__=='__main__':run()
