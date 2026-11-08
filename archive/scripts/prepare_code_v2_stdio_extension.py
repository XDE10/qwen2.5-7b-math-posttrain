"""Extend dev candidates from never-trained v1 prompts; no split is overwritten."""
import json
from pathlib import Path
from src.data.screening_io import rows,source_files,parquet_rows,sha,json_write
from src.data.prepare_math_data import write_jsonl
from src.data.screen_code_interfaces import python_code
from src.data.code_v2_dedup import ast_hash
from src.data.build_code_v2 import order


def run():
    base=Path('data/project2_code_v2')
    candidates=[r for r in rows('data/project2_code/train_prompts.jsonl')
                if r['interface']=='stdin_stdout' and 3<=r['test_case_count']<=4]
    candidates.sort(key=lambda r:(r['test_case_count'],order(r)))
    selected={r['sample_id']:r for r in candidates[:80]};result=[]
    source=source_files('code')
    for position,raw in enumerate(parquet_rows(source)):
        sid='code/'+str(raw.get('problem_id',position))
        if sid not in selected:continue
        metadata=selected[sid]
        assert ast_hash(raw['gold_standard_solution'])==metadata['reference_ast_sha256']
        result.append({**metadata,'gold_standard_solution':raw['gold_standard_solution'],
                       'verification_info':raw['verification_info']})
        if len(result)==len(selected):break
    assert len(result)==80
    path=base/'extra_stdio_dev_candidates.jsonl';write_jsonl(path,result)
    json_write(base/'stdio_extension_provenance.json',{'source_revision':source['revision'],
        'selected_candidates':80,'source_v1_train_sha256':sha('data/project2_code/train_prompts.jsonl'),
        'output_sha256':sha(path),'script_sha256':sha(__file__),
        'reason':'Legacy dev cannot supply 100 passing >=3-case stdout tasks; re-seal additional v2 dev from CPU-only, never-trained v1 prompts. Exclude all selected dev from v2 train.'})
    print('Additional stdin/stdout dev candidates prepared:',len(result),flush=True)


if __name__=='__main__':run()
