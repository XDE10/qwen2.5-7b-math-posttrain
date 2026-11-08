"""Freeze train-only calibration splits and a 50-question blind-solution pilot."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow.parquet as pq
from src.synthesis.deepseek_pilot import dump, sha, solve_payload, difficulty_payload
from src.data.screening_io import ScreeningSimilarityIndex

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/deepseek_pilot_20261023'
QUOTAS={'Algebra':15,'Geometry':10,'Number Theory':10,'Combinatorics':8,'Inequalities':5,'Logic and Puzzles':2}


def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_rows(p):return [json.loads(x) for x in Path(p).open(encoding='utf-8')]
def rank(row):return sha('pilot-v1/'+row.get('question_hash',row.get('sample_id','')))
def bucket(row):
    n=row['trajectory']['answer_tokens']
    return '500-1600' if n<=1600 else '1601-2200' if n<=2200 else '2201-3000'


def run():
    if (OUT/'manifest.json').exists():
        m=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        for path,h in m['files'].items():assert file_sha(OUT/path)==h
        print('Frozen pilot exists; verified, no overwrite');return
    OUT.mkdir(exist_ok=True)
    proposal=ROOT/'reports/project1_sft_v2_plan_20261023/coverage_proposal.json'
    rows=json.loads(proposal.read_text(encoding='utf-8'))
    picked=[]
    for kind,count in QUOTAS.items():
        candidates=sorted([r for r in rows if r['problem_type']==kind],key=rank)
        groups=defaultdict(list)
        for r in candidates:groups[bucket(r)].append(r)
        chosen=[]
        while len(chosen)<count:
            for b in sorted(groups):
                if groups[b] and len(chosen)<count:chosen.append(groups[b].pop(0))
        picked+=chosen
    assert len(picked)==50
    source_groups=defaultdict(list)
    for r in picked:source_groups[r['trajectory']['source_path']].append(r)
    references=[]
    source_hashes={x['path']:x['sha256'] for x in json.loads((ROOT/'reports/openr1_full_audit_final_20261023/identity.json').read_text(encoding='utf-8'))['source_files']}
    for path, group in source_groups.items():
        assert file_sha(ROOT/path)==source_hashes[path]
        indices={r['trajectory']['source_row']:r for r in group}
        offset=0
        for batch in pq.ParquetFile(ROOT/path).iter_batches(batch_size=512):
            for j,raw in enumerate(batch.to_pylist()):
                if offset+j not in indices:continue
                selected=indices[offset+j];t=selected['trajectory'];i=t['trajectory_index']
                assert raw['is_reasoning_complete'][i] and raw['correctness_math_verify'][i]
                solution=raw['generations'][i]
                assert hashlib.sha256(solution.encode()).hexdigest()==t['text_sha256']
                references.append({'sample_id':selected['question_hash'],'problem':raw['problem'],
                    'reference_answer':raw['answer'],'raw_solution':solution,'selection':selected,
                    'source_revision':'e4e141ec9dea9f8326f4d347be56105859b2bd68','split':'train'})
            offset+=batch.num_rows
    assert len(references)==50
    # Keep anchors/calibration problems disjoint and conservatively non-near-duplicate.
    index=ScreeningSimilarityIndex(.85)
    for r in references:index.add(r)
    math_path=ROOT/'data/math/math_train.jsonl'
    math=read_rows(math_path)
    by=defaultdict(list)
    for r in sorted(math,key=rank):
        if r['level'] in [f'Level {i}' for i in range(1,6)]:by[(r['problem_type'],r['level'])].append(r)
    subjects=sorted({k[0] for k in by})
    splits={}
    for split in ['anchors','calibration_dev','frozen_acceptance']:
        chosen=[]
        for level in range(1,6):
            order=subjects if split=='anchors' else subjects+[subjects[(level+j)%len(subjects)] for j in range(3)]
            for subject in order:
                candidates=by[(subject,f'Level {level}')]
                while candidates:
                    r=candidates.pop(0)
                    if index.match(r['problem']) is None:break
                else:raise ValueError('Insufficient disjoint calibration stratum')
                index.add(r)
                chosen.append({k:r[k] for k in ['sample_id','problem','problem_type','level','source','source_revision','split','problem_hash']})
        splits[split]=chosen
    assert [len(splits[k]) for k in splits]==[35,50,50]
    ids=[r['sample_id'] for rs in splits.values() for r in rs]
    assert len(ids)==len(set(ids))
    for k,rs in splits.items():dump(OUT/(k+'.json'),rs)
    references.sort(key=lambda x:x['sample_id']);dump(OUT/'verifier_references.json',references)
    requests=[{k:r[k] for k in ['sample_id','problem','source_revision','split']} for r in references]
    dump(OUT/'blind_requests.json',requests)
    # Freeze prompt and go/no-go before any labels are observed.
    policy={'version':'deepseek-pilot-v1','generation_mode':'blind_independent_compact_solution',
        'budget_usd_reserved':3.0,'max_requests':320,'max_generation_attempts':3,'generation_sample_count':50,
        'calibration_dev_n':50,'frozen_acceptance_n':50,'anchor_n':35,'no_prompt_tuning_this_run':True,
        'difficulty_gate':{'exact_agreement_min':.60,'within_one_min':.90,'low_to_high_false_positive_rate_max':.20,
            'valid_schema_min':1.0,'note':'Engineering pilot thresholds, not statistical certification or official difficulty labels'},
        'solution_gate':{'answer_tokens_min':500,'answer_tokens_max':1500,'total_tokens_max':3800,
            'math_verify_required':True,'process_review_required':True,'human_certification':False},
        'prompt_examples':{'solve':solve_payload('PLACEHOLDER'),
            'difficulty':difficulty_payload('PLACEHOLDER',splits['anchors'])},
        'prices_source':'https://api-docs.deepseek.com/quick_start/pricing/',
        'rate_date':'2026-10-23','difficulty_semantics':'estimated_level_only',
        'training_status':'pilot_not_approved_for_formal_training'}
    dump(OUT/'policy.json',policy)
    manifest={'files':{p.name:file_sha(p) for p in OUT.iterdir() if p.is_file()},
        'input_hashes':{str(p.relative_to(ROOT)):file_sha(p) for p in [proposal,math_path,ROOT/'data/math/math_dev.jsonl',ROOT/'data/project1_sft/sft_train.jsonl']},
        'split_counts':{k:len(v) for k,v in splits.items()},'pilot_types':dict(Counter(r['selection']['problem_type'] for r in references)),
        'pilot_length_bins':dict(Counter(bucket(r['selection']) for r in references)),
        'scope':'all API problem inputs from train only; dev/test files used only for existing frozen exclusion evidence',
        'excluded_unknown_level_rows':sum(r['level']=='Level ?' for r in math)}
    dump(OUT/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['split_counts','pilot_types','pilot_length_bins']},ensure_ascii=False))

if __name__=='__main__':run()
