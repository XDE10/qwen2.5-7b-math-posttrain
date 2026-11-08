"""Revalidate the sealed OpenR1 selection against source trajectories and export.

Preserves the original 1500 memberships/solutions. Allows documented SFT/RL
train overlap, forbids current dev/test overlap, and never invents difficulty.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from transformers import AutoTokenizer
from src.data.screening_io import rows,sha,json_write,ScreeningSimilarityIndex
from src.data.prepare_math_data import write_jsonl
from src.data.screen_openr1_sft import sealed_math,answer_is_complete,PROMPT

OUT=Path('data/project1_sft')


def bucket(n):
    if n<500:return '<500'
    if n<800:return '500-799'
    if n<1200:return '800-1199'
    if n<=1600:return '1200-1600'
    if n<=2048:return '1601-2048'
    if n<=4096:return '2049-4096'
    return '>4096'


def proxy(row):
    correct=row.get('correctness_math_verify');complete=row.get('is_reasoning_complete')
    if not isinstance(correct,list) or not correct or not isinstance(complete,list) or len(correct)!=len(complete):return 'unknown'
    if not all(isinstance(x,bool) for x in correct):return 'unknown'
    return str(sum(correct))+'/'+str(len(correct))


def audit_at(audits,position,sample_id):
    """Source UUIDs can repeat: use scan position, never collapse into a dict."""
    audit=audits[position]
    if audit['sample_id']!=sample_id:raise ValueError('Source scan/audit position mismatch')
    return audit


def strata_sample(records):
    ordered=sorted(records,key=lambda r:hashlib.sha256(('audit-v1/'+r['sample_id']).encode()).hexdigest())
    chosen={};groups={}
    for r in ordered:
        for key in [('type_length',r.get('problem_type'),bucket(r['answer_token_count'])),('source',r.get('source_category'))]:
            groups.setdefault(repr(key),r)
    for r in groups.values():chosen[r['sample_id']]=r
    if len(chosen)>50:raise ValueError('More audit strata than 50; explicit larger audit needed')
    for r in ordered:
        if len(chosen)>=50:break
        chosen.setdefault(r['sample_id'],r)
    return [{**r,'human_verdict':None,'human_notes':''} for r in chosen.values()]


def run():
    OUT.mkdir(exist_ok=True)
    prior_path=Path('data/project1_openr1/manifest.json')
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    assert sha(prior['output']['path'])==prior['output']['sha256']
    original=list(rows(prior['output']['path']));assert len(original)==1500
    source=json.loads(Path('data/screening_sources/openr1_source.json').read_text(encoding='utf-8'))
    assert source['revision']==prior['source_revision']
    _,protected=sealed_math()
    identities={'script_sha256':sha(__file__),'original_manifest_sha256':sha(prior_path),
                'original_selection_sha256':prior['output']['sha256'],
                'source_revision':source['revision'],'protected_hashes':prior['protected_hashes']}
    if (OUT/'manifest.json').exists():
        old=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        if old['identity']==identities:print('Project1 SFT export already sealed');return old
        raise ValueError('Use a new output version for changed SFT export')
    tokenizer_dir=Path('data/tokenizers/Qwen2.5-7B-Instruct')
    for name,digest in prior['tokenizer_files'].items():assert sha(tokenizer_dir/name)==digest
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,trust_remote_code=False)
    eval_index=ScreeningSimilarityIndex(.85);rl_index=ScreeningSimilarityIndex(.85);seen=ScreeningSimilarityIndex(.85)
    for key in ['math_dev','math500_test','gsm8k_test']:
        for r in protected[key]:eval_index.add(r)
    for r in protected['math_train']:rl_index.add(r)
    overlaps=[];prepared={};rechecks=[]
    for r in original:
        assert eval_index.match(r['problem']) is None
        assert seen.match(r['problem']) is None;seen.add(r)
        count=len(tokenizer.encode(r['solution'],add_special_tokens=False))
        total=len(tokenizer.apply_chat_template([{'role':'system','content':PROMPT},{'role':'user','content':r['problem']},
            {'role':'assistant','content':r['solution']}],tokenize=True,add_generation_prompt=False))
        assert count==r['response_token_count'] and 500<=count<=1600 and total==r['total_token_count'] and total<=4096
        assert answer_is_complete(r['solution'])
        hit=rl_index.match(r['problem'])
        if hit:overlaps.append({'sample_id':r['sample_id'],**hit})
        prepared[r['sample_id']]={**r,'answer_token_count':count,'difficulty':None,
            'difficulty_status':'source_has_no_reliable_difficulty_labels','rl_train_overlap':hit is not None}
    trajectory_audit=list(rows('data/project1_openr1/trajectory_audit.jsonl'))
    assert len(trajectory_audit)==prior['source_rows_scanned']
    before={k:Counter() for k in ['domain','problem_type','question_type','source_category','difficulty','teacher_success_proxy']}
    length_before=Counter();by_type_lengths={};raw_trajectory_count=0
    for audit in trajectory_audit:
        raw_trajectory_count+=sum(t['trajectory_index'] is not None for t in audit['trajectories'])
        for t in audit['trajectories']:
            if 'response_token_count' in t:length_before[bucket(t['response_token_count'])]+=1
    files=list(source['files']);random.Random(prior['seed']).shuffle(files)
    selected_uuids=pa.array([r['sample_id'].split('/',1)[1] for r in original]);scanned=0;source_files_checked=[]
    needed={'uuid','problem','problem_type','question_type','source','generations','is_reasoning_complete',
            'correctness_math_verify','finish_reasons'}
    for item in files:
        if scanned>=prior['source_rows_scanned']:break
        assert sha(item['path'])==item['sha256'];source_files_checked.append(item['path'])
        parquet=pq.ParquetFile(item['path']);columns=[c for c in parquet.schema_arrow.names if c in needed]
        for batch in parquet.iter_batches(batch_size=128,columns=columns):
            remaining=prior['source_rows_scanned']-scanned
            if remaining<=0:break
            if batch.num_rows>remaining:batch=batch.slice(0,remaining)
            meta=batch.select([c for c in columns if c not in {'generations','problem'}]).to_pylist()
            for offset,row in enumerate(meta):
                sid='openr1/'+str(row.get('uuid',scanned+offset))
                audit=audit_at(trajectory_audit,scanned+offset,sid)
                values={'domain':'math','problem_type':row.get('problem_type') or 'unknown',
                    'question_type':row.get('question_type') or 'unknown','source_category':row.get('source') or 'unknown',
                    'difficulty':'unknown','teacher_success_proxy':proxy(row)}
                for key,value in values.items():before[key][value]+=1
                for detail in audit['trajectories']:
                    if 'response_token_count' in detail:
                        by_type_lengths.setdefault(values['problem_type'],Counter())[bucket(detail['response_token_count'])]+=1
            mask=pc.is_in(batch.column(batch.schema.get_field_index('uuid')),value_set=selected_uuids)
            selected=batch.filter(mask).to_pylist()
            for offset,raw in zip(pc.indices_nonzero(mask).to_pylist(),selected):
                sid='openr1/'+raw['uuid'];record=prepared[sid];i=record['trajectory_index']
                audit=audit_at(trajectory_audit,scanned+offset,sid)
                eligible=[t for t in audit['trajectories'] if t['reason']=='eligible']
                if audit['problem_rejection'] is not None or not eligible:continue
                gens=raw['generations'];correct=raw['correctness_math_verify'];complete=raw['is_reasoning_complete']
                assert len(gens)==len(correct)==len(complete) and correct[i] is True and complete[i] is True
                assert gens[i]==record['solution'] and raw['problem']==record['problem']
                finishes=raw.get('finish_reasons')
                if finishes is not None:assert len(finishes)==len(gens) and finishes[i] in {None,'stop','eos','eos_token','stop_sequence'}
                assert min(eligible,key=lambda t:(t['response_token_count'],t['trajectory_index']))['trajectory_index']==i
                record['source_teacher_success_proxy']=proxy(raw)
                rechecks.append({'sample_id':sid,'trajectory_index':i,'source_flags_aligned':True,
                    'source_text_sha256':hashlib.sha256(gens[i].encode()).hexdigest(),
                    'answer_token_count':record['answer_token_count'],'total_token_count':record['total_token_count']})
            scanned+=batch.num_rows
            if scanned%4096==0:print('Source replay:',scanned,'verified selected:',len(rechecks),flush=True)
    assert scanned==prior['source_rows_scanned'] and len(rechecks)==len(prepared)==1500
    records=[prepared[r['sample_id']] for r in original]
    after={key:dict(Counter('unknown' if key=='difficulty' else r.get('source_teacher_success_proxy','unknown')
        if key=='teacher_success_proxy' else r.get(key) or 'unknown' for r in records)) for key in before}
    distributions={'before_questions':{k:dict(v) for k,v in before.items()},'after_questions':after,
        'scanned_questions':scanned,'source_trajectories_in_scan':raw_trajectory_count,
        'repeated_source_uuid_rows':len(trajectory_audit)-len({a['sample_id'] for a in trajectory_audit}),
        'length_before_compact_filter':dict(length_before),'length_before_denominator':sum(length_before.values()),
        'length_after_selected':dict(Counter(bucket(r['answer_token_count']) for r in records)),
        'length_population_note':'Before: trajectories passing indexed math-verification, completeness/finish/text checks; after: one compact selected trajectory per question. Incorrect/incomplete trajectories were not token-counted by original screen.',
        'before_verified_lengths_by_problem_type':{k:dict(v) for k,v in by_type_lengths.items()},
        'difficulty_available':False,'difficulty_missing_before':scanned,'difficulty_missing_after':1500,
        'difficulty_proxy_note':'Count of source math-verify true flags / aligned trajectory count; model-dependent solvability proxy, NOT a difficulty label or heldout evaluation.',
        'difficulty_conclusion':'Cannot establish absence of easy-problem bias without reliable difficulty labels or separate model evaluation. Source origin and short responses are not difficulty guarantees.'}
    write_jsonl(OUT/'sft_train.jsonl',records);write_jsonl(OUT/'rl_overlap_audit.jsonl',overlaps)
    write_jsonl(OUT/'trajectory_revalidation.jsonl',rechecks);write_jsonl(OUT/'human_audit_50.jsonl',strata_sample(records))
    json_write(OUT/'distributions.json',distributions)
    manifest={'identity':identities,'config':'default','split':'train','source_rows_scanned':scanned,
        'selected_rows':len(records),'membership_and_solutions_unchanged_from_original':True,
        'answer_tokens':{'min':min(r['answer_token_count'] for r in records),'max':max(r['answer_token_count'] for r in records)},
        'total_tokens_max':max(r['total_token_count'] for r in records),'tokenizer_revision':prior['tokenizer_revision'],
        'tokenizer_files':prior['tokenizer_files'],'source_files_rechecked':source_files_checked,
        'evaluation_overlap_rows':0,'rl_overlap_rows':len(overlaps),
        'rl_overlap_policy':'ALLOW train-to-train overlap; enumerate matched IDs/Jaccard evidence. Do not call this evaluation leakage. All SFT-started algorithm groups must share this same SFT and RL pools; report SFT-seen RL questions separately when analyzing.',
        'rl_overlap_kinds':dict(Counter(r['kind'] for r in overlaps)),
        'difficulty_labels':'unavailable; no guarantee of unbiased difficulty distribution',
        'quality_gate':'Source flags, exact indexed replay, lengths and protected-split checks passed. 50 stratified human audit rows exported; human reasoning/label audit PENDING.',
        'scan_scope':'Same seed42 shuffled-shard prefix as original; stop at first 1500 accepted; not whole-source random sampling.',
        'files':{p.name:{'path':p.as_posix(),'sha256':sha(p)} for p in OUT.iterdir() if p.suffix in {'.json','.jsonl'} and p.name!='manifest.json'},
        'gpu_operations':0,'paid_api_calls':0,'model_weights_loaded':False}
    json_write(OUT/'manifest.json',manifest)
    print('Project1 SFT finalized:',len(records),'RL overlaps:',len(overlaps),'difficulty labels unavailable',flush=True)
    return manifest


if __name__=='__main__':run()
