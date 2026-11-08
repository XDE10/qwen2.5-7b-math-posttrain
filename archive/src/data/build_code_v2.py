"""Pilot-gated, versioned 1250/1250 OPD pool and 100/100 executed dev.

No teacher calls or training. Source tests are checked for consistency and weak
constant mutants, not certified as an exhaustive semantic oracle.
"""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from src.data.code_v2_sources import source_rows,metadata
from src.data.code_function_screen import extract,InterfaceError
from src.data.audit_code_v2_pilot import decorate
from src.data.code_v2_dedup import Protected
from src.data.code_sandbox import run_jobs,function_job,environment
from src.data.screen_code_interfaces import python_code
from src.data.screening_io import rows,sha,json_write,ScreeningSimilarityIndex
from src.data.prepare_math_data import write_jsonl

OUT=Path('data/project2_code_v2')


def order(row):return hashlib.sha256(('42/'+row['sample_id']).encode()).hexdigest()


def check_gate():
    pilot=json.loads(Path('data/code_v2_pilot/report.json').read_text(encoding='utf-8'))
    if not pilot.get('bulk_gate_passed'):raise ValueError('Pilot acceptance gate is not passed')
    for path,digest in {**pilot['script_hashes'],**pilot['protected_files']}.items():
        if sha(path)!=digest:raise ValueError('Pilot identity changed: '+path)
    if sha('data/code_v2_pilot/agent_review.json')!=pilot['review_sha256']:raise ValueError('Pilot review changed')
    return pilot


def collect(shards):
    protected=Protected();candidates=[];counts={};audit=[]
    for source,limit in [('ace',3000),('magicoder',1000)]:
        counter=Counter();selected=0
        for raw in source_rows(source,shards=shards):
            counter['scanned']+=1;sid=source+'/'+str(raw.get('id') if source=='ace' else raw.get('raw_index'))
            evidence=None
            try:
                candidate=decorate(extract(raw,source),raw,source)
                evidence=protected.match(candidate)
                if evidence:raise InterfaceError(evidence['reason'])
                candidates.append(candidate);selected+=1;reason='static_candidate'
            except InterfaceError as exc:reason=str(exc)
            counter[reason]+=1;audit.append({'sample_id':sid,'reason':reason,'evidence':evidence})
            if counter['scanned']%2000==0:print(source,'scanned',counter['scanned'],'candidates',selected,flush=True)
            if selected>=limit:break
        counts[source]=dict(counter)
    # Prefer AceCode for its existing tests; Magicoder remains an OPD fallback.
    ordered=sorted(candidates,key=lambda r:(r['source_key']!='ace',order(r)))
    seen=ScreeningSimilarityIndex(.85);names=set();bodies=set();clean=[];rejections=Counter()
    for r in ordered:
        if r['entry_point'] in names:rejections['shared_entry_family']+=1;continue
        if r['reference_body_sha256'] in bodies:rejections['alpha_body_collision']+=1;continue
        if seen.match(r['problem']):rejections['near_prompt']+=1;continue
        seen.add(r);names.add(r['entry_point']);bodies.add(r['reference_body_sha256']);clean.append(r)
    write_jsonl(OUT/'function_candidates.jsonl',clean)
    write_jsonl(OUT/'function_source_audit.jsonl',audit)
    result={'sources':counts,'before_family_filter':len(candidates),'after_family_filter':len(clean),
            'family_rejections':dict(rejections),'protected_files':protected.files,
            'scan_scope':f'First {shards} AceCode shards, stop at 3000 eligible; Magicoder source order, stop at 1000 eligible; seed42 stable hash selection. Not whole-source random sampling.'}
    json_write(OUT/'source_scan.json',result)
    return clean,result


def cached_execute(jobs,cache_path):
    cache={r['sample_id']:r for r in rows(cache_path)} if cache_path.exists() else {}
    missing=[j for j in jobs if j['sample_id'] not in cache]
    # Two independent restricted containers; all cases of one problem stay together.
    batches=[missing[i:i+8] for i in range(0,len(missing),8)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for response in pool.map(run_jobs,batches):
            for result in response['results']:cache[result['sample_id']]=result
            write_jsonl(cache_path,cache.values())
            print(cache_path.stem,'completed',len(cache),flush=True)
    return cache


def function_dev(candidates):
    suitable=[]
    for r in candidates:
        if r['source_key']!='ace':continue
        distinct={repr((t['args'],t['kwargs'])) for t in r['test_cases']}
        if len(distinct)>=3:suitable.append(r)
    suitable=sorted(suitable,key=order)
    cache_path=OUT/'function_execution.jsonl';chosen=[];tested_bad=set()
    for offset in range(0,len(suitable),32):
        batch=suitable[offset:offset+32]
        cache=cached_execute([function_job(r) for r in batch],cache_path)
        for r in batch:
            result=cache[r['sample_id']]
            if result['passed'] and not any(result.get('constant_mutant_passes',[True])):
                chosen.append({**r,'execution':result})
            else:tested_bad.add(r['sample_id'])
        if len(chosen)>=100:break
    if len(chosen)<100:raise ValueError('Insufficient executed, nonconstant-test function dev candidates')
    return chosen[:100],tested_bad


def stdio_dev():
    candidates=[]
    source_records=list(rows('data/project2_code/dev_verification.jsonl'))
    extension=OUT/'extra_stdio_dev_candidates.jsonl'
    if extension.exists():source_records+=list(rows(extension))
    for r in source_records:
        if r['interface']!='stdin_stdout':continue
        tests=r['verification_info']['test_cases']
        if len(tests)<3 or not any(t['output'].strip() for t in tests):continue
        code,_=python_code(r['gold_standard_solution'])
        candidates.append({**r,'source_key':'openr1','reference_code':code,
                           'family_id':'legacy/'+r['source_family']})
    # A bounded-cost dev subset. Disclose this selection bias in the manifest.
    candidates.sort(key=lambda r:(len(r['verification_info']['test_cases']),order(r)))
    chosen=[]
    for offset in range(0,len(candidates),16):
        batch=candidates[offset:offset+16]
        jobs=[{'kind':'stdio','sample_id':r['sample_id'],'reference_code':r['reference_code'],
               'cases':r['verification_info']['test_cases']} for r in batch]
        cache=cached_execute(jobs,OUT/'stdio_execution.jsonl')
        for r in batch:
            result=cache[r['sample_id']]
            if result['passed']:chosen.append({**r,'execution':result})
        if len(chosen)>=100:break
    if len(chosen)<100:raise ValueError('Insufficient passing legacy stdin/stdout dev candidates')
    return chosen[:100]


def prompt_record(row,split):
    keys=['sample_id','source','source_key','source_revision','source_split','source_id','source_category',
          'problem','interface','entry_point','function_signature','family_id','problem_hash',
          'reference_ast_sha256','reference_body_sha256']
    result={k:row.get(k) for k in keys}
    result.update(split=split,domain='code',verification_status=row.get('verification_status','legacy_static_candidate'))
    result['prompt']=row.get('prompt') or row['problem']+'\nRequired Python interface (return the result; do not read stdin):\n'+row['function_signature']
    if split=='dev':result['execution_status']='all_source_tests_passed_in_restricted_container'
    else:result['execution_status']='OPD_prompt_only_not_a_correctness_label'
    return result


def run(shards=2,extend_stdio=False):
    pilot=check_gate();OUT.mkdir(exist_ok=True)
    identity={'version':'code-v2-20261023','seed':42,'ace_shards':shards,'builder_sha256':sha(__file__),
              'pilot_report_sha256':sha('data/code_v2_pilot/report.json'),
              'source_revisions':{key:metadata(key)['revision'] for key in ['ace','magicoder']},
              'source_helper_sha256':sha('src/data/code_v2_sources.py')}
    extension=OUT/'extra_stdio_dev_candidates.jsonl'
    if extension.exists():identity['stdio_extension_sha256']=sha(extension)
    if (OUT/'manifest.json').exists():
        old=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        if old['identity']==identity:print('v2 already sealed');return old
        raise ValueError('Use a new version for changed v2 configuration')
    if (OUT/'build_identity.json').exists():
        previous=json.loads((OUT/'build_identity.json').read_text(encoding='utf-8'))
        if previous!=identity:
            if not (extend_stdio and extension.exists() and
                    previous['builder_sha256']=='c769dff59a9171edcabf3ebed765a4927bd4085d0a80826835800f65c8adf815' and
                    all(identity.get(k)==v for k,v in previous.items() if k!='builder_sha256')):
                raise ValueError('Mismatched partial v2 build')
            json_write(OUT/'stdio_resume_migration.json',{'from':previous,'to':identity,
                'reason':'Explicit candidate-set extension only; same extractor, pilot, sources and sandbox. Existing per-ID execution evidence reused.',
                'preserved_cache_hashes':{p.name:sha(p) for p in OUT.glob('*.jsonl')}})
    json_write(OUT/'build_identity.json',identity)
    if (OUT/'source_scan.json').exists():
        candidates=list(rows(OUT/'function_candidates.jsonl'));scan=json.loads((OUT/'source_scan.json').read_text(encoding='utf-8'))
    else:candidates,scan=collect(shards)
    print('Clean standalone function pool:',len(candidates),flush=True)
    dev_functions,tested_bad=function_dev(candidates)
    print('Function dev sealed provisionally: 100',flush=True)
    dev_stdio=stdio_dev()
    print('stdio dev sealed provisionally: 100',flush=True)
    dev=dev_functions+dev_stdio;dev_ids={r['sample_id'] for r in dev}
    dev_index=ScreeningSimilarityIndex(.85)
    for r in dev:dev_index.add(r)
    functions=[r for r in candidates if r['sample_id'] not in dev_ids|tested_bad and not dev_index.match(r['problem'])]
    functions.sort(key=lambda r:(r['source_key']!='ace',order(r)))
    train_functions=functions[:1250]
    protected=Protected();stdio=[]
    for r in sorted(rows('data/project2_code/train_prompts.jsonl'),key=order):
        if r['interface']!='stdin_stdout' or dev_index.match(r['problem']) or protected.match(r):continue
        stdio.append({**r,'source_key':'openr1','family_id':'legacy/'+r['source_family']})
        if len(stdio)>=1250:break
    if len(train_functions)!=1250 or len(stdio)!=1250:raise ValueError('Cannot meet 1250/1250 target without relaxing rules')
    train=[prompt_record(r,'train') for r in train_functions+stdio]
    dev_meta=[prompt_record(r,'dev') for r in dev]
    dev_verify=[{**prompt_record(r,'dev'),'reference_code':r['reference_code'],
                 'test_cases':r.get('test_cases'),'verification_info':r.get('verification_info'),
                 'execution':r['execution']} for r in dev]
    for r in train:
        assert not any(k in r for k in ['reference_code','solution','test_cases','verification_info','gold_standard_solution'])
        assert not protected.match(r)
    for filename,digest in pilot['protected_files'].items():assert sha(filename)==digest
    assert not {r['family_id'] for r in train}&{r['family_id'] for r in dev_meta}
    write_jsonl(OUT/'train_prompts.jsonl',train);write_jsonl(OUT/'dev_metadata.jsonl',dev_meta)
    write_jsonl(OUT/'dev_verification.jsonl',dev_verify)
    write_jsonl(OUT/'agent_audit_queue_50.jsonl',sorted(dev_verify,key=order)[:50])
    manifest={'identity':identity,'source_scan':scan,'train_rows':len(train),'dev_rows':len(dev),
        'train_interfaces':dict(Counter(r['interface'] for r in train)),
        'dev_interfaces':dict(Counter(r['interface'] for r in dev_meta)),
        'train_sources':dict(Counter(r['source_key'] for r in train)),
        'dev_sources':dict(Counter(r['source_key'] for r in dev_meta)),
        'dev_cases':{'function':sum(len(r['test_cases']) for r in dev_functions),
                     'stdin_stdout':sum(len(r['verification_info']['test_cases']) for r in dev_stdio)},
        'quality_scope':'Pilot agent review passed. All 200 dev references pass every provided test in restricted container. Function dev rejects constant-None and constant-first-output mutants. This is consistency evidence, not an independent semantic oracle; full dev specification/coverage review PENDING.',
        'stdio_selection_bias':'Legacy dev plus explicitly re-sealed never-trained v1 prompts; >=3 cases, sorted by testcase count then seed42 hash; each problem capped at 20s, tokenized-whitespace stdout comparison.',
        'dedup_scope':'Official final tests and ALL legacy dev protected. Lexical Jaccard >=0.85, exact full AST, alpha-normalized target body; entry-name collisions excluded for review. Cross-source entry/body families never span splits. Not a proof of semantic or pretraining decontamination.',
        'sandbox':environment(),'paid_api_calls':0,'gpu_operations':0,
        'files':{p.name:{'path':p.as_posix(),'sha256':sha(p)} for p in OUT.glob('*.jsonl')}}
    if (OUT/'stdio_extension_provenance.json').exists():
        manifest['stdio_extension']=json.loads((OUT/'stdio_extension_provenance.json').read_text(encoding='utf-8'))
    old_dev_ids={r['sample_id'] for r in rows('data/project2_code/dev_verification.jsonl')}
    manifest['legacy_dev_overlap_rows']=sum(r['sample_id'] in old_dev_ids for r in dev)
    json_write(OUT/'manifest.json',manifest)
    print('v2 COMPLETE',json.dumps({k:manifest[k] for k in ['train_interfaces','dev_interfaces','train_sources','dev_cases']}),flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ace-shards',type=int,default=2);p.add_argument('--extend-stdio-dev',action='store_true');a=p.parse_args();run(a.ace_shards,a.extend_stdio_dev)
