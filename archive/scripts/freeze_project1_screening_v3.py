"""Conservative candidate exclusion and SFT parameter preregistration only.

Preserves original JSONL bytes/approvals, never registers new training files.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from src.data.decontamination_v2 import Index

SOURCE = Path('data/portable_release/project1_rl_priority_not_rollout_selected.jsonl')
REPORT = Path('reports/handover_20261024_v2/decontamination_20261024.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def screen_lines(lines, hits):
    heldout, sft = set(), set()
    reasons = defaultdict(list)
    for hit in hits:
        if hit['source'] == 'rl' and hit['target'] in ('dev', 'math500', 'gsm8k'):
            sid = hit['source_id']; heldout.add(sid)
        elif hit['source'] == 'sft' and hit['target'] == 'rl':
            sid = hit['target_id']; sft.add(sid)
        else:
            continue
        reasons[sid].append(hit)
    ids = [json.loads(line)['sample_id'] for line in lines if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate candidate IDs')
    if (heldout | sft) - set(ids):
        raise ValueError('Audit refers to IDs absent from candidate source')
    kept, removed = [], []
    for line in lines:
        if line.strip():
            (removed if json.loads(line)['sample_id'] in heldout | sft else kept).append(line)
    return kept, removed, heldout, sft, dict(reasons)


def freeze_sft(out):
    folder = out/'equal_exposure_frozen_v1'
    folder.mkdir()
    hashes = {}
    for arm in ('raw','compact'):
        source = Path(f'reports/handover_20261024_v2/formal_v1/equal_exposure/E1_{arm}.draft.json')
        c = json.loads(source.read_text(encoding='utf-8'))
        c.pop('matrix_entry')
        c.update(status='研究参数已冻结；执行未放行', parameter_status='frozen', execution_status='未验证：等待GPU验收与环境锁',
                 draft=True, preregistration_schema='project1-sft-equal-exposure-frozen-v1',
                 source_draft_sha256=sha(source),
                 blocking_reason='draft=true is an execution interlock; equal-token stopping is NOT a prerequisite')
        c['budget_protocol'].update(draft=False,status='等题曝光研究协议已冻结；真实7B未运行')
        path = folder/f'E1_{arm}.parameters_frozen.json'
        path.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
        hashes[path.name] = sha(path)
    identities = Path('reports/handover_20261024_v2/formal_v1/equal_exposure/identities.draft.json')
    identity = json.loads(identities.read_text(encoding='utf-8'))
    identity.update(draft=False,status='输入身份已封存；未新增训练批准')
    (folder/'input_identities.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    return hashes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',default='reports/handover_20261024_v3')
    a=p.parse_args()
    report=json.loads(REPORT.read_text(encoding='utf-8'))
    if sha(SOURCE)!=report['inputs']['rl']['sha256']:
        raise ValueError('Candidate audit is stale')
    for entry in report['inputs'].values():
        if sha(entry['path'])!=entry['sha256']:
            raise ValueError('Audit comparison input changed')
    lines=SOURCE.read_bytes().splitlines(keepends=True)
    kept,removed,heldout,sft,reasons=screen_lines(lines,report['hits'])
    rows=[json.loads(line) for line in kept]
    if any(r['training_approved'] is not False or r['requires_7b_rollout_selection'] is not True or r['level'] not in ('Level 3','Level 4','Level 5') for r in rows):
        raise ValueError('Unexpected candidate approval/selection/level state')
    # Re-run the same fixed lexical rule in both directions; no semantic claim.
    threshold=report['calibration']['threshold']
    recheck={}
    for target in ('dev','math500','gsm8k'):
        index=Index(read_rows(report['inputs'][target]['path']))
        recheck['rl->'+target]=sum(len(index.query(r['problem'],threshold)) for r in rows)
    index=Index(rows)
    recheck['sft->rl']=sum(len(index.query(r['problem'],threshold)) for r in read_rows(report['inputs']['sft']['path']))
    if any(recheck.values()):
        raise ValueError('Filtered snapshot still hits the frozen lexical threshold')
    out=Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    pool=out/'rl_candidates_4630_lexically_screened_v1.jsonl'
    excluded=out/'rl_candidates_excluded_v1.jsonl'
    pool.write_bytes(b''.join(kept));excluded.write_bytes(b''.join(removed))
    (out/'exclusion_reasons.json').write_text(json.dumps(reasons,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'scope':'candidate snapshot only; no training approval', 'decision':'conservative exclusion without claiming every hit is a true duplicate',
            'input':{'path':str(SOURCE),'sha256':sha(SOURCE),'rows':len(kept)+len(removed)},
            'audit':{'path':str(REPORT),'sha256':sha(REPORT),'char5_jaccard_threshold':threshold},
            'heldout_unique_ids':len(heldout),'sft_overlap_unique_rl_ids':len(sft),'intersection_ids':sorted(heldout&sft),
            'excluded_unique_ids':sorted(heldout|sft),'excluded_count':len(removed),'remaining_count':len(rows),
            'outputs':{pool.name:sha(pool),excluded.name:sha(excluded)},'remaining_level_counts':dict(Counter(r['level'] for r in rows)),
            'remaining_type_counts':dict(Counter(r['problem_type'] for r in rows)),
            'row_content':'kept/excluded lines copied byte-for-byte in original relative order; approval fields unchanged',
            'lexical_recheck':recheck,'semantic_embedding_check':'未验证',
            'rollout_selection':'未运行；8次采样通过率25%~75%后最多3000，数量无保证',
            'training_approved':False,'publication_manifest_changed':False,
            'sft_parameters':freeze_sft(out),'gpu_acceptance':'未验证','formal_rl_configs':'not frozen'}
    (out/'freeze_record.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'excluded':len(removed),'remaining':len(rows),'lexical_recheck':recheck},ensure_ascii=False))


if __name__=='__main__':main()
