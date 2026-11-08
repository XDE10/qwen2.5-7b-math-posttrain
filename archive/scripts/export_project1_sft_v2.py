"""Export sealed proposal indices as review-only native SFT candidates, CPU/local only."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.analyze_openr1_candidates import quantiles
from scripts.plan_project1_sft_v2 import QUOTAS
from src.data.prepare_math_data import fingerprint
from src.data.screen_openr1_sft import PROMPT, answer_is_complete, sealed_math
from src.data.openr1_quality import quality_signals
from src.data.screening_io import ScreeningSimilarityIndex, rows, sha

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path('reports/openr1_full_audit_final_20261023')
PLAN = Path('reports/project1_sft_v2_plan_20261023')
OUT = Path('data/project1_sft_v2')
REPORT = Path('reports/project1_sft_v2_export_20261023')
SCENARIOS = {'2200': 'balanced', '3000': 'coverage'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def put(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_existing(folder, identity):
    manifest = folder/'manifest.json'
    if manifest.exists():
        previous = load(manifest)
        check(previous['identity'] == identity, 'Refuse output identity change; choose a new version directory')
        for name, digest in previous['files'].items():
            check(sha(folder/name) == digest, 'Sealed output changed: '+name)
        return previous
    check(not folder.exists() or not any(folder.iterdir()), 'Refuse unsealed/partial nonempty output directory')
    return None


def replay_record(raw, selected, tokenizer):
    t = selected['trajectory']
    i = t['trajectory_index']
    check(raw['uuid'] == t['source_uuid'], 'Source UUID mismatch')
    check(fingerprint(raw['problem']) == selected['question_hash'], 'Problem hash mismatch')
    check(raw['is_reasoning_complete'][i] is True and raw['correctness_math_verify'][i] is True,
          'Source per-index verification flags changed')
    solution = raw['generations'][i]
    check(hashlib.sha256(solution.encode()).hexdigest() == t['text_sha256'], 'Trajectory text hash mismatch')
    check(answer_is_complete(solution), 'Incomplete solution structure')
    check(quality_signals(solution) == t['quality'] and t['quality']['automatic_pass'], 'Quality identity mismatch')
    finishes = raw.get('finish_reasons')
    finish = finishes[i] if finishes is not None else None
    check(finish in {None, 'stop', 'eos', 'eos_token', 'stop_sequence'}, 'Non-stop source finish')
    messages = [{'role':'system','content':PROMPT}, {'role':'user','content':raw['problem']}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages+[{'role':'assistant','content':solution}],
                                         tokenize=True, add_generation_prompt=False)
    check(full[:len(prefix)] == prefix, 'Assistant target boundary mismatch')
    counts = {'answer_tokens':len(tokenizer.encode(solution, add_special_tokens=False)),
              'total_tokens':len(full), 'prompt_tokens':len(prefix), 'supervised_tokens':len(full)-len(prefix)}
    check(all(counts[k] == t[k] for k in counts), 'Proposal tokenizer counts changed')
    return {'sample_id':'openr1-v2/'+selected['question_hash'], 'problem':raw['problem'],
        'solution':solution, 'reference_answer':raw['answer'], 'problem_hash':selected['question_hash'],
        'problem_raw_sha256':hashlib.sha256(raw['problem'].encode()).hexdigest(),
        'solution_sha256':t['text_sha256'], 'source':'open-r1/OpenR1-Math-220k',
        'source_category':selected['source'], 'source_revision':'e4e141ec9dea9f8326f4d347be56105859b2bd68',
        'source_split':'train', 'split':'train', 'domain':'math', 'family_id':selected['question_hash'],
        'problem_type':selected['problem_type'], 'source_path':t['source_path'], 'source_row':t['source_row'],
        'source_uuid':t['source_uuid'], 'trajectory_index':i, 'source_math_verify':True,
        'source_reasoning_complete':True, 'source_finish_reason':finish,
        'local_math_verify_rechecked':False, 'semantic_review':'pending', 'training_approved':False,
        'quality_status':'source_verified_pending_semantic_review',
        'answer_token_count':counts['answer_tokens'], 'response_token_count':counts['answer_tokens'],
        'total_token_count':counts['total_tokens'], 'supervised_token_count':counts['supervised_tokens'],
        'prompt_token_count':counts['prompt_tokens'], 'system_prompt':PROMPT,
        'linked_train_level':selected['linked_train_level'], 'level_link_kind':selected['level_link_kind'],
        'source_verification_ratio':selected['verification_ratio'], 'source_ratio_group':selected['ratio_group'],
        'rl_train_overlap':selected['rl_train_overlap'], 'automatic_quality':t['quality']}


def run(root=ROOT):
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    audit = load(root/AUDIT/'manifest.json')
    original = load(root/AUDIT/'identity.json')
    check(audit['scan_complete'], 'Full source audit is incomplete')
    check(audit['identity'] == original, 'Audit identity mismatch')
    for meta in audit['files'].values():
        check(sha(root/meta['path']) == meta['sha256'], 'Audit file changed: '+meta['path'])
    plan_manifest = load(root/PLAN/'manifest.json')
    for name, digest in plan_manifest['files'].items():
        check(sha(root/PLAN/name) == digest, 'Proposal file changed: '+name)
    check(plan_manifest['input_manifest_sha256'] == sha(root/AUDIT/'manifest.json'), 'Proposal audit identity changed')
    math_manifest, protected = sealed_math(root)
    check({k:v['sha256'] for k,v in math_manifest['files'].items()} == original['protected_files'],
          'Protected split identity changed')
    # Check every protected file, including source-test and quarantine, not only the exclusion pool.
    for meta in math_manifest['files'].values():
        check(sha(root/meta['path']) == meta['sha256'], 'Protected file changed: '+meta['path'])
    frozen = root/'data/project1_sft/sft_train.jsonl'
    check(sha(frozen) == original['compact_sft_sha256'], 'Original compact SFT changed')
    token_dir = root/'data/tokenizers/Qwen2.5-7B-Instruct'
    for name, digest in original['tokenizer_files'].items():
        check(sha(token_dir/name) == digest, 'Tokenizer identity changed')
    check(original['template_system'] == PROMPT, 'System prompt changed')
    identity = {'version':'native-review-export-v1', 'builder_sha256':sha(Path(__file__)),
        'audit_manifest_sha256':sha(root/AUDIT/'manifest.json'),
        'proposal_manifest_sha256':sha(root/PLAN/'manifest.json'), 'source_revision':original['source_revision'],
        'tokenizer_files':original['tokenizer_files'], 'system_prompt':PROMPT,
        'protected_files':original['protected_files'], 'compact_sft_sha256':sha(frozen),
        'source_files':original['source_files'], 'answer_min':500, 'answer_caps':[2200,3000], 'total_cap':3800,
        'exclusion':'frozen full-audit identities plus replayed normalized/3gram Jaccard >=0.85'}
    previous = verify_existing(root/OUT, identity)
    if previous:
        print('Sealed candidate export already complete; identities and files verified', flush=True)
        return previous
    proposals = {cap:load(root/PLAN/(name+'_proposal.json')) for cap,name in SCENARIOS.items()}
    selected = {r['question_hash']:r for group in proposals.values() for r in group}
    evidence = {}
    for q in rows(root/AUDIT/'questions.jsonl'):
        if q['question_hash'] in selected:
            check('evaluation_overlap' in q and 'internal_near_duplicate' in q, 'Missing exclusion evidence')
            check(not q['evaluation_overlap'] and not q['internal_near_duplicate'], 'Excluded question in proposal')
            for group in proposals.values():
                for r in group:
                    if r['question_hash'] == q['question_hash']:
                        check(r['trajectory'] in q['trajectories'], 'Proposal trajectory absent in frozen audit')
                        check(r['rl_train_overlap'] == q['rl_train_overlap'], 'Train overlap identity changed')
            evidence[q['question_hash']] = q
    check(len(evidence) == len(selected), 'Missing question audit records')
    tokenizer = AutoTokenizer.from_pretrained(token_dir, local_files_only=True)
    tokenizer.model_max_length = 10**9
    wanted = defaultdict(dict)
    for r in selected.values():
        t = r['trajectory']; wanted[t['source_path']][t['source_row']] = r
    replayed = {}
    for meta in original['source_files']:
        path = meta['path']
        check(sha(root/path) == meta['sha256'], 'Source shard changed: '+path)
        offset = 0
        for batch in pq.ParquetFile(root/path).iter_batches(batch_size=256):
            for j, raw in enumerate(batch.to_pylist()):
                if offset+j in wanted[path]:
                    r = wanted[path][offset+j]
                    replayed[r['question_hash']] = replay_record(raw, r, tokenizer)
            offset += batch.num_rows
        print('Replayed source shard', meta['name'], len(replayed), '/', len(selected), flush=True)
    check(len(replayed) == len(selected), 'Missing source indices')
    evaluation, train = ScreeningSimilarityIndex(.85), ScreeningSimilarityIndex(.85)
    for key in ('math_dev','math500_test','gsm8k_test'):
        for r in protected[key]: evaluation.add(r)
    for r in protected['math_train']: train.add(r)
    overlap_records = []
    for key, r in replayed.items():
        check(evaluation.match(r['problem']) is None, 'Replayed evaluation overlap')
        hit = train.match(r['problem'])
        check(hit == r['rl_train_overlap'], 'Replayed train overlap differs from audit')
        if hit: overlap_records.append({'question_hash':key, 'rl_train_overlap':hit})
    result = {}
    outputs = {}
    for cap, group in proposals.items():
        records = [replayed[r['question_hash']] for r in group]
        check(len({r['problem_hash'] for r in records}) == len(records), 'Duplicate question')
        check(len({r['solution_sha256'] for r in records}) == len(records), 'Duplicate solution')
        seen = ScreeningSimilarityIndex(.85)
        for r in records:
            check(seen.match(r['problem']) is None, 'Internal near duplicate')
            seen.add(r)
            check(500 <= r['answer_token_count'] <= int(cap) and r['total_token_count'] <= 3800, 'Length cap failed')
        counts = Counter(r['problem_type'] for r in records)
        result[cap] = {'count':len(records), 'problem_types':dict(counts),
            'quota_shortfalls':{k:max(0,v-counts[k]) for k,v in QUOTAS.items()},
            'answer_tokens':quantiles([r['answer_token_count'] for r in records]),
            'total_tokens':quantiles([r['total_token_count'] for r in records]),
            'max_total_tokens':max(r['total_token_count'] for r in records),
            'supervised_tokens':sum(r['supervised_token_count'] for r in records),
            'linked_train_levels':dict(Counter(r['linked_train_level'] or 'unknown' for r in records)),
            'rl_train_overlap_count':sum(bool(r['rl_train_overlap']) for r in records),
            'training_approved_count':0, 'semantic_reviewed_count':0, 'local_math_verify_rechecked_count':0}
        outputs['sft_train_'+cap+'.jsonl'] = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records)
    check(sha(frozen) == identity['compact_sft_sha256'], 'Compact SFT changed during export')
    for meta in math_manifest['files'].values():
        check(sha(root/meta['path']) == meta['sha256'], 'Protected file changed during export')
    (root/OUT).mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items(): (root/OUT/name).write_text(text, encoding='utf-8')
    put(root/OUT/'train_overlap.json', sorted(overlap_records,key=lambda x:x['question_hash']))
    readme = '''# OpenR1 SFT v2 原生基准候选导出

这两档是封存 balanced(2200)/coverage(3000) proposal 的原文回放，不是语义验收完成的训练集。
所有记录 `training_approved=false`、`semantic_review=pending`。未修改原 compact SFT 或任何 dev/test。
本次没有调用付费 API、GPU 或逐题本地 Math-Verify。`source_math_verify` 是源数据逐轨迹标签，不能解释为本次独立重验或过程正确性证明。

## 文件与字段

- `sft_train_2200.jsonl`、`sft_train_3000.jsonl`：原题 `problem`、原生轨迹 `solution`、验证端答案 `reference_answer`；回答 500 至对应上限，总模板不超过 3800。
- `problem_hash/family_id` 是既有规范化题面 SHA256；`problem_raw_sha256` 为逐字题面哈希；`solution_sha256` 为原轨迹 UTF-8 哈希。
- `source_path/source_row/trajectory_index/source_uuid/source_revision` 可回放 parquet 逐索引；`source_*` 标签均来自该索引。
- `answer_token_count/response_token_count` 不含模板；`prompt_token_count` 含生成前缀；`total_token_count` 为完整聊天模板；`supervised_token_count` 包含 assistant 结束标记。使用本地封存 Qwen2.5-7B-Instruct tokenizer 和原数学 system。
- `rl_train_overlap` 枚举与共享 MATH train 的精确或近重复关系；允许训练池重叠但必须披露，详情见 `train_overlap.json`。近重复不能迁移官方 Level。
- `linked_train_level` 仅继承可信精确 train 关联，缺失不伪造难度；源 `verification_ratio` 仅是教师通过率代理。
- `automatic_quality` 为启发式检查，不等于人工语义审核。原题保持原文，可能含待清洗后缀；后续同题 API 对照须另存清洗映射，不覆盖基线。
- `manifest.json` 封存来源、tokenizer、脚本、保护文件及产物哈希；再次运行只校验并复用，身份变化或未封存的部分产物拒绝覆盖。

## 使用边界

未完成题面/解法语义审核、可靠难度覆盖验收、本地答案独立重验和训练实测。候选导出不能直接升级为正式训练批准。评测排除沿用规范化文本和 3-gram Jaccard 0.85，不证明语义去重或预训练无污染。两档大量共享题，不能当作两组独立随机数据。

运行：` .\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.export_project1_sft_v2`。
'''
    (root/OUT/'README.md').write_text(readme,encoding='utf-8')
    manifest = {'identity':identity, 'status':'native_candidates_pending_semantic_review', 'training_approved':False,
        'scenarios':result, 'union_question_count':len(replayed), 'paid_api_calls':0, 'gpu_operations':0,
        'protected_files_unchanged':True, 'compact_unchanged':True,
        'files':{p.name:sha(p) for p in sorted((root/OUT).iterdir()) if p.is_file() and p.name!='manifest.json'}}
    put(root/OUT/'manifest.json',manifest)
    (root/REPORT).mkdir(parents=True, exist_ok=True)
    put(root/REPORT/'summary.json',{'scenarios':result, 'union_question_count':len(replayed),
        'output_manifest_sha256':sha(root/OUT/'manifest.json'), 'source_flags_replayed':True,
        'complete_template_retokenized':True,'heldout_overlap_count':0,'internal_near_duplicate_count':0,
        'local_math_verify_rechecked':False,'semantic_review':'pending','training_approved':False})
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return manifest


if __name__ == '__main__':
    run()
