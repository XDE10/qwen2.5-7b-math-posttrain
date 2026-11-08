"""Reproduce proposal-only selections from the sealed full audit; no SFT export."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.analyze_openr1_candidates import choose_for_budget, quantiles

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'reports/openr1_full_audit_final_20261023'
OUT = ROOT / 'reports/project1_sft_v2_plan_20261023'
QUOTAS = {'Algebra': 525, 'Geometry': 300, 'Number Theory': 300,
          'Combinatorics': 225, 'Inequalities': 100, 'Logic and Puzzles': 50}
SCENARIOS = {'compact': (1600, 2048), 'balanced': (2200, 2560), 'coverage': (3000, 3072)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def order(pair):
    return hashlib.sha256(('sft-v2-proposal-42/' + pair[0]['question_hash']).encode()).hexdigest()


def run():
    manifest = json.loads((AUDIT / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['scan_complete'] and not manifest['sft_data_written']
    for item in manifest['files'].values():
        assert digest(ROOT / item['path']) == item['sha256'], item['path']
    frozen = ROOT / 'data/project1_sft/sft_train.jsonl'
    frozen_hash = digest(frozen)
    stats = json.loads((AUDIT / 'statistics.json').read_text(encoding='utf-8'))
    assert frozen_hash == stats['frozen_compact_sha256_after']
    questions = [json.loads(line) for line in (AUDIT / 'questions.jsonl').open(encoding='utf-8')]
    OUT.mkdir(exist_ok=True)
    results = {}
    for name, (cap, rl_cap) in SCENARIOS.items():
        pool = []
        for q in questions:
            t = choose_for_budget(q, cap)
            if t:
                # Missing exclusion fields must fail closed, never mean "no match".
                assert 'evaluation_overlap' in q and 'internal_near_duplicate' in q
                if not q['evaluation_overlap'] and not q['internal_near_duplicate']:
                    pool.append((q, t))
        selected, strata = [], {}
        for kind, target in QUOTAS.items():
            candidates = sorted((x for x in pool if x[0]['problem_type'] == kind), key=order)
            disagreement = [x for x in candidates if 0 < x[0]['verification_ratio'] < 1]
            # A declared diversity policy, not a measured optimum or true difficulty.
            desired_proxy = round(target * .15)
            picked = disagreement[:min(desired_proxy, target)]
            used = {q['question_hash'] for q, _ in picked}
            remaining = sorted((x for x in candidates if x[0]['question_hash'] not in used), key=order)
            picked += remaining[:max(0, target-len(picked))]
            selected.extend(picked)
            strata[kind] = {'target': target, 'available': len(candidates), 'selected': len(picked),
                'shortfall': max(0, target-len(picked)), 'proxy_target': desired_proxy,
                'proxy_available': len(disagreement),
                'proxy_selected': sum(q['verification_ratio'] < 1 for q, _ in picked)}
        records = []
        for q, t in sorted(selected, key=order):
            assert t['quality']['automatic_pass'] and t['answer_tokens'] <= cap and t['total_tokens'] <= 3800
            records.append({'question_hash': q['question_hash'], 'problem_type': q['problem_type'],
                'source': q['source'], 'ratio_group': q['ratio_group'],
                'verification_ratio': q['verification_ratio'], 'unique_trajectory_count': q['unique_trajectory_count'],
                'linked_train_level': q.get('linked_train_level'), 'level_link_kind': q.get('level_link_kind'),
                'rl_train_overlap': q.get('rl_train_overlap'), 'trajectory': t,
                'semantic_review': 'pending', 'training_approved': False})
        assert len({r['question_hash'] for r in records}) == len(records)
        write(OUT / f'{name}_proposal.json', records)
        results[name] = {'answer_cap': cap, 'proposed_rl_cap': rl_cap, 'total_cap': 3800,
            'pool_count': len(pool), 'selected_count': len(records), 'strata': strata,
            'supervised_tokens': sum(t['supervised_tokens'] for _, t in selected),
            'answer_quantiles': quantiles([t['answer_tokens'] for _, t in selected]),
            'max_total_tokens': max(t['total_tokens'] for _, t in selected),
            'source_counts': dict(Counter(q['source'] for q, _ in selected)),
            'ratio_counts': dict(Counter(q['ratio_group'] for q, _ in selected)),
            'level_counts': dict(Counter(q.get('linked_train_level') or 'unknown' for q, _ in selected)),
            'rl_overlap_count': sum(bool(q.get('rl_train_overlap')) for q, _ in selected),
            'length_buckets': dict(Counter('500-1600' if t['answer_tokens'] <= 1600 else
                '1601-2200' if t['answer_tokens'] <= 2200 else '2201-3000' for _, t in selected)),
            'status': 'proposal_only_not_training_data'}
    summary = {'version': 'sft-v2-plan-20261023', 'source_revision': manifest['identity']['source_revision'],
        'statistics_sha256': digest(AUDIT/'statistics.json'), 'script_sha256': digest(Path(__file__)),
        'frozen_compact_sha256': frozen_hash, 'target': 1500, 'quotas': QUOTAS,
        'selection_policy': 'hashed question order; per-type up-to-15-percent source-disagreement preference; no length mix quota',
        'difficulty_status': 'unverified; source verification ratio is not a Level label',
        'scenarios': results, 'sft_data_written': False, 'gpu_operations': 0, 'paid_api_calls': 0}
    write(OUT/'summary.json', summary)
    lines = ['# OpenR1 全量统计与 SFT v2 方案试选证据', '',
        '日期：2026-10-23。以下为冻结缓存的只读再分析和题目/轨迹索引试选，不含可训练解答导出。', '',
        '## 统计口径', '',
        f"源 revision：`{summary['source_revision']}`；default/train：93,733 行、93,536 唯一题、193,767 条原始轨迹。",
        '完整正确指源逐索引验证标记、完整性与文本结构通过，不代表本轮重跑了 Math-Verify 或逐步证明正确。',
        '通过比例的分子是唯一轨迹中 math-verify=true 的条数，完整性不进入分子；选轨迹时另要求完整。',
        '分位数按每题最短合格轨迹计算，线性插值；全上下文含模板与结束标记。',
        '质量通过表示无当前启发式拒绝或待审信号。分位数在长度截取/评测排除之前；下面保留量在排除之后。', '',
        '## 各长度上限的候选保留情况', '',
        '回答下限 500、总上下文上限 3800；按既有评测重叠和池内近重复规则排除。监督 token 含 assistant 结束标记，不含 prompt。']
    for cap, groups in stats['retention_after_eval_and_internal_dedup'].items():
        lines += ['', f'### 回答上限 {cap}', '',
            f"总候选 {sum(v['retained'] for v in groups.values()):,}；监督 token {sum(v['supervised_tokens_sum'] for v in groups.values()):,}。"]
        for kind, v in groups.items():
            lines.append(f"- {kind}：{v['retained']:,}；占该题型全量 {v['retained_over_all_questions']:.2%}；占有质量合格完整正确轨迹的题 {v['retained_over_quality_eligible']:.2%}；监督 token {v['supervised_tokens_sum']:,}。")
    lines += ['', '## 两套分位数', '', '每项依次为回答 P50/P75/P90；完整上下文 P50/P75/P90，单位均为 token。']
    for key, label in [('source_complete_correct','源完整正确'), ('quality_pass','额外无质量告警')]:
        for dim, groups in stats[key].items():
            lines += ['', f'### {label} / {dim}', '']
            for group, v in groups.items():
                a, b = v['answer_tokens'], v['total_tokens']
                lines.append(f"- {group}（n={a['n']}）：{a['p50']} / {a['p75']} / {a['p90']}；{b['p50']} / {b['p75']} / {b['p90']}。")
    lines += ['', '## 三档配额试选', '', '配额是预先声明的设计选择，不是最优比例；不从测试集题型频数反推配额。']
    for name, v in results.items():
        lines += ['', f'### {name}', '',
            f"回答≤{v['answer_cap']}，拟议 RL={v['proposed_rl_cap']}；候选 {v['pool_count']}，试选 {v['selected_count']}，监督 token {v['supervised_tokens']:,}，上下文最大 {v['max_total_tokens']}。",
            f"实际长度分桶：{json.dumps(v['length_buckets'],ensure_ascii=False)}。",
            f"源教师通过率分桶：{json.dumps(v['ratio_counts'],ensure_ascii=False)}；可信 train Level：{json.dumps(v['level_counts'],ensure_ascii=False)}。"]
        for kind, d in v['strata'].items():
            lines.append(f"- {kind}：目标 {d['target']}，池 {d['available']}，试选 {d['selected']}，缺口 {d['shortfall']}；非全通过代理目标 {d['proxy_target']}，池 {d['proxy_available']}，实际 {d['proxy_selected']}。")
    lines += ['', '## 可复现与边界', '',
        '运行：` .\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.plan_project1_sft_v2`。',
        '脚本校验完整统计清单哈希与封存数据身份；缺失排除字段直接失败。输出仅写入本报告目录。',
        '池内去重沿用 3200 上限候选池的固定代表；低上限可能损失本可换代表保留的近重复题，因此是保守可行性估计。',
        '启发式质量过滤的误杀、题型标签与 MATH 分支映射、过程正确性和难度仍需审核。不得据此宣称中高难度占比或模型成绩。']
    (OUT/'统计与试选证据.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    assert digest(frozen) == frozen_hash
    write(OUT/'manifest.json', {'status':'proposal_only', 'input_manifest_sha256':digest(AUDIT/'manifest.json'),
        'frozen_unchanged':True, 'files':{p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}})
    print(json.dumps({k:{f:v[f] for f in ('selected_count','supervised_tokens','level_counts','length_buckets')} for k,v in results.items()},ensure_ascii=False))


if __name__ == '__main__':
    run()
