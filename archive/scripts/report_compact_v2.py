"""Rebuild a human-readable audit without paid calls or changes to frozen runs."""
import json
from pathlib import Path
from collections import Counter
from src.synthesis.deepseek_pilot import dump
from scripts.compact_distillation_v2 import ROOT, REPORT, DATA, read, file_sha, frozen_identity


def run():
    s=read(REPORT/'summary.json');f=read(REPORT/'format_robustness_review/summary.json')
    old=f['scenarios']['old_cache'];new=f['scenarios']['new_pilot'];policy=read(DATA/'policy.json')
    originals=read(REPORT/'cache_reaudit/summary.json')
    ledger=read(REPORT/'api_cache/ledger.json')
    statuses=dict(Counter(r['state'] for r in ledger))
    amounts={}
    def balances(path):
        return {r['currency']:float(r['total_balance']) for r in read(path).get('balance_infos',[])}
    before=balances(REPORT/'balance_before.json');after=balances(REPORT/'balance_after.json')
    for c,v in before.items():
        if c in after:amounts[c]=round(v-after[c],8)
    requests=read(DATA/'blind_requests.json');refs=read(DATA/'verifier_references.json')
    review_path=REPORT/'ai_process_spot_review.json'
    ai=read(review_path) if review_path.exists() else None
    validation=read(REPORT/'final_checks.json') if (REPORT/'final_checks.json').exists() else None
    if ai:
        decisions={(r['cohort'],r['sample_id']):r for r in ai['candidate_reviews']}
        review_out=REPORT/'reviewed_candidate_pairs';review_out.mkdir(exist_ok=True)
        for cohort in ('old_cache','new_pilot'):
            pairs=read(REPORT/'format_robustness_review'/(cohort+'_pairs.json'))
            checked=[];isolated=[];pending=[]
            for pair in pairs:
                decision=decisions.get((cohort,pair['sample_id']))
                updated={**pair,'ai_process_review':decision or {'decision':'pending'},
                         'human_review':'pending','training_approved':False}
                if not decision:pending.append(updated)
                elif decision['decision']=='pass':checked.append(updated)
                else:isolated.append(updated)
            dump(review_out/(cohort+'_ai_pass_pending_human.json'),checked)
            dump(review_out/(cohort+'_isolate_or_uncertain.json'),isolated)
            dump(review_out/(cohort+'_not_process_reviewed.json'),pending)
    dump(REPORT/'difficulty_negative_result_archive.json',{'status':'archived_negative_result',
        'source':'reports/deepseek_pilot_20261023/difficulty_gate.json',
        'source_sha256':file_sha(ROOT/'reports/deepseek_pilot_20261023/difficulty_gate.json'),
        'exact_agreement':.36,'within_one':.68,'bulk_labeling_enabled':False,
        'followup_difficulty_calls':0,'original_files_immutable':True})
    lines=['# 项目一精炼解答模块：缓存重审与独立 Pilot v2', '',
        '日期：2026-10-23。真实 API 已执行；未启动 GPU，未扩展到 1200/1500 题，全部候选未批准训练。', '',
        '## 1. 范围与实验身份', '',
        '比较固定 OpenR1 快照的原生轨迹与本次 API 盲生成精炼解答。不能把差异单独归因为教师年代、去冗或长度；教师、解法、风格和监督 token 都可能改变。',
        '主力仅 deepseek-flash，thinking.type=disabled、temperature=0.4、max_tokens=2400；Qwen 实测回答窗口150～1500，总上下文≤3800。',
        '每题至多3次盲生成，首个符合冻结自动门槛的候选停止；不发送标准答案、原解答、等级或验证反馈。不调用难度打标或过程裁判；不自动Pro补采。',
        '费用预留上限2 USD、请求上限135、4个工作线程。教师API token上限与Qwen计数是不同单位。',
        '请求和返回模型均记录为deepseek-flash；本次响应未提供可验证的不可变底层版本号，因此不能声称固定了某个2026检查点或“最新一代”因果比较。', '',
        '## 2. 零调用缓存重审与实现修订', '',
        f"50题144条缓存均重新执行本地Math-Verify（去重后合并重复输入）；与历史答案验证标记分歧{originals['math_verify_disagreements']}。原生轨迹也在本地重验。",
        '初版新解析器过于严格：将最终boxed的常规LaTeX显示定界符和闭合think之后的已有推导正文判为额外内容。冻结结果已完整保留，旧缓存0对，新pilot1对；不能据这个实现缺陷声称所有旧推导无效。',
        '随后新增独立 format_robustness_review，不改动冻结数据/代码/policy。允许移除成对的最终公式显示定界符，或将已有完整书面解答统一包装；保持全部boxed数学文本逐字一致，不补解答、不删除冲突结论。嵌套/缺失标签、boxed之后的解释/冲突正文仍待审。',
        f"此事后格式复审得到：旧缓存严格已有标签{old['strict_existing_wrapper_common_pairs']}对、规范化后{old['normalized_common_pairs']}对；新pilot严格已有标签{new['strict_existing_wrapper_common_pairs']}对、规范化后{new['normalized_common_pairs']}对。均是答案与结构自动候选，非过程正确认证。",
        '旧缓存用于规则开发，不能当新独立验证；新pilot的格式兼容统计也是观察结果后完成的事后分析，不能冒充冻结验收通过。', '',
        '## 3. 独立新题 Pilot 的原始结果', '',
        '固定抽取40道此前未用于上轮规则调试的coverage训练候选。与旧50题、135道锚点/开发/验收题及数学dev/test用规范化文本/Jaccard≥0.85排除；六类配额12/8/8/6/4/2。属于有界分层样本，不是全源无偏估计。',
        f"请求{s['generation_attempts']}次；首次答案通过{s['first_answer_verified']}/40（{s['first_answer_verified']/40:.1%}），至少一次答案通过{s['any_answer_verified']}/40（{s['any_answer_verified']/40:.1%}）。",
        f"严格冻结解析器的一次结构通过{s['first_strict_structure_pass']}/40，首次自动准入{s['first_strict_admitted']}/40，最终共同候选{s['normalized_common_pairs']}/40。engineering_gate_passed={s['engineering_gate_passed']}。",
        f"事后修订格式解析后共同自动候选{new['normalized_common_pairs']}/40（{new['normalized_common_pairs']/40:.1%}）；不能用该比例替代过程合格率，也没有达到可直接量产的结论。",
        f"本次运行墙钟{s['wall_seconds']:.2f}秒。响应usage估算峰/谷价{s['usage_estimated_peak_usd']:.6f}/{s['usage_estimated_offpeak_usd']:.6f} USD；累计预算预留{s['reserved_usd']:.6f} USD（预留不是账单）。",
        f"账户余额观测差额{json.dumps(amounts,ensure_ascii=False)}；与usage估算口径不同，可能有延迟或账号其他活动，不能把0.01元等观测差额当此次最终账单。请求状态{statuses}；不存在不确定reserved请求时才可确认完成。", '',
        '## 4. 源题清洗与过程审核', '',
        '清洗器只删除白名单中明确的独立翻译后缀，保留原始题面、编辑清单和前后哈希；含歧义/角色改变元指令的题隔离。不会用包含translate的宽泛正则删除数学平移条件。旧缓存已生成的解答仍对应旧题面，不将清洗追溯成新生成。',
        f"新40题实际清洗改动{sum(bool(r['source_cleaning']['edits']) for r in refs)}题；不能据没有自然污染样本的pilot宣称清洗收益已得到量化验证。已对上一轮真实翻译后缀做离线回归检查。",
        '确定性规则只做结构、终止、长度、噪声和答案必要条件；不以等号数量证明推理正确。人工审核没有完成。',
        (f"独立Codex AI逐题核查全部新19对：{ai['summary']['new_pilot']}；旧27对中分层抽12对：{ai['summary']['old_cache']}，其余15对未做过程审核。另审4个被拒案例，发现3个数学内容正确但格式可修复的假阴性。结果和局部符号计算见ai_process_spot_review.json/md，全部仍未人工审核或训练批准。" if ai else 'AI过程抽查仍在进行；未填写人工已通过。'),
        '已将AI通过待人工、隔离/不确定、未过程审核三类分别落盘在reviewed_candidate_pairs/。发现答案正确但函数方程推导错误，以及缺图仍假定射线顺序的真实反例，说明不能删除过程质量门槛。多数通过样本仍重复草稿和正式推导，不宜直接称高信噪比精炼成品。', '',
        '## 5. 已完成的CPU收尾与剩余门槛', '',
        '原生2200/3000两档各1500条已导出到data/project1_sft_v2/，保持各自配额和原文；训练重叠分别1/5题已枚举。仍training_approved=false，不把源验证标记当本地过程重验。',
        '项目三真实关联及MathVista审计见../project3_cpu_20261023/审计报告.md；训练入口审计见../training_readiness_20261023_v2.md。',
        '下一步先修订并冻结更稳健的序列化协议、补源题意/缺图隔离，依据AI与人工过程审核处置候选，再用未参与本次修订的新训练题开展下一轮独立验收；本轮不再自动追加付费实验。',
        '在数据与入口合格后，上A100执行完整更新、显存、生成截断及恢复验收；不以短SFT示范保证RL2048不截断。', '',
        '## 6. 复现与文件', '',
        '```powershell',
        '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.compact_distillation_v2 reaudit',
        '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.compact_distillation_v2 prepare',
        '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.compact_distillation_v2 run',
        '# 已执行付费命令：run --live --key-prompt；密钥仅经隐藏输入进入内存，不写命令或文件。',
        '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.review_compact_format',
        '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.report_compact_v2',
        '```', '',
        '首次付费响应已缓存；再次live复用相同缓存，不允许改变已冻结输入或代码身份。若代码/准入/样本改变，另立版本。不要为重建报告重新生成。',
        '旧pilot目录和原1500紧凑SFT均保持哈希；新data目录保存盲请求与本地验证参考分离、冻结policy和manifest。format_robustness_review中的pairs含参考数据，仅供审核，不能送入生成器。',
        '所有归档、原始失败、规范化操作和事后统计均保留，不把不同口径的候选数混为训练集条数。']
    if validation:
        lines += ['', '## 7. 最终检查', '',
            f"完整tests目录：{validation['pytest']}；{validation['checked_hashes']}项封存/来源/产物SHA复核无差异，更新文档本地链接与代码围栏检查通过。",
            '初次完整测试发现既有CPU venv缺Pillow，已仅新增Pillow==11.2.1后重跑通过；项目三另附独立requirements-project3-cpu.txt及实际Python3.11审计环境。未升级其他既有包，不把CPU环境作为CUDA锁。']
    (REPORT/'执行报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(REPORT/'final_integrity_manifest.json',{'protected_identity':frozen_identity(),
        'files':{str(p.relative_to(REPORT)):file_sha(p) for p in REPORT.rglob('*') if p.is_file() and p.name!='final_integrity_manifest.json'},
        'training_approved':False,'gpu_operations':0,'difficulty_calls':0,'llm_judge_calls':0})
    print('Report regenerated without network calls')


if __name__=='__main__':run()
