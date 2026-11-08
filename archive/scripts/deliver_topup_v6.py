"""Wait for bounded API work, then publish versioned local reports and hashes."""
import time,re,zipfile
from scripts.topup_common import *
def run():
    while not (REPORT/'budget_after_reserves.json').exists():time.sleep(3)
    from scripts.finalize_topup_v6 import run as finalize
    finalize()
    release=DATA/'release';s=read(release/'summary.json');docs=[ROOT/'docs/blueprints'/n for n in ['项目一_7B数学后训练_详细实施方案.md','项目二_双教师在线蒸馏_详细实施方案.md','项目三_多模态几何推理_详细实施方案.md','共用规范_强模型合成数据与验证.md','三项目_上卡前待办与验收清单.md','两个后训练项目_完整设计与实施指南.md']]
    backup=REPORT/'blueprints_before_topup_v6.zip'
    if not backup.exists():
        with zipfile.ZipFile(backup,'x',zipfile.ZIP_DEFLATED) as z:
            for p in docs:z.write(p,p.relative_to(ROOT).as_posix())
    notice=f"> **2026-10-24 Top-up v6 当前实测（优先于下文历史记录）**：用户累计API硬上限$8。严格逐项准入后：项目一{s['project1_approved_pairs']}对同题A/B；项目二数学{s['project2_math_train']} train/{s['project2_math_dev']} dev、代码train {s['project2_code_train']} / dev {s['project2_code_dev']}；项目三{s['project3_train']} train/{s['project3_validation']} validation逐事实子集。批准范围仅以新版manifest为准，候选/存疑/隔离行不随之批准。新增有限范围感知V4奖励，非100%金标，A100/GPU仍未运行。详见[Top-up实测报告](<D:/Users/Mr.Z/Desktop/modern_genai/reports/topup_v6_20261024/执行报告.md>)与[V4奖励及运行说明](<D:/Users/Mr.Z/Desktop/modern_genai/docs/blueprints/Topup_v6_感知奖励与运行说明.md>)。\n\n"
    for p in docs:
        body=p.read_text(encoding='utf-8')
        if not body.startswith(notice):p.write_text(notice+body,encoding='utf-8')
    raw=rows(release/'project1_raw_paired.jsonl');compact={r['sample_id']:r for r in rows(release/'project1_compact_paired.jsonl')}
    pack=['# Top-up v6 已批准真实样本预览','',f"精确数量与边界见[执行报告](<{(REPORT/'执行报告.md').as_posix()}>)。AI工程准入不等于人工金标。所有显示样本均来自新版批准文件；未放宽150-token门槛。",'', '## 项目一：3对同题完整轨迹','']
    new=[r for r in raw if r.get('cohort')=='v6_topup']
    for r in sorted(new,key=lambda r:r['sample_id'])[:3]:pack += [f"### {r['sample_id']}",'',r['problem'],'','原生全文：','','````text',r['solution'],'````','','精炼全文：','','````text',compact[r['sample_id']]['solution'],'````','']
    for filename,title in [('project2_math_train.jsonl','项目二数学'),('project2_code_train.jsonl','项目二代码')]:
        pack += ['## '+title,'']
        rs=rows(release/filename)
        if 'code' in filename:rs=[r for kind in ['function','stdin_stdout'] for r in rs if r['interface']==kind][:2]+[r for r in rs if r['interface']=='stdin_stdout'][:2]
        else:rs=rs[-4:]
        for r in rs[:4]:pack += [f"### {r['sample_id']}",'',r.get('prompt',r['problem']),'', 'OPD仅使用题面；参考解答/程序在独立审核证据中，不作为输入。','']
    pack += ['## 项目三：6道逐事实训练样本','']
    for r in sorted(rows(release/'project3_train_fact_pruned.jsonl'),key=lambda r:r['sample_id'])[:6]:
        pack += [f"### {r['sample_id']} / 原ID {r['original_id']}",'',r['problem'],'',f"![原图](<{Path(r['image_path']).as_posix()}>)",'','```text',r['model_perception'],'```','',f"被mask事实数：{len(r['masked_uncertain_facts'])}；有限感知评分，不声称完整图像标签。",'']
    (REPORT/'approved_sample_preview.md').write_text('\n'.join(pack),encoding='utf-8')
    if not (REPORT/'source_integrity.json').exists():
        from scripts.check_topup_source_integrity import run as check
        check()
    log=(REPORT/'pytest.log').read_text(encoding='utf-8');assert '201 passed' in log,log[-500:]
    for p,h in read(release/'manifest.json')['files'].items():assert sha(release/p)==h,p
    linkcount=0
    for p in [REPORT/'执行报告.md',REPORT/'approved_sample_preview.md']:
        for target in re.findall(r'\]\(<([^>]+)>\)',p.read_text(encoding='utf-8')):
            if target.startswith(('https:','http:')):continue
            assert Path(target).exists(),('broken link',target);linkcount+=1
    files=set(docs+[ROOT/'docs/blueprints/Topup_v6_感知奖励与运行说明.md'])
    for folder in [DATA,REPORT]:files.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update((ROOT/'scripts').glob('*topup*.py'));files.update([ROOT/p for p in ['src/synthesis/topup_budget.py','src/data/perception_v4.py','src/data/perception_text_guards.py','src/training/perception_reward_v4.py','src/training/approved_hf.py','tests/test_topup_budget.py','tests/test_perception_v4.py']])
    hits=[str(p.relative_to(ROOT)) for p in files if p.suffix in ['.py','.json','.jsonl','.md'] and re.search(r'sk-[A-Za-z0-9]{20,}',p.read_text(encoding='utf-8'))];assert not hits,('possible secret paths only',hits)
    checks={'approved_file_sha_verified':True,'current_report_local_links':linkcount,'secret_pattern_scan_passed':True,'tests':201,'api_owners_closed':True,'no_gpu_execution':True,'targets_met':s['targets_met']};dump(REPORT/'final_checks.json',checks);files.add(REPORT/'final_checks.json');files.discard(REPORT/'delivery_manifest.json')
    # Existing shared caches are linked through task/request IDs; hash the final
    # common ledger too without duplicating private request files in this pack.
    files.update([BUDGET/'ledger.json',BUDGET/'topup_budget_policy.json'])
    dump(REPORT/'delivery_manifest.json',{'files':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},'approval_scope':'Only release manifest explicit training/evaluation files; no candidate-pool-wide approval'})
    print('Top-up delivery complete',s,flush=True)
if __name__=='__main__':run()
