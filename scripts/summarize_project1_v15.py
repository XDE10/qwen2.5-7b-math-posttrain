"""Read-only metric aggregation; no inference, scoring changes or approval."""
import argparse
import json
from pathlib import Path
import re


def training_metrics(text):
    lines=[line for line in text.splitlines() if "'train_runtime'" in line]
    if not lines:return None
    return {k:float(v) for k,v in re.findall(r"'(train_runtime|train_loss|train_samples_per_second|train_steps_per_second|epoch)':\s*'?([-+\d.eE]+)'?",lines[-1])}


def render_report(comparison,metrics,math500):
    lines=['# 项目一：E0、raw、compact 实测汇总','',
           '同一固定7B模型起点，SFT两臂1500道同题、各一epoch、seed42。greedy评测预算2048；难题每题4次采样。单seed结果，不宣称统计显著或消除过度思考。','',
           '## MATH-500：三个模型','']
    for name in ['E0','raw','compact']:
        row=math500.get(name)
        if row:
            lines.append(f"- {name}：验证通过 {row['correct']}/{row['count']}（{row['correct_over_all']:.3%}）；长度截断 {row['finish_reason_counts'].get('length',0)}/{row['count']}；平均回答 {row['mean_generated_tokens']:.2f} token；不可评分 {row['count']-row['scorable_count']}。")
        else:lines.append(f'- {name}：未执行或结果尚未落盘，不能填为0。')
    if all(math500.get(k) for k in ['E0','raw','compact']):
        delta=math500['compact']['correct_over_all']-math500['raw']['correct_over_all']
        lines+=['',f'compact 相对 raw 的 MATH-500 正确率差为 {delta*100:+.3f} 个百分点；这只是本次固定评分器快照，拒绝项单列。']
    lines+=['','## 已完成的 dev 256题','']
    for name,row in [('E0',comparison['E0_dev']),('raw',comparison['raw']['dev']),('compact',comparison['compact']['dev'])]:
        lines.append(f"- {name}：{row['correct']}/{row['count']}（{row['correct_over_all']:.3%}）；length {row['finish_reason_counts'].get('length',0)}/{row['count']}（{row['finish_reason_counts'].get('length',0)/row['count']:.3%}）；平均 {row['mean_generated_tokens']:.2f} token；不可评分 {row['count']-row['scorable_count']}。")
    lines+=['','compact在dev比raw多5题、比E0多2题，但dev截断7条高于raw的6条；不能写成“compact几乎不截断”或据此宣称显著提升。','',
            '## 637道条件难题探针','']
    for name in ['raw','compact']:
        row=comparison[name]['hard']; n=row['trial_count']; length=row['finish_reason_counts'].get('length',0)
        lines.append(f"- {name}：至少一次验证正确 {row['questions_with_any_verified_correct']}/637；有效1～3/4候选 {row['potential_curriculum_count_1to3']}；截断 {length}/{n} 条（{length/n:.3%}）；分组拒绝 {row['group_rejections']}。")
    lines+=['','该集合按E0的0/4失败条件选出，是训练侧候选探针，不是独立最终测试。坏参考答案和评分误判仍可能存在；候选未自动批准。分组拒绝按先不可评分、再截断分类，不可用仅1个截断组替代逐回答length比例。','',
            '## 一epoch训练耗时','']
    for name in ['raw','compact']:
        row=metrics.get(name)
        if row:lines.append(f"- {name}：Trainer train_runtime={row['train_runtime']:.1f}秒（{row['train_runtime']/60:.2f}分钟）；训练平均loss={row['train_loss']:.4f}。")
        else:lines.append(f'- {name}：未读到train_runtime，未验证。')
    if metrics.get('raw') and metrics.get('compact'):
        r=metrics['raw']['train_runtime'];c=metrics['compact']['train_runtime']
        lines+=['',f'compact训练循环约快 {r/c:.3f} 倍，耗时减少 {(1-c/r)*100:.2f}%。train_runtime包含训练循环内保存开销，不包含初始模型加载和训练后合并导出，不能冒充完整作业墙钟时间。']
    lines+=['','两臂监督token约3516983对360882（约9.75倍差异），实际训练时间并不按该倍数缩放。训练目标文本不同，不能直接凭训练loss比较能力。','',
            '## 使用边界','',
            '只汇总既有dev/难题记录和本轮固定MATH-500测试，不新增训练批准。保留所有拒绝类别，不能把不可评分称为已确认数学答错。测试结果不得用于选模、改训练参数、生成变式题或调整评分规则。MATH-500未完成时本报告是阶段汇总，不是最终成绩报告。','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-root',default='/content/drive/MyDrive/runs/project1')
    p.add_argument('--math500-root');p.add_argument('--output-root',required=True)
    a=p.parse_args();runs=Path(a.runs_root);out=Path(a.output_root);out.mkdir(parents=True,exist_ok=True)
    comparison=json.loads((runs/'E1_eval_v13/comparison.json').read_text())
    metrics={name:training_metrics((runs/'E1_equal_exposure_v10'/f'{name}.log').read_text()) for name in ['raw','compact']}
    test=Path(a.math500_root) if a.math500_root else runs/'MATH500_v15'
    math500={name:json.loads((test/name/'summary.json').read_text()) if (test/name/'summary.json').exists() else None for name in ['E0','raw','compact']}
    payload={'dev_and_hard':comparison,'training_metrics':metrics,'MATH500':math500}
    report=render_report(comparison,metrics,math500)
    for path,body in [(out/'summary.json',json.dumps(payload,ensure_ascii=False,indent=2)),(out/'汇总对比报告.md',report)]:
        with path.open('x',encoding='utf-8') as stream:stream.write(body)
    print(report,flush=True)


if __name__=='__main__':main()
