# coding: utf-8
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'external_review_20261103_v92'
OLD = ROOT / 'reports' / 'claude_project1_review_20261031_v90' / '项目一_Claude审查包_v90.zip'
RUN = ROOT / 'artifacts' / 'qfft_incomplete_high_v91'
OUT.mkdir(parents=True, exist_ok=True)
status = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
results = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((RUN / 'results').glob('*.json'))]
counts = Counter(x['outcome'] for x in results)
assert len(results) == status['completed'] == sum(counts.values())
assert status['target'] == 62

overview = f"""# 项目一外部审查包：当前快照

生成日期：2026-11-03。v91状态文件最后更新：{status['updated_at']}（UTC+8）。此包是该时间的本地落盘快照，不代表今日重新执行训练或生成。

## 目标与边界

以同一批自建教师轨迹，从相同Qwen2.5-7B-Instruct底座独立初始化，分别训练SFT和QFFT，比较MATH500准确率、推理token与资源。约80% SFT和约83%后续GRPO只是目标。请重点建议尽快收尾的数据准入、最小可信双组比较和正式训练前必需工程。

## 数据事实

- 原1000题首轮：747条结构完整且QFFT式≤16K、164条完整超16K、89条不完整。结构完整不等于数学正确。
- v86对41条原max截断题有界补跑：27条新结构完整，其中11条≤16K、16条超16K；14条仍失败。若仅作候选替换，v91之前为758/180/62。这只是候选视图，原文件未覆盖。
- v91针对剩余62题，用high和抗重复提示词各一次，2并发。本快照结果{status['completed']}/62：{status['complete_within_16k']}条新完整≤16K、{status['complete_over_16k']}条新完整超16K、{status['incomplete']}条不完整、{status['transport_unknown_skip']}条传输未知跳过；另{62-status['completed']}条未形成结果。状态为{status['state']}，错误：{status.get('error', {}).get('message', '无')}。不能称62题跑完，更不能称1000条正确或可入训。
- 若按v86和本快照v91完整候选进行**假设性替换**，QFFT式≤16K完整上界为{758+status['complete_within_16k']}，完整超16K为{180+status['complete_over_16k']}，尚不完整或未形成结果为{62-status['complete_within_16k']-status['complete_over_16k']}。尚未合并、数学质检、SFT/QFFT双模板编码或批准训练，实际共同训练N未知。

## 请审查

1. 最小的数据质量门槛与疑难题处理：核验最终答案及关键推导，不按凑1000放宽标准；说明何时停止再造数。
2. 同一获批题集上的SFT/QFFT训练模板、think/题目位置、labels/mask/EOS、16K交集、训练步数随真实N动态计算、损失归一化、保存恢复及Drive持久化。
3. 两组各一次同底座独立训练的比较口径、评测隔离、准确率/长度/截断/成本记录；可选GRPO只作未来建议。
4. 按“已证实、推断、待验证”列出建议和最少的阻塞问题。不要把公开RomEscape结果当自训结果，不把已有短测当正式训练，也不要声称已逐题审完未附的完整轨迹。

## 包内文件

`证据/`沿用v90已核验的训练、评测和协议小文件；`v91/`提供最新提示词、说明与聚合状态。未包含密钥、数据库、1000条完整轨迹或历史日志。v90总结和旧审查提示词因状态已过时而不收录。需要逐题数学审查时，请明确请求最少的原始题目与轨迹。
"""

files = {'开始阅读.md': overview.encode('utf-8')}
files['给Argon的深度审查提示词.md'] = (OUT / '给Argon的深度审查提示词.md').read_bytes()
with zipfile.ZipFile(OLD) as z:
    for name in z.namelist():
        if name.startswith('证据/'):
            files[name] = z.read(name)
for src, dst in [
    (ROOT / 'reports/qfft_incomplete_high_20261031_v91/README.md', 'v91/补跑方案与边界.md'),
    (RUN / 'teacher_system_prompt.txt', 'v91/教师system提示词.txt'),
    (RUN / 'status.json', 'v91/原始状态.json'),
]:
    files[dst] = src.read_bytes()
summary = {'updated_at': status['updated_at'], 'state': status['state'], 'target': 62,
           'completed': len(results), 'outcomes': dict(counts),
           'complete_within_16k': status['complete_within_16k'],
           'complete_over_16k': status['complete_over_16k'],
           'training_approved': False,
           'results': [{'index': x['index'], 'outcome': x['outcome'],
                        'structural_flags': x.get('structural_flags', [])} for x in results]}
files['v91/逐题结构结果_无轨迹.json'] = (json.dumps(summary, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
manifest = [{'path': k, 'size': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for k,v in sorted(files.items())]
files['文件清单_SHA256.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
target = OUT / '项目一_外部模型审查包_20261103_v92.zip'
with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name, data in sorted(files.items()):
        z.writestr(name, data)
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert set(z.namelist()) == set(files)
    for name, data in files.items():
        assert z.read(name) == data
print(json.dumps({'zip': str(target), 'files': len(files), 'bytes': target.stat().st_size,
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  'status': status['state'], 'completed': status['completed']}, ensure_ascii=False))
