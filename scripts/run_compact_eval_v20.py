"""Run user-requested dev, fixed hard probe and MATH-500 at 4096; no training/login."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs-root',default='/content/drive/MyDrive/runs/project1')
    p.add_argument('--model-run')
    p.add_argument('--baseline-root')
    p.add_argument('--output-root')
    p.add_argument('--cache-root',default='/content/models')
    a=p.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    if importlib.util.find_spec('vllm') is None:
        raise SystemExit('缺少 vLLM；尚未开始评测。请提供当前 pip 安装情况，不要整体升级训练环境。')
    runs=Path(a.runs_root)
    model=Path(a.model_run) if a.model_run else runs/'compact_cosine_A10080_v18'/'E1_compact_r32_lr1e4_e3_b32_cosine_A10080_v18'
    if a.baseline_root:
        baseline=Path(a.baseline_root)
    else:
        found=sorted(d for d in runs.glob('E0_dev_v9_*') if (d/'hard_probe_identity.json').is_file() and (d/'summary.json').is_file())
        if not found:raise SystemExit('未找到已有 E0 dev / 固定637题记录；可用 --baseline-root 指定。')
        baseline=found[-1]
    from src.training.eval_identity_v20 import validate_completed_v18
    validate_completed_v18(read(model/'training_completion.json'),read(model/'run_identity.json'))
    if not (model/'merged'/'weight_manifest.json').is_file():
        raise SystemExit('合并导出尚未完成：缺少 weight_manifest.json。')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    out=Path(a.output_root) if a.output_root else runs/'compact_eval_v20'/stamp
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    commands=[
        [sys.executable,'-u','-m','scripts.evaluate_compact_dev_hard_v20','--arm','compact',
         '--model-run',str(model),'--baseline-root',str(baseline),'--sampling-root',str(runs/'rl_screening_v8_n4'),
         '--output-root',str(out/'dev_hard'),'--cache-root',a.cache_root,'--max-new-tokens','4096'],
        [sys.executable,'-u','-m','scripts.evaluate_compact_math500_v20','--model-key','compact_v18',
         '--runs-root',str(runs),'--model-run',str(model),'--output-root',str(out/'math500'),'--cache-root',a.cache_root],
    ]
    for i,command in enumerate(commands,1):
        print(f'[{i}/2] '+('dev256 + hard637x4' if i==1 else 'MATH-500')+'; max_new_tokens=4096; no SwanLab',flush=True)
        with (out/f'phase{i}.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=proc.wait()
            if code:raise subprocess.CalledProcessError(code,command)
    results={'model_run':str(model),'max_new_tokens':4096,
             'dev':read(out/'dev_hard'/'dev_summary.json'),
             'hard':read(out/'dev_hard'/'hard_summary.json'),
             'math500':read(out/'math500'/'summary.json'),
             'scope':'新compact4096预算；旧E0/raw/compact为2048，不是同预算对照。固定637题为条件探针；不新增批准；不据最终测试调参。'}
    (out/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# compact v18 评测汇总（4096 token）','',results['scope'],'']
    for label,key in [('dev','dev'),('MATH-500','math500')]:
        r=results[key]
        lines.append(f"- {label}：{r['correct']}/{r['count']}；不可评分 {r['count']-r['scorable_count']}；length {r['finish_reason_counts'].get('length',0)}；平均输出 {r['mean_generated_tokens']:.2f} token。")
    h=results['hard']
    lines.append(f"- 难题：至少一次验证正确 {h['questions_with_any_verified_correct']}/637；有效1～3/4候选 {h['potential_curriculum_count_1to3']}；拒绝 {h['group_rejections']}。候选未批准。")
    (out/'汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print('结果已保存：'+str(out),flush=True)


if __name__=='__main__':main()
