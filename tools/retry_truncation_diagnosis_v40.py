"""Retry only unfinished classifier cases with bounded-length evidence quotes.

No changes to saved accepted classifications or original model outputs.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('classifier', ROOT / 'tools/classify_truncated_dev_v38.py')
classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classifier)
FORMAT_SUFFIX = '''\n补充输出格式约束：每个evidence字段最多120个字符，只摘录一个很短的逐字片段。即使原文有无限循环或很长的小数，也只能复制短片段，绝对不要继续抄写整个重复段或整串数字。解释最多120个中文字；保证在输出预算内闭合JSON。分类规则不变，仍须阅读完整原输出。'''
FORMAT_SUFFIX += '''\n本次重试强制repetition_evidence字段返回空字符串，不复制任何小数或重复内容；重复模式仅在observable_mechanism_zh中用一句中文描述。正确结果和错误的证据各最多80个字符。不要输出长数字或重复公式。'''


def main():
    tool = classifier.tool
    directory = ROOT / 'artifacts/final_eval_local_snapshot/evaluations/final_dev256_32k_v32'
    output = directory / 'truncation_ds_v38'
    saved = tool.base.journal(output / 'results.jsonl')
    done = {r['sample_id'] for r in saved}
    raw = [r for r in tool.base.journal(directory / 'predictions.jsonl') if r['finish_reason'] == 'length' and r['sample_id'] not in done]
    original, _ = tool.sources('review-dev', ROOT, directory)
    golds = {r['sample_id']: r for r in original}
    api_key = tool.judge.credential(Path.home() / '.codex/private/project1/deepseek.json')
    classifier.SYSTEM += FORMAT_SUFFIX
    with tool.judge.locked(output / 'running.lock'):
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {}
            for row in raw:
                gold = golds[row['sample_id']]
                payload = {'sample_id': row['sample_id'], 'original_problem': gold['problem'],
                    'original_reference_answer': gold['answer'], 'complete_original_model_output': row['text'],
                    'finish_reason': 'length', 'generated_tokens': row['generated_tokens']}
                jobs[pool.submit(classifier.request, payload, api_key,
                    output / 'api_attempts' / (tool.judge.digest(row['sample_id']) + '.jsonl'))] = row['sample_id']
            for future in as_completed(jobs):
                result = {'sample_id': jobs[future], **future.result(),
                    'format_retry': {'system_suffix': FORMAT_SUFFIX, 'system_sha256': tool.judge.digest(classifier.SYSTEM)}}
                tool.base.append(output / 'results.jsonl', result)
                print(result['sample_id'] + ': ' + result['classification']['category'], flush=True)


if __name__ == '__main__':
    main()
