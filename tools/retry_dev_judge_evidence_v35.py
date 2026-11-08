"""Retry only unfinished dev reviews after a judge's non-verbatim evidence.

Keeps the complete original problem/gold/output and unchanged judge system.
Adds formatting feedback only; never tells the judge the desired verdict.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('continuation', ROOT / 'tools/continue_final_eval_v33.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def main():
    directory = ROOT / 'artifacts/final_eval_local_snapshot/evaluations/final_dev256_32k_v32'
    review = directory / 'native_ds_v33'
    raw = tool.base.journal(directory / 'predictions.jsonl')
    completed = {r['sample_id'] for r in tool.base.journal(review / 'results.jsonl')}
    original, _ = tool.sources('review-dev', ROOT, directory)
    golds = {r['sample_id']: r for r in original}
    key = tool.judge.credential(Path.home() / '.codex/private/project1/deepseek.json')
    with tool.judge.locked(directory / 'local_review_v34.lock'):
        for row in raw:
            if row['sample_id'] in completed:
                continue
            gold = golds[row['sample_id']]
            native = tool.judge.grade_bounded(row['text'], gold['answer'], ROOT)
            payload = tool.judge.judge_payload(gold['problem'], gold['answer'], row['text'], native, row['finish_reason'])
            payload['validation_feedback'] = (
                'An earlier API reply could not be accepted because its evidence was not an EXACT substring. '
                'Re-evaluate independently, without assuming the earlier verdict. Copy evidence verbatim, '
                'including original LaTeX spelling and newlines; do not normalize or reconstruct it. '
                'If no final answer can be located, return uncertain and empty evidence. '
                'Do not change the mathematical verdict merely to satisfy formatting.')
            draw_key = (row['sample_id'], 0)
            ds = tool.judge.call_ds(payload, key, 'deepseek-flash',
                                   review / 'api_attempts' / (tool.judge.digest(draw_key) + '.jsonl'))
            result = {'sample_id': row['sample_id'], 'draw': 0, 'text': row['text'],
                'output_tokens': row['generated_tokens'], 'finish_reason': row['finish_reason'],
                'original_problem': gold['problem'], 'original_reference_answer': gold['answer'],
                'native_math_verify': native, 'ds_review': ds,
                'final_verdict': ds['judgment']['verdict'],
                'format_retry': {'feedback': payload['validation_feedback'], 'payload_sha256': tool.judge.digest(payload)}}
            tool.base.append(review / 'results.jsonl', result)
            print(row['sample_id'] + ': validated ' + result['final_verdict'], flush=True)


if __name__ == '__main__':
    main()
