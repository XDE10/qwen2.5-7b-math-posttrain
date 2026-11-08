"""Archive existing QFFT responses for later handling. No API or GPU imports/calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/qfft_ds_dataset_v66'
RECIPE = ROOT / 'reports/runs/qfft_ds_dataset_20261029_v66'
OUT = ROOT / 'reports/runs/qfft_followups_20261029_v70'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def relative(path):
    return path.relative_to(ROOT).as_posix()


NOTES = {
    7: ['Published reference solution is inconsistent; final N=21 is unreliable. New teacher final solution checked valid; full CoT remains pending.'],
    21: ['Published reference uses floor outside allowed expression grammar. Do not use that reference as proof gold.'],
    27: ['Subquestion B.4 lacks preceding neutrino-energy context. Do not silently insert reference-only numeric values.'],
    29: ['Missing figure and h/H definitions. Teacher assigns H to main-body surface; option letter not certified.'],
    45: ['Published reference has intermediate algebra errors. New teacher final solution checked valid; full CoT remains pending.'],
    46: ['Subquestion (d) lacks rotating-frame and angular-velocity orientation context; conditional solution may require explicit assumptions.'],
}


def main():
    protocol = read(RECIPE / 'protocol.json')
    questions = {r['index']: r for r in read(RECIPE / 'questions1000.json')}
    paths = sorted((SOURCE / 'candidates').glob('*.json'))
    records, source_hashes = [], {}
    groups = {'complete_over_student_window': [], 'teacher_budget_exhausted': [], 'other_incomplete': []}
    OUT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        result = read(path)
        row = questions[result['index']]
        call = SOURCE / 'calls' / f'{row["index"]:04d}' / result['provider']
        request_path, response_path = call / 'request.json', call / 'response.json'
        dispatch, response = read(request_path), read(response_path)
        msg = response['response']['choices'][0]['message']
        assert result['reasoning_content_raw'] == (msg.get('reasoning_content') or '')
        assert result['final_content_raw'] == (msg.get('content') or '')
        assert dispatch['body']['messages'][1]['content'] == row['problem']
        assert result['training_approved'] is False
        for src in [path, request_path, response_path]:
            source_hashes[relative(src)] = sha(src)
        exhausted = result['finish_reason'] == 'length'
        over = result['cleaned_student_tokens'] > protocol['student_window_qwen']
        category = ('teacher_budget_exhausted' if exhausted else
                    'complete_over_student_window' if result['structurally_complete'] and over else
                    'other_incomplete' if not result['structurally_complete'] else None)
        record = {
            'sample_id': row['sample_id'], 'index': row['index'], 'generation_batch': row['batch'],
            'question_sha256': row['question_sha256'], 'source': row['source'],
            'source_revision': row['source_revision'], 'source_position': row['source_position'],
            'task_type': row['task_type'], 'category': row['category'], 'difficulty': row['difficulty'],
            'provider': result['provider'], 'model_requested': result['model_requested'],
            'model_returned': result['model_returned'],
            'reasoning_effort_requested': result['reasoning_effort_requested'],
            'teacher_api_max_output_tokens': dispatch['body']['max_tokens'],
            'teacher_usage': response['response'].get('usage', {}),
            'teacher_usage_note': 'Provider/API tokenizer; do not equate to Qwen student tokens.',
            'student_window_qwen': protocol['student_window_qwen'],
            'raw_student_tokens': result['raw_student_tokens'],
            'cleaned_student_tokens': result['cleaned_student_tokens'],
            'over_student_window': over, 'teacher_budget_exhausted': exhausted,
            'finish_reason': result['finish_reason'],
            'reasoning_nonempty': bool(result['reasoning_content_raw'].strip()),
            'final_nonempty': bool(result['final_content_raw'].strip()),
            'structurally_complete': result['structurally_complete'], 'length_class': result['length_class'],
            'followup_category': category,
            'source_files': {relative(p): source_hashes[relative(p)] for p in [path, request_path, response_path]},
            'completed_utc': response['completed_utc'], 'latency_seconds': response['latency_seconds'],
            'system_prompt_sha256': hashlib.sha256(dispatch['body']['messages'][0]['content'].encode()).hexdigest(),
            'known_local_review_notes': NOTES.get(row['index'], []),
            'proof_quality': 'pending_full_independent_review', 'training_approved': False,
            'followup_status': 'pending_user_decision' if category else 'pending_quality_review',
            'followup_model': None, 'followup_result_path': None, 'followup_approved_to_generate': False,
            'original_response_never_overwrite': True,
        }
        records.append(record)
        if category:
            groups[category].append(record)
            case_dir = OUT / category / f'{row["index"]:04d}'
            save(case_dir / 'case_record.json', record)
            save(case_dir / 'question_only.json', {'sample_id': row['sample_id'], 'index': row['index'],
                 'problem': row['problem'], 'question_sha256': row['question_sha256'],
                 'task_type': row['task_type'], 'training_approved': False})
            # A separate full original trajectory for forensic review, not independent regeneration input.
            save(case_dir / 'original_trace.json', result)
            assert read(case_dir / 'original_trace.json') == result
            (case_dir / 'question_only.md').write_text(
                f'# {row["sample_id"]}\n\n' + row['problem'] + '\n', encoding='utf-8')

    for name, items in groups.items():
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / 'index.json', {'category': name, 'count': len(items), 'records': items})
        save(folder / 'questions_only.json', {'category': name, 'count': len(items),
             'scope': 'Original questions only. No old solution, reference answer, or review verdict supplied.',
             'records': [read(folder / f'{r["index"]:04d}' / 'question_only.json') for r in items]})

    deferred_path = SOURCE / 'reviewed_unresolved_v73.json'
    deferred = read(deferred_path).get('records', []) if deferred_path.exists() else []
    unresolved_questions = []
    for item in deferred:
        row = questions[item['index']]
        question = {'sample_id': row['sample_id'], 'index': row['index'], 'problem': row['problem'],
                    'question_sha256': row['question_sha256'], 'task_type': row['task_type'],
                    'training_approved': False}
        unresolved_questions.append(question)
        case_dir = OUT / 'unknown_after_dispatch' / f'{row["index"]:04d}'
        save(case_dir / 'question_only.json', question)
        stream_evidence = {}
        dispatch_dir = SOURCE / 'calls' / f'{row["index"]:04d}' / item['provider']
        for name in ['stream_chunks.jsonl', 'stream_partial.json', 'stream_complete.json']:
            source = dispatch_dir / name
            if source.exists():
                target = case_dir / name
                target.write_bytes(source.read_bytes())
                assert sha(source) == sha(target)
                stream_evidence[name] = {'original_path': relative(source), 'sha256': sha(source)}
        save(case_dir / 'case_record.json', {**item, 'response_received_locally': False,
             'proof_quality': 'no_response_available', 'training_approved': False,
             'stream_evidence': stream_evidence,
             'reason': 'Original dispatch outcome unknown; preserve evidence, never fabricate a completed trajectory.'})
    save(OUT / 'unknown_after_dispatch' / 'index.json', {'count': len(deferred), 'records': deferred})
    save(OUT / 'unknown_after_dispatch' / 'questions_only.json', {'count': len(deferred),
         'scope': 'Original questions only; no returned trajectory exists locally.', 'records': unresolved_questions})

    save(OUT / 'all_generated_sample_ledger.json', {'source_version': 66, 'records': records,
         'manual_updates': 'Append to followup_events.jsonl; never edit original request/response/candidate.',
         'none_automatically_training_approved': True})
    # This file is user/editor-owned and must never be reset on re-export.
    events = OUT / 'followup_events.jsonl'
    if not events.exists():
        events.write_text('', encoding='utf-8')
    schema = {
        'event_id': 'unique identifier', 'recorded_at': 'ISO8601 timezone included',
        'sample_id': 'same original sample_id',
        'action': 'review | propose_regenerate | authorize_regenerate | receive_new_solution | verify | approve_training | reject',
        'actor_or_model': 'human or exact model/version, not just Gemini',
        'authorization_evidence': 'required before any new generation',
        'new_result_path': 'separate relative path; never overwrite original_trace',
        'new_result_sha256': 'hash of saved new response', 'new_prompt_path': 'saved new prompt',
        'new_usage_and_budget': 'actual values or explicitly unavailable for web UI',
        'original_question_changed': False,
        'review_notes_and_quotes': [], 'training_approved': False,
    }
    save(OUT / 'followup_event_schema.json', schema)
    (OUT / 'Gemini后续独立解题提示词.txt').write_text(
        '请独立解答附件questions_only.json中的题目，逐题按sample_id覆盖。附件没有提供旧答案或推理。\n'
        '给出完整、可核验的推导，核对定义域、分类穷尽、边界、等号可达条件和关键计算；最终结论明确。'
        '不要因为题目以前被其他模型做过，就假设它一定可解或参考一定正确。'
        '如果题面缺图、缺前文、数据矛盾或只能在额外假设下求解，明确指出缺失信息或条件，不能悄悄补造。\n'
        '不为了缩短而省略必要推导，也不强迫制造失败分支。逐题输出sample_id、推导、最终结论和待确认信息；'
        '无法完成时明确说明，不声称已经证明。最后列出已解和未解ID。\n'
        '这是独立新解答，不是对旧输出的原文修复。审查与训练批准将另行进行。\n', encoding='utf-8')

    counts = {name: len(items) for name, items in groups.items()}
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_version': 66,
               'generated_count': len(records), 'followup_category_counts': counts,
               'over_student_window_total_including_exhausted': sum(r['over_student_window'] for r in records),
               'teacher_budget_exhausted_ids': [r['sample_id'] for r in groups['teacher_budget_exhausted']],
               'complete_over_student_window_ids': [r['sample_id'] for r in groups['complete_over_student_window']],
               'distinct_followup_ids': [r['sample_id'] for r in records if r['followup_category']],
               'pending_quality_review_count': sum(r['followup_category'] is None for r in records),
               'unknown_dispatches_count_not_in_generated_ledger': len(deferred),
               'unknown_dispatches_ids': [r['sample_id'] for r in deferred],
               'zero_new_generation_requests': True, 'source_files_unmodified': True,
               'all_training_approved_false': True}
    assert len(summary['distinct_followup_ids']) == sum(counts.values())
    for rel, expected in source_hashes.items():
        assert sha(ROOT / rel) == expected
    save(OUT / 'summary.json', summary)
    save(OUT / 'source_hashes.json', source_hashes)
    (OUT / 'README.md').write_text(f'''# 已有生成数据后续处理台账 v70（共{len(records)}条）

按现有v66返回结果独立归档，不发API，不生成Gemini答案，不变更原题、轨迹或训练批准。

- `complete_over_student_window/`：正常结束且思考/最终回答非空，但学生Qwen序列超过16384。完整保留，数学质量待审；不是教师API截断。
- `teacher_budget_exhausted/`：finish_reason=length，教师API预算32768耗尽，本批这些题最终回答为空。不能当已完成解题。
- `other_incomplete/`：为未来其他结构不完整情况预留，本批没有记录。
- `unknown_after_dispatch/`：{len(deferred)}条请求结果未知，与已返回候选分开登记，不伪造原始轨迹；包含中断或Go服务端错误后暂缓的原题和证据索引。
- `all_generated_sample_ledger.json`：全部{len(records)}条台账。{summary['pending_quality_review_count']}条未归入长度/结构异常队列的样本仍需质量审查，并非已批准。

分类计数：完整超学生窗口{counts['complete_over_student_window']}条、教师预算耗尽{counts['teacher_budget_exhausted']}条、其他结构不完整{counts['other_incomplete']}条。分类队列互斥，共{sum(counts.values())}条；非互斥的“全部超过学生窗口”标记共{summary['over_student_window_total_including_exhausted']}条，不能与分类数量相加。

每题单独目录都有case_record.json（详细索引）、question_only.json/md（原题）、original_trace.json（完整旧轨迹）。case记录教师模型/渠道/思考档、API预算和usage、学生token长度、停止原因、来源、完成时间、原文件路径和SHA256、已知题面/参考问题及后续状态。API token与学生token口径分开。

## 将来交给Gemini

仅让Gemini独立生成时，发相应类别的questions_only.json并复制Gemini后续独立解题提示词.txt，不发旧答案和原生判分。也可逐题发送question_only.md。

要审查旧推理时，再单独发该题original_trace.json；不要把“独立新解”与“参考旧解纠错”混成同一来源。

Gemini输出必须另存新文件并记录精确模型/日期、提示词、是否可见旧解、结果SHA、可获得的预算/usage、验证结论与授权。网页端没有的usage不能编造。题目若补上下文或修改，必须记录新题版本，不能假称同一原题的盲生成。Gemini网页答案通常也不代表有完整可见内部思考，不能直接称作reasoning_content轨迹。

后续人工/模型处理事件追加到followup_events.jsonl，格式参考followup_event_schema.json；重导出脚本不会清空此文件。原始轨迹始终保留，不用Gemini新答案覆盖，审查后再决定是否进入新训练数据版本。

## 分享方式

先用“待后续处理_仅题目与提示词_v70.zip”：体积小，不含旧答案。
如果要诊断旧模型为什么耗尽/超长，再用“待后续处理_完整归档_v70.zip”，逐题上传，避免再次超出上下文。旧“11题完整归档”ZIP保留为首批历史快照，不作为全量最新结果。

自动分组只是结构与长度状态，不意味着错误根因已确定，也不触发补采、压缩、裁剪或自动入训。
''', encoding='utf-8')

    question_paths = [OUT / 'README.md', OUT / 'Gemini后续独立解题提示词.txt']
    question_paths += [OUT / name / 'questions_only.json' for name in groups if groups[name]]
    if deferred:
        question_paths.append(OUT / 'unknown_after_dispatch' / 'questions_only.json')
    archives = [('待后续处理_仅题目与提示词_v70.zip', question_paths),
                ('待后续处理_完整归档_v70.zip', sorted(p for p in OUT.rglob('*') if p.is_file()
                  and p.suffix != '.zip' and p.name != 'followup_events.jsonl'))]
    for name, files in archives:
        with ZipFile(OUT / name, 'w', ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(OUT).as_posix())
        with ZipFile(OUT / name) as archive:
            assert archive.testzip() is None
            for path in files:
                assert archive.read(path.relative_to(OUT).as_posix()) == path.read_bytes()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
