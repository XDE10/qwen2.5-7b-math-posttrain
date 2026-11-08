import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ds_v29', ROOT / 'tools/evaluate_sft_ds_v29.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class NativeDSReviewTests(unittest.TestCase):
    def test_native_equivalence_and_final_box(self):
        cases = [(r'Final answer: \boxed{\frac{1}{2}}', '0.5', True),
                 (r'First guess \boxed{2}. Correction: \boxed{3}', '3', True),
                 (r'First guess \boxed{3}. Final answer: \boxed{2}', '3', False),
                 (r'Final answer: \boxed{x=3}', '3', True)]
        for output, gold, expected in cases:
            with self.subTest(output=output):
                result = mod.grade_bounded(output, gold, ROOT)
                self.assertEqual(result['correct'], expected)

    def test_full_original_payload(self):
        output = 'Beginning of reasoning\n' + 'a' * 40000 + '\nFinal answer is 7.'
        result = mod.judge_payload('original problem', 'original gold', output,
                                   {'status': 'mismatch'}, 'stop')
        self.assertEqual(result['complete_original_model_output'], output)
        self.assertEqual(result['original_reference_answer'], 'original gold')

    def test_invalid_judge_evidence_rejected(self):
        good = {'verdict': 'correct', 'model_final_answer': '7', 'reason': 'Equivalent', 'evidence': 'answer 7'}
        self.assertEqual(mod.validate_judgment(good, 'Final answer 7'), good)
        with self.assertRaises(ValueError):
            mod.validate_judgment(good, 'Final answer 8')
        with self.assertRaises(ValueError):
            mod.validate_judgment({**good, 'evidence': ''}, 'Final answer 7')

    def test_review_resume_and_uncertain_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data.jsonl'
            records = [{'sample_id': str(i), 'problem': 'Find value', 'answer': str(i)} for i in range(3)]
            source.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
            mod.save(root / 'reports/aimo_sft_20261026_v26/evaluation_sources.json',
                     {'dev': {'file': 'data.jsonl', 'sha256': mod.file_sha(source)}})
            directory = root / 'eval'
            directory.mkdir()
            mod.save(directory / 'identity.json', {'model': 'test'})
            for i in range(3):
                mod.append(directory / 'raw.jsonl', {'sample_id': str(i), 'draw': 0,
                           'text': str(i), 'finish_reason': 'stop'})
            def auto(output, reference, project_root):
                return {'correct': output == '0', 'status': 'equivalent' if output == '0' else 'mismatch'}
            def judge(payload, key, model, attempts):
                verdict = 'correct' if payload['original_reference_answer'] == '1' else 'uncertain'
                return {'judgment': {'verdict': verdict, 'evidence': payload['complete_original_model_output']}}
            with patch.object(mod, 'grade_bounded', side_effect=auto) as native, \
                 patch.object(mod, 'call_ds', side_effect=judge) as ds:
                first = mod.review_directory(directory, 'dev', root, 'fake', 'fake-model')
                second = mod.review_directory(directory, 'dev', root, 'fake', 'fake-model')
                self.assertEqual(first, second)
                self.assertEqual(native.call_count, 3)
                self.assertEqual(ds.call_count, 2)
                self.assertEqual(first['uncertain'], 1)
                self.assertEqual(first['reviewed_accuracy_lower_bound'], 2 / 3)
                self.assertEqual(first['reviewed_accuracy_upper_bound'], 1)
                raw = mod.rows(directory / 'raw.jsonl')
                raw[0]['text'] = 'changed'
                (directory / 'raw.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in raw), encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, 'input changed'):
                    mod.review_directory(directory, 'dev', root, 'fake', 'fake-model')


if __name__ == '__main__':
    unittest.main()
