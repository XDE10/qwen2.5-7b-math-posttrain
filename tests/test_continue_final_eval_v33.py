import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('continuation', ROOT / 'tools/continue_final_eval_v33.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ContinuationTests(unittest.TestCase):
    def test_actual_portable_dev_and_math_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(ROOT / 'reports/aimo_sft_20261026_v26/project1_aimo_sft_v26.zip') as archive:
                archive.extractall(tmp)
            portable = Path(tmp) / 'aimo-sft-v26'
            dev, settings = mod.sources('review-dev', portable, Path(tmp))
            math, msettings = mod.sources('math500', portable, Path(tmp))
            self.assertEqual(len(dev), 256)
            self.assertEqual(len(math), 500)
            self.assertEqual(settings['temperature'], 0)
            self.assertEqual(msettings['samples'], 1)
            self.assertEqual(msettings['temperature'], 0.6)

    def test_resume_and_full_output_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            mod.base.save(directory / 'identity.json', {'model': 'fake-final'})
            long_output = 'Unabridged start ' + 'x' * 100000 + ' final 2'
            data = [{'sample_id': 'a', 'problem': 'original problem a', 'answer': '1'},
                    {'sample_id': 'b', 'problem': 'original problem b', 'answer': '2'}]
            raw = [{'sample_id': 'a', 'draw': 0, 'text': 'final 1', 'output_tokens': 2, 'finish_reason': 'stop'},
                   {'sample_id': 'b', 'draw': 0, 'text': long_output, 'output_tokens': 30000, 'finish_reason': 'stop'}]
            def native(output, gold, project):
                return {'correct': gold == '1', 'status': 'equivalent' if gold == '1' else 'mismatch'}
            def judge(payload, *_):
                self.assertEqual(payload['complete_original_model_output'], long_output)
                self.assertEqual(payload['original_problem'], 'original problem b')
                self.assertEqual(payload['original_reference_answer'], '2')
                return {'judgment': {'verdict': 'uncertain'}}
            with patch.object(mod.judge, 'grade_bounded', side_effect=native) as n, \
                 patch.object(mod.judge, 'call_ds', side_effect=judge) as ds:
                first = mod.review(raw, data, {'samples': 1}, directory, ROOT, 'fake')
                second = mod.review(raw, data, {'samples': 1}, directory, ROOT, 'fake')
                self.assertEqual(first, second)
                self.assertEqual(n.call_count, 2)
                self.assertEqual(ds.call_count, 1)
                self.assertEqual(first['uncertain'], 1)
                self.assertEqual(first['reviewed_accuracy_lower_bound'], 0.5)
                self.assertEqual(first['reviewed_accuracy_upper_bound'], 1)

    def test_missing_draws_do_not_publish_full_score(self):
        data = [{'sample_id': 'a'}]
        raw = [{'sample_id': 'a', 'draw': 0}]
        self.assertEqual(len(mod.inspect_raw(raw, data, 4)), 3)
        with self.assertRaises(RuntimeError):
            mod.inspect_raw(raw * 2, data, 4)

    def test_next_is_dev_review_then_math_then_probe(self):
        with patch.object(sys, 'argv', ['continue', '--stage', 'next']), \
             patch.object(mod.subprocess, 'run') as run:
            mod.main()
            stages = [call.args[0][call.args[0].index('--stage') + 1] for call in run.call_args_list]
            self.assertEqual(stages, ['review-dev', 'math500', 'hard637'])

    def test_generation_only_skips_dev_and_passes_flag(self):
        with patch.object(sys, 'argv', ['continue', '--stage', 'next', '--generate-only']), \
             patch.object(mod.subprocess, 'run') as run:
            mod.main()
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual([cmd[cmd.index('--stage') + 1] for cmd in commands], ['math500', 'hard637'])
            self.assertTrue(all('--generate-only' in cmd for cmd in commands))


if __name__ == '__main__':
    unittest.main()
