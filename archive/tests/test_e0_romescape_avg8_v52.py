import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('romescape', ROOT / 'tools/evaluate_e0_romescape_avg8_v52.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TestPairedAvg8(unittest.TestCase):
    def test_sealed_questions_and_budget(self):
        self.assertEqual(len(m.math_data(ROOT)[0]), 500)
        self.assertEqual((m.REASON, m.ANSWER, m.TOTAL, m.CONTEXT), (7936, 256, 8192, 16384))
        self.assertEqual(m.MODEL_REVISION, '47277850077799b7b416611b76ca201231ff270d')

    def test_native_system_prompt_and_batchencoding(self):
        from transformers.tokenization_utils_base import BatchEncoding
        class Tok:
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return BatchEncoding({'input_ids': [[1, 2, 3]]})
        t = Tok()
        self.assertEqual(m.prompt_token_ids(t, 'original question'), [1, 2, 3])
        self.assertEqual(t.messages, [{'role': 'system', 'content': m.SYSTEM}, {'role': 'user', 'content': 'original question'}])
        self.assertFalse(t.kwargs['return_dict'])

    def test_identity_never_migrates_old_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            m.prepare_identity(p, {'model': 'a'})
            with self.assertRaises(RuntimeError):
                m.prepare_identity(p, {'model': 'b'})
            self.assertEqual(m.read(p / 'identity.json'), {'model': 'a'})

    def test_gpu_busy_refused(self):
        with patch.object(m.subprocess, 'run', return_value=SimpleNamespace(stdout='123, python')):
            with self.assertRaises(RuntimeError):
                m.assert_gpu_idle()
        with patch.object(m.subprocess, 'run', return_value=SimpleNamespace(stdout='')):
            m.assert_gpu_idle()

    def test_continuation_preserves_original_ids(self):
        class Tok:
            def encode(self, text, **kwargs): return [8, 9]
        row = {'text': 'working', 'token_ids': [3, 4], 'finish_reason': 'length'}
        self.assertTrue(m.needs_answer(row))
        self.assertEqual(m.continuation_ids([1, 2], row, Tok()), [1, 2, 3, 4, 8, 9])
        self.assertEqual(m.answer_suffix('<|im_start|>answer\n42').count('<|im_start|>answer'), 0)

    def test_paired_seeds_are_distinct(self):
        self.assertEqual(m.SAMPLES, 8)
        self.assertEqual(len({m.seed('question', i) for i in range(8)}), 8)

    def test_both_launches_e0_then_romescape(self):
        with patch.object(m.sys, 'argv', ['entry']), patch.object(m.subprocess, 'run') as run:
            m.main()
        self.assertEqual(run.call_count, 2)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual([c[c.index('--stage') + 1] for c in commands], ['e0', 'romescape'])
        self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))

    def test_stage_failure_stops_chain(self):
        with patch.object(m.sys, 'argv', ['entry']), patch.object(m.subprocess, 'run', side_effect=RuntimeError('failed')) as run:
            with self.assertRaises(RuntimeError):
                m.main()
        self.assertEqual(run.call_count, 1)

if __name__ == '__main__':
    unittest.main()
