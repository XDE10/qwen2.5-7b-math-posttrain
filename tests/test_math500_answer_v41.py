import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('math500_v41', ROOT / 'tools/evaluate_math500_final_answer_v41.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TestAnswerProtocol(unittest.TestCase):
    def test_intervene_only_for_unfinished_or_missing_box(self):
        self.assertFalse(m.needs_answer({'finish_reason': 'stop', 'text': r'\boxed{3}'}))
        self.assertTrue(m.needs_answer({'finish_reason': 'length', 'text': r'\boxed{3}'}))
        self.assertTrue(m.needs_answer({'finish_reason': 'stop', 'text': 'I think it is 3'}))
        self.assertEqual(m.REASON + m.ANSWER, 8192)

    def test_preserve_exact_original_tokens_and_close_only_open_think(self):
        class Tok:
            def encode(self, text, add_special_tokens):
                self.text = text
                return [17, 18]
        tok = Tok()
        row = {'text': '<think>work', 'token_ids': [91, 92]}
        self.assertEqual(m.continuation_ids([1, 2], row, tok), [1, 2, 91, 92, 17, 18])
        self.assertTrue(tok.text.startswith('\n</think>'))
        self.assertNotIn('</think>', m.answer_suffix('ordinary reasoning'))
        self.assertNotIn('</think>', m.answer_suffix('<think>work</think>'))

    def test_sealed500_and_resume_identity(self):
        data, _ = m.math_data(ROOT)
        self.assertEqual(len(data), 500)
        expected = {r['sample_id'] for r in data}
        self.assertEqual(len(m.by_id([{'sample_id': data[0]['sample_id']}], expected)), 1)
        with self.assertRaises(RuntimeError):
            m.by_id([{'sample_id': 'fake'}], expected)
        with self.assertRaises(RuntimeError):
            m.by_id([{'sample_id': data[0]['sample_id']}] * 2, expected)


if __name__ == '__main__':
    unittest.main()
