import importlib.util
from pathlib import Path
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('s11_eval', ROOT / 'tools/evaluate_public_s11_math500_v43.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TestPublicS1(unittest.TestCase):
    def test_batchencoding_flat_and_nested_are_plain_json_tokens(self):
        from transformers.tokenization_utils_base import BatchEncoding
        class Tok:
            def __init__(self, value):
                self.value = value
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return self.value
        for value in ([1, 151644, 7], BatchEncoding({'input_ids': [1, 151644, 7]}),
                      BatchEncoding({'input_ids': [[1, 151644, 7]]})):
            tok = Tok(value)
            result = m.prompt_token_ids(tok, 'sample')
            self.assertEqual(result, [1, 151644, 7])
            self.assertFalse(tok.kwargs['return_dict'])
            self.assertEqual(m.json.loads(m.json.dumps({'sample': result}))['sample'], result)
        with self.assertRaises(RuntimeError):
            m.prompt_token_ids(Tok([[1], [2]]), 'sample')

    def test_repair_only_known_initial_failure_without_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            old = {'script_sha256': m.FAILED_INITIAL_SCRIPT, 'budget': 8192}
            new = {**old, 'script_sha256': 'new'}
            m.save(output / 'identity.json', old)
            m.prepare_identity(output, new)
            self.assertEqual(m.read(output / 'identity_before_batchencoding_fix.json'), old)
            self.assertEqual(m.read(output / 'identity.json'), new)
            m.save(output / 'identity.json', old)
            (output / 'reasoning.jsonl').write_text('{}\n', encoding='utf8')
            with self.assertRaises(RuntimeError):
                m.prepare_identity(output, new)

    def test_model_budget_and_data(self):
        self.assertEqual(m.MODEL_ID, 'simplescaling/s1.1-7B')
        self.assertEqual(m.MODEL_REVISION, 'ff0270ab653ee96811d034acba1a880e48612cc1')
        self.assertIsNone(m.ROPE)
        self.assertEqual((m.REASON, m.ANSWER, m.TOTAL, m.CONTEXT), (7936, 256, 8192, 16384))
        self.assertEqual(len(m.math_data(ROOT)[0]), 500)

    def test_s1_boundary_and_exact_continuation(self):
        class Tok:
            def encode(self, value, add_special_tokens):
                self.value = value
                return [151644, 2, 3]
        tok = Tok()
        r = {'text': '<|im_start|>think\nwork', 'token_ids': [17, 18]}
        ids = m.continuation_ids([1], r, tok)
        self.assertEqual(ids, [1, 17, 18, 151644, 2, 3])
        self.assertIn('<|im_start|>answer', tok.value)
        self.assertNotIn('</think>', tok.value)
        self.assertNotIn('<|im_start|>answer', m.answer_suffix('<|im_start|>answer\nmore work'))

    def test_existing_complete_answers_and_cache_location(self):
        self.assertFalse(m.needs_answer({'finish_reason': 'stop', 'text': r'answer \boxed{42}'}))
        self.assertTrue(m.needs_answer({'finish_reason': 'length', 'text': 'work'}))
        run = Path('/content/drive/MyDrive/runs/project1/aimo_sft_v26')
        self.assertEqual(run.parents[2] / 'models', Path('/content/drive/MyDrive/models'))


if __name__ == '__main__':
    unittest.main()
