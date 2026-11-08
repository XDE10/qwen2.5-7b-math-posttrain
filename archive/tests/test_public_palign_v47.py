"""CPU tests: public adapter identity, prompt, continuation, actual LoRA routing."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('palign_eval', ROOT / 'tools/evaluate_public_palign_math500_v47.py')
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


class TokenizerStub:
    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{'role': 'user', 'content': g.INSTRUCTION + '\nProblem text'}]
        return {'input_ids': [7, 8, 9]}

    def encode(self, text, **kwargs):
        assert '\\boxed{}' in text
        return [10, 11]


class PalignTests(unittest.TestCase):
    def test_original_published_adapter_layout(self):
        p = ROOT / 'artifacts/research_s1_7b/two_repos/palign/published_adapter_config.json'
        c = json.loads(p.read_text())
        g.validate_adapter(c)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            g.validate_adapter(dict(c, r=32))

    def test_prompt_batchencoding_and_exact_continuation(self):
        tok = TokenizerStub()
        ids = g.prompt_token_ids(tok, 'Problem text')
        self.assertEqual(ids, [7, 8, 9])
        self.assertEqual(g.continuation_ids(ids, {'text': 'draft', 'token_ids': [42, 43]}, tok),
                         [7, 8, 9, 42, 43, 10, 11])
        suffix = g.answer_suffix('draft')
        self.assertNotIn('<|im_start|>answer', suffix)
        self.assertNotIn('<think>', suffix)

    def test_both_generation_phases_use_adapter(self):
        tree = ast.parse(Path(g.__file__).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'generate']
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertTrue(any(k.arg == 'lora_request' and isinstance(k.value, ast.Name)
                                and k.value.id == 'lora_request' for k in call.keywords))
        params = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'SamplingParams']
        self.assertEqual(len(params), 2)
        for call in params:
            self.assertTrue(any(k.arg == 'repetition_penalty' and isinstance(k.value, ast.Name)
                                and k.value.id == 'REPETITION_PENALTY' for k in call.keywords))
        self.assertEqual(g.REPETITION_PENALTY, 1.05)

    def test_identity_change_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            g.prepare_identity(out, {'revision': 'one'})
            g.prepare_identity(out, {'revision': 'one'})
            with self.assertRaises(RuntimeError):
                g.prepare_identity(out, {'revision': 'two'})

    def test_sealed_questions_budget_and_model(self):
        data, digest = g.math_data(ROOT)
        self.assertEqual(len(data), 500)
        self.assertEqual(g.REASON + g.ANSWER, 8192)
        self.assertEqual(g.ANSWER, 256)
        self.assertEqual(g.MODEL_ID, 'qizheyanger/P-ALIGN')
        self.assertIsNone(g.ROPE)
        self.assertTrue(g.needs_answer({'text': 'unfinished', 'finish_reason': 'length'}))
        self.assertFalse(g.needs_answer({'text': '\\boxed{1}', 'finish_reason': 'stop'}))

    def test_published_tokenizer_keeps_base_vocabulary(self):
        from tokenizers import Tokenizer
        author = ROOT / 'artifacts/research_s1_7b/two_repos/palign/published_tokenizer.json'
        base = ROOT / 'data/tokenizers/Qwen2.5-7B-Instruct/tokenizer.json'
        self.assertEqual(Tokenizer.from_file(str(author)).get_vocab(), Tokenizer.from_file(str(base)).get_vocab())


if __name__ == '__main__':
    unittest.main()
