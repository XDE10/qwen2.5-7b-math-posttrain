import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('romescape', ROOT / 'tools/evaluate_public_romescape_qfft_math500_v58.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TestRomEscape(unittest.TestCase):
    def test_sealed_questions_and_budget(self):
        self.assertEqual(len(m.math_data(ROOT)[0]), 500)
        self.assertEqual((m.REASON, m.ANSWER, m.TOTAL, m.CONTEXT), (7936, 256, 8192, 16384))
        self.assertEqual(m.MODEL_REVISION, 'e8a801fb66a54b86f4219689a6d9973d91330d25')

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

class TestQfftProtocol(unittest.TestCase):
    def test_qfft_checkpoint_and_single_sampling(self):
        self.assertEqual(m.MODEL_ID, 'PJiNH/QFFT-repro-S1.1-QFFT-7B')
        for cap in (m.REASON, m.ANSWER):
            value = m.sampling_params(lambda **kw: kw, 'math500/test/00000', cap)
            self.assertEqual(value['n'], 1)
            self.assertEqual((value['temperature'], value['top_p'], value['top_k'], value['repetition_penalty']), (.6, .95, -1, 1.0))
            self.assertEqual(value['max_tokens'], cap)

    def test_tokens_include_original_and_continuation(self):
        a = [{'sample_id':'a','output_tokens':3,'token_ids':[1,2,3],'finish_reason':'length','text':'work'}]
        b = [{'sample_id':'a','forced':True,'output_tokens':2,'token_ids':[4,5],'total_generated_tokens':5,'text':r'\boxed{1}'}]
        value = m.token_statistics(a,b)
        self.assertEqual(value['original_mean_tokens'],3)
        self.assertEqual(value['total_generated_mean_tokens'],5)
        self.assertEqual(value['original_length_count'],1)
        b[0]['total_generated_tokens']=2
        with self.assertRaises(RuntimeError):m.token_statistics(a,b)

    def test_shard_sizes_and_native_generation_config(self):
        self.assertEqual(sum(m.EXPECTED_SHARDS.values()),15231271864)
        source = Path(m.__file__).read_text(encoding='utf-8')
        self.assertIn("generation_config='vllm'",source)
        self.assertIn('public_romescape_s1_qfft_7b_math500_v58',source)
        self.assertNotIn('import torch',source)

if __name__ == '__main__':
    unittest.main()
