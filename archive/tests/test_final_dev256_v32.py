import tempfile
import unittest
import zipfile
from pathlib import Path
from tools.evaluate_final_dev256_v32 import protocol_data, journal

ROOT = Path(__file__).resolve().parents[1]


class FinalDevTest(unittest.TestCase):
    def test_portable_dev_matches_historical_all_256(self):
        from experiments.aimo_sft_v26.common import prompt_ids, SUFFIX
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(ROOT / 'reports/aimo_sft_20261026_v26/project1_aimo_sft_v26.zip') as archive:
                archive.extractall(tmp)
            protocol, _, data = protocol_data(Path(tmp) / 'aimo-sft-v26')
            old = journal(ROOT / 'data/math/math_dev.jsonl')
            self.assertEqual(len(data), 256)
            self.assertEqual([(r['sample_id'], r['problem'], r['reference_answer']) for r in data],
                             [(r['sample_id'], r['problem'], r['reference_answer']) for r in old])
            class CaptureTokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    self.messages = messages
                    self.kwargs = kwargs
                    return [1, 2, 3]
            capture = CaptureTokenizer()
            for row in data:
                self.assertEqual(prompt_ids(capture, row['problem']), [1, 2, 3])
                self.assertEqual(capture.messages, [{'role': 'user', 'content': row['problem'] + SUFFIX}])
                self.assertTrue(capture.kwargs['add_generation_prompt'])
            self.assertEqual(protocol['max_new_tokens'], 32768)
            self.assertEqual(protocol['temperature'], 0)
            self.assertEqual(protocol['max_num_seqs'], 16)

    def test_local_dev_matches_same_protocol(self):
        protocol, _, data = protocol_data(ROOT)
        self.assertEqual(len(data), protocol['count'])
        self.assertEqual(protocol['count'], 256)


if __name__ == '__main__':
    unittest.main()
