import unittest
import tempfile
import zipfile
from pathlib import Path
from tools.evaluate_last_e0_protocol_v30 import messages, read, sha

ROOT = Path(__file__).resolve().parents[1]


class HistoricalProtocolTests(unittest.TestCase):
    def test_all_500_prompt_messages_match_old_e0(self):
        from scripts.evaluate_project1_dev_v9 import messages as historical_messages
        from tools.evaluate_last_e0_protocol_v30 import SYSTEM
        from tools.evaluate_last_e0_protocol_v30 import journal
        data = journal(ROOT / 'data/math/math500_test.jsonl')
        self.assertEqual(len(data), 500)
        for row in data:
            self.assertEqual(messages(row['problem'], SYSTEM), historical_messages(row))

    def test_frozen_e0_data_and_scorer_match(self):
        from src.training.frozen_verifier_v4 import digest, grader_identity
        protocol = read(ROOT / 'configs/project1_math500_protocol_v15.json')
        self.assertEqual(sha(ROOT / protocol['data']), protocol['data_sha256'])
        self.assertEqual(digest(grader_identity()), protocol['grader_identity_sha256'])
        self.assertEqual(protocol['max_new_tokens'], 2048)
        self.assertEqual(protocol['temperature'], 0)
        self.assertEqual(protocol['max_num_seqs'], 128)

    def test_actual_portable_bundle_without_legacy_modules(self):
        from tools.evaluate_last_e0_protocol_v30 import protocol_data
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(ROOT / 'reports/aimo_sft_20261026_v26/project1_aimo_sft_v26.zip') as archive:
                archive.extractall(tmp)
            portable = Path(tmp) / 'aimo-sft-v26'
            self.assertFalse((portable / 'src/training/rollout_pipeline_v4.py').exists())
            self.assertFalse((portable / 'src/training/runner_support_v4.py').exists())
            protocol, path, data = protocol_data(portable)
            from tools.evaluate_last_e0_protocol_v30 import journal
            old = journal(ROOT / 'data/math/math500_test.jsonl')
            self.assertEqual([(r['sample_id'], r['problem'], r['reference_answer']) for r in data],
                             [(r['sample_id'], r['problem'], r['reference_answer']) for r in old])
            self.assertEqual(len(data), 500)


if __name__ == '__main__':
    unittest.main()
