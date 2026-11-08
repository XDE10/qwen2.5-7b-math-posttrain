import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hard_v31', ROOT / 'tools/evaluate_last_with_hard637_v31.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class HardProbeSummaryTest(unittest.TestCase):
    def test_groups_with_bad_output_are_not_eligible(self):
        records = []
        for sid, categories in [('a', ['equivalent', 'mismatch', 'mismatch', 'mismatch']),
                                ('b', ['equivalent', 'parse_failure', 'mismatch', 'mismatch'])]:
            records.extend({'sample_id': sid, 'trial': i, 'category': c, 'finish_reason': 'stop'}
                           for i, c in enumerate(categories))
        result = mod.summarize(records)
        self.assertEqual(result['questions_with_at_least_one_correct'], 2)
        self.assertEqual(result['valid_intermediate_1_to_3_of_4'], 1)
        self.assertEqual(result['groups_with_unsupported_or_abnormal_output'], 1)
        self.assertFalse(result['complete'])


if __name__ == '__main__':
    unittest.main()
