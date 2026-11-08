import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot', ROOT / 'tools/run_pilot16_rope_decode_v39.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotTests(unittest.TestCase):
    def test_selection_is_fixed_unique_and_stratified(self):
        selection = pilot.read(ROOT / 'reports/rope_decode_pilot_20261026_v39/pilot16_rope_decode_v39.json')
        data = pilot.validate_selection(selection)
        source = ROOT / 'artifacts/final_eval_local_snapshot/evaluations/final_dev256_32k_v32/predictions.jsonl'
        self.assertEqual(pilot.sha(source), selection['source_predictions_sha256'])
        dev = {r['sample_id']: r for r in pilot.rows(ROOT / 'reports/aimo_sft_20261026_v26/dev.jsonl')}
        for row in data:
            self.assertEqual(row['problem'], dev[row['sample_id']]['problem'])
            self.assertEqual(row['answer'], dev[row['sample_id']]['answer'])

    def test_only_decoding_recipe_changes_within_rope_pair(self):
        for sid in ['math/train/1', 'math/train/2']:
            greedy = pilot.sampling_config('greedy', sid)
            sample = pilot.sampling_config('sample', sid)
            for field in ('n', 'max_tokens', 'top_k', 'repetition_penalty', 'seed'):
                self.assertEqual(greedy[field], sample[field])
            self.assertEqual(greedy['max_tokens'], 8192)
            self.assertEqual(greedy['temperature'], 0)
            self.assertEqual(sample['temperature'], 0.6)
            self.assertEqual(sample['top_p'], 0.95)
            self.assertEqual(len(pilot.ARMS), 4)

    def test_repetition_detector(self):
        self.assertGreaterEqual(pilot.tail_repeat(list(range(32)) * 10), 4)
        self.assertEqual(pilot.tail_repeat(list(range(100))), 1)


if __name__ == '__main__':
    unittest.main()
