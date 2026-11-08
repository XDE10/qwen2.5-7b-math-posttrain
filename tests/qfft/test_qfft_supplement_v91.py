from pathlib import Path
import sys
import importlib.machinery
import importlib.util
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
loader=importlib.machinery.SourceFileLoader('v91',str(Path(__file__).resolve().parents[1]/'tools/supplement_incomplete_high_v91.pyw'))
spec=importlib.util.spec_from_loader(loader.name,loader);m=importlib.util.module_from_spec(spec);loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_repaired_ids_excluded_from_incomplete_selection(self):
        original={1:{'structurally_complete':False},2:{'structurally_complete':False},3:{'structurally_complete':True}}
        view=m.effective_view(original,{1:{'structurally_complete':True}})
        self.assertEqual([i for i,x in view.items() if not x['structurally_complete']],[2])
        self.assertFalse(original[1]['structurally_complete'])

    def test_never_promote_incomplete_repair(self):
        with self.assertRaises(ValueError):m.effective_view({}, {1:{'structurally_complete':False}})

    def test_wire_high_blind_prompt_and_32k_unchanged(self):
        row={'problem':'fixture problem','reference_answer':'DO_NOT_SEND_GOLD','difficulty':{'reasoning_effort':'max'}}
        for provider in ('go','official'):
            body=m.wire_payload(row,provider)
            self.assertEqual(body['reasoning_effort'],'high');self.assertEqual(body['max_tokens'],32768)
            self.assertTrue(body['stream']);self.assertEqual(len(body['messages']),2)
            self.assertEqual(body['messages'][1]['content'],'fixture problem')
            self.assertNotIn('DO_NOT_SEND_GOLD',str(body));self.assertEqual(body['thinking'],{'type':'enabled'})

    def test_prompt_variant_keeps_math_requirements(self):
        self.assertIn('Avoid repetitive self-doubt loops.',m.SYSTEM)
        self.assertIn('recompute every affected coefficient',m.SYSTEM)
        self.assertIn('equality is attainable',m.SYSTEM)
        self.assertNotIn('Use alternative approaches and independent checks',m.SYSTEM)
        self.assertNotEqual(m.SYSTEM,m.c.SYSTEM)

    def test_structure_not_quality_or_hard_truncation(self):
        self.assertEqual(m.outcome({'structurally_complete':True,'fits_current_training_window':False}),'complete_over_16k')
        self.assertEqual(m.outcome({'structurally_complete':False}),'incomplete')

if __name__=='__main__':unittest.main()
