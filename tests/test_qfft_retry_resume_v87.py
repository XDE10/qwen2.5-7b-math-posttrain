from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_qfft_ds_dataset_v66 as b
import qfft_ds_common_v66 as c


class Tests(unittest.TestCase):
    def fixture(self,out):
        p=out/'calls/10295/go/request.json'
        c.save(p,{'identity':'fixture','sample_id':'row/295','body':{}})
        c.save(out/'reviewed_unresolved_v73.json',{'identity':'fixture','action':'defer_without_retry',
            'records':[{'index':10295,'provider':'go','sample_id':'row/295','request_sha256':c.sha_file(p)}]})
        return p

    def test_original_default_still_rejects_high_slot(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.fixture(out)
            with self.assertRaises(RuntimeError):b.reviewed_unknowns(out,'fixture')

    def test_explicit_slot_keeps_hash_and_response_guards(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);p=self.fixture(out)
            self.assertEqual(set(b.reviewed_unknowns(out,'fixture',{10295})),{10295})
            with self.assertRaises(RuntimeError):b.reviewed_unknowns(out,'fixture',{10301})
            c.save(p,{'changed':True})
            with self.assertRaises(RuntimeError):b.reviewed_unknowns(out,'fixture',{10295})

if __name__=='__main__':unittest.main()
