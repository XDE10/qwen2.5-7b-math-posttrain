"""Explicit circuit review preserves lifetime history and keeps subsequent limits."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_qfft_ds_dataset_v66 as b
import qfft_ds_common_v66 as c


def policy():
    return {'identity':'id','consecutive_go_5xx':0,'total_go_5xx':1,
            'consecutive_go_transient':0,'total_go_transient':11,'reviewed_go_transient_total':11}


class ReviewTests(unittest.TestCase):
    @patch.object(b,'check_time')
    def test_review_preserves_lifetime_and_next_three_failures_still_stop(self,_):
        def transport(*args):raise b.Rejected(503,'unavailable')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);c.save(out/'server_error_policy_v74.json',policy())
            r=b.Router(out,'id',transport=transport,key_loader=lambda _:'fixture')
            self.assertEqual(r.server_policy['total_go_transient'],11)
            for i in range(3):
                row={'index':i,'sample_id':f'test/{i}','batch':1,'problem':'1+1?',
                     'difficulty':{'reasoning_effort':'high'}}
                with self.assertRaises(b.DeferredGoServerFailure) as caught:r.generate(row)
                self.assertEqual(caught.exception.circuit_open,i==2)
            self.assertEqual(r.server_policy['total_go_transient'],14)
            self.assertEqual(b.unreviewed_transient_total(r.server_policy),3)
            with self.assertRaises(RuntimeError):b.Router(out,'id')

    def test_invalid_review_baseline_cannot_disable_circuit(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);p=policy();p['reviewed_go_transient_total']=12
            c.save(out/'server_error_policy_v74.json',p)
            with self.assertRaises(RuntimeError):b.Router(out,'id')


if __name__=='__main__':unittest.main()
