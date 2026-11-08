from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import qfft_ds_common_v66 as c


def denied():
    error=PermissionError('Windows destination temporarily busy');error.winerror=5
    return error


class Tests(unittest.TestCase):
    def test_transient_commit_keeps_old_until_atomic_success(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'status.json';c.save(target,{'old':True})
            replace=Path.replace;calls=[]
            def commit(src,dst):
                calls.append(src)
                if len(calls)<3:
                    self.assertEqual(c.read(target),{'old':True});raise denied()
                return replace(src,dst)
            with patch.object(Path,'replace',commit),patch('time.sleep') as sleep:
                c.save(target,{'new':True})
            self.assertEqual(c.read(target),{'new':True});self.assertEqual(len(calls),3)
            self.assertEqual(len(set(calls)),1);self.assertEqual(sleep.call_count,2)

    def test_persistent_denial_bounded_preserves_old_and_temp(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'status.json';c.save(target,{'old':True})
            with patch.object(Path,'replace',side_effect=denied()) as replace,patch('time.sleep'):
                with self.assertRaises(PermissionError):c.save(target,{'new':True})
            self.assertEqual(replace.call_count,8);self.assertEqual(c.read(target),{'old':True})
            files=list(Path(d).glob('*.writing'));self.assertEqual(len(files),1)
            self.assertEqual(c.read(files[0]),{'new':True})

    def test_other_permission_error_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(Path,'replace',side_effect=PermissionError('permanent')) as replace,patch('time.sleep') as sleep:
                with self.assertRaises(PermissionError):c.save(Path(d)/'x.json',{})
            self.assertEqual(replace.call_count,1);sleep.assert_not_called()

if __name__=='__main__':unittest.main()
