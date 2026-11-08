"""Offline process lifecycle, stale-diagnostic and thread-isolation regression tests."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/qfft'))
import build_qfft_ds_dataset_v66 as b
import continue_qfft_all_v71 as s
import qfft_ds_common_v66 as c


class ProcessTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='win32','Windows lifecycle check')
    def test_child_has_no_console(self):
        flags=s.background_subprocess_kwargs()
        self.assertTrue(flags['creationflags'] & subprocess.CREATE_NO_WINDOW)
        p=subprocess.run([sys.executable,'-c','import ctypes;print(ctypes.windll.kernel32.GetConsoleWindow())'],
                         capture_output=True,text=True,check=True,timeout=15,**flags)
        self.assertEqual(p.stdout.strip(),'0')

    def test_current_exit_replaces_and_archives_stale_network_pause(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';control=Path(d)/'control';out.mkdir();control.mkdir()
            original={'failures':[{'sample_id':'old0266','message':'old network error'}]}
            c.save(out/'last_pause.json',original)
            log=control/'generation.log';log.write_text('recent invocation')
            with patch.object(s,'OUT',out),patch.object(s,'CONTROL',control):
                s.record_process_exit(3221225786,log,3221225786)
            current=c.read(out/'last_pause.json')
            self.assertEqual(current['scope'],'supervisor_process_exit')
            self.assertEqual(current['failures'][0]['error_type'],'WindowsProcessInterrupted')
            backups=list((control/'pause_history').glob('*.json'))
            self.assertEqual(len(backups),1)
            self.assertEqual(c.read(backups[0]),original)

    def test_four_threads_have_independent_reused_judge_modules(self):
        barrier=threading.Barrier(4)
        def make_module(spec):
            module=SimpleNamespace()
            module.grade_bounded=lambda *args:{'module_id':id(module),'correct':True}
            return module
        spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None))
        def work(i):
            barrier.wait()
            return [b.grade('2','2')['module_id'] for _ in range(2)]
        with patch.object(b.importlib.util,'spec_from_file_location',return_value=spec), \
             patch.object(b.importlib.util,'module_from_spec',side_effect=make_module) as creator:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results=list(pool.map(work,range(4)))
        self.assertTrue(all(a==z for a,z in results))
        self.assertEqual(len({r[0] for r in results}),4)
        self.assertEqual(creator.call_count,4)

    def test_real_native_verifier_reused_without_changing_scores(self):
        try:
            a=b.grade(r'\boxed{2}','2')
            worker=b._JUDGE_LOCAL.module._NATIVE_WORKER
            pid=worker.proc.pid
            z=b.grade(r'\boxed{3}','2')
            self.assertTrue(a['correct'])
            self.assertFalse(z['correct'])
            self.assertEqual(worker.proc.pid,pid)
        finally:
            module=getattr(b._JUDGE_LOCAL,'module',None)
            if module is not None and getattr(module,'_NATIVE_WORKER',None) is not None:
                module._NATIVE_WORKER.close()


if __name__ == '__main__': unittest.main()
