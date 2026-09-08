"""Local failure-injection for the same publisher used by ModelArts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import window_run_io as io


class IOTests(unittest.TestCase):
    def test_double_write_incremental_and_failure(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); source=p/'source'; source.mkdir()
            (source/'checkpoint_latest.pt').write_bytes(b'complete')
            (source/'checkpoint_latest.pt.tmp').write_bytes(b'incomplete')
            (source/'metrics.jsonl').write_text('{"step":1}\n')
            roots=[str(p/'current'),str(p/'mirror')]
            io.publish(source,roots)
            for root in roots:
                self.assertEqual((Path(root)/'checkpoint_latest.pt').read_bytes(),b'complete')
                self.assertFalse((Path(root)/'checkpoint_latest.pt.tmp').exists())
            actual=io.copy_file; copied=[]
            def record(src,dst): copied.append(dst); return actual(src,dst)
            with patch.object(io,'copy_file',record): io.publish(source,roots)
            self.assertTrue(all(Path(x).name=='publication_status.json' for x in copied))
            (source/'metrics.jsonl').write_text('{"step":2}\n')
            def fail_one(src,dst):
                if str(p/'current') in dst: raise OSError('simulated unavailable output mount')
                return actual(src,dst)
            with patch.object(io,'copy_file',fail_one), self.assertRaises(RuntimeError):
                io.publish(source,roots)
            self.assertIn('2',(p/'mirror/metrics.jsonl').read_text())
            self.assertEqual(json.loads((source/'publication_status.json').read_text())['status'],'failed')
            io.publish(source,roots)
            self.assertIn('2',(p/'current/metrics.jsonl').read_text())

    def test_double_read_and_atomic_destination(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); (p/'mirror.pt').write_bytes(b'checkpoint')
            io.stage(p/'staged.pt',[str(p/'missing.pt'),str(p/'mirror.pt')])
            self.assertEqual((p/'staged.pt').read_bytes(),b'checkpoint')
            with self.assertRaises(FileNotFoundError): io.stage(p/'staged.pt',[str(p/'missing.pt')])
            self.assertEqual((p/'staged.pt').read_bytes(),b'checkpoint')


if __name__=='__main__': unittest.main()
