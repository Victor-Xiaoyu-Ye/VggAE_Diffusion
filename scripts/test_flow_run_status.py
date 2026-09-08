#!/usr/bin/env python3
"""Pure-standard-library regression tests for completion vs memory quality."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.flow_run_status import FlowRunStatus, atomic_json, verify_run


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        args = SimpleNamespace(max_steps=2000, stop_after_steps=0,
                               prediction='direct_velocity', noise_mode='random')
        self.status = FlowRunStatus(self.root, args)

    def update(self, **values):
        with contextlib.redirect_stdout(io.StringIO()):
            self.status.update(**values)

    def test_incomplete_run_cannot_succeed(self):
        self.update(step=500, phase='evaluating')
        with self.assertRaises(ValueError):
            verify_run(self.root, 2000)
        self.update(status='completed', step=500, checkpoint='checkpoint_final.pt')
        (self.root/'checkpoint_final.pt').write_bytes(b'placeholder')
        with self.assertRaises(ValueError):
            verify_run(self.root, 2000)

    def test_quality_failure_does_not_mean_training_failure(self):
        self.update(status='completed', step=2000, checkpoint='checkpoint_final.pt',
                    quality_gate='failed')
        (self.root/'checkpoint_final.pt').write_bytes(b'placeholder')
        self.assertEqual(verify_run(self.root, 2000)['quality_gate'], 'failed')

    def test_paused_is_only_valid_when_requested(self):
        self.update(status='paused', step=1, checkpoint='checkpoint_latest.pt')
        (self.root/'checkpoint_latest.pt').write_bytes(b'placeholder')
        self.assertEqual(verify_run(self.root, 2000, 1)['status'], 'paused')
        with self.assertRaises(ValueError):
            verify_run(self.root, 2000)

    def test_missing_or_temporary_checkpoint_not_accepted(self):
        self.update(status='completed', step=2000, checkpoint='checkpoint_final.pt')
        (self.root/'checkpoint_final.pt.tmp-123').write_bytes(b'incomplete')
        with self.assertRaises(ValueError):
            verify_run(self.root, 2000)

    def test_atomic_json_leaves_no_temporary_file(self):
        atomic_json(self.root/'status.json', {'status': 'running'})
        atomic_json(self.root/'status.json', {'status': 'completed'})
        self.assertEqual(json.loads((self.root/'status.json').read_text())['status'], 'completed')
        self.assertFalse(list(self.root.glob('*.tmp-*')))


if __name__ == '__main__':
    unittest.main()
