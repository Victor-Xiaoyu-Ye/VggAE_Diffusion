import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_window_ae


class DomainReplayTests(unittest.TestCase):
    def arguments(self, root):
        return ['audit_window_ae', '--manifest', 'unused', '--stats', 'unused',
                '--r7_ckpt', 'unused', '--encoder_ckpt', 'unused',
                '--output_dir', str(root / 'out'), '--report_only']

    def test_report_only_does_not_hide_runtime_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sys.argv', self.arguments(root)), \
                    patch.object(audit_window_ae, 'get_device_name', return_value='cpu'):
                with self.assertRaisesRegex(RuntimeError, 'requires an accelerator'):
                    audit_window_ae.main()
            result = json.loads((root / 'out/audit.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertIn('requires an accelerator', result['error'])

    def test_duplicate_baseline_ids_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / 'baseline.json'
            baseline.write_text(json.dumps({'clips': [{'video_id': 'same'}] * 2}))
            argv = self.arguments(root) + ['--clips', '2', '--baseline_json', str(baseline)]
            with patch('sys.argv', argv):
                with self.assertRaises(SystemExit) as caught:
                    audit_window_ae.main()
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse((root / 'out').exists())
