"""Synthetic contracts for the CPU saved-teacher audit, not teacher accuracy."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from scripts.audit_scene_control_teacher import audit_payload, receipt_status


def teacher(t):
    points = torch.tensor([[2., 3.], [4., 5.]])
    return dict(tracks=points[None].repeat(t, 1, 1), query_points=points,
        visibility=torch.ones(t, 2), track_confidence=torch.ones(t, 2),
        image_size_hw=[14, 14], reference_frame=0, teacher={'backend': 'synthetic'})


def payload():
    return dict(full=teacher(5), prefix=teacher(2), window_key='synthetic',
                provenance=dict(frame_ids=[10, 20, 30, 40, 50]))


class SavedTeacherAuditTests(unittest.TestCase):
    def test_nonzero_and_independent_source_destination_intersection(self):
        value = payload()
        value['prefix']['visibility'][:, 0] = 0
        value['full']['visibility'][3, 1] = 0
        report, _ = audit_payload(value)
        self.assertEqual(report['reference_eligibility']['observed_source_queries'], 1)
        self.assertEqual(report['reference_eligibility']['source_destination_pairs'], 2)
        self.assertEqual(report['reference_eligibility']['future_frames'][0]['source_frame_id'], 30)
        self.assertEqual(report['findings'], [])

    def test_zero_visibility_and_confidence_have_distinct_reasons(self):
        for key, reason in (('visibility', 'visibility_only'), ('track_confidence', 'confidence_only')):
            value = payload()
            value['full'][key].zero_()
            report, _ = audit_payload(value)
            self.assertIn('FULL_TRACK_WEIGHTS_ALL_ZERO', report['findings'])
            self.assertIn('NO_OBSERVED_TO_FUTURE_REFERENCE_SUPPORT', report['findings'])
            self.assertEqual(report['full']['per_frame'][0]['score_rejection'][reason], 2)
            self.assertTrue(report['full']['reference_query_check']['matches_queries'])

    def test_threshold_boundaries_nonfinite_and_outside(self):
        value = payload()
        value['full']['visibility'].fill_(.5)
        value['full']['track_confidence'].fill_(.2)
        value['full']['tracks'][1, 0, 0] = 14  # out of image
        value['full']['visibility'][2, 0] = float('nan')
        value['full']['track_confidence'][3, 0] = float('inf')
        report, _ = audit_payload(value)
        frames = report['full']['per_frame']
        self.assertEqual(frames[0]['filter_funnel']['positive_weight'], 2)
        self.assertEqual(frames[1]['filter_funnel']['inside'], 1)
        self.assertEqual(frames[2]['visibility']['finite_count'], 1)
        self.assertEqual(report['full']['positive_weight_count'], 7)
        self.assertIn('FULL_INVALID_PROBABILITY_VALUES', report['findings'])
        json.dumps(report, allow_nan=False)

    def test_query_alignment_and_first_frame_drift(self):
        value = payload()
        value['full']['tracks'][0, 0, 0] += 1
        report, _ = audit_payload(value)
        self.assertIn('FULL_REFERENCE_QUERY_MISMATCH', report['findings'])
        value['prefix']['query_points'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'query identities'):
            audit_payload(value)

    def test_receipt_missing_empty_verified_and_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'teacher.pt'
            torch.save(payload(), str(path))
            self.assertEqual(receipt_status(path)['status'], 'missing_receipt')
            receipt_path = Path(str(path)+'.json')
            receipt_path.write_bytes(b'')
            self.assertEqual(receipt_status(path)['status'], 'empty_receipt')
            receipt = dict(size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            receipt_path.write_text(json.dumps(receipt))
            self.assertTrue(receipt_status(path)['verified'])
            bad = copy.deepcopy(receipt)
            bad['sha256'] = '0'*64
            receipt_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                receipt_status(path)
            bad['size'] += 1
            receipt_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'size mismatch'):
                receipt_status(path)
            receipt_path.write_text('{broken')
            with self.assertRaisesRegex(ValueError, 'Invalid nonempty receipt'):
                receipt_status(path)

    def test_direct_cli_from_other_directory_multiple_files_and_compact_export(self):
        script = Path(__file__).resolve().parents[1]/'scripts/audit_scene_control_teacher.py'
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            files = [directory/'first.pt', directory/'second.pt']
            for path in files:
                torch.save(payload(), str(path))
            output = directory/'report.json'
            args = [sys.executable, str(script), '--output', str(output), '--export-npz']
            for path in files:
                args.extend(['--teacher-file', str(path)])
            result = subprocess.run(args, cwd=tmp, env=os.environ.copy(), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(len(report['cases']), 2)
            self.assertFalse(report['cases'][0]['receipt_verification']['verified'])
            with np.load(report['cases'][0]['compact_npz'], allow_pickle=False) as values:
                self.assertEqual(values['full_tracks'].shape, (5, 2, 2))
                self.assertNotIn('full_depth', values)


if __name__ == '__main__':
    unittest.main()
