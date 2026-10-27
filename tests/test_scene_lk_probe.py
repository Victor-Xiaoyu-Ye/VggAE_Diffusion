"""CPU synthetic LK checks. Passing these does not validate real data or NPU."""
import copy
import json
import unittest
from unittest import mock

import cv2
import numpy as np
import torch

from utils.scene_lk_probe import LKProbeConfig, probe_lk


def fixture(shifts=((0, 0), (3, 2), (6, 4)), queries=None):
    rng = np.random.default_rng(20261027)
    base = cv2.GaussianBlur(rng.integers(0, 256, (192, 224), dtype=np.uint8), (3, 3), .6)
    images = [cv2.warpAffine(base, np.float32([[1, 0, x], [0, 1, y]]), (224, 192),
                            borderMode=cv2.BORDER_REFLECT_101) for x, y in shifts]
    frames = torch.from_numpy(np.stack(images)[:, None].repeat(3, axis=1).astype(np.float32) / 255.)
    q = torch.tensor(queries if queries is not None else [[60., 60.], [100., 70.], [140., 110.], [80., 130.]])
    t = len(shifts)
    teacher = dict(query_points=q, tracks=q[None].repeat(t, 1, 1),
        visibility=torch.zeros(t, len(q)), track_confidence=torch.zeros(t, len(q)),
        depth=torch.ones(t, 192, 224), depth_conf=torch.ones(t, 192, 224),
        extrinsics=torch.eye(3, 4)[None].repeat(t, 1, 1),
        intrinsics=torch.eye(3)[None].repeat(t, 1, 1), image_size_hw=[192, 224],
        reference_frame=0, teacher_context_frames=t, teacher={'backend': 'synthetic_stream', 'coverage': {'matched': 3}},
        camera_convention='OpenCV_world_to_camera')
    return frames, teacher


class LKProbeTests(unittest.TestCase):
    def test_translation_and_preserved_geometry(self):
        frames, teacher = fixture()
        before = copy.deepcopy(teacher)
        out, report = probe_lk(frames, teacher)
        expected = teacher['query_points'][None] + torch.tensor([[0., 0.], [3., 2.], [6., 4.]])[:, None]
        self.assertLess(float((out['tracks'] - expected).abs().max()), .15)
        self.assertTrue(bool((out['visibility'] == 1).all()))
        self.assertTrue(bool((out['track_confidence'] >= .2).all()))
        self.assertTrue(bool((out['track_confidence'] < 1).all()))
        for key in ('depth', 'depth_conf', 'extrinsics', 'intrinsics', 'query_points'):
            self.assertIs(out[key], teacher[key])
            torch.testing.assert_close(teacher[key], before[key])
        self.assertEqual(teacher['teacher'], before['teacher'])
        torch.testing.assert_close(teacher['tracks'], before['tracks'])
        self.assertEqual(out['teacher']['backend'], 'diagnostic_lk')
        self.assertEqual(out['teacher']['geometry_backend'], 'synthetic_stream')
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['qualified_fraction'], 1.)
        json.dumps(report, allow_nan=False)

    def test_repeat_frames_and_boundaries(self):
        frames, teacher = fixture(shifts=((0, 0),) * 3,
                                  queries=[[60., 60.], [0., 0.], [223., 191.], [-2., 8.]])
        out, report = probe_lk(frames, teacher)
        torch.testing.assert_close(out['tracks'][:, 0], teacher['tracks'][:, 0], atol=.01, rtol=0)
        self.assertTrue(bool((out['visibility'][:, 0] == 1).all()))
        self.assertEqual(int(out['visibility'][:, 1:].sum()), 0)
        self.assertEqual(int(out['track_confidence'][:, 1:].sum()), 0)
        self.assertEqual(report['per_frame'][0]['inside_patch'], 1)

    def test_textureless_and_disappeared_stay_unknown(self):
        frames, teacher = fixture(shifts=((0, 0),) * 3)
        blank, report = probe_lk(torch.zeros_like(frames), teacher)
        self.assertEqual(float(blank['visibility'].sum()), 0.)
        self.assertEqual(report['qualified_fraction'], 0.)
        frames[1] = 0
        out, _ = probe_lk(frames, teacher)
        self.assertGreater(float(out['visibility'][0].sum()), 0.)
        self.assertEqual(float(out['visibility'][1:].sum()), 0.)
        self.assertEqual(float(out['track_confidence'][1:].sum()), 0.)
        self.assertTrue(bool(torch.isfinite(out['tracks']).all()))

    def test_failed_forward_backward_is_unknown(self):
        frames, teacher = fixture()
        original = cv2.calcOpticalFlowPyrLK
        count = [0]

        def shifted_backward(*args, **kwargs):
            count[0] += 1
            points, status, error = original(*args, **kwargs)
            if count[0] == 2:
                points = points + 10.
            return points, status, error

        with mock.patch('cv2.calcOpticalFlowPyrLK', side_effect=shifted_backward):
            out, report = probe_lk(frames, teacher)
        self.assertEqual(report['per_frame'][1]['forward_backward_pass'], 0)
        self.assertEqual(float(out['visibility'][1:].sum()), 0.)
        self.assertEqual(float(out['track_confidence'][1:].sum()), 0.)

    def test_photometric_rejection_even_with_consistent_coordinates(self):
        frames, teacher = fixture(shifts=((0, 0),) * 3)
        frames[1] = 1 - frames[0]

        def identity_flow(previous_image, current_image, points, initial, **kwargs):
            return points.copy(), np.ones((len(points), 1), np.uint8), np.zeros((len(points), 1), np.float32)

        with mock.patch('cv2.calcOpticalFlowPyrLK', side_effect=identity_flow):
            out, report = probe_lk(frames, teacher)
        self.assertEqual(report['per_frame'][1]['forward_backward_pass'], 4)
        self.assertEqual(report['per_frame'][1]['texture_pass'], 4)
        self.assertEqual(report['per_frame'][1]['photometric_pass'], 0)
        self.assertEqual(float(out['visibility'][1:].sum()), 0.)
        self.assertEqual(float(out['track_confidence'][1:].sum()), 0.)

    def test_tracker_error_and_prefix_causality(self):
        frames, teacher = fixture()
        full, _ = probe_lk(frames, teacher)
        prefix = dict(teacher)
        for key in ('depth', 'depth_conf', 'extrinsics', 'intrinsics', 'tracks', 'visibility', 'track_confidence'):
            prefix[key] = teacher[key][:2]
        prefix['teacher_context_frames'] = 2
        short, _ = probe_lk(frames[:2], prefix)
        for key in ('tracks', 'visibility', 'track_confidence'):
            torch.testing.assert_close(full[key][:2], short[key])
        with mock.patch('cv2.calcOpticalFlowPyrLK', side_effect=RuntimeError('synthetic LK failure')):
            out, report = probe_lk(frames, teacher)
        self.assertEqual(report['status'], 'completed_with_tracking_errors')
        self.assertEqual(len(report['errors']), 1)
        self.assertEqual(float(out['visibility'][1:].sum()), 0.)
        self.assertEqual(float(out['track_confidence'][1:].sum()), 0.)

    def test_mismatched_window_and_bad_input_rejected(self):
        frames, teacher = fixture()
        with self.assertRaisesRegex(ValueError, 'match'):
            probe_lk(frames[:2], teacher)
        with self.assertRaisesRegex(ValueError, 'reference_frame'):
            probe_lk(frames, {**teacher, 'reference_frame': 1})
        with self.assertRaisesRegex(ValueError, 'floating RGB'):
            probe_lk(frames.numpy().astype(np.uint8), teacher)
        with self.assertRaisesRegex(ValueError, 'floating RGB'):
            probe_lk(frames.to(torch.uint8), teacher)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            probe_lk(frames, teacher, config=LKProbeConfig(fb_max_pixels=0))


if __name__ == '__main__':
    unittest.main()
