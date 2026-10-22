"""Geometry/motion contracts independent of a teacher and accelerator runtime."""
import unittest
from dataclasses import replace

import numpy as np

from utils.scene_trackmask import TrackMaskConfig, camera_compensated_track_mask


class TrackMaskTests(unittest.TestCase):
    def scene(self, motion='translation', frames=5):
        height = width = 128
        intrinsic = np.tile([[90., 0., 64.], [0., 90., 64.], [0., 0., 1.]], (frames, 1, 1))
        extrinsic = np.tile(np.eye(4), (frames, 1, 1))
        grid = np.array([(x, y) for y in (-1., -.5, 0., .5, 1.) for x in (-1., -.5, 0., .5, 1.)])
        world = np.c_[grid, np.full(len(grid), 5.)]
        tracks, depth = [], []
        yy, xx = np.mgrid[:height, :width]
        rays = np.stack(((xx-64)/90, (yy-64)/90, np.ones_like(xx)), -1)
        for t in range(frames):
            if motion == 'translation':
                extrinsic[t, 0, 3] = -.12*t
            elif motion == 'rotation':
                angle = .025*t
                c, s = np.cos(angle), np.sin(angle)
                extrinsic[t, :3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
            rotation, translation = extrinsic[t, :3, :3], extrinsic[t, :3, 3]
            camera = world @ rotation.T+translation
            homogeneous = camera @ intrinsic[t].T
            tracks.append(homogeneous[:, :2]/homogeneous[:, 2:])
            # Intersect every camera ray with the stationary world plane z=5.
            normal = rotation[:, 2]
            distance = 5.+normal@translation
            depth.append(distance/(rays@normal))
        return dict(tracks=np.array(tracks), depth=np.array(depth),
                    world_to_camera=extrinsic, intrinsics=intrinsic)

    def run_mask(self, scene, **kwargs):
        return camera_compensated_track_mask(**scene, **kwargs)

    def test_translating_camera_static_building(self):
        scene = self.scene()
        self.assertGreater(np.linalg.norm(scene['tracks'][-1]-scene['tracks'][0], axis=-1).mean(), 8.)
        out = self.run_mask(scene)
        self.assertTrue(np.all(out['track_labels'] == 0))
        self.assertFalse(out['persistent_dynamic'].any())
        self.assertLess(np.nanmax(out['residual_px']), 1e-5)

    def test_pure_rotation_static_building(self):
        out = self.run_mask(self.scene('rotation'))
        self.assertTrue(out['frame_reliable'][1:].all())
        self.assertTrue(np.all(out['track_labels'] == 0))
        self.assertLess(np.nanmax(out['residual_px']), .001)

    def test_independently_moving_point_survives_camera_compensation(self):
        scene = self.scene()
        scene['tracks'][1:, 12, 1] += np.arange(1, 5)*2.
        out = self.run_mask(scene)
        self.assertEqual(out['track_labels'][12], 1)
        self.assertEqual(out['diagnostics']['dynamic_tracks'], 1)
        self.assertTrue(np.all(out['track_labels'][np.arange(25) != 12] == 0))
        self.assertTrue(out['labels'][1:, 12].max() == 1)
        self.assertTrue(out['bidirectional_agreement'][1:, 12].all())
        self.assertTrue((out['reverse_residual_px'][1:, 12] > 1.).all())

    def test_localized_wrong_query_depth_does_not_make_static_building_dynamic(self):
        scene = self.scene()
        # The scene and tracks remain static. Only one teacher depth patch is
        # wrong: the former one-way test falsely called this a moving point.
        scene['depth'][0, 62:67, 62:67] = 3.
        out = self.run_mask(scene)
        self.assertGreater(out['residual_px'][-1, 12], 5.)
        self.assertLess(np.nanmax(out['reverse_residual_px'][:, 12]), 1e-5)
        self.assertEqual(out['track_labels'][12], -1)
        self.assertTrue(np.all(out['labels'][:, 12] == -1))
        self.assertTrue(np.all(out['reason_bits'][1:, 12] & 128))
        self.assertFalse(out['persistent_dynamic'].any())

    def test_localized_wrong_target_depth_is_unknown_not_false_static(self):
        scene = self.scene()
        for t in range(1, len(scene['tracks'])):
            x, y = np.floor(scene['tracks'][t, 12]).astype(int)
            scene['depth'][t, y-2:y+4, x-2:x+4] = 3.
        out = self.run_mask(scene)
        self.assertLess(np.nanmax(out['residual_px'][:, 12]), 1e-5)
        self.assertGreater(out['reverse_residual_px'][-1, 12], 3.)
        self.assertEqual(out['track_labels'][12], -1)
        self.assertTrue(np.all(out['labels'][:, 12] == -1))
        self.assertTrue(np.any(out['reason_bits'][1:, 12] & 128))
        self.assertFalse(out['persistent_dynamic'].any())

    def test_low_confidence_unknown_not_dynamic(self):
        scene = self.scene()
        scene['tracks'][1:, 12, 1] += np.arange(1, 5)*2.
        confidence = np.ones((5, 25)); confidence[:, 12] = .01
        out = self.run_mask(scene, track_confidence=confidence)
        self.assertEqual(out['track_labels'][12], -1)
        self.assertFalse(out['valid'][:, 12].any())
        self.assertTrue(np.all(out['reason_bits'][:, 12] & 2))

    def test_occluded_rigid_projection_is_unknown(self):
        scene = self.scene(motion='none')
        # A nearer surface covers the old query ray, so no motion claim is safe.
        scene['depth'][1:, 62:67, 62:67] = 2.
        out = self.run_mask(scene)
        self.assertEqual(out['track_labels'][12], -1)
        self.assertTrue(np.all(out['reason_bits'][1:, 12] & 16))
        self.assertFalse(out['persistent_dynamic'].any())

    def test_depth_translation_joint_scale_invariance(self):
        scene = self.scene()
        scene['tracks'][1:, 12, 1] += np.arange(1, 5)*2.
        first = self.run_mask(scene)
        scene['depth'] *= 37.
        scene['world_to_camera'][:, :3, 3] *= 37.
        second = self.run_mask(scene)
        for key in ('labels', 'track_labels', 'valid', 'frame_reliable', 'bidirectional_agreement'):
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_allclose(first['residual_px'], second['residual_px'], atol=1e-5)
        np.testing.assert_allclose(first['reverse_residual_px'], second['reverse_residual_px'], atol=1e-5)
        np.testing.assert_allclose(first['effective_thresholds_px'], second['effective_thresholds_px'], atol=1e-5)
        np.testing.assert_allclose(first['reverse_effective_thresholds_px'], second['reverse_effective_thresholds_px'], atol=1e-5)

    def test_world_coordinate_gauge_and_nonzero_reference(self):
        scene = self.scene('rotation')
        first = self.run_mask(scene, reference_frame=2)
        # An arbitrary world-coordinate redefinition must not change motion.
        angle = .3
        c, s = np.cos(angle), np.sin(angle)
        gauge = np.eye(4)
        gauge[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
        gauge[:3, 3] = [2., -1., .4]
        scene['world_to_camera'] = scene['world_to_camera'] @ np.linalg.inv(gauge)
        second = self.run_mask(scene, reference_frame=2)
        np.testing.assert_array_equal(first['labels'], second['labels'])
        np.testing.assert_allclose(first['residual_px'], second['residual_px'], atol=1e-5)
        np.testing.assert_allclose(first['reverse_residual_px'], second['reverse_residual_px'], atol=1e-5)
        self.assertTrue(np.all(second['track_labels'] == 0))
        self.assertTrue(np.all(second['labels'][2] == -1))

    def test_visibility_and_relative_depth_confidence_are_unknown_evidence(self):
        scene = self.scene('none')
        visibility = np.ones((5, 25)); visibility[:, 0] = 0.
        # Scores deliberately exceed one: depth confidence is not a probability.
        confidence = np.full_like(scene['depth'], 100.)
        confidence[:, 62:67, 62:67] = 2.
        out = self.run_mask(scene, visibility=visibility, depth_confidence=confidence)
        self.assertEqual(out['track_labels'][0], -1)
        self.assertEqual(out['track_labels'][12], -1)
        self.assertTrue(np.all(out['reason_bits'][:, 0] & 2))
        self.assertTrue(np.all(out['reason_bits'][:, 12] & 4))

    def test_global_camera_failure_does_not_label_whole_image_dynamic(self):
        scene = self.scene(motion='none')
        scene['world_to_camera'][1:, 0, 3] = -.8
        out = self.run_mask(scene)
        self.assertFalse(out['frame_reliable'].any())
        self.assertTrue(np.all(out['track_labels'] == -1))
        self.assertFalse(out['persistent_dynamic'].any())
        self.assertTrue(np.all(out['reason_bits'][1:] & 32))

    def test_one_frame_jump_is_unknown_not_persistent_motion(self):
        scene = self.scene()
        scene['tracks'][2, 12, 1] += 8.
        out = self.run_mask(scene)
        self.assertFalse(out['persistent_dynamic'].any())
        self.assertEqual(out['track_labels'][12], -1)
        self.assertEqual(out['labels'][2, 12], -1)
        self.assertTrue(out['reason_bits'][2, 12] & 64)

    def test_reference_frame_has_no_independent_motion_evidence(self):
        out = self.run_mask(self.scene())
        self.assertTrue(np.isnan(out['residual_px'][0]).all())
        self.assertTrue(np.all(out['labels'][0] == -1))
        self.assertTrue(out['diagnostics']['uses_future_frames'])

    def test_depth_edge_and_out_of_bounds_unknown(self):
        scene = self.scene('none')
        scene['tracks'][:, 0] = [-1., 20.]
        scene['tracks'][:, 12] = [64.5, 64.5]
        scene['depth'][:, 65:, :] = 10.
        out = self.run_mask(scene)
        self.assertEqual(out['track_labels'][0], -1)
        self.assertEqual(out['track_labels'][12], -1)
        self.assertTrue(np.all(out['reason_bits'][:, 0] & 1))
        self.assertTrue(np.all(out['reason_bits'][:, 12] & 8))

    def test_positive_depth_and_shape_validation(self):
        scene = self.scene()
        scene['depth'][:] = -1
        out = self.run_mask(scene)
        self.assertFalse(out['valid'].any())
        with self.assertRaises(ValueError):
            self.run_mask(self.scene(), visibility=np.ones((5, 24)))
        with self.assertRaises(ValueError):
            self.run_mask(self.scene(), config=replace(TrackMaskConfig(), depth_uncertainty_relative=1.))


if __name__ == '__main__':
    unittest.main()
