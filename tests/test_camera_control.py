"""Camera-coordinate tests; synthetic geometry is not dataset-quality evidence."""
import unittest

import numpy as np

from utils.camera_control import (
    build_window_camera_controls, normalize_quaternions,
    normalized_intrinsics_to_pixel, parse_indexes_text, quaternion_to_matrix,
    slerp_xyzw, pack_window_camera_features,
)


def poses_from_cameras(centers, quaternions):
    q = normalize_quaternions(quaternions)
    r = quaternion_to_matrix(q)
    w2c_q = q * np.array([-1., -1., -1., 1.])
    translation = -np.einsum('nji,nj->ni', r, np.asarray(centers))
    return np.concatenate((translation, w2c_q), axis=1)


class CameraControlTests(unittest.TestCase):
    def setUp(self):
        self.k = np.tile([.9, 1.2, .5, .5], (3, 1))
        self.poses = poses_from_cameras(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0, 1]]*3)

    def test_indexes_are_original_frames_not_row_positions(self):
        np.testing.assert_array_equal(parse_indexes_text('\ufeff0\n6\n12\n', 3), [0, 6, 12])
        np.testing.assert_array_equal(
            parse_indexes_text('# total 3 indexes\n0 0\n1 5\n2 10\n', 3), [0, 5, 10])
        for text in ('0\n0\n12', '0\n12\n6', '0,6,12', '0 6 12', '0\n1.0\n2', '-1\n6\n12'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_indexes_text(text, 3)
        with self.assertRaises(ValueError):
            parse_indexes_text('0\n6\n12', 2)

    def test_annotation_row_mapping_is_validated(self):
        for text in ('# total 4 indexes\n0 0\n1 5\n2 10',
                     '1 0\n2 5\n3 10', '0 0\n0 5\n2 10',
                     '0 0\n2 5\n1 10', '0 0\n5\n2 10',
                     '0 0\n1 5\n2 5', '0 0\n1 10\n2 5'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_indexes_text(text, 3)

    def test_alignment_w2c_sign_anchor_and_original_timestamps(self):
        result = build_window_camera_controls(self.poses, self.k, [0, 6, 12], [3, 6, 9])
        np.testing.assert_allclose(result['relative_c2w'][:, :3, 3], [[0, 0, 0], [.5, 0, 0], [1, 0, 0]])
        np.testing.assert_array_equal(result['annotation_brackets'], [[0, 6], [6, 6], [6, 12]])
        np.testing.assert_allclose(result['interpolation_weights'], [.5, 0, .5])
        np.testing.assert_allclose(result['relative_c2w'][0], np.eye(4))
        self.assertEqual(result['contract']['translation_units'], 'source_annotation_units_unverified')

    def test_shortest_rotation_arc_and_camera_center_interpolation(self):
        half = np.sqrt(.5)
        q = np.array([[0, 0, 0, 1], [0, 1, 0, 0]])
        poses = poses_from_cameras([[0, 0, 0], [2, 0, 0]], q)
        result = build_window_camera_controls(poses, self.k[:2], [0, 10], [0, 5, 10])
        np.testing.assert_allclose(result['relative_c2w'][1, :3, :3],
                                   quaternion_to_matrix([0, half, 0, half]), atol=1e-14)
        np.testing.assert_allclose(result['relative_c2w'][1, :3, 3], [1, 0, 0])
        np.testing.assert_allclose(quaternion_to_matrix(slerp_xyzw(q[0], -q[0], .5)), np.eye(3))

    def test_world_gauge_changes_cancel_and_magnitude_is_preserved(self):
        relative = build_window_camera_controls(self.poses, self.k, [0, 6, 12], [0, 6, 12])
        half = np.sqrt(.5)
        q = [0, 0, half, half]
        rotation = quaternion_to_matrix(q)
        centers = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]) @ rotation.T + [9, 7, 3]
        changed = poses_from_cameras(centers, [q]*3)
        transformed = build_window_camera_controls(changed, self.k, [0, 6, 12], [0, 6, 12])
        np.testing.assert_allclose(relative['relative_c2w'], transformed['relative_c2w'], atol=1e-14)
        twice = self.poses.copy()
        twice[:, :3] *= 2
        scaled = build_window_camera_controls(twice, self.k, [0, 6, 12], [0, 6, 12])
        np.testing.assert_allclose(scaled['relative_c2w'][:, :3, 3], 2*relative['relative_c2w'][:, :3, 3])

    def test_normalized_intrinsics_survive_anisotropic_resize(self):
        original = normalized_intrinsics_to_pixel(self.k, 1920, 1080)
        resized = normalized_intrinsics_to_pixel(self.k, 518, 518)
        transform = np.diag([518/1920, 518/1080, 1])
        np.testing.assert_allclose(resized, transform @ original)

    def test_invalid_alignment_cannot_extrapolate_or_silently_repair(self):
        for indices in ([0, 6, 13], [-1, 6, 12], [0., 6., 12.], [0, 6, 6]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                build_window_camera_controls(self.poses, self.k, [0, 6, 12], indices)
        with self.assertRaises(ValueError):
            build_window_camera_controls(self.poses, self.k, [0, 6, 12], [0, 3, 6], 5)
        with self.assertRaises(ValueError):
            build_window_camera_controls(self.poses[:2], self.k, [0, 6, 12], [0, 6])
        bad = self.poses.copy()
        bad[1, 3:] = 0
        with self.assertRaises(ValueError):
            build_window_camera_controls(bad, self.k, [0, 6, 12], [0, 6])
        bad = self.k.copy()
        bad[1, 0] = -1
        with self.assertRaises(ValueError):
            build_window_camera_controls(self.poses, bad, [0, 6, 12], [0, 6])

    def test_quaternion_normalization_is_stable(self):
        reference = normalize_quaternions([1, 2, 3, 4])
        for factor in (1e-200, 1e200, -8):
            np.testing.assert_allclose(quaternion_to_matrix(np.array([1, 2, 3, 4])*factor),
                                       quaternion_to_matrix(reference))
        for invalid in ([0, 0, 0, 0], [0, 0, 0, np.inf], [0, 0, np.nan, 1]):
            with self.assertRaises(ValueError):
                normalize_quaternions(invalid)

    def test_feature_packing_keeps_rotation_columns_scale_and_seconds(self):
        half = np.sqrt(.5)
        poses = poses_from_cameras([[0, 0, 0], [4, 0, 0]],
                                  [[0, 0, 0, 1], [0, half, 0, half]])
        controls = build_window_camera_controls(poses, self.k[:2], [10, 18], list(range(10, 19)))
        features = pack_window_camera_features(controls, fps=8, translation_scale=2)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(features.shape, (9, 14))
        np.testing.assert_allclose(features[0, :6], [1, 0, 0, 0, 1, 0])
        np.testing.assert_allclose(features[-1, :6], [0, 0, -1, 0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(features[:, 6], np.arange(9)/4)
        np.testing.assert_allclose(features[:, 9:13], np.tile(self.k[0], (9, 1)))
        np.testing.assert_allclose(features[:, 13], np.arange(9)/8)
        # Packing cannot mutate source-scale camera controls.
        self.assertAlmostEqual(controls['relative_c2w'][-1, 0, 3], 4.)
        twice_scale = pack_window_camera_features(controls, fps=8, translation_scale=4)
        np.testing.assert_allclose(twice_scale[:, 6:9], features[:, 6:9]/2)
        for fps, scale in ((0, 2), (8, 0), (np.nan, 2), (8, np.inf), (8, [1, 2])):
            with self.subTest(fps=fps, scale=scale), self.assertRaises(ValueError):
                pack_window_camera_features(controls, fps, scale)
        short = build_window_camera_controls(self.poses, self.k, [0, 6, 12], [0, 6, 12])
        with self.assertRaises(ValueError):
            pack_window_camera_features(short, 8, 2)


if __name__ == '__main__':
    unittest.main()
