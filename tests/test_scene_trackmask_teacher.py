"""Contracts for the separate official VGGT diagnostic adapter (CPU fake teacher)."""
import tempfile
from pathlib import Path
import types
import unittest
from unittest import mock

import torch
from torch import nn

from utils.scene_trackmask_teacher import (
    VGGTTrackTeacher, VENDORED_ROOT, checkpoint_state, import_official_vggt,
    query_grid, validate_checkpoint_coverage,
)


class FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, images):
        assert not torch.is_grad_enabled()
        return [images.mean((-1, -2, -3))[..., None, None] * self.weight], 0


class FakeCamera(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, tokens):
        return [tokens[0][..., 0].repeat(1, 1, 9) * self.weight]


class FakeDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, tokens, images, patch_start_idx):
        b, t, _, h, w = images.shape
        assert tokens[0].dtype == torch.float32
        depth = (1 + tokens[0][..., 0, 0])[:, :, None, None, None].expand(b, t, h, w, 1)
        return depth * self.weight, torch.ones(b, t, h, w, device=images.device)


class FakeTrack(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, tokens, images, patch_start_idx, query_points):
        b, t = images.shape[:2]
        displacement = tokens[0][..., 0, 0][:, :, None, None]
        track = query_points[:, None] + displacement * self.weight
        vis = torch.ones(b, t, query_points.shape[1], device=images.device)
        return [track], vis, vis * .9


class FakeVGGT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.aggregator = FakeAggregator()
        self.camera_head = FakeCamera()
        self.depth_head = FakeDepth()
        self.track_head = FakeTrack()


def fake_pose_decoder(pose, image_size):
    b, t = pose.shape[:2]
    extri = torch.eye(4, device=pose.device)[None, None, :3].repeat(b, t, 1, 1)
    extri[..., 0, 3] = pose[..., 0]
    intri = torch.eye(3, device=pose.device)[None, None].repeat(b, t, 1, 1)
    return extri, intri


class TrackTeacherTests(unittest.TestCase):
    def make_teacher(self, path, **kwargs):
        torch.save(FakeVGGT().state_dict(), path)
        with mock.patch('utils.scene_trackmask_teacher.import_official_vggt',
                        return_value=(FakeVGGT, fake_pose_decoder, {'code_sha256': 'fake'})):
            return VGGTTrackTeacher(path, query_stride=7, **kwargs)

    def test_full_checkpoint_and_inference_preserve_pixels_and_time_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = self.make_teacher(Path(tmp) / 'teacher.pt')
            frames = torch.tensor([.1, .2, .3])[:, None, None, None].expand(3, 3, 28, 42)
            original = frames.clone()
            result = teacher.infer(frames, reference_frame=2)
            self.assertFalse(teacher.model.training)
            self.assertTrue(all(not p.requires_grad for p in teacher.model.parameters()))
            self.assertEqual(result['depth'].shape, (3, 28, 42))
            torch.testing.assert_close(result['depth'][:, 0, 0], torch.tensor([1.1, 1.2, 1.3]))
            torch.testing.assert_close(result['extrinsics'][:, 0, 3], torch.tensor([.1, .2, .3]))
            torch.testing.assert_close(result['tracks'][:, 0, 0], torch.tensor([7.1, 7.2, 7.3]))
            torch.testing.assert_close(frames, original)
            self.assertEqual(result['image_size_hw'], [28, 42])
            self.assertEqual(result['reference_frame'], 2)
            self.assertEqual(result['teacher_context_frames'], 3)
            self.assertEqual(result['teacher']['role'], 'offline_bidirectional_pseudo_label_teacher')
            for value in result.values():
                if torch.is_tensor(value):
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertEqual(value.device.type, 'cpu')

    def test_missing_tracking_weight_is_rejected_and_unused_point_head_is_explicit(self):
        model = FakeVGGT()
        state = model.state_dict()
        state['point_head.extra'] = torch.ones(1)
        self.assertEqual(validate_checkpoint_coverage(model, state)['ignored_point_head_keys'], 1)
        del state['track_head.weight']
        with self.assertRaisesRegex(ValueError, 'track_head.weight'):
            validate_checkpoint_coverage(model, state)

    def test_shape_and_unknown_backbone_weights_are_rejected(self):
        model = FakeVGGT()
        state = model.state_dict()
        state['camera_head.weight'] = torch.zeros(2)
        state['omega.unexpected'] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, 'shape_mismatches'):
            validate_checkpoint_coverage(model, state)

    def test_aggregator_only_model_cannot_claim_full_coverage(self):
        model = nn.Module()
        model.aggregator = FakeAggregator()
        with self.assertRaisesRegex(ValueError, 'model has no required module'):
            validate_checkpoint_coverage(model, model.state_dict())

    def test_checkpoint_wrappers_and_non_tensor_payload(self):
        state = {'module.aggregator.weight': torch.ones(1)}
        self.assertEqual(list(checkpoint_state({'model': state})), ['aggregator.weight'])
        with self.assertRaises(TypeError):
            checkpoint_state({'bad': 7})

    def test_bad_input_and_nonfinite_head_fail_without_fabricated_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = self.make_teacher(Path(tmp) / 'teacher.pt')
            with self.assertRaisesRegex(ValueError, 'multiples of14'):
                teacher.infer(torch.ones(2, 3, 27, 28))
            with self.assertRaisesRegex(ValueError, r'\[0,1\]'):
                teacher.infer(torch.full((2, 3, 28, 28), 255.))
            with torch.no_grad():
                teacher.model.depth_head.weight.fill_(float('nan'))
            with self.assertRaisesRegex(ValueError, 'Nonfinite teacher depth'):
                teacher.infer(torch.zeros(2, 3, 28, 28))

    def test_query_chunking_is_recorded_as_context_changing_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = self.make_teacher(Path(tmp) / 'teacher.pt', query_chunk_size=3)
            result = teacher.infer(torch.zeros(2, 3, 28, 42))
            self.assertEqual(result['tracks'].shape[1], len(query_grid(28, 42, 7)))
            self.assertTrue(result['teacher']['query_chunk_changes_tracker_context'])

    def test_explicit_checkout_rejects_streamvggt_and_existing_foreign_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, 'Full official VGGT'):
                import_official_vggt(tmp)
        foreign = types.ModuleType('vggt')
        foreign.__file__ = '/another/project/vggt/__init__.py'
        with mock.patch.dict('sys.modules', {'vggt': foreign}):
            with self.assertRaisesRegex(RuntimeError, 'Another vggt package'):
                import_official_vggt(VENDORED_ROOT)


if __name__ == '__main__':
    unittest.main()
