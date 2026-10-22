"""Contracts for the separate official VGGT diagnostic adapter (CPU fake teacher)."""
import tempfile
import argparse
import hashlib
import json
from pathlib import Path
import types
import unittest
from unittest import mock

import torch
from torch import nn

from utils.scene_trackmask_teacher import (
    VGGTTrackTeacher, VENDORED_ROOT, checkpoint_state, import_official_vggt,
    query_grid, validate_checkpoint_coverage,
    teacher_source_identity, import_streamvggt_teacher, STREAM_VENDORED_ROOT,
)


class FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, images, use_cache=False):
        assert not torch.is_grad_enabled()
        assert use_cache is False
        self.received_images = images.clone()
        return [images.mean((-1, -2, -3))[..., None, None] * self.weight], 0


class FakeCamera(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, tokens, use_cache=False):
        assert use_cache is False
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
        self.received_images = images.clone()
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
        for wrapper in ('model_state_dict', 'model', 'state_dict', 'ema_state_dict'):
            with self.subTest(wrapper=wrapper):
                self.assertEqual(list(checkpoint_state({wrapper: state})), ['aggregator.weight'])
        with self.assertRaises(TypeError):
            checkpoint_state({'bad': 7})
        with self.assertRaisesRegex(TypeError, 'wrapper model'):
            checkpoint_state({'model': 7, 'state_dict': state})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            checkpoint_state({'module.x': torch.ones(1), 'x': torch.ones(1)})

    def test_stream_full_window_tracking_preserves_history_and_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoints.pth'
            state = {'module.' + k: v for k, v in FakeVGGT().state_dict().items()}
            torch.save({'model_state_dict': state, 'args': argparse.Namespace(epoch=12)}, path)
            real_load = torch.load
            with mock.patch('utils.scene_trackmask_teacher.import_streamvggt_teacher',
                            return_value=(FakeVGGT, fake_pose_decoder, {'code_sha256': 'fake-stream'})), \
                    mock.patch('torch.load', wraps=real_load) as loader:
                teacher = VGGTTrackTeacher(path, backend='streamvggt', query_stride=7)
            self.assertIsInstance(loader.call_args.args[0], str)
            self.assertFalse(loader.call_args.kwargs['weights_only'])
            frames = torch.tensor([.1, .3, .6])[:, None, None, None].expand(3, 3, 28, 42)
            result = teacher.infer(frames)
            torch.testing.assert_close(teacher.model.aggregator.received_images[0], frames)
            torch.testing.assert_close(teacher.model.track_head.received_images[0], frames)
            self.assertEqual(teacher.model.track_head.received_images.shape[1], 3)
            torch.testing.assert_close(result['tracks'][:, 0, 0], torch.tensor([7.1, 7.3, 7.6]))
            metadata = result['teacher']
            self.assertTrue(metadata['causal_backbone'])
            self.assertTrue(metadata['causal_camera_head'])
            self.assertTrue(metadata['tracking_uses_future_within_supplied_window'])
            self.assertFalse(metadata['online_conditioner'])
            self.assertFalse(metadata['kv_cache'])
            with self.assertRaisesRegex(ValueError, 'chronological causal history'):
                teacher.infer(frames, reference_frame=1)
            with self.assertRaisesRegex(ValueError, 'override is not supported'):
                VGGTTrackTeacher(path, backend='streamvggt', code_root=tmp)

    def test_stream_aggregator_only_checkpoint_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'partial.pth'
            torch.save({'model': {'aggregator.weight': torch.ones(())}}, path)
            with mock.patch('utils.scene_trackmask_teacher.import_streamvggt_teacher',
                            return_value=(FakeVGGT, fake_pose_decoder, {})):
                with self.assertRaisesRegex(ValueError, 'camera_head.weight'):
                    VGGTTrackTeacher(path, backend='streamvggt')

    def test_source_identity_tracks_production_and_vendor_without_model_import(self):
        with mock.patch('utils.scene_trackmask_teacher.importlib.import_module',
                        side_effect=AssertionError('source hashing must not import models')):
            identity = teacher_source_identity('streamvggt')
        self.assertEqual(len(identity['code_sha256']), 64)
        self.assertEqual(len(identity['code_roots']), 2)
        self.assertTrue(any(Path(p).name == 'streamvggt' for p in identity['code_roots']))
        with self.assertRaisesRegex(ValueError, 'override'):
            teacher_source_identity('streamvggt', 'ignored-custom-checkout')
        with self.assertRaises(ValueError):
            teacher_source_identity('typo')

    def test_vendor_matches_pinned_upstream_with_only_declared_import_adaptation(self):
        manifest = json.loads((STREAM_VENDORED_ROOT / 'UPSTREAM.json').read_text(encoding='utf-8'))
        for rel, item in manifest['files'].items():
            data = (STREAM_VENDORED_ROOT / rel).read_bytes()
            if rel == 'heads/camera_head.py':
                data = data.replace(b'from .head_act import activate_pose',
                                    b'from streamvggt.heads.head_act import activate_pose')
            self.assertEqual(hashlib.sha256(data).hexdigest(), item['upstream_sha256'], rel)

    def test_real_stream_camera_head_is_causal_and_cpu_dpt_positions_match(self):
        from utils.scene_trackmask_vendor import import_stream_vendor
        CameraHead = import_stream_vendor('heads.camera_head').CameraHead
        DPTHead = import_stream_vendor('heads.dpt_head').DPTHead
        from utils.scene_trackmask_stream import DeviceCompatibleDPTHead
        torch.manual_seed(13)
        head = CameraHead(dim_in=32, trunk_depth=1, num_heads=4).eval()
        tokens = torch.randn(1, 4, 2, 32)
        changed = tokens.clone()
        changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 10
        with torch.inference_mode():
            before = head([tokens])[-1]
            after = head([changed])[-1]
        torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)
        self.assertFalse(torch.equal(before[:, 2:], after[:, 2:]))
        features = torch.randn(2, 32, 7, 9)
        expected = DPTHead._apply_pos_embed(None, features, 126, 98)
        actual = DeviceCompatibleDPTHead._apply_pos_embed(None, features, 126, 98)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_real_tracker_full_sequence_updates_future_but_single_frame_locks_query(self):
        from utils.scene_trackmask_vendor import import_stream_vendor
        BaseTrackerPredictor = import_stream_vendor('heads.track_modules.base_track_predictor').BaseTrackerPredictor
        torch.manual_seed(17)
        tracker = BaseTrackerPredictor(latent_dim=16, hidden_size=32, depth=1,
                                       corr_levels=1, corr_radius=1, use_spaceatt=False).eval()
        # Make a deterministic positive coordinate update; real tracker retains
        # its correlation, temporal transformer and first-frame reset behavior.
        with torch.no_grad():
            tracker.updateformer.flow_head.weight.zero_()
            tracker.updateformer.flow_head.bias.zero_()
            tracker.updateformer.flow_head.bias[:2] = torch.tensor([.4, .2])
        points = torch.tensor([[[3., 4.], [5., 6.]]])
        feature_maps = torch.randn(1, 3, 16, 12, 12)
        with torch.inference_mode():
            tracks, _, _ = tracker(points, fmaps=feature_maps, iters=2)
            single, _, _ = tracker(points, fmaps=feature_maps[:, :1], iters=2)
        torch.testing.assert_close(tracks[-1][:, 0], points)
        torch.testing.assert_close(tracks[-1][:, 1], points + torch.tensor([.8, .4]))
        torch.testing.assert_close(single[-1][:, 0], points)

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
