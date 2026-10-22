"""Content recovery, sparse support and real media contracts for the mask audit."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from utils.scene_trackmask import camera_compensated_track_mask
from utils import scene_trackmask_audit as audit


class TrackMaskAuditTests(unittest.TestCase):
    def setUp(self):
        # Keep rendered temporary media on the repository's disk (D: locally).
        parent = Path(__file__).resolve().parents[1]/'outputs'/'test_tmp'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='trackmask_', dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def metadata(self):
        return dict(case_index=4, seed=101, step=6000, weights='ema',
                    history_frames=4, future_frames=5, frame_ids=list(range(9)),
                    columns=['RAW', 'AE', 'GENERATED', 'COPY'])

    def receipt_fixture(self, root, video=b'content checked preview'):
        prefix = 'eval/step0006000/ema/case004_seed101'
        payloads = {prefix+'.json': json.dumps(self.metadata()).encode('utf8'),
                    prefix+'.mp4': video}
        receipt = dict(status='completed', contract_hash='source-contract', files={})
        for name, payload in payloads.items():
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            receipt['files'][name] = dict(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        # Source receipts also contain large tensors; this audit must not fetch
        # or require them. The intentionally absent file makes that observable.
        receipt['files'][prefix+'.pt'] = dict(size=999999999, sha256='absent-weight-hash')
        (root/(prefix+'.done.json')).write_text(json.dumps(receipt), encoding='utf8')
        return prefix, receipt

    def test_receipt_reads_only_verified_media_subset_without_missing_pt(self):
        primary, local = self.root/'source', self.root/'local'
        prefix, receipt = self.receipt_fixture(primary)
        with mock.patch.object(audit, 'copy_file', wraps=audit.copy_file) as copies:
            path, row, identity = audit.read_case(local, [str(primary)], 4)
        self.assertEqual(path.read_bytes(), b'content checked preview')
        self.assertEqual(row, self.metadata())
        self.assertEqual(identity['source_contract'], 'source-contract')
        self.assertEqual(identity['video_sha256'], receipt['files'][prefix+'.mp4']['sha256'])
        self.assertEqual(len(copies.call_args_list), 2)
        self.assertTrue(all(str(call.args[0]).endswith(('.json', '.mp4')) for call in copies.call_args_list))
        self.assertFalse(list(local.rglob('*.pt')))
        # A verified cache is usable without revisiting its media sources.
        with mock.patch.object(audit, 'copy_file', side_effect=AssertionError('unexpected redownload')):
            audit.read_case(local, [str(primary)], 4)

    def test_corrupt_primary_media_falls_back_to_verified_second_copy(self):
        first, second, local = (self.root/name for name in ('first', 'second', 'local'))
        prefix, _ = self.receipt_fixture(first)
        self.receipt_fixture(second)
        original = (first/(prefix+'.mp4')).read_bytes()
        # Equal length forces checksum validation, not merely size validation.
        (first/(prefix+'.mp4')).write_bytes(b'X'*len(original))
        stale = local/(prefix+'.mp4')
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b'Y'*len(original))
        with mock.patch.object(audit, 'copy_file', wraps=audit.copy_file) as copies:
            path, _, _ = audit.read_case(local, [str(first), str(second)], 4)
        self.assertEqual(path.read_bytes(), original)
        sources = [str(call.args[0]) for call in copies.call_args_list]
        self.assertTrue(any(str(first) in source and source.endswith('.mp4') for source in sources))
        self.assertTrue(any(str(second) in source and source.endswith('.mp4') for source in sources))
        self.assertFalse(list(local.rglob('*.partial')))

    def test_corrupt_all_copies_does_not_publish_unverified_payload(self):
        primary, local = self.root/'source', self.root/'local'
        prefix, _ = self.receipt_fixture(primary)
        (primary/(prefix+'.mp4')).write_bytes(b'bad')
        with self.assertRaisesRegex(RuntimeError, 'Cannot recover verified'):
            audit.read_case(local, [str(primary)], 4)
        self.assertFalse((local/(prefix+'.mp4')).exists())

    def test_crop_removes_exactly_32_pixels_without_resizing(self):
        side = 12
        frames = np.arange(9*(32+side)*side*4*3, dtype=np.int64).reshape(9, 32+side, side*4, 3).astype(np.uint8)
        columns = audit.split_preview(frames, self.metadata())
        for i, name in enumerate(self.metadata()['columns']):
            np.testing.assert_array_equal(columns[name], frames[:, 32:, i*side:(i+1)*side])
            self.assertEqual(columns[name].shape, (9, side, side, 3))
        columns['RAW'][:] = 0
        self.assertTrue(frames[:, 32:, :side].any(), 'split must not expose an aliased mutable source')

    def test_oracle_and_inconsistent_frame_layout_are_rejected(self):
        frames = np.zeros((9, 44, 48, 3), dtype=np.uint8)
        oracle = self.metadata(); oracle['columns'][2] = 'GT_ASSISTED_ORACLE'
        with self.assertRaisesRegex(ValueError, 'not an oracle'):
            audit.split_preview(frames, oracle)
        with self.assertRaisesRegex(ValueError, 'frame count'):
            audit.split_preview(frames[:8], self.metadata())
        mismatched = self.metadata(); mismatched['frame_ids'] = list(range(8))
        with self.assertRaisesRegex(ValueError, 'frame count'):
            audit.split_preview(frames, mismatched)
        with self.assertRaisesRegex(ValueError, 'square'):
            audit.split_preview(np.zeros((9, 45, 48, 3), np.uint8), self.metadata())

    def test_sparse_support_unknown_and_finite_radius(self):
        tracks = np.array([[[2., 2.], [4., 2.], [8., 8.], [np.nan, np.nan]]])
        labels = np.array([[1, -1, 0, 1]], dtype=np.int8)
        support = audit.point_support(tracks, labels, height=12, width=12, radius=3.)
        self.assertEqual(support[0, 2, 2], 1)
        self.assertEqual(support[0, 2, 4], -1, 'nearest unknown must override farther dynamic point')
        self.assertEqual(support[0, 2, 5], -1)
        self.assertEqual(support[0, 8, 8], 0)
        self.assertEqual(support[0, 11, 0], -1, 'uncovered pixels are not background')
        self.assertEqual(support[0, 0, 0], 1)
        self.assertEqual(support[0, 11, 11], -1, 'outside the Euclidean disk remains unknown')
        empty = audit.point_support(np.zeros((1, 0, 2)), np.zeros((1, 0), np.int8), 12, 12, 3.)
        self.assertTrue(np.all(empty == -1))

    def test_roi_metrics_use_identical_support_and_future_frames_only(self):
        raw = np.zeros((9, 4, 4, 3), dtype=np.uint8)
        ae = raw.copy(); generated = raw.copy(); copied = raw.copy()
        # Extreme history errors must not affect reported future metrics.
        ae[:4] = 255; generated[:4] = 255; copied[:4] = 255
        ae[4:, :2] = 51; generated[4:, :2] = 102; copied[4:, :2] = 153
        ae[4:, 2:] = 204; generated[4:, 2:] = 51; copied[4:, 2:] = 102
        support = np.zeros((9, 4, 4), dtype=np.int8); support[:, :2] = 1
        result = audit.region_metrics(dict(RAW=raw, AE=ae, GENERATED=generated, COPY=copied), support, 4)
        self.assertEqual([row['frame_index'] for row in result['per_future_frame']], [4, 5, 6, 7, 8])
        self.assertEqual(result['summary']['dynamic']['pixels'], 40)
        self.assertEqual(result['summary']['static']['pixels'], 40)
        for key, expected in (('AE', .2), ('GENERATED', .4), ('COPY', .6)):
            self.assertAlmostEqual(result['summary']['dynamic']['rgb_l1'][key], expected, places=6)
        self.assertAlmostEqual(result['summary']['static']['rgb_l1']['AE'], .8, places=6)
        self.assertIn('Identical GT-teacher support', result['scope'])
        support[:] = 0
        empty = audit.region_metrics(dict(RAW=raw, AE=ae, GENERATED=generated, COPY=copied), support, 4)
        self.assertEqual(empty['summary']['dynamic']['pixels'], 0)
        self.assertTrue(all(value is None for value in empty['summary']['dynamic']['rgb_l1'].values()))
        self.assertTrue(all(row['regions']['dynamic']['rgb_l1']['GENERATED'] is None for row in empty['per_future_frame']))

    def test_conflicting_equidistant_support_is_unknown_in_both_orders(self):
        points = np.array([[[2., 2.], [4., 2.]]])
        for labels in (np.array([[1, -1]]), np.array([[0, 1]])):
            first = audit.point_support(points, labels, 8, 8, 3.)
            second = audit.point_support(points[:, ::-1], labels[:, ::-1], 8, 8, 3.)
            self.assertEqual(first[0, 2, 3], -1)
            self.assertEqual(second[0, 2, 3], -1)

    def test_displacement_baseline_does_not_hide_camera_or_depth_rejections(self):
        raw = np.zeros((2, 48, 48, 3), np.uint8)
        tracks = np.array([[[10., 10.]], [[20., 10.]]])
        result = dict(labels=np.full((2, 1), -1), valid=np.zeros((2, 1), bool),
                      reason_bits=np.full((2, 1), 32 | 4, np.uint16),
                      thresholds_px=np.full(2, 100.), diagnostics={'reference_frame': 0})
        naive = audit.mask_overlay(raw, tracks, result, naive=True)
        np.testing.assert_array_equal(naive[1, 10, 20], [255, 50, 65])

    def synthetic_geometry(self, frames):
        points = np.array([(x, y) for y in (10., 20., 30.) for x in (10., 20., 30.)])
        tracks = np.tile(points, (frames, 1, 1))
        tracks[:, 4, 0] += np.arange(frames)*1.5
        args = dict(tracks=tracks, depth=np.full((frames, 48, 48), 5.),
                    world_to_camera=np.tile(np.eye(4), (frames, 1, 1)),
                    intrinsics=np.tile([[40., 0., 24.], [0., 40., 24.], [0., 0., 1.]], (frames, 1, 1)))
        result = camera_compensated_track_mask(**args)
        result['tracks'] = tracks
        return result

    def test_actual_review_render_has_even_video_dimensions_and_separate_prefix(self):
        import imageio_ffmpeg
        yy, xx = np.mgrid[:48, :48]
        raw = np.stack([np.stack(((xx*5+t*3)%256, (yy*5)%256, ((xx+yy)*2)%256), -1)
                        for t in range(9)]).astype(np.uint8)
        columns = dict(RAW=raw, AE=raw.copy(), GENERATED=np.roll(raw, 1, axis=2), COPY=np.tile(raw[3:4], (9, 1, 1, 1)))
        full, prefix = self.synthetic_geometry(9), self.synthetic_geometry(4)
        original_full = copy.deepcopy(full['diagnostics'])
        output = self.root/'render'; output.mkdir()
        labels = []
        draw_text = ImageDraw.ImageDraw.text
        def record_text(draw, xy, text, *args, **kwargs):
            labels.append(text)
            return draw_text(draw, xy, text, *args, **kwargs)
        with (mock.patch.object(audit, 'mask_overlay', wraps=audit.mask_overlay) as overlays,
              mock.patch.object(ImageDraw.ImageDraw, 'text', new=record_text)):
            names = audit.render_mask_review(columns, full, prefix, output, history=4, fps=8.)
        self.assertEqual(set(names), {'mask_review.mp4', 'mask_review.jpg', 'observed_prefix.jpg'})
        self.assertEqual(len(overlays.call_args_list), 3)
        last = overlays.call_args_list[-1]
        self.assertEqual(len(last.args[0]), 4)
        self.assertIs(last.args[2], prefix, 'observed view must use separately computed prefix teacher')
        self.assertEqual(full['diagnostics'], original_full)
        self.assertTrue(full['diagnostics']['uses_future_frames'])
        self.assertTrue(any(text.startswith('FULL-WINDOW GT TEACHER') for text in labels))
        self.assertTrue(any(text.startswith('OBSERVED PREFIX ONLY') for text in labels))
        self.assertIn('GENERATED F8', labels)
        self.assertIn('PREFIX F3', labels)
        self.assertNotIn('PREFIX F4', labels)
        reader = imageio_ffmpeg.read_frames(str(output/'mask_review.mp4'), pix_fmt='rgb24')
        try:
            info = next(reader)
            width, height = info['size']
            decoded = list(reader)
        finally:
            reader.close()
        self.assertEqual((width, height), (1296, 326))
        self.assertEqual(width % 2, 0); self.assertEqual(height % 2, 0)
        self.assertEqual(len(decoded), 9)
        self.assertTrue(all(len(frame) == width*height*3 for frame in decoded))
        with Image.open(output/'mask_review.jpg') as image:
            self.assertEqual(image.size, (1296, 978))
        with Image.open(output/'observed_prefix.jpg') as image:
            self.assertEqual(image.size, (1036, 326))
        audit.input_contact_sheet(columns, 4, output/'input.jpg')
        with Image.open(output/'input.jpg') as image:
            self.assertEqual(image.size, (1554, 911))


if __name__ == '__main__':
    unittest.main()
