"""Real MP4 I/O and resume contracts with a recording fake VGGT teacher.

These tests establish orchestration and future-input isolation, not mask quality.
Windows temporary artifacts stay on D: when the project's audit drive exists.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import imageio.v2 as imageio
import numpy as np
import torch

import audit_scene_trackmask as audit
from utils.native_audit_io import recover_case, seal
from utils.scene_run import ArtifactStore, atomic
from utils.scene_trackmask_audit import decode_preview, read_case


def temporary():
    root = Path('D:/workspace/VggAE_DataAudit') if os.name == 'nt' else None
    return tempfile.TemporaryDirectory(prefix='trackmask-pipeline-',
        dir=str(root) if root is not None and root.is_dir() else None)


class RecordingTeacher:
    def __init__(self):
        self.calls = []
        self.metadata = {'backend': 'fake', 'not_real_teacher_quality': True}

    def infer(self, frames, reference_frame=0):
        self.calls.append(np.array(frames, copy=True))
        t, _, h, w = frames.shape
        y, x = np.meshgrid(np.arange(7, h-7, 3), np.arange(7, w-7, 3), indexing='ij')
        query = np.stack((x, y), -1).reshape(-1, 2).astype(np.float32)
        tracks = np.broadcast_to(query[None], (t, len(query), 2)).copy()
        world_to_camera = np.broadcast_to(np.eye(4, dtype=np.float32)[None, :3], (t, 3, 4)).copy()
        intrinsics = np.broadcast_to(np.array([[20, 0, w/2], [0, 20, h/2], [0, 0, 1]],
                                              np.float32)[None], (t, 3, 3)).copy()
        arrays = dict(depth=np.ones((t, h, w), np.float32), depth_conf=np.ones((t, h, w), np.float32),
            tracks=tracks, visibility=np.ones((t, len(query)), np.float32),
            track_confidence=np.ones((t, len(query)), np.float32), extrinsics=world_to_camera,
            intrinsics=intrinsics, query_points=query)
        # Match the actual adapter's Tensor API, not an easier NumPy-only mock.
        return dict({name: torch.from_numpy(value) for name, value in arrays.items()},
                    reference_frame=reference_frame, image_size_hw=[h, w], teacher={'kind': 'fake'},
                    teacher_context_frames=t, camera_convention='OpenCV_world_to_camera')


def fixture(root, index=0):
    prefix = f'eval/step0006000/ema/case{index:03d}_seed101'
    video = Path(root)/(prefix+'.mp4')
    video.parent.mkdir(parents=True, exist_ok=True)
    # Exact square tiles and even H/W: codec cannot alter the coordinate system.
    tiles = np.zeros((9, 28, 28*4, 3), np.uint8)
    for t in range(9):
        for column in range(4):
            tiles[t, :, column*28:(column+1)*28] = [20+t*13+column*2, 80, 120]
    frames = np.pad(tiles, ((0, 0), (32, 0), (0, 0), (0, 0)))
    imageio.mimwrite(str(video), frames, fps=8, macro_block_size=1,
                    codec='libx264', quality=8, pixelformat='yuv420p')
    metadata = dict(case_index=index, seed=101, step=6000, weights='ema', id=f'街景样本{index}',
                    columns=['RAW', 'AE', 'GENERATED', 'COPY'], history_frames=4, future_frames=5,
                    frame_ids=list(range(9)), nominal_fps=8)
    atomic(Path(root)/(prefix+'.json'), metadata)
    seal(root, prefix+'.done.json', 'fixture-source-contract',
         [prefix+'.json', prefix+'.mp4'], metadata)
    return video, metadata


def arguments(source, output, mirror=None):
    return argparse.Namespace(source_local=str(source), source_root=[], output=str(output),
        root=[str(mirror)] if mirror else [], teacher_kind='streamvggt', teacher_code=None,
        teacher_checkpoint='explicit-fake-test-only', device='cpu', precision='float32', query_stride=7,
        node_rank=0, nodes=1, allocated_devices=1, cases='0', seed=101, step=6000, weights='ema',
        prepare_only=False, collect_only=False)


class TrackMaskPipelineTests(unittest.TestCase):
    def run_fake(self, args, teacher):
        with mock.patch.object(audit, 'checkpoint_identity', return_value={'test_only': 'fake-teacher'}), \
             mock.patch('utils.scene_trackmask_teacher.VGGTTrackTeacher', return_value=teacher) as factory:
            result = audit.run(args)
            if factory.called:
                self.assertEqual(factory.call_args.kwargs['backend'], args.teacher_kind)
            return result

    def test_default_backend_is_existing_stream_without_external_source(self):
        args = audit.build_parser().parse_args(['--output', 'unused'])
        self.assertEqual(args.teacher_kind, 'streamvggt')
        self.assertIsNone(args.teacher_code)

    def test_changed_teacher_backend_cannot_reuse_an_old_case_receipt(self):
        with temporary() as directory:
            root = Path(directory)
            fixture(root/'source')
            args = arguments(root/'source', root/'output')
            teacher = RecordingTeacher()
            self.assertEqual(self.run_fake(args, teacher), 0)
            teacher.calls.clear()
            args.teacher_kind = 'vggt'
            with self.assertRaisesRegex(ValueError, 'another contract'):
                self.run_fake(args, teacher)
            self.assertEqual(teacher.calls, [])

    def test_incomplete_teacher_is_reported_as_setup_failure_not_bad_videos(self):
        with temporary() as directory:
            root = Path(directory)
            fixture(root/'source')
            args = arguments(root/'source', root/'output', root/'mirror')
            with mock.patch.object(audit, 'checkpoint_identity', return_value={'test_only': 'fake'}), \
                 mock.patch('utils.scene_trackmask_teacher.VGGTTrackTeacher',
                            side_effect=ValueError('missing track_head.weight')):
                with self.assertRaisesRegex(audit.TeacherSetupError, 'track_head'):
                    audit.run(args)
            report = json.loads((root/'output/nodes/teacher_setup000.json').read_text(encoding='utf8'))
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['teacher_kind'], 'streamvggt')
            self.assertTrue((root/'mirror/nodes/teacher_setup000.json').is_file())
            self.assertFalse((root/'output/cases/case000/teacher.done.json').exists())

    def test_end_to_end_real_mp4_seals_results_and_separates_prefix(self):
        with temporary() as directory:
            root = Path(directory)/'中文场景路径'
            fixture(root/'source')
            teacher = RecordingTeacher()
            args = arguments(root/'source', root/'output', root/'mirror')
            self.assertEqual(self.run_fake(args, teacher), 0)
            self.assertEqual([len(x) for x in teacher.calls], [9, 4])
            np.testing.assert_array_equal(teacher.calls[0][:4], teacher.calls[1])
            summary = json.loads((root/'output/summary.json').read_text(encoding='utf8'))
            self.assertEqual(summary['status'], 'completed')
            self.assertFalse(summary['teacher_quality_validated'])
            setup = json.loads((root/'output/nodes/teacher_setup000.json').read_text(encoding='utf8'))
            self.assertEqual(setup['status'], 'loaded')
            self.assertEqual(setup['teacher_kind'], 'streamvggt')
            self.assertFalse(summary['cases'][0]['prefix']['uses_future_frames'])
            receipt_path = root/'output/cases/case000/done.json'
            receipt = json.loads(receipt_path.read_text(encoding='utf8'))
            recovered = recover_case(root/'output', 'cases/case000/done.json',
                                     receipt['contract_hash'], [str(root/'mirror')])
            self.assertTrue(recovered)
            self.assertTrue((root/'mirror/cases/case000/teacher.done.json').is_file())
            self.assertTrue((root/'output/cases/case000/mask_review.mp4').stat().st_size > 0)
            # Re-running a completed case never constructs or invokes the teacher.
            teacher.calls.clear()
            self.assertEqual(self.run_fake(args, teacher), 0)
            self.assertEqual(teacher.calls, [])

    def test_render_failure_preserves_teacher_commit_and_resume_does_not_reinfer(self):
        with temporary() as directory:
            root = Path(directory)
            fixture(root/'source')
            args = arguments(root/'source', root/'output', root/'mirror')
            teacher = RecordingTeacher()
            with mock.patch.object(audit, 'render_mask_review', side_effect=RuntimeError('injected render interruption')):
                self.assertEqual(self.run_fake(args, teacher), 1)
            self.assertTrue((root/'output/cases/case000/teacher.done.json').is_file())
            self.assertFalse((root/'output/cases/case000/done.json').exists())
            summary = json.loads((root/'output/summary.json').read_text(encoding='utf8'))
            self.assertEqual(summary['status'], 'completed_with_errors')
            self.assertEqual(summary['failed'], 1)
            self.assertIn('render interruption', summary['cases'][0]['error'])
            # Lose a local inference payload: recovery must use the sealed mirror.
            (root/'output/cases/case000/full_teacher.npz').unlink()
            teacher.calls.clear()
            self.assertEqual(self.run_fake(args, teacher), 0)
            self.assertEqual(teacher.calls, [])
            self.assertTrue((root/'output/cases/case000/full_teacher.npz').is_file())
            self.assertEqual(json.loads((root/'output/summary.json').read_text(encoding='utf8'))['failed'], 0)

    def test_future_rgb_mutation_never_reaches_prefix_teacher(self):
        with temporary() as directory:
            root = Path(directory)
            video, metadata = fixture(root/'source')
            raw = decode_preview(video, metadata)['RAW']
            changed = raw.copy()
            changed[4:] = 255-changed[4:]
            teacher = RecordingTeacher()
            store = ArtifactStore(root/'output', [])
            for name, rgb in [('original', raw), ('changed', changed)]:
                (store.local/name).mkdir()
                audit.obtain_teacher(store, name, rgb, 4, name, lambda: teacher)
            self.assertEqual([len(x) for x in teacher.calls], [9, 4, 9, 4])
            self.assertFalse(np.array_equal(teacher.calls[0][4:], teacher.calls[2][4:]))
            np.testing.assert_array_equal(teacher.calls[1], teacher.calls[3])

    def test_incomplete_node_collection_is_partial_and_never_completed(self):
        with temporary() as directory:
            store = ArtifactStore(Path(directory)/'output', [])
            store.write_json('nodes/node000.json', dict(contract_hash='same', requested_cases=[0, 1],
                cases=[dict(case_index=0, status='completed')]))
            result = audit.collect(store, 2, write_partial=True)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['completed'], 1)
            self.assertFalse((store.local/'summary.json').exists())
            self.assertTrue((store.local/'collection_partial.json').exists())
            store.write_json('nodes/node001.json', dict(contract_hash='same', requested_cases=[0, 1],
                cases=[dict(case_index=1, status='failed', error='visible failure')]))
            result = audit.collect(store, 2)
            self.assertEqual(result['status'], 'completed_with_errors')
            self.assertEqual(result['failed'], 1)
            self.assertIn('visible failure', result['cases'][1]['error'])

    def test_source_content_corruption_is_not_silently_accepted(self):
        with temporary() as directory:
            root = Path(directory)
            video, _ = fixture(root/'source')
            video.write_bytes(b'corrupted')
            with self.assertRaisesRegex(RuntimeError, 'Cannot recover verified'):
                read_case(root/'source', [], 0)

    def test_unsupported_teacher_runtime_stops_without_sealing_fake_results(self):
        with temporary() as directory:
            store = ArtifactStore(Path(directory)/'output', [])
            (store.local/'case').mkdir()
            teacher = mock.Mock()
            teacher.infer.side_effect = NotImplementedError('unsupported device operator')
            with self.assertRaises(audit.TeacherSetupError):
                audit.obtain_teacher(store, 'case', np.zeros((9, 28, 28, 3), np.uint8),
                                     4, 'fixture', lambda: teacher)
            self.assertEqual(teacher.infer.call_count, 1)
            self.assertFalse((store.local/'case/teacher.done.json').exists())


if __name__ == '__main__':
    unittest.main()
