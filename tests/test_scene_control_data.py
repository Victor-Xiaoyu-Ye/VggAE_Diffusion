"""CPU contract tests: future-control labels are separate from appearance."""
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.scene_control_data import (build_controls, camera_control_rays, replay_rgb,
                                     select_windows, validate_controls, window_key)
from utils.scene_run import ArtifactStore


def teacher(frames=9, side=28, query_side=4):
    y, x = np.meshgrid(np.linspace(5, side-6, query_side),
                       np.linspace(5, side-6, query_side), indexing='ij')
    points = torch.from_numpy(np.stack((x, y), -1).reshape(-1, 2)).float()
    ext = torch.eye(4)[None, :3].repeat(frames, 1, 1)
    intrinsic = torch.tensor([[25., 0., (side-1)/2], [0., 25., (side-1)/2], [0., 0., 1.]])
    return dict(depth=torch.full((frames, side, side), 4.),
        depth_conf=torch.ones(frames, side, side), extrinsics=ext,
        intrinsics=intrinsic[None].repeat(frames, 1, 1),
        tracks=points[None].repeat(frames, 1, 1), query_points=points,
        visibility=torch.ones(frames, len(points)),
        track_confidence=torch.ones(frames, len(points)))


class ControlDataTests(unittest.TestCase):
    def test_balanced_selection_reproducible_and_deduplicated(self):
        entries = [dict(id='spatialvid:'+str(i), dataset='spatialvid', file=f'{i}.pt',
                        index=w, frame_ids=list(range(w*10, w*10+9)))
                   for i in range(4) for w in range(4)]
        first = select_windows(entries, 5)
        self.assertEqual(first, select_windows(list(reversed(entries))+entries[:2], 5))
        self.assertEqual(len({s['id'] for s in first[:4]}), 4)
        self.assertEqual(len({window_key(s) for s in first}), 5)
        self.assertEqual(len(select_windows(entries, 0)), len(entries))

    def test_camera_scale_never_depends_on_future_span(self):
        full, prefix = teacher(), teacher(4)
        full['extrinsics'][4:, 0, 3] = torch.arange(1, 6).float()
        first, scale = camera_control_rays(full, prefix, 4)
        changed = copy.deepcopy(full)
        changed['extrinsics'][8, 0, 3] = 1000
        changed['depth'][4:] = 1000
        second, scale2 = camera_control_rays(changed, prefix, 4)
        self.assertEqual(scale, 4.)
        self.assertEqual(scale, scale2)
        torch.testing.assert_close(first[:8], second[:8], rtol=0, atol=0)
        self.assertFalse(torch.equal(first[8], second[8]))

    def test_schema_static_evidence_unknown_and_no_future_depth(self):
        value = build_controls(teacher(), teacher(4), history=4, grid=12)
        validate_controls(value, history=4, grid=12, frames=9)
        self.assertEqual(value['motion'].shape, (9, 144, 8))
        self.assertTrue((value['motion'][..., 7] == 0).all())
        self.assertGreater(value['motion'][1:, :, 5].sum(), 0)  # static camera/scene
        empty = value['motion'][..., 2] == 0
        self.assertTrue(empty.any())
        self.assertTrue((value['motion'][..., 6][empty] == 1).all())
        self.assertFalse(value['metadata']['tracks_are_semantic_instances'])

    def test_future_teacher_does_not_replace_observed_tracks(self):
        full, prefix = teacher(), teacher(4)
        first = build_controls(full, prefix, grid=4)
        full['tracks'][2:, :, 0] += 1
        changed = build_controls(full, prefix, grid=4)
        torch.testing.assert_close(first['prefix_tracks'], changed['prefix_tracks'], rtol=0, atol=0)
        self.assertFalse(torch.equal(first['tracks'], changed['tracks']))

    def test_query_id_mismatch_rejected(self):
        prefix = teacher(4)
        prefix['query_points'] = prefix['query_points'].flip(0)
        with self.assertRaisesRegex(ValueError, 'query identities'):
            build_controls(teacher(), prefix)

    def test_invisible_and_outside_tracks_have_zero_transport_weight(self):
        full, prefix = teacher(), teacher(4)
        full['visibility'][:, 0] = 0
        full['tracks'][:, 1, 0] = 100
        prefix['visibility'][:, 2] = 0
        value = build_controls(full, prefix, grid=4)
        self.assertTrue((value['track_weight'][:, :2] == 0).all())
        self.assertTrue((value['prefix_track_weight'][:, 2] == 0).all())

    def test_zero_coverage_is_valid_unknown_condition_not_gate(self):
        full, prefix = teacher(), teacher(4)
        full['visibility'].zero_()
        value = build_controls(full, prefix, grid=4)
        self.assertEqual(value['motion'][..., 2].sum().item(), 0)
        self.assertTrue((value['motion'][..., 6] == 1).all())

    def test_sealed_controls_roundtrip_dual_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = ArtifactStore(root/'local', [str(root/'a'), str(root/'b')])
            value = build_controls(teacher(), teacher(4), grid=4)
            store.save('controls.pt', {'controls': value}, 'identity')
            (root/'local'/'controls.pt').unlink()
            (root/'a'/'controls.pt').write_bytes(b'corrupt')
            recovered = store.load('controls.pt', 'identity', True)['controls']
            torch.testing.assert_close(recovered['rays'], value['rays'], rtol=0, atol=0)

    def test_cleanup_uses_artifact_specific_ack_and_output_error_is_fatal(self):
        from scripts import prepare_scene_controls as prep
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = ArtifactStore(root/'local', [str(root/'a'), str(root/'b')])
            prep.save_output(store, 'teacher.pt', {'x':torch.zeros(1)}, 'id')
            store._control_publications['teacher.pt'] = [str(root/'a')]
            prep.save_output(store, 'controls.pt', {'x':torch.zeros(1)}, 'id')
            prep.trim_published(store, 'teacher.pt')
            self.assertTrue((root/'local'/'teacher.pt').exists())
            prep.trim_published(store, 'controls.pt')
            self.assertFalse((root/'local'/'controls.pt').exists())
            with patch.object(store, 'save', side_effect=RuntimeError('synthetic publication failed')):
                with self.assertRaises(prep.OutputFailure):
                    prep.save_output(store, 'failed.pt', {}, 'id')

    def test_exact_raw_replay_uses_saved_indices_crop_and_uint8_provenance(self):
        # Mock only the unavailable video backend. Real production RGB crop,
        # resizing, channel conversion, frame indexing and checksum run here.
        requested = []
        class Capture:
            def isOpened(self): return True
            def get(self, prop): return 10 if prop == 0 else 8.
            def set(self, prop, index): self.index = index; requested.append(index)
            def read(self):
                image = np.empty((32, 48, 3), np.uint8)
                image[:] = (self.index*20, 20, 40)
                return True, image
            def release(self): pass
        cv2 = SimpleNamespace(VideoCapture=lambda _:Capture(), CAP_PROP_FRAME_COUNT=0,
            CAP_PROP_FPS=1, CAP_PROP_POS_FRAMES=2, COLOR_BGR2RGB=3,
            cvtColor=lambda x, _:x[..., ::-1].copy())
        with patch.dict('sys.modules', {'cv2':cv2}):
            sample = dict(id='spatialvid:abc', dataset='spatialvid', frame_ids=[1, 3, 8])
            row = dict(id=sample['id'], dataset='spatialvid', kind='video', root='original.mp4')
            frames, provenance = replay_rgb(sample, row, lambda p:p, size=28)
            self.assertEqual(requested, [1, 3, 8])
            self.assertEqual(frames.shape, (3, 3, 28, 28))
            self.assertEqual(provenance['frame_ids'], [1, 3, 8])
            self.assertEqual(provenance['source_dimensions_hw'], [[32, 48]]*3)
            sample['raw'] = (frames*255).round().to(torch.uint8)
            _, second = replay_rgb(sample, row, lambda p:p, size=28)
            self.assertTrue(second['eval_raw_exact_uint8_match'])
            sample['raw'][0, 0, 0, 0] ^= 1
            with self.assertRaisesRegex(ValueError, 'differs'):
                replay_rgb(sample, row, lambda p:p, size=28)

    def test_plan_two_workers_resume_merge_and_common_bad_skip(self):
        from scripts import prepare_scene_controls as prep
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cohort = dict(seed=42, records=[dict(id=f'spatialvid:{i}', dataset='spatialvid',
                split='train' if i < 3 else 'val', kind='video', root=f'{i}.mp4') for i in range(4)])
            cohort_path = root/'cohort.json'
            cohort_path.write_text(json.dumps(cohort), encoding='utf8')
            selection_path = root/'selection.json'
            selection_path.write_text(json.dumps(dict(schema='scene-street-ids-v1',
                count=3, allowed_ids=[f'spatialvid:{i}' for i in range(3)],
                cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
                cohort_canonical_sha256=hashlib.sha256(json.dumps(cohort,
                    sort_keys=True, separators=(',', ':')).encode()).hexdigest())), encoding='utf8')
            source = ArtifactStore(root/'source', [])
            source.write_json('contract.json', dict(identity='prefix',
                cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest()))
            def sample(index, raw=False):
                z = torch.full((9, 16, 8), float(index), dtype=torch.bfloat16)
                value = dict(id=f'spatialvid:{index}', dataset='spatialvid',
                    frame_ids=list(range(9)), target=z, anchor=z[:1].clone())
                if raw: value['raw'] = torch.zeros(9, 3, 28, 28, dtype=torch.uint8)
                return value
            source.save('data/a.pt', dict(samples=[sample(0), sample(1)]), 'prefix')
            source.save('data/b.pt', dict(samples=[sample(2)]), 'prefix')
            source.save('data/eval.pt', dict(samples=[sample(3, True)]), 'prefix')
            manifest = dict(identity='prefix', mean=[0.]*8, std=[1.]*8, shards=[
                dict(file='data/a.pt', sources=['spatialvid']*2),
                dict(file='data/b.pt', sources=['spatialvid'])])
            source.write_json('data/cache_complete.json', manifest)
            checkpoint = root/'teacher.pt'; checkpoint.write_bytes(b'fakecheckpoint')
            a = SimpleNamespace(cohort=str(cohort_path), selection=str(selection_path),
                teacher_checkpoint=str(checkpoint), precision='float32', query_stride=7,
                query_margin=5, query_chunk_size=0, train_windows=3, eval_cases=1,
                history=4, grid=4, size=28, seed=42, workers=2, worker_rank=0,
                stage='plan', raw_cache=str(root/'raw'), device='cpu', source_shard_limit=0)
            store = ArtifactStore(root/'controlled', [str(root/'replica1'), str(root/'replica2')])
            calls = []
            class FakeTeacher:
                metadata = dict(backend='fake_test_only')
                def __init__(self, *args, **kwargs): pass
                def infer(self, frames, reference_frame=0):
                    calls.append(len(frames)); return teacher(len(frames))
            def fake_replay(s, row, localize, size):
                if s['id'] == 'spatialvid:1': raise ValueError('synthetic missing raw input')
                return torch.zeros(9, 3, 28, 28), dict(frame_ids=s['frame_ids'], source='synthetic')
            with patch.object(prep, 'teacher_source_identity', return_value={'code_sha256':'test'}):
                contract, original, selection = prep.make_contract(a, source)
            prep.plan(a, source, store, contract, original, selection)
            with patch.object(prep, 'VGGTTrackTeacher', FakeTeacher), patch.object(prep, 'replay_rgb', fake_replay):
                for rank in range(2):
                    a.worker_rank, a.stage = rank, 'work'
                    prep.work(a, source, store, contract, original)
                first_calls = len(calls)
                prep.work(a, source, store, contract, original)
                self.assertEqual(len(calls), first_calls)  # sealed recovery, no reinference
            a.worker_rank, a.stage = 0, 'merge'
            complete = prep.merge(a, store, contract, original)
            self.assertEqual(complete['samples'], 2)
            self.assertEqual(complete['eval_cases'], 1)
            self.assertEqual(len(complete['bad']), 1)
            for shard in complete['shards']:
                payload = store.load(shard['file'], complete['identity'], True)
                for item in payload['samples']:
                    index = int(item['id'].split(':')[1])
                    self.assertTrue(torch.equal(item['target'], sample(index)['target']))
                    self.assertTrue(torch.equal(item['anchor'], sample(index)['anchor']))
            self.assertEqual(complete['mean'], manifest['mean'])
            self.assertEqual(complete['std'], manifest['std'])
            # A separate bounded smoke never reads shard b; production still
            # pins the entire original manifest rather than pretending that
            # this small prefix is the full-quality dataset.
            smoke = copy.copy(a)
            smoke.stage, smoke.source_shard_limit = 'plan', 1
            smoke_store = ArtifactStore(root/'smoke', [])
            with patch.object(prep, 'teacher_source_identity', return_value={'code_sha256':'test'}):
                smoke_contract, _, _ = prep.make_contract(smoke, source)
            self.assertNotEqual(smoke_contract['identity'], contract['identity'])
            self.assertEqual(smoke_contract['source_cache_manifest_sha'], contract['source_cache_manifest_sha'])
            original_load = source.load
            read_names = []
            def checked_load(name, *args, **kwargs):
                read_names.append(name)
                if name == 'data/b.pt': raise AssertionError('Smoke scanned an excluded shard')
                return original_load(name, *args, **kwargs)
            with patch.object(source, 'load', side_effect=checked_load):
                smoke_plan = prep.plan(smoke, source, smoke_store, smoke_contract, original, selection)
                self.assertEqual(smoke_plan['selected_windows'], 2)
                self.assertEqual(smoke_plan['selected_pool']['scanned_source_shards'], 1)
                count = len(read_names)
                prep.plan(smoke, source, smoke_store, smoke_contract, original, selection)
                self.assertEqual(len(read_names), count)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                prep.plan(a, source, smoke_store, contract, original, selection)


if __name__ == '__main__':
    unittest.main()
