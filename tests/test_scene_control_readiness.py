"""CPU signal-presence contracts, not tracking or image-quality validation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from utils.scene_control_readiness import assess_controls, raw_teacher_scores, scan_prepared_payload, summarize
from utils.scene_control_data import window_key
from utils.scene_run import ArtifactStore


def controls(positive=True):
    points = torch.tensor([[.25, .25], [.75, .75]])
    value = dict(tracks=points[None].repeat(4, 1, 1),
        track_weight=torch.full((4, 2), float(positive)),
        prefix_tracks=points[None].repeat(2, 1, 1),
        prefix_track_weight=torch.full((2, 2), float(positive)),
        motion=torch.zeros(4, 4, 8), rays=torch.zeros(4, 4, 8), metadata={})
    value['rays'][..., -1] = 1
    value['motion'][..., 2 if positive else 6] = 1
    return value


def teacher_scores():
    return {name:dict(visibility=torch.full((frames, 2), .025),
                     track_confidence=torch.full((frames, 2), 1e-10))
            for name,frames in [('full',4),('prefix',2)]}


def sample(value=None):
    return dict(id='spatialvid:synthetic', dataset='spatialvid', frame_ids=[0, 1, 2, 3],
                controls=value if value is not None else controls())


class ControlReadinessTests(unittest.TestCase):
    def test_global_absence_is_explicit_but_camera_remains_available(self):
        result = summarize([assess_controls(controls(False), split='train'),
                            assess_controls(controls(False), split='eval')])
        for report in result['splits'].values():
            self.assertEqual(report['motion_status'], 'INVALID_ZERO_SUPPORT')
            self.assertEqual(report['reference_status'], 'INVALID_ZERO_SUPPORT')
            self.assertEqual(report['camera_status'], 'NONZERO_SUPPORT')
            self.assertEqual(report['zero_motion_windows'], 1)

    def test_individual_zero_coverage_does_not_invalidate_nonzero_dataset(self):
        records = [assess_controls(controls(False), split='train'),
                   assess_controls(controls(True), split='train')]
        result = summarize(records)
        report = result['splits']['train']
        self.assertEqual(report['motion_status'], 'NONZERO_SUPPORT')
        self.assertEqual(report['reference_status'], 'NONZERO_SUPPORT')
        self.assertEqual(report['zero_reference_windows'], 1)
        self.assertEqual(report['reference_eligible_pairs'], 4)
        self.assertEqual(result['splits']['eval']['motion_status'], 'UNAVAILABLE_EMPTY_SPLIT')

    def test_observed_only_tracks_do_not_imply_future_motion(self):
        value = controls()
        value['track_weight'][2:] = 0
        value['motion'][2:, :, 2] = 0
        row = assess_controls(value)
        self.assertGreater(row['track_positive'], 0)
        self.assertFalse(row['future_motion_present'])
        self.assertFalse(row['reference_eligible'])

    def test_reference_requires_intersection_of_observed_and_future_query_ids(self):
        value = controls()
        value['prefix_track_weight'][:, 1] = 0
        value['track_weight'][2:, 0] = 0
        row = assess_controls(value)
        self.assertTrue(row['future_motion_present'])
        self.assertEqual(row['reference_source_queries'], 1)
        self.assertEqual(row['future_track_positive'], 2)
        self.assertEqual(row['reference_eligible_pairs'], 0)

    def test_coordinates_match_transport_inbounds_qualification(self):
        value = controls()
        value['tracks'][2:, 0, 0] = 1.1
        value['prefix_tracks'][:, 1, 1] = float('nan')
        row = assess_controls(value)
        self.assertEqual(row['future_track_positive'], 2)
        self.assertEqual(row['reference_source_queries'], 1)
        self.assertFalse(row['reference_eligible'])
        json.dumps(row, allow_nan=False)

    def test_raw_scores_are_actual_preserved_or_explicitly_unavailable(self):
        empty = assess_controls(controls())
        self.assertFalse(empty['raw_teacher_scores']['available'])
        current = assess_controls(controls(False), teacher=teacher_scores())
        score = current['raw_teacher_scores']['full']['per_frame'][0]
        self.assertLess(score['track_confidence']['quantiles']['100'], 1e-9)
        self.assertEqual(score['visibility']['pass_count'], 0)
        value = controls(False)
        value['metadata']['raw_teacher_scores'] = current['raw_teacher_scores']
        self.assertEqual(assess_controls(value)['raw_teacher_scores'], current['raw_teacher_scores'])

    def test_legacy_payload_scan_requires_no_new_metadata(self):
        payload = dict(samples=[sample(controls()), sample(controls(False))])
        rows = scan_prepared_payload(payload, split='eval', history=2)
        report = summarize(rows)['splits']['eval']
        self.assertEqual(report['windows'], 2)
        self.assertEqual(report['motion_status'], 'NONZERO_SUPPORT')
        self.assertEqual(report['raw_teacher_score_windows'], 0)

    def test_malformed_payload_not_silently_counted_as_zero_or_dropped(self):
        payload = dict(samples=[sample(), dict(id='bad', controls=None)])
        rows = scan_prepared_payload(payload, split='train')
        report = summarize(rows)['splits']['train']
        self.assertEqual(report['windows'], 2)
        self.assertEqual(report['invalid_records'], 1)
        self.assertEqual(report['motion_status'], 'UNAVAILABLE_INVALID_RECORDS')

    def test_prepared_controls_resume_rebuilds_sidecar_without_teacher(self):
        from scripts import prepare_scene_controls as prep
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root/'local', [str(root/'a'), str(root/'b')])
            item = sample(controls(False))
            item['controls']['metadata']['raw_teacher_scores'] = raw_teacher_scores(teacher_scores())
            key = window_key(item)
            path = f'prepare/cases/{key}/controls.pt'
            store.save(path, dict(window_key=key, controls=item['controls']), 'identity')
            (store.local/path).unlink()  # recover from durable replica
            def forbidden():
                raise AssertionError('Readiness restoration must not rerun teacher')
            restored, frames = prep.prepare_case(item, SimpleNamespace(history=2, grid=2),
                store, 'identity', {}, None, forbidden, split='train')
            self.assertEqual(frames, 0)
            sidecar = store.json(f'prepare/cases/{key}/readiness.json', True)
            self.assertFalse(sidecar['future_motion_present'])
            self.assertTrue(sidecar['raw_teacher_scores']['available'])
            for destination in ('a', 'b'):
                self.assertTrue((root/destination/f'prepare/cases/{key}/readiness.json').is_file())
            torch.testing.assert_close(restored['tracks'], item['controls']['tracks'])

    def test_legacy_sidecar_absence_does_not_fabricate_teacher_scores(self):
        from scripts import prepare_scene_controls as prep
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, [])
            item = sample(controls(False))
            report = prep.write_case_readiness(item, item['controls'], store, 'identity', split='eval')
            self.assertFalse(report['raw_teacher_scores']['available'])
            known = copy.deepcopy(report)
            known['raw_teacher_scores'] = raw_teacher_scores(teacher_scores())
            store.write_json(f'prepare/cases/{window_key(item)}/readiness.json', known)
            restored = prep.write_case_readiness(item, item['controls'], store, 'identity', split='eval')
            self.assertEqual(restored['raw_teacher_scores'], known['raw_teacher_scores'])

    def test_merge_exposes_invalid_conditions_without_blocking_camera_data(self):
        from scripts import prepare_scene_controls as prep
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, [])
            jobs = [dict(file=f'data/prepared/job{i:05d}.pt', job_index=i, kind=kind,
                         eval_index=0 if kind=='eval' else None, count=1, sources=['spatialvid'])
                    for i,kind in enumerate(('train', 'eval'))]
            for job in jobs:
                store.save(job['file'], dict(samples=[sample(controls(False))]), 'identity')
            store.write_json('prepare/plan.json', dict(jobs=jobs, bad_source_shards=[],
                selected_windows=1, selected_pool={}))
            store.write_json('prepare/nodes/worker000.json', dict(identity='identity',
                status='completed', jobs=jobs, errors=[]))
            ready = prep.merge(SimpleNamespace(workers=1, train_windows=1), store,
                dict(identity='identity', source_prefix='source', source_cache_manifest_sha='manifest'),
                dict(mean=[0.], std=[1.]))
            self.assertEqual(ready['samples'], 1)
            self.assertEqual(ready['control_readiness']['splits']['train']['motion_status'], 'INVALID_ZERO_SUPPORT')
            self.assertEqual(store.json('data/control_readiness.json')['splits']['eval']['reference_status'],
                             'INVALID_ZERO_SUPPORT')


if __name__ == '__main__':
    unittest.main()
