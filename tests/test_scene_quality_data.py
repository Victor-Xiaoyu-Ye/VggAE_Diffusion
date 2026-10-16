"""Street filtering must not leak replay, held-out clips or auxiliary domains."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from utils.scene_quality_data import (
    FilteredLatentSamples, filter_street_manifest,
    load_street_selection, select_street_samples,
)


def sample(ident, dataset='spatialvid'):
    return dict(id=ident, dataset=dataset, anchor=['saved-independent-anchor'],
                target=['saved-causal-target'], has_caption=True)


class FakeStore:
    def __init__(self, files):
        self.files, self.calls = files, []

    def load(self, name, identity, required=False):
        self.calls.append((name, identity, required))
        value = self.files[name]
        if isinstance(value, Exception):
            raise value
        return value


class SceneQualityDataTests(unittest.TestCase):
    def setUp(self):
        self.selection = dict(allowed_ids=['spatialvid:a', 'spatialvid:b'])
        self.manifest = dict(identity='original-cache', mean=[1., 2.], std=[3., 4.],
            normalization='original train per-channel', shards=[
                dict(file='a.pt', count=1, sources=['spatialvid']),
                dict(file='b.pt', count=1, sources=['spatialvid'])])
        self.files = {'a.pt': dict(samples=[sample('spatialvid:a')]),
                      'b.pt': dict(samples=[sample('spatialvid:b')])}

    def test_real_selection_is_exact_role_split_and_pins_cohort(self):
        repo = Path(__file__).resolve().parents[1]
        selection = load_street_selection(repo/'configs/scene_cohort_v1.json')
        self.assertEqual(selection['count'], 8192)
        self.assertEqual(len(selection['val_ids']), 128)
        self.assertEqual(len(selection['test_ids']), 128)
        self.assertFalse(set(selection['allowed_ids']) & set(selection['val_ids']))
        self.assertFalse(set(selection['allowed_ids']) & set(selection['test_ids']))
        cohort = json.loads((repo/'configs/scene_cohort_v1.json').read_text())
        from_dict = load_street_selection(cohort)
        self.assertEqual(from_dict, selection)
        old = Path(r'D:/workspace/VggAE_DataAudit/rebuild_data_20260923/selection_v1/train_candidates.jsonl')
        if old.exists():
            original = load_street_selection(repo/'configs/scene_cohort_v1.json', old)
            self.assertEqual(original['allowed_ids'], selection['allowed_ids'])
            self.assertEqual(original['source_selection_sha256'], selection['source_selection_sha256'])
        cohort['seed'] = 'different-source-cohort'
        with self.assertRaisesRegex(ValueError, 'another cohort'):
            load_street_selection(cohort)

    def test_role_filter_excludes_replay_even_when_motion_label_matches(self):
        cohort = dict(records=[
            dict(id='spatialvid:a', dataset='spatialvid', split='train'),
            dict(id='spatialvid:r', dataset='spatialvid', split='train'),
            dict(id='spatialvid:v', dataset='spatialvid', split='val')])
        rows = [dict(cohort['records'][0], role='real_street_video_candidate', motion_stratum='moderate'),
                dict(cohort['records'][1], role='appearance_replay', motion_stratum='moderate'),
                dict(cohort['records'][2], role='real_street_video_candidate')]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'train_candidates.jsonl'
            path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf8')
            chosen = load_street_selection(cohort, path)
            self.assertEqual(chosen['allowed_ids'], ['spatialvid:a'])
            rows[0]['id'] = 'spatialvid:v'
            path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'non-train'):
                load_street_selection(cohort, path)

    def test_selection_rejects_source_provenance_change(self):
        cohort = dict(input_sha256={'train_candidates.jsonl': 'expected-source-hash'}, records=[
            dict(id='spatialvid:a', dataset='spatialvid', split='train')])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'train_candidates.jsonl'
            path.write_text(json.dumps(dict(cohort['records'][0],
                role='real_street_video_candidate')), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_street_selection(cohort, path)

    def test_select_samples_drops_replay_and_auxiliary_but_rejects_bad_tags(self):
        street = sample('spatialvid:a')
        values = [sample('dl3dv:other', 'dl3dv'), sample('spatialvid:replay'), street]
        self.assertEqual(select_street_samples(values, self.selection), [street])
        self.assertIs(select_street_samples(values, self.selection)[0], street)
        with self.assertRaises(ValueError):
            select_street_samples([sample('mystery:x', 'mystery')], self.selection)
        with self.assertRaises(ValueError):
            select_street_samples([sample('dl3dv:a')], self.selection)

    def test_manifest_view_preserves_payload_positions_and_original_statistics(self):
        original = dict(self.manifest, shards=[dict(file='mixed.pt', count=4,
            sources=['dl3dv', 'spatialvid', 'spatialvid', 'spatialvid'],
            ids=['dl3dv:x', 'spatialvid:a', 'spatialvid:replay', 'spatialvid:b'])])
        before = copy.deepcopy(original)
        filtered = filter_street_manifest(original, self.selection)
        self.assertEqual(filtered['shards'][0]['indices'], [1, 3])
        self.assertEqual(filtered['shards'][0]['count'], 4)
        self.assertEqual(filtered['selected_count'], 2)
        self.assertEqual(filtered['mean'], before['mean'])
        self.assertEqual(filtered['std'], before['std'])
        self.assertEqual(original, before)
        narrower = filter_street_manifest(original, self.selection, {'mixed.pt': [3]})
        self.assertEqual(narrower['shards'][0]['indices'], [3])
        with self.assertRaisesRegex(ValueError, 'index'):
            filter_street_manifest(original, indexes={'mixed.pt': [-1]})
        invalid = copy.deepcopy(original)
        invalid['shards'][0]['sources'][0] = 'unexpected'
        with self.assertRaisesRegex(ValueError, 'Unknown dataset'):
            filter_street_manifest(invalid)

    def test_cursor_sampling_is_order_independent_and_resume_exact(self):
        one = FilteredLatentSamples(FakeStore(self.files), self.manifest, self.selection, seed=19)
        expected = {i: one.get(i)['id'] for i in range(100)}
        two = FilteredLatentSamples(FakeStore(self.files), self.manifest, self.selection, seed=19)
        two.load_state_dict(one.state_dict())
        for i in reversed(range(100)):
            self.assertEqual(two.get(i)['id'], expected[i])
        self.assertEqual(set(expected.values()), {'spatialvid:a', 'spatialvid:b'})
        self.assertEqual(one.manifest['mean'], [1., 2.])
        other_seed = FilteredLatentSamples(FakeStore(self.files), self.manifest, self.selection, seed=20)
        with self.assertRaises(ValueError):
            other_seed.load_state_dict(one.state_dict())

    def test_bad_shard_is_logged_once_and_remains_excluded_after_repair_on_resume(self):
        errors = []
        files = dict(self.files, **{'a.pt': OSError('bad replica payload')})
        bank = FilteredLatentSamples(FakeStore(files), self.manifest, self.selection,
                                     max_attempts=64, on_bad_shard=errors.append)
        for i in range(50):
            self.assertEqual(bank.get(i)['id'], 'spatialvid:b')
        self.assertEqual([r['file'] for r in errors], ['a.pt'])
        resumed = FilteredLatentSamples(FakeStore(self.files), self.manifest, self.selection,
                                        max_attempts=64)
        resumed.load_state_dict(bank.state_dict())
        for i in range(50, 100):
            self.assertEqual(resumed.get(i)['id'], bank.get(i)['id'])
            self.assertEqual(resumed.get(i)['id'], 'spatialvid:b')

    def test_replay_contamination_is_not_returned_and_total_failure_is_bounded(self):
        errors = []
        files = {'a.pt': dict(samples=[sample('spatialvid:replay')]),
                 'b.pt': RuntimeError('hash mismatch')}
        store = FakeStore(files)
        bank = FilteredLatentSamples(store, self.manifest, self.selection,
                                     max_attempts=16, on_bad_shard=errors.append)
        with self.assertRaisesRegex(RuntimeError, 'after 16 attempts'):
            bank.get(0)
        self.assertLessEqual(len(store.calls), 2)
        self.assertTrue(any('Non-allowlisted' in e['error'] for e in errors))
        with self.assertRaises(ValueError):
            bank.get(-1)

    def test_exact_old_shard_view_can_skip_known_auxiliary_slots(self):
        manifest = dict(self.manifest, shards=[dict(file='mixed.pt', count=3,
            sources=['dl3dv', 'spatialvid', 'spatialvid'], indices=[2])])
        store = FakeStore({'mixed.pt': dict(samples=[sample('dl3dv:x', 'dl3dv'),
                                                    sample('spatialvid:replay'), sample('spatialvid:b')])})
        bank = FilteredLatentSamples(store, manifest, self.selection)
        for i in range(10):
            self.assertEqual(bank.get(i)['id'], 'spatialvid:b')


if __name__ == '__main__':
    unittest.main()
