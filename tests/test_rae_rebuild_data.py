import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_rae_rebuild_data import (
    digest, direct_ids, group_partition, prepare, safe_relative, spatial_stratum,
)
from scripts import audit_rae_rebuild_metadata as audit


ROOT = 'data/external/x00445638/data/train_spatial/open_datasets/'
REPO = Path(__file__).resolve().parents[1]


class RebuildDataTests(unittest.TestCase):
    def test_partition_is_order_independent_and_reserves_whole_scene(self):
        ids = [f'scene{i}' for i in range(15)]
        a = group_partition(ids, 'seed', 3, 2, ['scene0', 'scene1'])
        self.assertEqual(a, group_partition(ids[::-1], 'seed', 3, 2, ['scene1', 'scene0']))
        self.assertEqual((a['scene0'], a['scene1']), ('test', 'test'))
        # Every derived frame/window inherits this map, never a separate hash.
        self.assertEqual(sum(s == 'test' for s in a.values()), 4)
        with self.assertRaises(ValueError):
            group_partition(ids + ['scene0'], 'seed', 3, 2)

    def test_truncated_or_unrelated_listing_cannot_define_training_pool(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'listing.json'
            p.write_text(json.dumps({'items': [], 'truncated': True}))
            with self.assertRaises(ValueError):
                direct_ids(p, 'dataset', '[0-9]+')
            p.write_text(json.dumps({'items': [{'prefix': ROOT + 'other/0001/'}], 'truncated': False}))
            with self.assertRaises(ValueError):
                direct_ids(p, 'dataset', '[0-9]+')

    def test_unsafe_paths_and_nonfinite_metadata_are_rejected(self):
        for value in ('../data', 'obs://bucket/file', '/root', 'a//b', 'a/./b'):
            with self.assertRaises(ValueError):
                safe_relative(value)
        policy = json.loads((REPO / 'configs/rae_rebuild_data_v1.json').read_text())['spatialvid']
        row = dict(sceneType='Urban;Street Scene', timeOfDay='Daytime', brightness='Bright',
            fps='24', resolution='1280x720', dynamicRatio='0.1', trajTurns='0')
        row.update({'num frames': '240', 'aesthetic score': '5', 'ocr score': '.01', 'motion score': '4'})
        self.assertEqual(spatial_stratum(row, policy), 'moderate_dynamic_proxy')
        row['dynamicRatio'] = 'nan'
        self.assertIsNone(spatial_stratum(row, policy))

    def test_object_audit_rejects_missing_paired_camera(self):
        with tempfile.TemporaryDirectory() as d, patch.object(audit, 'OUT', Path(d)):
            with patch.object(audit, 'listing', side_effect=[{f'{i:04}.png': 1 for i in range(9)}, {f'{i:04}.npz': 1 for i in range(8)}]):
                result = audit.probe(('dl3dv', 'scene', 'example/scene'))
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['missing_cameras'], 1)
            self.assertFalse(result['media_downloaded'])

    def test_full_materialization_excludes_test_parents_and_never_claims_decode(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            snapshots = p / 'snapshots'
            snapshots.mkdir()
            (snapshots / 'probes').mkdir()
            def write(path, data):
                path.write_text(json.dumps(data), encoding='utf8')
            def listing(rel, ids):
                write(snapshots / (rel.replace('/', '__') + '.json'), dict(truncated=False,
                    items=[{'prefix': ROOT + rel + '/' + i + '/'} for i in ids]))
            def receipt(dataset, ident, root, **fields):
                write(snapshots / 'probes' / f'{dataset}_{ident}.json',
                    dict(dataset=dataset, id=ident, root=root, status='metadata_pass', **fields))
            policy = json.loads((REPO / 'configs/rae_rebuild_data_v1.json').read_text())
            policy['dl3dv'] = {'val_scenes': 1, 'test_scenes': 1}
            policy['mvssynth'] = {'val_scenes': 1, 'test_scenes': 1}
            policy['omniworld']['val_scenes'] = 1
            policy['spatialvid'].update(train_max=4, val_clips=1, test_clips=1, replay_train_max=1)
            policy_path = p / 'policy.json'
            write(policy_path, policy)
            valid_rows = []
            dl_ids = [f'{i:064x}' for i in range(1, 8)]
            for k, ident in enumerate(dl_ids, 1):
                rel = f'DL3DV_v1/DL3DV-ALL-480P-NEW/processed_dl3dv_ours/{k}K'
                listing(rel, [ident])
                listing(f'DL3DV_v1/processed_dl3dv_ours/{k}K', [ident])
                receipt('dl3dv', ident, rel + '/' + ident, rgb_frames=100, camera_frames=100)
                valid_rows.append({'hash': ident, 'batch': f'{k}K'})
            with (snapshots / 'DL3DV-valid.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, ['hash', 'batch']); w.writeheader(); w.writerows(valid_rows)
            write(snapshots / 'dl3dv_stored_test_index.json', ['DL3DV-10K/1K/' + dl_ids[0]])
            rel = 'MVS-Synth/GTAV_1080'
            listing(rel, ['0000', '0001', '0002'])
            for ident in ('0000', '0001', '0002'):
                receipt('mvssynth', ident, rel + '/' + ident, rgb_frames=100)
            omni_ids = ['000000000001', '000000000002', '000000000003']
            rel = 'OmniWorld/videos/OmniWorld-Game'
            listing(rel, omni_ids)
            listing('OmniWorld/annotations/OmniWorld-Game', omni_ids)
            for ident in omni_ids:
                receipt('omniworld', ident, rel + '/' + ident, split_count=2, split_frames=[100, 100],
                    rgb_frames=200, caption_files=2, split_info_sha256='abc')
            with (snapshots / 'omniworld_game_metadata.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, ['UID', 'Split Num', 'Test Split Index', 'FPS', 'Metric Scale'])
                w.writeheader()
                w.writerows({'UID': i, 'Split Num': 2, 'Test Split Index': '0' if i == omni_ids[0] else '',
                             'FPS': 24, 'Metric Scale': 1} for i in omni_ids)
            spatial_rows = []
            inventory_rows = []
            for i in range(24):
                ident = f'video{i}'
                row = {'id': ident, 'video path': f'videos/group_0001/{ident}.mp4',
                       'annotation path': f'annotations/group_0001/{ident}', 'sceneType': 'Urban;Street Scene',
                       'timeOfDay': 'Daytime', 'brightness': 'Bright', 'resolution': '1280x720', 'num frames': '240',
                       'fps': '24', 'aesthetic score': '5', 'ocr score': '.01', 'motion score': '4',
                       'dynamicRatio': '.2' if i % 3 == 0 else '.1', 'trajTurns': '0'}
                spatial_rows.append(row)
                inventory_rows.append({'key': policy['spatialvid_root'].split('/', 3)[3] + 'videos/SpatialVID/' + row['video path'], 'size': 100})
            metadata = p / 'metadata.csv'
            with metadata.open('w', newline='') as f:
                w = csv.DictWriter(f, spatial_rows[0].keys()); w.writeheader(); w.writerows(spatial_rows)
            inventory = p / 'inventory.json'
            write(inventory, dict(items=inventory_rows, truncated=False))
            history = p / 'history.json'
            write(history, {'metadata_sha256': digest(metadata), 'availability': {'inventory_sha256': digest(inventory)},
                'splits': {'eval_street_128': ['video0'], 'test_street_128': ['video1'],
                           'train_mixed_2048': ['video2', 'video3', 'video4', 'video5'],
                           'train_single_2048': ['video2', 'video3', 'video4', 'video5']}})
            out = p / 'out'
            report = prepare(snapshots, metadata, inventory, history, policy_path, out)
            train = [json.loads(x) for x in (out / 'train_candidates.jsonl').read_text().splitlines()]
            self.assertNotIn('omniworld:' + omni_ids[0], {x['id'] for x in train})
            self.assertNotIn('dl3dv:' + dl_ids[0], {x['id'] for x in train})
            self.assertFalse({'spatialvid:video0', 'spatialvid:video1'} & {x['id'] for x in train})
            self.assertTrue(all(x['training_ready'] is False for x in train))
            self.assertFalse(report['physical_scene_disjoint_verified'])
            self.assertEqual(report['old_train_replay'], 1)
            # Re-running into an existing output must not silently mutate splits.
            with self.assertRaises(ValueError):
                prepare(snapshots, metadata, inventory, history, policy_path, out)
            out2 = p / 'out2'
            again = prepare(snapshots, metadata, inventory, history, policy_path, out2)
            self.assertEqual(report['files_sha256'], again['files_sha256'])


if __name__ == '__main__':
    unittest.main()
