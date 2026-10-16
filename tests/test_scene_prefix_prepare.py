"""Exercise real local dual ArtifactStore cache preparation and interruption.

Only metric presentation is replaced. The source read, checksum verification,
torch serialization, filtering, cursor state and both publication roots remain
real. Fixtures are tiny and never access OBS or pretrained weights.
"""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import train_scene_prefix as trainer
import utils.scene_run as scene_run
from utils.scene_run import ArtifactStore, StageBudget


SOURCE_ID = 'frozen-cache-fixture'
OUTPUT_ID = 'street-prefix-fixture'


def sample(ident, dataset='spatialvid', offset=0.):
    target = (torch.arange(9*4*4).reshape(9, 4, 4).float()/100 + offset).bfloat16()
    return dict(id=ident, dataset=dataset, target=target, anchor=target[:1].clone(),
                frame_ids=list(range(0, 36, 4)), text=torch.ones(1, 8),
                rays=torch.ones(9, 4, 8), has_caption=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrefixPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='scene-prefix-prepare-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.production = ArtifactStore(self.root/'production', [str(self.root/'source-mirror')])
        self.source = ArtifactStore(self.root/'private-inputs',
            [str(self.production.local), str(self.root/'source-mirror')])
        self.output = ArtifactStore(self.root/'output',
            [str(self.root/'output-primary'), str(self.root/'output-mirror')])
        self.selection = dict(allowed_ids=['spatialvid:a', 'spatialvid:b'])
        self.manifest = dict(identity=SOURCE_ID, mean=[.1, .2, .3, .4],
                             std=[.8, .9, 1.1, 1.2], shards=[])
        self.input_hashes = {}

    def shard(self, name, samples):
        name = 'cache/'+name+'.pt'
        self.production.save(name, dict(samples=samples), SOURCE_ID)
        self.manifest['shards'].append(dict(file=name, count=len(samples),
            sources=[s['dataset'] for s in samples]))
        for root in self.source.roots:
            for suffix in ('', '.json'):
                path = Path(root)/(name+suffix)
                self.input_hashes[path] = sha(path)
        return name

    def run_prepare(self):
        with patch.object(trainer, 'metrics'):
            return trainer.compact_cache(self.source, self.output, self.manifest,
                self.selection, OUTPUT_ID, StageBudget(3600), torch.device('cpu'), 0, 1)

    def assert_sources_intact(self):
        for path, expected in self.input_hashes.items():
            self.assertTrue(path.is_file(), path)
            self.assertEqual(sha(path), expected, path)

    def test_filter_preserves_original_statistics_anchor_and_only_cleans_private_staging(self):
        first, second = sample('spatialvid:a'), sample('spatialvid:b', offset=1.)
        source_name = self.shard('mixed', [sample('dl3dv:aux', 'dl3dv'),
            first, sample('spatialvid:replay'), second])
        replay_name = self.shard('replay-only', [sample('spatialvid:replay')])
        before = copy.deepcopy(self.manifest)
        result = self.run_prepare()

        self.assertEqual(self.manifest, before)
        self.assertEqual(result['mean'], before['mean'])
        self.assertEqual(result['std'], before['std'])
        self.assertEqual(result['source_identity'], SOURCE_ID)
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['bad'], [])
        self.assertEqual(len(result['shards']), 1)
        self.assertEqual(result['shards'][0]['sources'], ['spatialvid', 'spatialvid'])
        name = result['shards'][0]['file']
        values = self.output.load(name, OUTPUT_ID, True)['samples']
        self.assertEqual([s['id'] for s in values], ['spatialvid:a', 'spatialvid:b'])
        for got, expected in zip(values, (first, second)):
            self.assertEqual(set(got), {'id', 'dataset', 'anchor', 'target', 'frame_ids'})
            torch.testing.assert_close(got['target'], expected['target'], rtol=0, atol=0)
            torch.testing.assert_close(got['anchor'], expected['anchor'], rtol=0, atol=0)
            self.assertEqual(got['frame_ids'], expected['frame_ids'])
        self.assertEqual(self.output.json('prepare/rank000.json', True)['cursor'], 2)
        for root in self.output.roots:
            for suffix in ('', '.json'):
                self.assertEqual(sha(Path(root)/(name+suffix)),
                                 sha(self.output.local/(name+suffix)))
            self.assertTrue((Path(root)/'data/cache_complete.json').is_file())
        for name in (source_name, replay_name):
            self.assertFalse((self.source.local/name).exists())
            self.assertFalse((self.source.local/(name+'.json')).exists())
        self.assert_sources_intact()

    def test_corrupt_input_and_invalid_anchor_are_recorded_then_good_input_continues(self):
        corrupt = self.shard('corrupt', [sample('spatialvid:a')])
        invalid = sample('spatialvid:a')
        invalid['anchor'] = invalid['anchor'] + 1
        self.shard('invalid-anchor', [invalid])
        self.shard('good', [sample('spatialvid:b')])
        for root in self.source.roots:
            (Path(root)/corrupt).write_bytes(b'broken-checksum-payload')
        result = self.run_prepare()
        self.assertEqual(result['samples'], 1)
        self.assertEqual(len(result['bad']), 2)
        self.assertIn('Unrecoverable committed artifact', result['bad'][0]['error'])
        self.assertIn('anchor contract', result['bad'][1]['error'])
        state = self.output.json('prepare/rank000.json', True)
        self.assertEqual(state['cursor'], 3)
        self.assertEqual(state['bad'], result['bad'])
        self.assertEqual(self.output.load(result['shards'][0]['file'], OUTPUT_ID, True)
                         ['samples'][0]['id'], 'spatialvid:b')

    def test_both_output_roots_failing_cannot_acknowledge_cursor_or_mark_bad_input(self):
        source_name = self.shard('good', [sample('spatialvid:a')])
        original_copy = scene_run.copy_file
        roots = [Path(root).resolve() for root in self.output.roots]
        def fail_output(source, destination):
            path = Path(destination).resolve()
            if any(root == path or root in path.parents for root in roots):
                raise OSError('fixture output destination unavailable')
            return original_copy(source, destination)

        with patch.object(scene_run, 'copy_file', side_effect=fail_output):
            with self.assertRaisesRegex(RuntimeError, 'No durable output destination'):
                self.run_prepare()
        self.assertIsNone(self.output.json('prepare/rank000.json'))
        self.assertIsNone(self.output.json('data/cache_complete.json'))
        status = self.output.json(self.output.status_name, True)
        self.assertEqual(status['success'], [])
        self.assertEqual(len(status['errors']), 2)
        self.assertFalse((self.source.local/source_name).exists())
        self.assert_sources_intact()

        # A local receipt can exist even though neither remote accepted it.
        # The retry must publish it successfully before advancing the cursor.
        with patch.object(self.source, 'load', side_effect=AssertionError('source should not be reread')):
            result = self.run_prepare()
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['bad'], [])
        self.assertEqual(self.output.json('prepare/rank000.json', True)['cursor'], 1)
        for root in self.output.roots:
            self.assertTrue((Path(root)/result['shards'][0]['file']).is_file())

    def test_interruption_after_shard_publication_resumes_without_read_or_duplicate(self):
        self.shard('good', [sample('spatialvid:a'), sample('spatialvid:b')])
        original_save = self.output.save
        def publish_then_interrupt(name, payload, identity):
            original_save(name, payload, identity)
            raise RuntimeError('fixture interruption after durable shard publication')

        with patch.object(self.output, 'save', side_effect=publish_then_interrupt):
            with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                self.run_prepare()
        self.assertIsNone(self.output.json('prepare/rank000.json'))
        receipt = self.output.json('data/shards/source00000.pt.json', True)
        with patch.object(self.source, 'load', side_effect=AssertionError('source should not be reread')):
            with patch.object(self.output, 'save', wraps=original_save) as write:
                result = self.run_prepare()
                write.assert_not_called()
        self.assertEqual(result['samples'], 2)
        self.assertEqual(len(result['shards']), 1)
        self.assertEqual(self.output.json('data/shards/source00000.pt.json', True), receipt)
        state = self.output.json('prepare/rank000.json', True)
        self.assertEqual(state['cursor'], 1)
        self.assertEqual(len(state['shards']), 1)
        self.assertEqual(state['bad'], [])
        self.assert_sources_intact()
        with patch.object(self.source, 'load', side_effect=AssertionError('completed cache reread')):
            self.assertEqual(self.run_prepare(), result)


if __name__ == '__main__':
    unittest.main()
