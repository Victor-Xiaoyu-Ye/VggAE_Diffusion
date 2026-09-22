"""Small CPU fixtures for immutable, shared camera-pilot initialization.

Only metadata uses the full model architecture; fake EMA tensors stay tiny.
These tests neither download source checkpoints nor allocate a 1.65B model.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage_camera_pilot import fetch_inputs, freeze, publish_inputs, sha256
from utils.moxing_io import copy_file
from utils.window_codec import runtime_contract


class CameraStagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root/'leader'
        self.output.mkdir()
        self.roots = [str(self.root/'current'), str(self.root/'mirror')]
        self.source = self.root/'source.pt'
        self.names = ('train_manifest.txt', 'eval_manifest.txt',
                      'camera_audit.json', 'camera_bank.pt')
        self.payload = dict(
            schema='r7-window-trainer-v1', step=6000,
            model_args=dict(channels=192, grid=18, future=4, temporal_factor=2,
                            width=1536, depth=24, heads=24, text_dim=4096),
            statistics=dict(representation=dict(window_codec_runtime=runtime_contract('legacy'))),
            contract=dict(identity=dict(source='fixture')),
            ema={'weight': torch.arange(6.)},
            optimizer={'sentinel': torch.ones(10)},
            model={'weight': torch.zeros(6)},
        )
        torch.save(self.payload, self.source)

    def prepare(self):
        freeze(self.output, self.roots, [str(self.source)])
        for name in self.names:
            (self.output/name).write_text('fixture '+name, encoding='utf-8')
        publish_inputs(self.output, self.roots, self.names)
        return json.loads((self.output/'initialization.json').read_text())

    def test_compact_freeze_is_shared_and_live_source_cannot_change_it(self):
        receipt = self.prepare()
        frozen = self.output/receipt['file']
        compact = torch.load(frozen, map_location='cpu', weights_only=False, mmap=True)
        self.assertEqual(compact['step'], 6000)
        self.assertTrue(compact['initialization_only'])
        self.assertNotIn('optimizer', compact)
        self.assertNotIn('model', compact)
        torch.testing.assert_close(compact['ema']['weight'], self.payload['ema']['weight'])
        self.assertEqual(receipt['source_sha256'], sha256(self.source))
        self.assertFalse((self.output/'source_checkpoint.pt').exists())
        for root in self.roots:
            self.assertEqual(sha256(Path(root)/receipt['file']), receipt['sha256'])
        self.payload['step'] = 9000
        self.payload['ema']['weight'].add_(100)
        torch.save(self.payload, self.source)
        freeze(self.output, self.roots, [str(self.source)])
        self.assertEqual(json.loads((self.output/'initialization.json').read_text()), receipt)
        self.assertEqual(sha256(frozen), receipt['sha256'])

    def test_corrupt_first_copy_falls_back_and_both_corrupt_copies_fail(self):
        self.prepare()
        worker = self.root/'worker'
        worker.mkdir()
        original = (self.output/'camera_bank.pt').read_bytes()
        (Path(self.roots[0])/'camera_bank.pt').write_bytes(b'corrupt first copy')
        self.assertEqual(fetch_inputs(worker, self.roots), 0)
        self.assertEqual((worker/'camera_bank.pt').read_bytes(), original)
        (worker/'camera_bank.pt').write_bytes(b'corrupt local copy')
        (Path(self.roots[1])/'camera_bank.pt').write_bytes(b'corrupt second copy')
        with self.assertRaisesRegex(RuntimeError, 'No valid copy of camera_bank.pt'):
            fetch_inputs(worker, self.roots)

    def test_existing_cohort_cannot_be_replaced(self):
        self.prepare()
        saved = [json.loads((Path(root)/'inputs.json').read_text()) for root in self.roots]
        original = (Path(self.roots[0])/'train_manifest.txt').read_bytes()
        (self.output/'train_manifest.txt').write_text('a different cohort', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Refusing to replace frozen camera input cohort'):
            publish_inputs(self.output, self.roots, self.names)
        for index, root in enumerate(self.roots):
            self.assertEqual(json.loads((Path(root)/'inputs.json').read_text()), saved[index])
            self.assertEqual((Path(root)/'train_manifest.txt').read_bytes(), original)

    def test_leader_reuse_repairs_partial_double_publication(self):
        freeze(self.output, self.roots, [str(self.source)])
        for name in self.names:
            (self.output/name).write_text('fixture '+name, encoding='utf-8')

        def fail_second_root(source, destination):
            if destination == str(Path(self.roots[1])/'camera_bank.pt'):
                raise OSError('simulated second-destination failure')
            return copy_file(source, destination)

        with mock.patch('scripts.stage_camera_pilot.copy_file', side_effect=fail_second_root):
            with self.assertRaisesRegex(RuntimeError, 'double publication failed'):
                publish_inputs(self.output, self.roots, self.names)
        self.assertTrue((self.output/'inputs.json').exists())
        self.assertTrue((Path(self.roots[0])/'inputs.json').exists())
        self.assertFalse((Path(self.roots[1])/'inputs.json').exists())
        with mock.patch('scripts.stage_camera_pilot.copy_file', wraps=copy_file) as copies:
            self.assertEqual(fetch_inputs(self.output, self.roots, allow_missing=True), 0)
        # The existing initialization commit avoids resending its checkpoint.
        initialization = json.loads((self.output/'initialization.json').read_text())
        self.assertFalse(any(Path(call.args[0]).name == initialization['file']
                             for call in copies.call_args_list))
        for root in self.roots:
            receipt = json.loads((Path(root)/'inputs.json').read_text())
            for name, info in receipt['files'].items():
                self.assertEqual(sha256(Path(root)/name), info['sha256'])

    def test_leader_reuse_repairs_missing_committed_file(self):
        self.prepare()
        missing = Path(self.roots[1])/'camera_bank.pt'
        missing.unlink()
        self.assertEqual(fetch_inputs(self.output, self.roots, allow_missing=True), 0)
        self.assertEqual(sha256(missing), sha256(self.output/'camera_bank.pt'))

    def test_wrong_source_architecture_is_rejected_before_publication(self):
        self.payload['model_args']['width'] = 768
        torch.save(self.payload, self.source)
        with self.assertRaisesRegex(ValueError, '1.65B source architecture'):
            freeze(self.output, self.roots, [str(self.source)])
        self.assertFalse((self.output/'initialization.json').exists())
        self.assertTrue(all(not (Path(root)/'initialization.json').exists() for root in self.roots))


if __name__ == '__main__':
    unittest.main()
