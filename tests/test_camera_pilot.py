"""Local tar/annotation integration tests; no remote access or accelerator work."""
import contextlib
import csv
import io
import json
from pathlib import Path
import random
import statistics
import tarfile
import tempfile
import unittest

import numpy as np
import torch

from scripts.prepare_camera_pilot import prepare


class CameraPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.annotation = self.root / 'annotations'
        self.rows = []
        self.shards = []
        self.samples = {}
        for i in range(4):
            vid = f'train{i}'
            self.add_video(vid, i + 1)
            samples = [self.sample(vid, list(range(j, j + 9))) for j in range(10)]
            shard = self.root / f'train{i}.tar'
            self.write_tar(shard, samples)
            self.shards.append(str(shard))
            self.samples[shard] = samples
        evaluation = []
        for i in range(4):
            vid = f'eval{i}'
            self.add_video(vid, 1000 + i)
            evaluation.append(self.sample(vid, list(range(9))))
        self.eval_shard = self.root / 'eval.tar'
        self.write_tar(self.eval_shard, evaluation)
        self.eval_samples = evaluation
        self.metadata = self.root / 'metadata.csv'
        self.write_metadata()
        self.train_manifest = self.root / 'train.txt'
        self.train_manifest.write_text('\n'.join(self.shards) + '\n')
        self.eval_manifest = self.root / 'eval.txt'
        self.eval_manifest.write_text(str(self.eval_shard) + '\n')

    @staticmethod
    def sample(vid, frames):
        return dict(video_id=vid, frame_indices=frames,
                    target=torch.zeros(4, 1, 1), cond=torch.zeros(1, 1, 1))

    def add_video(self, vid, speed):
        self.rows.append({'id': vid, 'video path': f'videos/group_0001/{vid}.mp4',
                          'fps': '32', 'motionTags': 'forward;right'})
        folder = self.annotation / 'group_0001' / vid
        folder.mkdir(parents=True)
        poses = np.zeros((25, 7), dtype=np.float64)
        poses[:, 0] = -np.arange(25) * speed
        poses[:, 6] = 1
        np.save(folder / 'poses.npy', poses)
        np.save(folder / 'intrinsics.npy', np.tile([1., 1., .5, .5], (25, 1)))
        (folder / 'indexes.txt').write_text('# total 25 indexes\n' +
            '\n'.join(f'{i} {i}' for i in range(25)) + '\n')

    def write_metadata(self):
        with self.metadata.open('w', newline='') as f:
            writer = csv.DictWriter(f, self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    @staticmethod
    def write_tar(path, samples):
        with tarfile.open(path, 'w') as archive:
            for i, sample in enumerate(samples):
                buffer = io.BytesIO()
                torch.save(sample, buffer)
                content = buffer.getvalue()
                info = tarfile.TarInfo(f'{i}.pt')
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))

    def run_prepare(self, **kwargs):
        args = dict(metadata=str(self.metadata), train_manifest=str(self.train_manifest),
                    eval_manifest=str(self.eval_manifest), annotation_root=str(self.annotation),
                    output_dir=str(self.root / 'out'), train_shards=3, seed=42,
                    expected_eval_windows=4, progress_dir=str(self.root / 'logs'))
        args.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(**args)

    def test_selection_exact_frames_train_only_scale_and_existing_files(self):
        out = self.root / 'out'
        out.mkdir()
        (out / 'initialization.json').write_text('{"step":12000}')
        audit = self.run_prepare()
        selected = sorted(random.Random(42).sample(self.shards, 3))
        self.assertEqual((out / 'train_manifest.txt').read_text().splitlines(), selected)
        self.assertEqual((out / 'eval_manifest.txt').read_text().splitlines(), [str(self.eval_shard)])
        bank = torch.load(out / 'camera_bank.pt', weights_only=False)
        expected_scale = statistics.median([8 * (int(Path(s).stem[-1]) + 1) for s in selected])
        self.assertEqual(bank['translation_scale'], expected_scale)
        self.assertEqual(audit['counts']['train'], {'attempted': 30, 'valid': 30, 'invalid': 0})
        self.assertEqual(len(bank['train_keys']), 30)
        self.assertEqual(len(bank['eval_keys']), 4)
        self.assertEqual(audit['status'], 'complete')
        self.assertEqual(audit['DI_throughput'], 0.)
        self.assertTrue((out / 'initialization.json').exists())
        self.assertEqual(audit['motion_tags']['train'], {'forward': 30, 'right': 30})
        key = bank['train_keys'][0]
        features = bank['features'][key]
        self.assertEqual(features.shape, (9, 14))
        self.assertEqual(features.dtype, torch.float32)
        self.assertAlmostEqual(float(features[-1, -1]), 8 / 32)
        self.assertEqual(len(audit['output_sha256']['camera_bank.pt']), 64)
        self.assertEqual(len(audit['annotation_sources']), 7)
        # Existing committed inputs must never be reselected or overwritten.
        (out / 'inputs.json').write_text('{}')
        original = (out / 'camera_bank.pt').read_bytes()
        with self.assertRaisesRegex(ValueError, 'committed'):
            self.run_prepare(seed=99)
        self.assertEqual(original, (out / 'camera_bank.pt').read_bytes())

    def test_invalid_train_window_skipped_and_ledger_keeps_exact_frames(self):
        selected = sorted(random.Random(42).sample(self.shards, 3))
        shard = Path(selected[0])
        self.samples[shard][0]['frame_indices'] = list(range(8)) + [99]
        self.write_tar(shard, self.samples[shard])
        audit = self.run_prepare()
        self.assertEqual(audit['counts']['train']['valid'], 29)
        self.assertEqual(audit['counts']['train']['invalid'], 1)
        self.assertEqual(audit['failures'][0]['frame_indices'][-1], 99)
        self.assertIn('extrapolation', audit['failures'][0]['error'])

    def test_eval_failure_is_fatal_and_audited(self):
        (self.annotation / 'group_0001/eval0/poses.npy').unlink()
        with self.assertRaisesRegex(ValueError, 'reviewed evaluation'):
            self.run_prepare()
        out = self.root / 'out'
        audit = json.loads((out / 'camera_audit.json').read_text())
        self.assertEqual(audit['status'], 'failed')
        self.assertEqual(audit['counts']['eval']['invalid'], 1)
        self.assertEqual(json.loads((self.root / 'logs/camera_audit.json').read_text()), audit)
        self.assertFalse((out / 'camera_bank.pt').exists())

    def test_training_failure_budget_and_shard_empty_are_fatal(self):
        selected = sorted(random.Random(42).sample(self.shards, 3))
        shard = Path(selected[0])
        self.write_tar(shard, [self.sample('missing', list(range(9)))])
        with self.assertRaisesRegex(ValueError, 'no valid windows'):
            self.run_prepare()
        audit = json.loads((self.root / 'out/camera_audit.json').read_text())
        self.assertEqual(audit['counts']['train']['valid'], 20)
        self.assertEqual(audit['status'], 'failed')
        self.write_tar(shard, [self.sample('missing', list(range(9))) for _ in range(10)])
        with self.assertRaisesRegex(ValueError, '90%'):
            self.run_prepare()

    def test_split_overlap_and_default_eval_count_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '128'):
            self.run_prepare(expected_eval_windows=128)
        selected = sorted(random.Random(42).sample(self.shards, 3))
        vid = Path(selected[0]).stem
        self.eval_samples[0]['video_id'] = vid
        self.write_tar(self.eval_shard, self.eval_samples)
        with self.assertRaisesRegex(ValueError, 'video IDs overlap'):
            self.run_prepare()


if __name__ == '__main__':
    unittest.main()
