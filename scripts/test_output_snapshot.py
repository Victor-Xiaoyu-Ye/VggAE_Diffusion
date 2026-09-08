#!/usr/bin/env python3
"""No torch/OBS needed: exercise stable output sync and failure propagation."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.output_snapshot import build_snapshot, cleanup_snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.src = self.base/'output'; self.src.mkdir()

    def write(self, name, data='test'):
        p = self.src/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)
        return p

    def snapshot(self):
        root, stats = build_snapshot(self.src)
        self.addCleanup(cleanup_snapshot, root)
        return Path(root), stats

    def test_intermediates_and_snapshot_subtrees_excluded(self):
        self.write('checkpoint_latest.pt')
        for name in ('checkpoint_latest.pt.tmp-123','a.pending.json','x.partial',
                     'x.part','stage.tmp-1/inner.dat','.vggae-sync-123/checkpoint.pt'):
            self.write(name)
        self.write('tmp_stats.json')
        root, _ = self.snapshot()
        self.assertEqual({p.name for p in root.rglob('*') if p.is_file()},
                         {'checkpoint_latest.pt','tmp_stats.json'})
        self.assertTrue((self.src/'checkpoint_latest.pt.tmp-123').exists())

    def test_replace_does_not_mutate_snapshot(self):
        p = self.write('checkpoint_latest.pt','old')
        root, stats = self.snapshot()
        replacement = self.write('checkpoint_latest.pt.tmp-1','new')
        os.replace(replacement,p)
        self.assertEqual((root/p.name).read_text(),'old')
        self.assertEqual(p.read_text(),'new')

    def test_logs_are_copied_not_linked(self):
        p = self.write('metrics.jsonl','first\n')
        root, stats = self.snapshot()
        with p.open('a') as f: f.write('second\n')
        self.assertEqual((root/p.name).read_text(),'first\n')
        self.assertEqual(stats['copied'],1)

    def test_link_fallback(self):
        self.write('checkpoint_latest.pt','weights')
        with patch('utils.output_snapshot.os.link',side_effect=OSError('unsupported')):
            root, stats = self.snapshot()
        self.assertEqual((root/'checkpoint_latest.pt').read_text(),'weights')
        self.assertEqual(stats['copied'],1)

    def test_vanished_committed_file_fails_and_cleans(self):
        self.write('checkpoint_latest.pt')
        with patch('utils.output_snapshot.os.link',side_effect=FileNotFoundError('vanished')):
            with self.assertRaises(FileNotFoundError): build_snapshot(self.src)
        self.assertFalse(list(self.base.glob('.vggae-sync-*')))
        self.assertTrue((self.src/'checkpoint_latest.pt').exists())

    def test_cleanup_refuses_unowned_source(self):
        self.write('keep')
        with self.assertRaises(ValueError): cleanup_snapshot(self.src)
        self.assertTrue((self.src/'keep').exists())

    def test_symlinks_not_uploaded(self):
        outside = self.base/'outside'; outside.write_text('private')
        try: os.symlink(outside,self.src/'link')
        except OSError: self.skipTest('symlinks unavailable')
        root, stats = self.snapshot()
        self.assertFalse((root/'link').exists())
        self.assertEqual(stats['skipped_symlink'],1)

    def test_nested_log_watch_snapshot_is_pruned(self):
        self.write('logs/train.log')
        logroot,_=build_snapshot(self.src/'logs')
        self.addCleanup(cleanup_snapshot,logroot)
        root,_=self.snapshot()
        self.assertFalse(any(p.name.startswith('.vggae-sync-') for p in root.rglob('*')))


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('transfer_test_target',ROOT/'scripts/moxing_transfer.py')
        cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)

    def test_remote_directory_download_not_skipped(self):
        args=SimpleNamespace(directory=True,source='obs://bucket/input',destination='/tmp/input',watch=False)
        with patch.object(self.module,'copy_directory') as copy:
            self.module.transfer(args)
            copy.assert_called_once_with(args.source,args.destination)

    def test_missing_one_shot_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(directory=True,source=str(Path(tmp)/'absent'),destination=str(Path(tmp)/'dest'),watch=False)
            with self.assertRaises(FileNotFoundError): self.module.transfer(args)

    def test_failed_upload_cleans_only_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'src';src.mkdir();(src/'checkpoint_latest.pt').write_text('weights')
            args=SimpleNamespace(directory=True,source=str(src),destination='obs://bucket/out',watch=False)
            with patch.object(self.module,'copy_directory',side_effect=RuntimeError('upload failed')):
                with self.assertRaises(RuntimeError): self.module.transfer(args)
            self.assertTrue((src/'checkpoint_latest.pt').exists())
            self.assertFalse(list(Path(tmp).glob('.vggae-sync-*')))


if __name__=='__main__': unittest.main()
