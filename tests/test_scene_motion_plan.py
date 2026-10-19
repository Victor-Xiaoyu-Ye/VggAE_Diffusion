"""Paired motion pilot must pin the original cache, online weights and codec."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from audit_scene_flow_weights import receipt_spec
from train_scene_motion import build_plan
from utils.scene_run import ArtifactStore, digest


def fixture(root):
    source = ArtifactStore(root/'prefix', [])
    codec = ArtifactStore(root/'codec', [])
    codec.save('ae/best_joint.pt', dict(model={'w': torch.tensor([.5])}), 'original-codec')
    selection = dict(allowed_ids=['spatialvid:b', 'spatialvid:a'])
    previous = dict(config=dict(history=4, views=9),
        street_ids_sha=digest(sorted(selection['allowed_ids'])),
        ae=receipt_spec(codec, 'ae/best_joint.pt'),
        model=dict(channels=4, grid=2, width=24), mean=[.1, .2, .3, .4], std=[1., 2., 3., 4.])
    previous['identity'] = digest(previous)
    source.write_json('contract.json', previous)
    source.write_json('complete.json', dict(identity=previous['identity'], step=31069))
    manifest = dict(identity=previous['identity'], samples=8,
        shards=[dict(file='data/shards/one.pt', sources=['spatialvid']*8)],
        mean=previous['mean'], std=previous['std'])
    source.write_json('data/cache_complete.json', manifest)
    source.save('train/weights_step0031069.pt', dict(model={'w': torch.tensor([.75])},
        model_args=previous['model'], step=31069), previous['identity'])
    source.save('data/eval.pt', dict(samples=[dict(id='spatialvid:val')]), previous['identity'])
    cfg = dict(history=4, views=9, source_weights='train/weights_step0031069.pt',
        motion_objective=dict(strength=1., cap=3., log_activity=True),
        convergence=dict(min_steps=4000, patience=4, relative_tol=.002))
    return source, codec, cfg, selection, previous, manifest


class MotionPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='scene-motion-plan-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_accepts_matching_receipts_and_pins_unchanged_assets_for_both_arms(self):
        source, codec, cfg, selection, previous, manifest = fixture(self.root)
        before = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob('*') if p.is_file()}
        plan, got_manifest = build_plan(source, codec, cfg, selection, 24)
        self.assertEqual(got_manifest, manifest)
        self.assertEqual(plan['source_prefix'], previous['identity'])
        self.assertEqual(plan['source_weights'], receipt_spec(source, cfg['source_weights']))
        self.assertEqual(plan['ae'], previous['ae'])
        self.assertEqual(plan['evaluation'], receipt_spec(source, 'data/eval.pt'))
        self.assertEqual(plan['mean'], previous['mean'])
        self.assertEqual(plan['std'], previous['std'])
        self.assertEqual(plan['cache_manifest_sha'], digest(manifest))
        self.assertEqual(plan['identity'], digest({k:v for k,v in plan.items() if k!='identity'}))
        control_cfg = copy.deepcopy(cfg)
        control_cfg['motion_objective']['strength'] = 0.
        control, _ = build_plan(source, codec, control_cfg, selection, 24)
        self.assertNotEqual(plan['identity'], control['identity'])
        for key in ('source_prefix','source_weights','ae','evaluation','model','mean','std',
                    'cache_identity','cache_manifest_sha','street_ids_sha'):
            self.assertEqual(plan[key], control[key])
        after = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(after, before)

    def test_changed_codec_payload_is_rejected_even_with_the_same_codec_identity(self):
        source, codec, cfg, selection, _, _ = fixture(self.root)
        codec.save('ae/best_joint.pt', dict(model={'w': torch.tensor([9.])}), 'original-codec')
        with self.assertRaisesRegex(ValueError, 'Frozen codec differs'):
            build_plan(source, codec, cfg, selection, 24)

    def test_foreign_completion_cache_weights_and_evaluation_identities_are_rejected(self):
        for name in ('complete.json', 'data/cache_complete.json',
                     'train/weights_step0031069.pt', 'data/eval.pt'):
            with self.subTest(artifact=name):
                source, codec, cfg, selection, _, _ = fixture(self.root/name.replace('/', '_'))
                if name.endswith('.pt'):
                    source.save(name, dict(model={'w':torch.tensor([1.])}), 'foreign-prefix')
                else:
                    payload = source.json(name, True)
                    source.write_json(name, dict(payload, identity='foreign-prefix'))
                with self.assertRaisesRegex(ValueError, 'identities differ|another prefix run'):
                    build_plan(source, codec, cfg, selection, 24)

    def test_changed_selection_or_prefix_window_is_rejected(self):
        source, codec, cfg, selection, _, _ = fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'Source street IDs differ'):
            build_plan(source, codec, cfg, dict(allowed_ids=['spatialvid:a','spatialvid:replay']), 24)
        with self.assertRaisesRegex(ValueError, 'prediction window differ'):
            build_plan(source, codec, dict(cfg, history=1), selection, 24)


if __name__ == '__main__':
    unittest.main()
