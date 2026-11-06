"""Two real Gloo ranks exercise Stage65 updates, restart and failed inputs.

The native backbone is tiny; the RGB/RAE state widths and 17-frame causal
prefix grids are the production contracts. No accelerator or 14B-memory claim.
"""
from datetime import timedelta
import importlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_vace_edit_pilot import cpu_native_import
from utils.scene_run import ArtifactStore
from utils.vace_edit_runtime import native_modules
from utils.vace_state_runtime import StateConditionedVACE

REPO = Path(__file__).resolve().parents[1]


class OnceBrokenStore(ArtifactStore):
    def __init__(self, local, rank, corrupt=False):
        super().__init__(local, [])
        self.rank, self.corrupt, self.failed = rank, corrupt, False

    def load(self, name, identity=None, required=False):
        if self.corrupt and self.rank == 1 and name.startswith('states/train') and not self.failed:
            self.failed = True
            raise OSError('fixture: one rank cannot read a training state')
        return super().load(name, identity, required)


def configuration():
    return dict(seed=27, frames=17, shift=3., context_scale=1., lora_rank=2,
        lora_alpha=2, bridge_width=8, state_channels=dict(rae=514, rgb=1538),
        training=dict(max_steps=4, min_steps=20, lr=.003, warmup_steps=1,
            accum=1, save_every=2, eval_every=2, log_every=1, ema_decay=.9,
            patience=4, min_improvement=0., monitor_cases=2,
            monitor_sigmas=[.15, .35, .6, .85], identity_fraction=.2,
            text_dropout=.1, state_dropout=.1, grad_clip=1.,
            flow_edit_fraction=.5, preview_every=0))


def create_fixture(root):
    from train_vace_edit_state import encoded_reference
    torch.manual_seed(101)
    manifest = dict(identity='state-gloo-fixture', records=[])
    payloads = []
    for i in range(6):
        part = 'train' if i < 4 else 'monitor'
        name = part + str(i)
        # Every row belongs to a separate source group, including train/monitor.
        row = dict(id=name, group='source-' + name, partition=part, status='completed',
            state_file='states/' + name + '.pt', cache_file='native/' + name + '.pt',
            supervision_file='supervision/' + name + '.pt')
        manifest['records'].append(row)
        source = torch.rand(17, 3, 16, 32) * .4 + .2
        masks = torch.zeros(17, 1, 16, 32)
        masks[..., :16] = 1.
        reference = dict(source=source, reference=(source + masks * .1).clamp(0, 1),
            anchor=(source[0] + masks[0] * .1).clamp(0, 1), masks=masks,
            confidence=torch.ones_like(masks) * .8,
            spatial=dict(canvas=[16, 32], valid_rect=[0, 0, 16, 32]))
        state = dict(reference=encoded_reference(reference),
            rae_state=torch.randn(514, 6, 1, 2).half(),
            text=torch.randn(5, 12), negative=torch.randn(3, 12), empty=torch.randn(1, 12))
        context = torch.randn(96, 6, 2, 4) * .2
        native = dict(context=context, identity_context=context * .9)
        labels = dict(clean=torch.randn(16, 5, 2, 4) * .15,
                      identity_clean=torch.randn(16, 5, 2, 4) * .15)
        payloads.append((row, state, native, labels))
    for run in ('direct', 'resumed', 'corrupt'):
        out = ArtifactStore(root / run, [])
        for row, state, native, labels in payloads:
            out.save(row['state_file'], state, manifest['identity'])
            out.save(row['cache_file'], native, manifest['identity'])
            out.save(row['supervision_file'], labels, manifest['identity'])
    torch.save(dict(manifest=manifest, config=configuration()), str(root / 'fixture.pt'))


def native_backbone():
    prefix, _ = native_modules()
    cls = importlib.import_module(prefix + '.modules.vace_model').VaceWanModel
    value = cls(dim=64, ffn_dim=96, freq_dim=16, text_dim=12, num_heads=4,
        num_layers=2, in_dim=16, out_dim=16, text_len=8, vace_layers=[0], vace_in_dim=96)
    nn.init.normal_(value.head.head.weight, std=.1)
    # Released VACE has trained control projections; constructor zeros would
    # prevent a representative first-step control-bridge gradient.
    for block in value.vace_blocks:
        for name in ('before_proj', 'after_proj'):
            if hasattr(block, name):
                nn.init.normal_(getattr(block, name).weight, std=.08)
    return value


def worker(rank, directory, phase):
    import train_vace_edit_state as trainer
    root = Path(directory)
    torch.set_num_threads(1)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    dist.init_process_group('gloo', init_method=(root / ('rendezvous-' + phase)).as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        fixture = torch.load(str(root / 'fixture.pt'), weights_only=False)
        cfg, manifest = fixture['config'], fixture['manifest']
        random.seed(31)
        np.random.seed(31)
        torch.manual_seed(31)
        arm = 'rgb' if phase == 'corrupt' else 'rae'
        with cpu_native_import():
            model = StateConditionedVACE(native_backbone(), cfg, cfg['state_channels'][arm])
            initial = model.deltas()
            frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
            run = 'direct' if phase == 'direct' else 'corrupt' if phase == 'corrupt' else 'resumed'
            out = OnceBrokenStore(root / run, rank, corrupt=(phase == 'corrupt'))
            trainer.train_loop(SimpleNamespace(arm=arm, stop_after=2 if phase == 'partial' else 0),
                cfg, out, manifest, manifest['records'], model, torch.device('cpu'), rank, 2)
            for name, value in model.named_parameters():
                if name in frozen:
                    if value.grad is not None:
                        raise AssertionError('Frozen native parameter acquired a gradient: ' + name)
                    torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
            torch.save(dict(model=model.deltas(), initial=initial, injected_failure=out.failed),
                str(root / f'{phase}-rank{rank}.pt'))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def assert_exact(test, left, right):
    """Compare nested optimizer and RNG payloads, including numpy RNG arrays."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_exact(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_exact(test, a, b)
    else:
        test.assertEqual(left, right)


@unittest.skipUnless(dist.is_gloo_available(), 'Gloo required')
class EditStateDDPTests(unittest.TestCase):
    def test_rae_updates_and_four_vs_two_plus_two_resume_are_exact(self):
        with tempfile.TemporaryDirectory(prefix='vace state gloo ', dir=REPO.parent) as temp:
            root = Path(temp)
            create_fixture(root)
            for phase in ('direct', 'partial', 'resume'):
                mp.spawn(worker, args=(temp, phase), nprocs=2, join=True)
            states = [torch.load(str(root / f'{p}-rank{r}.pt'), weights_only=False)
                      for p in ('direct', 'resume') for r in range(2)]
            base = states[0]
            self.assertGreater(float(base['model']['bridge.2.weight'].abs().sum()), 0.)
            self.assertFalse(torch.equal(base['model']['bridge.2.weight'], base['initial']['bridge.2.weight']))
            changed_backbone = [key for key in base['model'] if key.startswith('backbone.')
                and not torch.equal(base['model'][key], base['initial'][key])]
            self.assertTrue(changed_backbone, 'Native LoRA must actually update')
            for result in states[1:]:
                assert_exact(self, base['model'], result['model'])
            a = ArtifactStore(root / 'direct/arms/rae', []).load('checkpoint_latest.pt', required=True)
            b = ArtifactStore(root / 'resumed/arms/rae', []).load('checkpoint_latest.pt', required=True)
            for key in ('step', 'updates', 'stale', 'best', 'converged', 'world',
                        'model', 'ema', 'optimizer', 'rng'):
                assert_exact(self, a[key], b[key])
            self.assertEqual(a['step'], 4)
            self.assertEqual(a['updates'], 4)
            for run in ('direct', 'resumed'):
                store = ArtifactStore(root / run / 'arms/rae', [])
                selected = store.load('selected.pt', required=True)
                self.assertIsNotNone(selected['score'])
                self.assertEqual(store.json('training_summary.json', True)['status'], 'completed')
            run = ArtifactStore(root / 'direct/arms/rae', []).json('run.json', True)
            self.assertEqual((run['train_cases'], run['monitor_cases']), (4, 2))

    def test_rgb_one_rank_read_failure_skips_collectively_then_trains(self):
        with tempfile.TemporaryDirectory(prefix='vace state failure gloo ', dir=REPO.parent) as temp:
            root = Path(temp)
            create_fixture(root)
            mp.spawn(worker, args=(temp, 'corrupt'), nprocs=2, join=True)
            states = [torch.load(str(root / f'corrupt-rank{r}.pt'), weights_only=False) for r in range(2)]
            self.assertFalse(states[0]['injected_failure'])
            self.assertTrue(states[1]['injected_failure'])
            assert_exact(self, states[0]['model'], states[1]['model'])
            self.assertGreater(float(states[0]['model']['bridge.2.weight'].abs().sum()), 0.)
            store = ArtifactStore(root / 'corrupt/arms/rgb', [])
            report = store.json('training_summary.json', True)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual((report['step'], report['updates'], report['skipped']), (4, 3, 1))
            metrics = [json.loads(line) for line in (store.local / 'logs/training.jsonl').read_text().splitlines()]
            self.assertEqual(metrics[0]['DI_throughput'], 0.)
            self.assertGreater(metrics[-1]['DI_throughput'], 0.)
            self.assertIsNotNone(store.load('selected.pt', required=True)['score'])


if __name__ == '__main__':
    unittest.main()
