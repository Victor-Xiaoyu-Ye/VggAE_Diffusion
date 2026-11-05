"""Real two-rank Gloo updates, restart and one-rank input failure for Stage64."""
from datetime import timedelta
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from models.edit_transport import EditTransport
from utils.scene_run import ArtifactStore

REPO = Path(__file__).resolve().parents[1]


class InputOnly(dict):
    """Input deserialization succeeds; accessing an edited future does not."""
    def __getitem__(self, name):
        if name in ('target', 'future_target', 'target_z'):
            raise AssertionError('Edited future supervision reached the input branch')
        return super().__getitem__(name)

    def get(self, name, default=None):
        if name in ('target', 'future_target', 'target_z'):
            raise AssertionError('Edited future supervision reached the input branch')
        return super().get(name, default)


class ObservedTransport(EditTransport):
    def __init__(self, cfg):
        super().__init__(cfg, input_channels=4)
        self.calls = 0

    def forward(self, source_z, anchor_z, source, proposal, confidence, masks, scale):
        # A strict signature means passing target labels into conditioning fails.
        if anchor_z.shape[1] != 1:
            raise AssertionError('Edited condition must contain one observed frame')
        if source_z.shape[1] != source.shape[1]:
            raise AssertionError('Source timing changed')
        self.calls += 1
        return super().forward(source_z, anchor_z, source, proposal, confidence, masks, scale)


class FrozenDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:, :3].copy_(torch.eye(3))
        self.requires_grad_(False)

    def forward(self, z):
        return F.interpolate(self.proj(z[:, 0]).permute(0, 3, 1, 2),
                             size=(8, 8), mode='bilinear', align_corners=False)


class OnceBrokenStore(ArtifactStore):
    def __init__(self, local, rank, corrupt=False):
        super().__init__(local, [])
        self.rank, self.corrupt, self.failed = rank, corrupt, False

    def load(self, name, identity=None, required=False):
        if self.corrupt and self.rank == 1 and name.startswith('inputs/train') and not self.failed:
            self.failed = True
            raise OSError('fixture: one rank could not load its training input')
        return super().load(name, identity, required)


def configuration():
    return dict(seed=27,
        learned=dict(width=16, depth=1, heads=4, max_frames=3, dropout=.1,
            delta_limit=.75, support_dilation=3, trusted_confidence=.95),
        training=dict(max_steps=4, min_steps=20, lr=.003, warmup_steps=1,
            accum=1, save_every=2, eval_every=2, log_every=1, ema_decay=.9,
            patience=4, min_improvement=0., monitor_cases=2, identity_fraction=.2,
            grad_clip=1., edit_weight=.75, keep_weight=.25, latent_weight=.0001))


def create_fixture(root):
    prepared = dict(identity='learned-gloo-fixture', source_scale=[1.] * 4, records=[])
    torch.manual_seed(101)
    payloads = []
    for i in range(6):
        partition = 'train' if i < 4 else 'monitor'
        name = partition + str(i)
        source = torch.full((3, 3, 8, 8), 65 + i * 3, dtype=torch.uint8)
        proposal = source.clone()
        proposal[:, 0] += 18
        z = torch.randn(3, 2, 2, 4) * .05
        anchor_z = z[:1] + .1
        inputs = InputOnly(source_z=z, anchor_z=anchor_z, source=source,
            proposal=proposal, confidence=torch.zeros(3, 1, 8, 8),
            masks=torch.ones(3, 1, 8, 8), target='POISON_NOT_MODEL_INPUT')
        truth = proposal.clone()
        truth[1:, 0] += 16
        truth[1:, 1] -= 8
        row = dict(id=name, group=name, partition=partition, status='completed',
                   inputs='inputs/' + name + '.pt', supervision='supervision/' + name + '.pt')
        prepared['records'].append(row)
        payloads.append((row, inputs, dict(target=truth)))
    for run in ('direct', 'resumed', 'corrupt'):
        store = ArtifactStore(root / run, [])
        for row, inputs, supervision in payloads:
            store.save(row['inputs'], inputs, prepared['identity'])
            store.save(row['supervision'], supervision, prepared['identity'])
    torch.save(dict(prepared=prepared, config=configuration()), str(root/'fixture.pt'))


def worker(rank, directory, phase):
    import train_edit_transport as trainer
    root = Path(directory)
    torch.set_num_threads(1)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    dist.init_process_group('gloo', init_method=(root/('rendezvous-'+phase)).as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        fixture = torch.load(str(root/'fixture.pt'), weights_only=False)
        cfg = fixture['config']
        torch.manual_seed(31)
        model = ObservedTransport(cfg['learned'])
        initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
        decoder = FrozenDecoder()
        frozen = copy.deepcopy(decoder.state_dict())
        name = 'direct' if phase == 'direct' else 'corrupt' if phase == 'corrupt' else 'resumed'
        out = OnceBrokenStore(root/name, rank, corrupt=(phase=='corrupt'))
        trainer.run_training(cfg, out, fixture['prepared'], model, decoder,
            torch.device('cpu'), rank, 2, stop_after=2 if phase=='partial' else 0)
        if any(p.grad is not None for p in decoder.parameters()):
            raise AssertionError('Frozen decoder acquired parameter gradients')
        for key, value in decoder.state_dict().items():
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
        torch.save(dict(model=model.state_dict(), initial=initial, forward_calls=model.calls,
                        decoder=frozen, injected_failure=out.failed),
                   str(root/f'{phase}-rank{rank}.pt'))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_gloo_available(), 'Gloo required')
class LearnedTransportDDPTests(unittest.TestCase):
    def test_actual_updates_are_synchronized_and_restart_is_exact(self):
        with tempfile.TemporaryDirectory(prefix='learned edit gloo ', dir=REPO.parent) as temp:
            root = Path(temp)
            create_fixture(root)
            for phase in ('direct', 'partial', 'resume'):
                mp.spawn(worker, args=(temp, phase), nprocs=2, join=True)
            states = [torch.load(str(root/f'{phase}-rank{rank}.pt'), weights_only=False)
                      for phase in ('direct', 'resume') for rank in range(2)]
            baseline = states[0]
            self.assertGreater(float(baseline['model']['output.weight'].abs().sum()), 0.)
            self.assertFalse(torch.equal(baseline['model']['output.weight'],
                                         baseline['initial']['output.weight']))
            for value in states:
                self.assertGreater(value['forward_calls'], 0)
                for key, expected in baseline['model'].items():
                    torch.testing.assert_close(expected, value['model'][key], rtol=0, atol=0)
            direct = ArtifactStore(root/'direct', []).load('checkpoint_latest.pt', required=True)
            resumed = ArtifactStore(root/'resumed', []).load('checkpoint_latest.pt', required=True)
            self.assertEqual(direct['step'], 4)
            self.assertEqual(resumed['step'], 4)
            for key, expected in direct['ema'].items():
                torch.testing.assert_close(expected, resumed['ema'][key], rtol=0, atol=0)
            for key, state in direct['optimizer']['state'].items():
                for field, value in state.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, resumed['optimizer']['state'][key][field], rtol=0, atol=0)
            self.assertEqual(direct['best_score'], resumed['best_score'])

    def test_one_rank_bad_input_skips_collectively_and_later_updates_complete(self):
        with tempfile.TemporaryDirectory(prefix='learned badinput gloo ', dir=REPO.parent) as temp:
            root = Path(temp)
            create_fixture(root)
            mp.spawn(worker, args=(temp, 'corrupt'), nprocs=2, join=True)
            states = [torch.load(str(root/f'corrupt-rank{r}.pt'), weights_only=False) for r in range(2)]
            self.assertFalse(states[0]['injected_failure'])
            self.assertTrue(states[1]['injected_failure'])
            self.assertGreater(float(states[0]['model']['output.weight'].abs().sum()), 0.)
            for key, value in states[0]['model'].items():
                torch.testing.assert_close(value, states[1]['model'][key], rtol=0, atol=0)
            report = ArtifactStore(root/'corrupt', []).json('training_summary.json', True)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['step'], 4)
            self.assertEqual(report['skipped'], 1)
            self.assertEqual(report['updates'], 3)
            metrics = [json.loads(line) for line in (root/'corrupt/logs/training.jsonl').read_text().splitlines()]
            self.assertEqual(metrics[0]['DI_throughput'], 0., 'Skipped data must not inflate processing throughput')
            self.assertGreater(metrics[-1]['DI_throughput'], 0.)
            self.assertIsNotNone(ArtifactStore(root/'corrupt', []).load('selected.pt', required=True))


if __name__ == '__main__':
    unittest.main()
