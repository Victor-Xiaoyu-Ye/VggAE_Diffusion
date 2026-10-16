"""Real tiny SceneFlow training/resume tests, without pretrained RGB assets.

ArtifactStore, filtered latent sampling, AdamW, EMA, training RNG, the actual
training loop and prefix sampler remain real. Only the expensive RGB codec and
media/TensorBoard presentation are replaced. These are CPU/Gloo checks, not an
Ascend full-shape rehearsal or a video quality claim.
"""
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

from models.scene_flow import SceneFlow
from utils.scene_run import ArtifactStore, StageBudget
from utils.training import append_metrics, capture_rng_state
import train_scene_prefix as trainer


MODEL = dict(channels=4, grid=2, width=24, depth=1, head_width=48,
             head_depth=1, heads=4, text_dim=8, checkpoint_blocks=True)
IDENTITY = 'prefix-train-cpu-fixture-v1'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyRAE(nn.Module):
    def __init__(self, source_config, channels):
        super().__init__()
        self.grid = source_config['latent_grid']
        self.channels = channels
        self.config = SimpleNamespace(decoder_temporal_blocks=0)
        # Constructor RNG is intentional: restoring the training RNG before
        # codec construction must fail the interrupted-versus-continuous test.
        self.bias = nn.Parameter(torch.randn(3)*.01)
        self.constructor_random = (random.random(), float(np.random.rand()))

    def decode(self, z):
        return (z[..., :3]+self.bias).sigmoid()


def fake_preview(store, name, frames):
    if not torch.isfinite(frames).all():
        raise ValueError('Nonfinite fixture preview')
    path = store.local/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'fixture-media; generation tensors checked separately')
    return [name]


def record_metrics(store, row, rank=0):
    append_metrics(str(store.local/f'metrics-r{rank:03d}.jsonl'), dict(row, rank=rank))


def fixture(root):
    seed_all(5301)
    source = ArtifactStore(root/'source', [])
    model = SceneFlow(**MODEL)
    # A nonzero pretrained head makes first-step training and warm-start
    # equality meaningful, rather than comparing a zero initialized output.
    with torch.no_grad():
        model.output[-1].weight.normal_(0, .025)
        model.output[-1].bias.normal_(0, .01)
    source.save('weights.pt', dict(model=model.state_dict(),
        ema={k: torch.full_like(v, 321, dtype=torch.bfloat16)
             for k, v in model.state_dict().items()},
        model_args=MODEL, step=71609), 'historical-source')
    codec = TinyRAE({'latent_grid': 2}, 4)
    source.save('rae.pt', dict(model=codec.state_dict(),
        spec=dict(source_config={'latent_grid': 2}, channels=4)), 'historical-ae')
    samples = []
    for index in range(4):
        z = torch.randn(9, 4, 4)*.3 + index*.07
        samples.append(dict(id=f'spatialvid:street{index}', dataset='spatialvid',
            anchor=z[:1].clone(), target=z, frame_ids=list(range(0, 36, 4)),
            raw=(z.reshape(9, 2, 2, 4)[..., :3].sigmoid().permute(0, 3, 1, 2)*255).round().byte(),
            fps=32., nominal_fps=8.))
    validation = dict(samples[0], id='spatialvid:heldout')
    source.save('eval.pt', dict(samples=[validation]), 'fixture-validation')
    cfg = dict(seed=53, history=4, views=9, accum=2, max_steps=3,
        lr_steps=8, lr=1e-3, min_lr=2e-4, warmup_steps=2,
        train_hours=100., reserve_hours=0., hours=100.,
        save_seconds=100000., weights_seconds=100000., full_snapshot_seconds=100000.,
        eval_seconds=100000., log_every=1, monitor_cases=1, sample_steps=2,
        eval_seeds=[101])
    plan = dict(identity=IDENTITY, model=MODEL,
                mean=[.1, -.2, .05, .3], std=[.7, 1.2, .8, 1.1])
    selection = dict(allowed_ids=[s['id'] for s in samples])
    return dict(source=source, samples=samples, cfg=cfg, plan=plan, selection=selection)


def new_store(root, label, data, *, malformed=True):
    store = ArtifactStore(root/label, [str(root/(label+'-primary')), str(root/(label+'-mirror'))])
    shards = []
    # Match the real bank's cursor RNG so cursor0 definitely encounters the
    # malformed shard before retrying a good sample.
    draw_seed = int(hashlib.sha256(f'{data["cfg"]["seed"]}:0'.encode()).hexdigest()[:16], 16)
    bad_index = random.Random(draw_seed).choice(list(range(len(data['samples']))))
    bad_name = None
    for index, sample in enumerate(data['samples']):
        name = f'data/shards/sample{index}.pt'
        is_bad = malformed and index == bad_index
        store.save(name, dict(samples=[] if is_bad else [sample]), IDENTITY)
        shards.append(dict(file=name, sources=['spatialvid'], count=1))
        if is_bad:
            bad_name = name
    return store, dict(identity=IDENTITY, shards=shards), bad_name, bad_index


def run_train(data, store, manifest, *, stage='train', max_steps=3, rank=0, world=1,
              capture_models=None):
    cfg = dict(data['cfg'], max_steps=max_steps)
    source = data['source']
    with ExitStack() as stack:
        stack.enter_context(patch.object(trainer, 'SceneRAE', TinyRAE))
        stack.enter_context(patch.object(trainer, 'preview', fake_preview))
        stack.enter_context(patch.object(trainer, 'metrics', record_metrics))
        if capture_models is not None:
            def make_model(**kwargs):
                model = SceneFlow(**kwargs)
                capture_models.append(model)
                return model
            stack.enter_context(patch.object(trainer, 'SceneFlow', make_model))
        trainer.train(SimpleNamespace(stage=stage), cfg, data['plan'], source, store,
            manifest, data['selection'], source.local/'rae.pt', source.local/'weights.pt',
            source.local/'eval.pt', StageBudget(100*3600), torch.device('cpu'), rank, world)
    return store.load('train/checkpoint_latest.pt', IDENTITY, True)


def bank_states(payload):
    for key in ('bank_states', 'banks'):
        if key in payload:
            return payload[key]
    raise AssertionError('Checkpoint must retain per-rank filtered bank state')


def ddp_worker(rank, root):
    root = Path(root)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        saved = torch.load(str(root/'ddp_fixture.pt'), weights_only=False)
        saved['source'] = ArtifactStore(root/'source', [])
        store = ArtifactStore(root/'ddp', [str(root/'ddp-primary'), str(root/'ddp-mirror')])
        seed_all(100+rank)
        models = []
        run_train(saved, store, saved['manifest'], stage='preflight',
                  rank=rank, world=2, capture_models=models)
        torch.save(models[0].state_dict(), root/f'rank{rank}_model.pt')
        dist.barrier()
    finally:
        dist.destroy_process_group()


class PrefixTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_nested_equal(value, reference)
        else:
            self.assertEqual(actual, expected)

    def test_source_online_only_and_fresh_fp32_ema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture(root)
            source_file = data['source'].local/'weights.pt'
            before = hashlib.sha256(source_file.read_bytes()).hexdigest()
            store, manifest, _, _ = new_store(root, 'initialize', data, malformed=False)
            seed_all(42)
            state = run_train(data, store, manifest, max_steps=0)
            warm = data['source'].load('weights.pt', 'historical-source', True)
            self.assertEqual(state['step'], 0)
            self.assertEqual(state['cursors'], [0])
            self.assertFalse(state['optimizer']['state'])
            self.assertEqual(state['ema_metadata']['num_updates'], 0)
            self.assertEqual(state['ema_metadata']['dtype'], 'torch.float32')
            self.assert_nested_equal(state['model'], warm['model'])
            self.assert_nested_equal(state['ema'], warm['model'])
            self.assertTrue(all(v.dtype == torch.float32 for v in state['ema'].values()))
            self.assertTrue(any(not torch.equal(state['ema'][k], v.float()) for k, v in warm['ema'].items()))
            self.assertEqual(hashlib.sha256(source_file.read_bytes()).hexdigest(), before)
            self.assertEqual(store.json('eval/source71609/summary.json', True)['samples'][0]['source'], 'source71609')

    def test_preflight_two_then_resume_one_matches_continuous_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture(root)
            interrupted, manifest, bad, index = new_store(root, 'interrupted', data)
            continuous, other_manifest, _, _ = new_store(root, 'continuous', data)
            seed_all(771)
            two = run_train(data, interrupted, manifest, stage='preflight')
            self.assertEqual(two['step'], 2)
            self.assertEqual(two['cursors'], [4])
            self.assertEqual(two['ema_metadata']['num_updates'], 2)
            self.assertIn(bad, bank_states(two)[0]['bad_shards'])
            self.assertTrue(interrupted.json('preflight_complete.json', True))
            self.assertFalse(interrupted.json('complete.json'))
            self.assertTrue(interrupted.json('train/full_step0000002.pt.json', True))

            # Restore from the second durable replica: remove local and corrupt
            # primary payload, preserving its receipt. The source bad shard is
            # also repaired, but the run's persisted exclusion must survive.
            interrupted.save(bad, dict(samples=[data['samples'][index]]), IDENTITY)
            latest = 'train/checkpoint_latest.pt'
            (interrupted.local/latest).unlink()
            (Path(interrupted.roots[0])/latest).write_bytes(b'corrupt-primary')
            seed_all(99991)  # Fresh-process RNG must not influence the next update.
            three = run_train(data, interrupted, manifest)
            resumed_rng = capture_rng_state()
            seed_all(771)
            reference = run_train(data, continuous, other_manifest)
            continuous_rng = capture_rng_state()
            self.assertEqual(three['step'], 3)
            self.assertEqual(three['cursors'], [6])
            for field in ('model', 'optimizer', 'ema', 'ema_metadata', 'cursors', 'rng'):
                self.assert_nested_equal(three[field], reference[field])
            self.assert_nested_equal(bank_states(three), bank_states(reference))
            self.assert_nested_equal(resumed_rng, continuous_rng)
            self.assertEqual(three['ema_metadata']['num_updates'], 3)
            self.assertTrue(all(value['step'].item() == 3 for value in three['optimizer']['state'].values()))
            self.assertIn(bad, bank_states(three)[0]['bad_shards'])
            self.assertEqual(interrupted.json('complete.json', True)['step'], 3)
            for replica in interrupted.roots:
                committed = ArtifactStore(Path(replica), [])
                self.assertEqual(committed.load(latest, IDENTITY, True)['step'], 3)
            rows = [json.loads(line) for line in (interrupted.local/'metrics-r000.jsonl').read_text().splitlines()]
            train_rows = [row for row in rows if row['stage'] == 'train']
            self.assertEqual([row['step'] for row in train_rows], [1, 2, 3])
            self.assertTrue(all('DI_throughput' in row and 'pre_clip_grad_norm' in row for row in train_rows))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_two_rank_real_prefix_forward_backward_and_recovery_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture(root)
            _, manifest, _, _ = new_store(root, 'ddp', data, malformed=False)
            serializable = {k: v for k, v in data.items() if k != 'source'}
            torch.save(dict(serializable, manifest=manifest), root/'ddp_fixture.pt')
            torch.multiprocessing.spawn(ddp_worker, args=(str(root),), nprocs=2, join=True)
            first = torch.load(str(root/'rank0_model.pt'), weights_only=False)
            second = torch.load(str(root/'rank1_model.pt'), weights_only=False)
            self.assert_nested_equal(first, second)
            store = ArtifactStore(root/'ddp', [])
            state = store.load('train/checkpoint_latest.pt', IDENTITY, True)
            self.assertEqual(state['step'], 2)
            self.assertEqual(state['world'], 2)
            self.assertEqual(state['cursors'], [4, 4])
            self.assertEqual(len(state['rng']), 2)
            self.assertEqual(len(bank_states(state)), 2)
            self.assertTrue(all(v.dtype == torch.float32 for v in state['ema'].values()))


if __name__ == '__main__':
    unittest.main()
