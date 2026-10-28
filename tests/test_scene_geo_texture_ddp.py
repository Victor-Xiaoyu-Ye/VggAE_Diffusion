"""Real two-rank Gloo train/save/fresh-process resume for the geo oracle.

Only the existing tiny RGB codec/media fixtures are substituted. DDP, AdamW,
the C256 conditioned model, EMA and checkpoint IO run through the real trainer.
This tests distributed correctness, not Ascend memory or generated image quality.
"""
from datetime import timedelta
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

import train_scene_prefix as trainer
from tests import test_scene_geo_texture_train as fixtures
from tests import test_scene_prefix_train as prefix_fixtures
from utils.scene_geo_condition import GeoConditionRuntime
from utils.scene_run import ArtifactStore


def _state_copy(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _geo_ddp_worker(rank, directory, stage):
    root = Path(directory)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(root/f'rendezvous_{stage}').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        data = torch.load(str(root/'fixture.pt'), weights_only=False)
        data['source'] = ArtifactStore(root/'source', [])
        bank = ArtifactStore(root/'bank', [str(root/'bank-primary'), str(root/'bank-mirror')])
        output = ArtifactStore(root/'output', [str(root/'output-primary'), str(root/'output-mirror')])
        # Rank-dependent constructor RNG makes an EMA initialized before DDP's
        # parameter broadcast observably wrong. Training RNG is reset/restored
        # by the real runtime after constructor and source-baseline evaluation.
        prefix_fixtures.seed_all(921 + rank)
        captured = {}
        original_ema = trainer.EMA

        def capture_ema(model, *args, **kwargs):
            ema = original_ema(model, *args, **kwargs)
            captured.update(model=model, ema=ema, initial_ema=_state_copy(ema.state_dict()),
                            restored_ema=None, gradients=[])
            update, restore = ema.update, ema.load_state_dict

            def record_update(current):
                parameters = dict(current.named_parameters())
                if any(parameter.grad is None for parameter in parameters.values()):
                    raise AssertionError('A distributed geo parameter was absent from the training graph')
                if not all(torch.isfinite(parameter.grad).all() for parameter in parameters.values()):
                    raise AssertionError('A distributed geo gradient was nonfinite')
                captured['gradients'].append(dict(
                    before_updates=ema.num_updates,
                    adapter_output_l1=float(parameters['geo_condition.2.weight'].grad.abs().sum()),
                    backbone_input_l1=float(parameters['input.weight'].grad.abs().sum())))
                update(current)

            def record_restore(state):
                restore(state)
                captured['restored_ema'] = _state_copy(ema.state_dict())

            ema.update, ema.load_state_dict = record_update, record_restore
            return ema

        with patch.object(trainer, 'EMA', capture_ema):
            saved = fixtures.execute(data, output, bank, data['manifest'], 'oracle_geo',
                                     stage=stage, steps=3, rank=rank, world=2)
        runtime = GeoConditionRuntime('oracle_geo', data['plan']['mean'], data['plan']['std'])
        runtime.validate_checkpoint_metadata(saved, resume=True)
        metadata = runtime.checkpoint_metadata()
        result = dict(model=_state_copy(captured['model'].state_dict()),
            ema=_state_copy(captured['ema'].state_dict()),
            initial_ema=captured['initial_ema'], restored_ema=captured['restored_ema'],
            gradients=captured['gradients'], ema_metadata=captured['ema'].metadata(),
            saved_model=_state_copy(saved['model']), saved_ema=_state_copy(saved['ema']),
            saved_ema_metadata=saved['ema_metadata'],
            runtime_metadata=metadata, saved_metadata={key: saved.get(key) for key in metadata},
            step=saved['step'], world=saved['world'], cursors=saved['cursors'])
        torch.save(result, root/f'{stage}_rank{rank}.pt')
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
class GeoTextureDistributedTests(unittest.TestCase):
    def assert_state_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for name in first:
            with self.subTest(parameter=name):
                torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_two_rank_preflight_then_fresh_process_resume_preserves_online_and_ema(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
            root = Path(temporary)
            data = fixtures.geo_fixture(root)
            fixtures.two_eval(data)
            _, manifest, _, _ = fixtures.new_store(root, 'bank', data, malformed=False)
            # Passing only serialized fixture data makes each spawned phase
            # reconstruct its own stores, models, optimizer and EMA instance.
            torch.save({**{key: value for key, value in data.items() if key != 'source'},
                        'manifest': manifest}, root/'fixture.pt')
            results = {}
            for stage, expected_step in (('preflight', 2), ('train', 3)):
                torch.multiprocessing.spawn(_geo_ddp_worker, args=(str(root), stage),
                                            nprocs=2, join=True)
                ranks = [torch.load(str(root/f'{stage}_rank{rank}.pt'), weights_only=False)
                         for rank in range(2)]
                results[stage] = ranks
                for rank, result in enumerate(ranks):
                    with self.subTest(stage=stage, rank=rank):
                        self.assertEqual(result['step'], expected_step)
                        self.assertEqual(result['world'], 2)
                        self.assertEqual(result['cursors'], [expected_step*data['cfg']['accum']]*2)
                        self.assertEqual(result['ema_metadata']['num_updates'], expected_step)
                        self.assertEqual(result['ema_metadata'], result['saved_ema_metadata'])
                        self.assertEqual(result['runtime_metadata'], result['saved_metadata'])
                        self.assert_state_equal(result['model'], result['saved_model'])
                        self.assert_state_equal(result['ema'], result['saved_ema'])
                        self.assertTrue(all(value.dtype == torch.float32 for value in result['ema'].values()))
                        self.assertTrue(result['gradients'])
                        self.assertTrue(all(row['adapter_output_l1'] > 0 and row['backbone_input_l1'] > 0
                                            for row in result['gradients']))
                self.assert_state_equal(ranks[0]['model'], ranks[1]['model'])
                self.assert_state_equal(ranks[0]['ema'], ranks[1]['ema'])
                self.assert_state_equal(ranks[0]['initial_ema'], ranks[1]['initial_ema'])
                self.assertEqual(ranks[0]['saved_metadata'], ranks[1]['saved_metadata'])
                self.assertEqual(ranks[0]['gradients'], ranks[1]['gradients'])
            for rank in range(2):
                first, resumed = results['preflight'][rank], results['train'][rank]
                self.assertIsNone(first['restored_ema'])
                self.assert_state_equal(first['ema'], resumed['restored_ema'])
                self.assertEqual([row['before_updates'] for row in first['gradients']], [0, 1])
                self.assertEqual([row['before_updates'] for row in resumed['gradients']], [2])
                self.assertGreater(float(resumed['model']['geo_condition.2.weight'].abs().sum()), 0)
                self.assertFalse(torch.equal(first['model']['geo_condition.2.weight'],
                                             resumed['model']['geo_condition.2.weight']))
            output = ArtifactStore(root/'output', [])
            self.assertEqual(output.json('preflight_complete.json', True)['step'], 2)
            self.assertEqual(output.json('complete.json', True)['evaluation_status'], 'completed')
            for destination in ('output-primary', 'output-mirror'):
                mirror = ArtifactStore(root/destination, [])
                saved = mirror.load('train/checkpoint_latest.pt', data['plan']['identity'], True)
                self.assertEqual(saved['step'], 3)
                self.assertEqual(saved['model_family'], 'GeoConditionedSceneFlow')
                self.assert_state_equal(saved['ema'], results['train'][0]['ema'])


if __name__ == '__main__':
    unittest.main()
