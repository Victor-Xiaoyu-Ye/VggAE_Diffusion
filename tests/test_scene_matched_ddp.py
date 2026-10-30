"""Fresh-process, two-rank Gloo rehearsal of the actual matched pipeline.

Only media encoding and TensorBoard presentation use the existing fixture
substitutes. Codec, flow, gradients, DDP, AdamW, EMA, mirrored artifacts,
distributed cache/evaluation and rank-specific resume states remain real.
"""
from contextlib import ExitStack
from datetime import timedelta
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from tests import test_scene_matched_train as helpers
from utils.scene_run import ArtifactStore


def _cpu_state(module):
    return {name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()}


def _worker(rank, directory, phase):
    """Each spawn is a genuinely new interpreter and process group."""
    root = Path(directory)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = '2'
    torch.set_num_threads(1)
    dist.init_process_group('gloo',
        init_method=(root/f'rendezvous-{phase}').as_uri(), rank=rank,
        world_size=2, timeout=timedelta(seconds=45))
    try:
        config = torch.load(str(root/'fixture.pt'), weights_only=False)
        bank, cfg = config['bank'], config['cfg']
        data = ArtifactStore(root/'data', [str(root/'data-mirror')])
        out = ArtifactStore(root/'run',
                            [str(root/'primary'), str(root/'mirror')])
        trainer = helpers.trainer
        if phase == 'cache':
            codec = trainer.load_codec(out, cfg, 8, 'matched-run', 'cpu')
            with patch.object(trainer, 'metrics', helpers.record_metrics):
                trainer.cache_latents(codec, data, bank, out, cfg,
                    'matched-run', torch.device('cpu'), rank, 2)
            return

        stage, action = phase.split('-')
        stop = 2 if action == 'preflight' else 0
        captured = {}
        original_wrap, original_ema = trainer.wrap, trainer.EMA

        def wrap(model, device, world):
            captured['model'] = model
            return original_wrap(model, device, world)

        def ema(*args, **kwargs):
            result = original_ema(*args, **kwargs)
            captured['ema'] = result
            return result

        # The live rank replicas must agree, not merely their reads of the
        # leader's checkpoint. Capture the actual DDP-wrapped trainable module.
        with ExitStack() as stack:
            stack.enter_context(patch.object(trainer, 'wrap', wrap))
            stack.enter_context(patch.object(trainer, 'EMA', ema))
            state = helpers.execute(data, bank, cfg, out, stage,
                stop=stop, rank=rank, world=2)
        torch.save(dict(model=_cpu_state(captured['model']),
                        ema=_cpu_state(captured['ema']),
                        ema_metadata=captured['ema'].metadata(),
                        saved_step=state['step'], pid=os.getpid()),
                   str(root/f'{phase}-rank{rank}.pt'))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(),
                     'PyTorch Gloo is required for the two-rank rehearsal')
class MatchedDDPTests(unittest.TestCase):
    assert_nested_equal = helpers.MatchedTrainingTests.assert_nested_equal

    def _phase(self, root, phase, expected_step):
        torch.multiprocessing.spawn(_worker, args=(str(root), phase),
                                    nprocs=2, join=True)
        stage = phase.split('-')[0]
        store = ArtifactStore(root/'run', [])
        saved = store.load(f'{stage}/checkpoint_latest.pt', 'matched-run', True)
        replicas = [torch.load(str(root/f'{phase}-rank{rank}.pt'),
                               weights_only=False) for rank in range(2)]
        for key in ('model', 'ema', 'ema_metadata'):
            self.assert_nested_equal(replicas[0][key], replicas[1][key])
            self.assert_nested_equal(replicas[0][key], saved[key])
        self.assertEqual(saved['step'], expected_step)
        self.assertEqual(saved['world'], 2)
        self.assertEqual(saved['cursors'], [2*expected_step]*2)
        self.assertEqual(len(saved['rng']), 2)
        self.assertEqual(saved['ema_metadata']['num_updates'], expected_step)
        self.assertTrue(saved['optimizer']['state'])
        self.assertTrue(all(item['step'].item() == expected_step
                            for item in saved['optimizer']['state'].values()))
        self.assertTrue(all(value.dtype == torch.float32
                            for value in saved['ema'].values()))
        for rank, replica in enumerate(replicas):
            self.assertEqual(replica['saved_step'], expected_step)
            self.assertTrue((store.local/f'metrics-r{rank:03d}.jsonl').exists())
        # A successful publication must actually contain the resumed optimizer,
        # rank cursors and EMA on both destinations.
        for mirror in ('primary', 'mirror'):
            other = ArtifactStore(root/mirror, []).load(
                f'{stage}/checkpoint_latest.pt', 'matched-run', True)
            self.assertEqual(other['step'], expected_step)
            self.assertEqual(other['cursors'], saved['cursors'])
            self.assert_nested_equal(other['ema'], saved['ema'])
        return saved

    def test_two_rank_codec_cache_flow_fresh_process_resume(self):
        # All temporary feature/weight payloads live on the repository's drive.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root = Path(tmp)
            _data, bank, cfg = helpers.fixture(root)
            torch.save(dict(bank=bank, cfg=cfg), str(root/'fixture.pt'))
            store = ArtifactStore(root/'run', [])

            two = self._phase(root, 'codec-preflight', 2)
            self.assertIsNone(store.json('codec/complete.json'))
            self.assertIsNone(store.json('codec/selected.pt.json'))
            three = self._phase(root, 'codec-resume', 3)
            self.assertTrue(any(not torch.equal(value, three['model'][name])
                                for name, value in two['model'].items()))
            self.assertEqual(store.json('codec/complete.json')['evaluation_status'],
                             'completed')
            reconstruction = store.json('codec/eval/step0000003/summary.json', True)
            self.assertTrue(reconstruction['full'])
            self.assertEqual(len(reconstruction['rows']), 4)
            self.assertEqual({(row['split'], row['id']) for row in reconstruction['rows']},
                {('train_eval', entry['id']) for entry in bank['train_eval']}
                | {('dev', entry['id']) for entry in bank['dev']})

            torch.multiprocessing.spawn(_worker, args=(str(root), 'cache'),
                                        nprocs=2, join=True)
            manifest = store.json('cache/complete.json', True)
            self.assertEqual(len(manifest['train']), 3)
            self.assertEqual(len(manifest['dev']), 2)
            self.assertEqual(len(manifest['train_eval']), 2)
            targets = torch.cat([store.load(entry['file'], manifest['identity'], True)
                ['target'].flatten(0, 1) for entry in manifest['train']]).double()
            torch.testing.assert_close(torch.tensor(manifest['mean']),
                                       targets.mean(0).float())

            two = self._phase(root, 'flow-preflight', 2)
            self.assertIsNone(store.json('flow/complete.json'))
            three = self._phase(root, 'flow-resume', 3)
            self.assertTrue(any(not torch.equal(value, three['model'][name])
                                for name, value in two['model'].items()))
            self.assertEqual(store.json('flow/complete.json')['evaluation_status'],
                             'completed')
            evaluation = store.json('flow/eval/step0000003/summary.json', True)
            self.assertTrue(evaluation['full'])
            self.assertEqual(len(evaluation['rows']), 16)
            self.assertEqual({(row['split'], row['id'], row['weights'], row['seed'])
                              for row in evaluation['rows']},
                {(split, entry['id'], weights, seed)
                 for split in ('train_eval', 'dev') for entry in bank[split]
                 for weights in ('online', 'ema') for seed in cfg['eval_seeds']})
            self.assertTrue(all(row['DI_throughput'] > 0
                               for row in evaluation['rows']))


if __name__ == '__main__':
    unittest.main()
