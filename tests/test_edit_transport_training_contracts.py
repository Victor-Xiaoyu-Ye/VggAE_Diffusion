"""Training-input augmentation and valid-canvas supervision contracts."""
import copy
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F

from models.edit_transport import EditTransport
from train_edit_transport import corrupt_observations, training_loss
import train_edit_transport as runner
from utils.scene_run import ArtifactStore


class TrainingContracts(unittest.TestCase):
    def _selection_fixture(self, root, monitor=False):
        cfg = dict(seed=19, learned=dict(width=8, heads=2, depth=1, max_frames=3,
            support_dilation=1, dropout=0.), training=dict(max_steps=4, min_steps=20,
            lr=.004, warmup_steps=1, accum=1, save_every=1, eval_every=2,
            log_every=1, ema_decay=.9, patience=4, min_improvement=0.,
            monitor_cases=1, identity_fraction=0., observation_dropout=0.,
            grad_clip=1., edit_weight=.75, keep_weight=.25, latent_weight=1e-4))
        store = ArtifactStore(root/'run', [])
        prepared = dict(identity='selection-contract', source_scale=[1.]*3, records=[])
        torch.manual_seed(13)
        for i in range(2+int(monitor)):
            partition = 'monitor' if i == 2 else 'train'
            key = f'{partition}{i}'
            source = torch.full((3,3,8,8), 60+i*4, dtype=torch.uint8)
            proposal = source.clone(); proposal[:,0] += 20
            target = proposal.clone(); target[1:,0] += 20; target[1:,1] -= 9
            z = torch.randn(3,2,2,3)*.05
            inputs = dict(source_z=z, anchor_z=z[:1]+.1, source=source,
                proposal=proposal, confidence=torch.zeros(3,1,8,8),
                masks=torch.ones(3,1,8,8), valid_rect=[0,0,8,8])
            row = dict(id=key, group=key, partition=partition, status='completed',
                inputs=f'inputs/{key}.pt', supervision=f'supervision/{key}.pt')
            prepared['records'].append(row)
            store.save(row['inputs'], inputs, prepared['identity'])
            store.save(row['supervision'], dict(target=target), prepared['identity'])
        model = EditTransport(cfg['learned'], 3)
        def decoder(z):
            return F.interpolate(z[:,0].permute(0,3,1,2), (8,8), mode='bilinear', align_corners=False)
        return cfg, store, prepared, model, decoder

    def test_no_monitor_fallback_selection_refreshes_after_real_resumed_updates(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(prefix='stage64 selection ', dir=Path(__file__).resolve().parents[2]) as directory, \
                 patch.object(runner.StageBudget, 'install_signals', return_value=None), \
                 contextlib.redirect_stdout(io.StringIO()):
                cfg, store, prepared, model, decoder = self._selection_fixture(Path(directory))
                device = torch.device('cpu')
                runner.run_training(cfg, store, prepared, model, decoder, device, 0, 1, stop_after=1)
                first = store.load('selected.pt', prepared['identity'], True)
                first_hash = store.json('selected.pt.json', True)['sha256']
                self.assertEqual(first['step'], 1)
                self.assertIsNone(first['score'])
                # New process/model semantics: recovery must restore and keep training.
                resumed = EditTransport(cfg['learned'], 3)
                runner.run_training(cfg, store, prepared, resumed, decoder, device, 0, 1)
                latest = store.load('checkpoint_latest.pt', prepared['identity'], True)
                selected = store.load('selected.pt', prepared['identity'], True)
                self.assertEqual(latest['updates'], 4)
                self.assertEqual(selected['step'], 4)
                self.assertIsNone(selected['score'])
                self.assertNotEqual(store.json('selected.pt.json', True)['sha256'], first_hash)
                self.assertFalse(torch.equal(first['model']['output.weight'], selected['model']['output.weight']))
                for key, value in latest['ema'].items():
                    torch.testing.assert_close(selected['model'][key], value.cpu(), rtol=0, atol=0)
        finally:
            torch.set_num_threads(threads)

    def test_real_best_monitor_selection_is_not_overwritten_by_final_ema(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(prefix='stage64 best selection ', dir=Path(__file__).resolve().parents[2]) as directory, \
                 patch.object(runner.StageBudget, 'install_signals', return_value=None), \
                 contextlib.redirect_stdout(io.StringIO()):
                cfg, store, prepared, model, decoder = self._selection_fixture(Path(directory), monitor=True)
                device = torch.device('cpu')
                runner.run_training(cfg, store, prepared, model, decoder, device, 0, 1, stop_after=2)
                best = store.load('selected.pt', prepared['identity'], True)
                best_hash = store.json('selected.pt.json', True)['sha256']
                self.assertEqual(best['step'], 2)
                self.assertIsNotNone(best['score'])
                real_validate = runner.validate_training
                def worse_monitor(*args, **kwargs):
                    report = real_validate(*args, **kwargs)
                    self.assertIsNotNone(report['score'])
                    report['score'] = best['score']+1.
                    return report
                with patch.object(runner, 'validate_training', side_effect=worse_monitor):
                    runner.run_training(cfg, store, prepared, EditTransport(cfg['learned'], 3),
                                        decoder, device, 0, 1)
                selected = store.load('selected.pt', prepared['identity'], True)
                latest = store.load('checkpoint_latest.pt', prepared['identity'], True)
                self.assertEqual(latest['updates'], 4)
                self.assertEqual(selected['step'], 2)
                self.assertEqual(selected['score'], best['score'])
                self.assertEqual(store.json('selected.pt.json', True)['sha256'], best_hash)
                self.assertFalse(torch.equal(latest['ema']['output.weight'], selected['model']['output.weight']))
                for key, value in best['model'].items():
                    torch.testing.assert_close(selected['model'][key], value, rtol=0, atol=0)
        finally:
            torch.set_num_threads(threads)

    def test_missing_observations_preserve_anchor_source_and_original_inputs(self):
        source=torch.full((1,5,3,16,24),.2)
        values=dict(source=source,proposal=source+.3,confidence=torch.ones(1,5,1,16,24),
                    masks=torch.ones(1,5,1,16,24))
        original=copy.deepcopy(values)
        output=corrupt_observations(values,random.Random(12),1.)
        torch.testing.assert_close(output['source'],source,rtol=0,atol=0)
        torch.testing.assert_close(output['proposal'][:,0],original['proposal'][:,0],rtol=0,atol=0)
        torch.testing.assert_close(output['confidence'][:,0],original['confidence'][:,0],rtol=0,atol=0)
        self.assertGreater(int((output['confidence']==0).sum()),0)
        selected=output['confidence']==0
        torch.testing.assert_close(output['proposal'][selected.expand_as(source)],
                                   source[selected.expand_as(source)],rtol=0,atol=0)
        for key in original:torch.testing.assert_close(values[key],original[key],rtol=0,atol=0)
        disabled=corrupt_observations(values,random.Random(12),0.)
        self.assertIs(disabled,values)

    def test_padding_never_changes_training_score(self):
        torch.manual_seed(4)
        cfg=dict(learned=dict(width=8,heads=2,depth=1,max_frames=3,support_dilation=1),
                 training=dict(edit_weight=.75,keep_weight=.25,latent_weight=1e-4))
        model=EditTransport(cfg['learned'],3)
        source=torch.full((1,3,3,8,8),.2)
        source[:,:, :,0:2]=0
        support=torch.ones(1,3,1,8,8);support[:,:,:,0:2]=0
        values=dict(source_z=torch.randn(1,3,2,2,3),anchor_z=torch.randn(1,1,2,2,3),
                    source=source,proposal=source+.1*support,confidence=torch.zeros_like(support),masks=support)
        def decoder(z):return F.interpolate(z[:,0].permute(0,3,1,2),(8,8),mode='nearest')
        target=values['proposal'].clone()+.03*support
        other=target.clone();other[:,:,:,0:2]=.99
        loss,metrics=training_loss(model,decoder,values,target,torch.ones(3),cfg,[1,2],[2,0,6,8])
        second,changed=training_loss(model,decoder,values,other,torch.ones(3),cfg,[1,2],[2,0,6,8])
        torch.testing.assert_close(loss,second,rtol=0,atol=0)
        self.assertEqual(metrics,changed)


if __name__=='__main__':unittest.main()
