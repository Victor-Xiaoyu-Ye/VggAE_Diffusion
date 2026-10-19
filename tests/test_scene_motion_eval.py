"""Small diagnostic contracts; no claim of real codec/NPU or motion quality."""
import copy
import json
import unittest

import torch
from torch import nn

from utils.scene_motion_eval import oracle_branches, probe_case
from utils.scene_motion_objective import x0_error_metrics
from utils.scene_prefix import prepare_prefix
from utils.scene_prefix_eval import stable_case_seed


class TinyRAE(nn.Module):
    grid = 2
    channels = 4

    def __init__(self):
        super().__init__()
        self.calls = []
        self.child = nn.Dropout()

    def decode(self, z):
        self.calls.append(z.detach().clone())
        return (z[..., :3]+.1*z[..., :3].mean(1, keepdim=True)).sigmoid()


class ProbeModel(nn.Module):
    args = {'text_dim': 3}

    def __init__(self, truth=None, fail=False):
        super().__init__()
        self.truth = truth
        self.fail = fail
        self.calls = []
        self.child = nn.Dropout()

    def forward(self, noisy, u, anchor, text, text_valid, ref_present=None, rays=None):
        self.calls.append(dict(noisy=noisy.clone(), u=u.clone(), anchor=anchor.clone(),
                               text=text.clone(), rays=rays))
        if self.fail:
            raise RuntimeError('fixture failure')
        return noisy if self.truth is None else self.truth


class SceneMotionEvalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(54)
        self.target = torch.randn((6, 4, 4), generator=generator)
        self.sample = dict(id='fixture', target=self.target, anchor=self.target[:1]+2,
            frame_ids=[2, 4, 6, 8, 10, 12], raw=torch.arange(72, dtype=torch.uint8).reshape(6, 3, 2, 2),
            text=torch.full((2, 3), 99.), rays=torch.full((6, 4, 8), 99.))
        self.mean = torch.tensor([.1, -.2, .3, -.4])
        self.std = torch.tensor([1., 2., .5, 3.])
        self.rae = TinyRAE()

    def probe(self, model, **kwargs):
        return probe_case(model, self.rae, self.sample, self.mean, self.std, 'cpu',
                          history=2, **kwargs)

    def test_perfect_x0_has_zero_common_unweighted_errors(self):
        normalized = ((self.target-self.mean)/self.std).unsqueeze(0)
        row = self.probe(ProbeModel(normalized[:, 1:]))
        self.assertEqual(row['probe_global_mse'], 0.)
        self.assertEqual(row['probe_change_mse'], 0.)
        self.assertEqual(len(row['per_time']), 4)
        self.assertEqual([x['frame_index'] for x in row['per_future_frame']], [2, 3, 4, 5])
        self.assertTrue(all(x['global_mse'] == 0 for x in row['per_future_frame']))
        self.assertIn('NOT free generation', row['kind'])
        json.dumps(row, allow_nan=False)

    def test_fixed_noise_reused_at_all_times_conditions_modes_and_rng_unchanged(self):
        model = ProbeModel().train()
        model.child.eval()
        state = torch.get_rng_state().clone()
        first = self.probe(model, seed=303, times=(.25, .5, .95))
        torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        part = prepare_prefix(self.sample, self.mean, self.std, hist=2)
        noise = torch.randn(part.target.shape, generator=torch.Generator().manual_seed(
            stable_case_seed(self.sample, 303)))
        for call in model.calls:
            u = call['u'].item()
            torch.testing.assert_close(call['noisy'][:, :1], part.known, atol=0, rtol=0)
            torch.testing.assert_close(call['noisy'][:, 1:], (1-u)*part.target+u*noise)
            torch.testing.assert_close(call['anchor'], part.anchor, atol=0, rtol=0)
            self.assertEqual(call['text'].count_nonzero(), 0)
            self.assertIsNone(call['rays'])
        second = self.probe(model, seed=303, times=(.25, .5, .95))
        self.assertEqual(first, second)
        self.assertNotEqual(first['probe_global_mse'], self.probe(model, seed=304)['probe_global_mse'])

    def test_mask_includes_boundary_and_empty_mask_falls_back_to_global(self):
        self.sample['target'] = torch.zeros_like(self.target)
        self.sample['target'][1] = 4.
        row = self.probe(ProbeModel(), times=(.5,))
        self.assertEqual(row['change_fraction'], .25)
        self.assertEqual(row['per_future_frame'][0]['change_fraction'], 1.)
        self.assertTrue(row['per_future_frame'][1]['empty_change_fallback'])
        self.sample['target'][1] = 0.
        empty = self.probe(ProbeModel(), times=(.5,))
        self.assertTrue(empty['empty_change_fallback'])
        self.assertEqual(empty['probe_global_mse'], empty['probe_change_mse'])
        # history=1 must use the separately stored anchor, not target[0].
        one = probe_case(ProbeModel(), self.rae, self.sample, self.mean, self.std,
                         'cpu', history=1, times=(.5,))
        self.assertGreater(one['change_fraction'], 0.)

    def test_model_mode_restored_on_probe_failure(self):
        model = ProbeModel(fail=True).train()
        model.child.eval()
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            self.probe(model)
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)

    def test_probe_mask_matches_objective_log_activity_and_floor(self):
        # Raw RMS means select only 100; log1p means select both 10 and 100.
        steps = torch.tensor([1., 5., 10., 100.])
        self.sample['target'] = torch.zeros_like(self.target)
        for i in range(4):
            self.sample['target'][i+2] = (i+1)*steps[:, None]
        model = ProbeModel()
        result = probe_case(model, self.rae, self.sample, torch.zeros(4), torch.ones(4),
                            'cpu', history=2, times=(.25, .75))
        self.assertEqual(result['change_fraction'], .5)
        self.assertEqual(result['activity_transform'], 'log1p')
        self.assertEqual(result['activity_floor'], 1e-4)
        clean = self.sample['target'][2:].unsqueeze(0)
        last = self.sample['target'][1:2].unsqueeze(0)
        common = []
        for call, row in zip(model.calls, result['per_time']):
            metrics = x0_error_metrics(call['noisy'][:, 1:], clean, last)
            self.assertEqual(row['probe_global_mse'], metrics['x0_mse'])
            self.assertEqual(row['probe_change_mse'], metrics['changed_region_x0_mse'])
            self.assertEqual(result['change_fraction'], metrics['changed_fraction'])
            common.append(metrics)
        self.assertAlmostEqual(result['probe_change_mse'],
            sum(x['changed_region_x0_mse'] for x in common)/len(common), places=3)
        self.sample['target'].zero_()
        self.sample['target'][1].fill_(5e-5)
        small = probe_case(ProbeModel(), self.rae, self.sample, torch.zeros(4), torch.ones(4),
                           'cpu', history=2, times=(.5,))
        self.assertTrue(small['empty_change_fallback'])

    def test_oracles_mix_exact_channel_halves_use_actual_anchor_and_label_gt(self):
        generated = self.target[2:]+torch.tensor([1., 1., 2., 2.])
        self.rae.train()
        self.rae.child.eval()
        rng = torch.get_rng_state().clone()
        metadata, preview = oracle_branches(self.rae, self.sample, generated, 'cpu', history=2)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        self.assertTrue(self.rae.training)
        self.assertFalse(self.rae.child.training)
        self.assertEqual(tuple(preview.shape), (6, 2, 10, 3))
        self.assertEqual(metadata['columns'], ['RAW', 'AE', 'GEN', 'GT_GEO+PRED_TEX', 'PRED_GEO+GT_TEX'])
        self.assertNotIn('GEN', metadata['gt_assisted_columns'])
        self.assertIn('NOT extra free-generation', metadata['kind'])
        self.assertEqual(metadata['latent_branch_mse'], {'geo': 1., 'texture': 4.})
        observed = torch.cat((self.sample['anchor'], self.target[1:2]))
        ae, gen, gt_geo, gt_tex = [call.flatten(2, 3)[0] for call in self.rae.calls]
        for call in (ae, gen, gt_geo, gt_tex):
            torch.testing.assert_close(call[:2], observed, atol=0, rtol=0)
        torch.testing.assert_close(ae[2:], self.target[2:], atol=0, rtol=0)
        torch.testing.assert_close(gen[2:], generated, atol=0, rtol=0)
        torch.testing.assert_close(gt_geo[2:, :, :2], self.target[2:, :, :2], atol=0, rtol=0)
        torch.testing.assert_close(gt_geo[2:, :, 2:], generated[:, :, 2:], atol=0, rtol=0)
        torch.testing.assert_close(gt_tex[2:, :, :2], generated[:, :, :2], atol=0, rtol=0)
        torch.testing.assert_close(gt_tex[2:, :, 2:], self.target[2:, :, 2:], atol=0, rtol=0)
        raw = self.sample['raw'].float().permute(0, 2, 3, 1)/255
        torch.testing.assert_close(preview[:2, :, 4:6], raw[:2], atol=0, rtol=0)
        self.assertEqual(len(metadata['per_transition']), 4)
        for row in metadata['per_transition']:
            start, end = row['from_frame'], row['to_frame']
            for i, label in enumerate(metadata['columns']):
                expected = (preview[end, :, i*2:(i+1)*2]-preview[start, :, i*2:(i+1)*2]).abs().mean()
                self.assertAlmostEqual(row['rgb_difference'][label], expected.item())
        json.dumps(metadata, allow_nan=False)

    def test_gen_decode_never_reads_future_gt_even_when_decoder_mixes_time(self):
        generated = torch.full_like(self.target[2:], .25)
        _, original = oracle_branches(self.rae, self.sample, generated, 'cpu', history=2)
        changed = copy.deepcopy(self.sample)
        changed['target'][2:] += 30.
        _, replaced = oracle_branches(self.rae, changed, generated, 'cpu', history=2)
        torch.testing.assert_close(original[:, :, 4:6], replaced[:, :, 4:6], atol=0, rtol=0)
        self.assertFalse(torch.equal(original[2:, :, 6:8], replaced[2:, :, 6:8]))


if __name__ == '__main__':
    unittest.main()
