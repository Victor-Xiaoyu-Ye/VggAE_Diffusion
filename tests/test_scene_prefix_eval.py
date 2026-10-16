"""Small real-SceneFlow sampling tests, not a claim of NPU/encoder validation."""
import copy
import unittest

import torch
from torch import nn

from models.scene_flow import SceneFlow
from utils.scene_prefix_eval import sample_case, stable_case_seed


class FixtureRAE(nn.Module):
    grid = 2
    channels = 4

    def __init__(self):
        super().__init__()
        self.calls = []

    def decode(self, latent):
        self.calls.append(latent.detach().clone())
        # Deliberately mixes time: a contaminated decode input would change all
        # generated RGB, even when the denoiser itself ignores future truth.
        return (latent[..., :3] + .2*latent[..., :3].mean(1, keepdim=True)).sigmoid()


class ScenePrefixEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(53)
        self.model = SceneFlow(channels=4, grid=2, width=24, depth=1,
            head_width=48, head_depth=1, heads=4, text_dim=8, checkpoint_blocks=False)
        # Nonzero gates/head prevent a trivial constant-output leakage test.
        for parameter in self.model.parameters():
            nn.init.normal_(parameter, std=.1)
        self.rae = FixtureRAE()
        target = torch.randn(9, 4, 4)
        colors = torch.tensor([5, 15, 35, 60, 90, 120, 150, 180, 210], dtype=torch.uint8)
        raw = colors[:, None, None, None].expand(9, 3, 2, 2).clone()
        self.sample = dict(id='spatialvid:fixture', dataset='spatialvid', target=target,
            anchor=target[:1].clone(), raw=raw, frame_ids=list(range(17, 44, 3)),
            text=torch.randn(4, 8), has_caption=True, rays=torch.randn(9, 4, 8), nominal_fps=8.)
        self.mean = torch.tensor([.1, -.2, .3, -.4])
        self.std = torch.tensor([1., 2., .5, 3.])

    def run_case(self, sample=None, **kwargs):
        return sample_case(self.model, self.rae, sample or self.sample, self.mean, self.std,
                           torch.device('cpu'), steps=3, **kwargs)

    def test_future_truth_replacement_cannot_change_generation_even_with_temporal_decoder(self):
        a, frames_a, z_a = self.run_case()
        changed = copy.deepcopy(self.sample)
        changed['target'][4:] = changed['target'][4:]*9 + 50
        changed['raw'][4:] = 255-changed['raw'][4:]
        changed['text'].fill_(999.)
        changed['rays'].fill_(999.)
        b, frames_b, z_b = self.run_case(changed)
        torch.testing.assert_close(z_a, z_b, rtol=0, atol=0)
        torch.testing.assert_close(frames_a[:, :, 4:6], frames_b[:, :, 4:6], rtol=0, atol=0)
        self.assertFalse(torch.equal(frames_a[4:, :, 2:4], frames_b[4:, :, 2:4]))
        self.assertNotEqual(a['raw_l1'], b['raw_l1'])
        self.assertNotEqual(a['ae_l1'], b['ae_l1'])
        # Generation and oracle are separate full-window decode invocations.
        generated_input = self.rae.calls[0].flatten(2, 3)[0]
        torch.testing.assert_close(generated_input[:4], self.sample['target'][:4], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(generated_input[4:], z_a, rtol=0, atol=0)
        self.assertEqual(tuple(z_a.shape), (5, 4, 4))

    def test_first_unknown_slots_are_pure_seeded_noise_and_prefix_is_clamped(self):
        calls = []
        handle = self.model.register_forward_pre_hook(lambda module, args:
            calls.append((args[0].detach().clone(), args[1].detach().clone())))
        try:
            self.run_case()
        finally:
            handle.remove()
        generator = torch.Generator().manual_seed(stable_case_seed(self.sample, 101))
        expected = torch.randn((1, 5, 4, 4), generator=generator)
        torch.testing.assert_close(calls[0][0][:, 3:], expected, rtol=0, atol=0)
        torch.testing.assert_close(calls[0][1], torch.ones(1), rtol=0, atol=0)
        prefix = ((self.sample['target'][1:4]-self.mean)/self.std).unsqueeze(0)
        for full, _ in calls:
            torch.testing.assert_close(full[:, :3], prefix, rtol=0, atol=0)
            self.assertEqual(tuple(full.shape), (1, 8, 4, 4))

    def test_copy_and_metrics_use_last_observed_frame_and_future_only(self):
        row, frames, _ = self.run_case()
        raw = self.sample['raw'].float().permute(0, 2, 3, 1)/255
        expected = (raw[4:]-raw[3]).abs().mean().item()
        self.assertAlmostEqual(row['copy_l1'], expected)
        torch.testing.assert_close(frames[:4, :, 4:6], raw[:4], rtol=0, atol=0)
        torch.testing.assert_close(frames[4:, :, 6:8], raw[3:4].expand_as(raw[4:]), rtol=0, atol=0)
        self.assertEqual(row['copy_motion_l1'], 0.)
        self.assertAlmostEqual(row['copy_laplacian_abs'], 0., places=6)
        self.assertAlmostEqual(row['gt_seam_l1'], 30/255, places=6)
        self.assertEqual(row['history_frames'], 4)
        self.assertEqual(row['future_frames'], 5)
        self.assertEqual(row['nominal_future_seconds'], 5/8)
        self.assertIn('[4:9]', row['metric_scope'])
        self.assertEqual(row['columns'], ['RAW', 'AE', 'GENERATED', 'COPY'])
        self.assertFalse(row['has_caption'])
        self.assertFalse(row['camera_present'])

    def test_seed_is_stable_across_rank_metadata_and_modes_rng_are_preserved(self):
        changed = dict(self.sample, rank=17, world=24, checkpoint='another')
        self.assertEqual(stable_case_seed(changed), stable_case_seed(self.sample))
        self.assertNotEqual(stable_case_seed(self.sample, 101), stable_case_seed(self.sample, 102))
        changed = dict(self.sample, frame_ids=list(range(18, 45, 3)))
        self.assertNotEqual(stable_case_seed(changed), stable_case_seed(self.sample))
        before = torch.get_rng_state().clone()
        self.model.train()
        self.rae.train()
        self.run_case()
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertTrue(self.model.training)
        self.assertTrue(self.rae.training)


if __name__ == '__main__':
    unittest.main()
