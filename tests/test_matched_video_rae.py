"""CPU contracts for matched feature/codec interfaces, not NPU or quality proof."""
import unittest

import torch
from torch import nn

from models.matched_video_rae import (FrozenMatchedFeatures, MatchedVideoRAE,
                                      extract_matched_features)
from models.scene_flow import SceneFlow
from utils.scene_prefix_eval import sample_case


class FakePatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, 2, stride=2)

    def forward(self, frames):
        return {'x_norm_patchtokens': self.proj(frames).flatten(2).transpose(1, 2)}


class FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = FakePatch()
        self.register_buffer('_resnet_mean', torch.full((1, 1, 3, 1, 1), .4))
        self.register_buffer('_resnet_std', torch.full((1, 1, 3, 1, 1), .2))
        self.calls = 0
        self.fail = False

    def forward(self, frames):
        self.calls += 1
        b, t, c, h, w = frames.shape
        patch = self.patch_embed(((frames-self._resnet_mean)/self._resnet_std)
                                .reshape(b*t, c, h, w))['x_norm_patchtokens']
        if self.fail:
            raise RuntimeError('injected extraction failure')
        patch = patch.reshape(b, t, -1, 8)
        # Causal history dependence approximates the actual interface.
        history = patch.cumsum(1)/torch.arange(1, t+1).view(1, t, 1, 1)
        stream = torch.cat((patch, history), dim=-1)
        specials = torch.full((b, t, 2, 16), 999.)
        return [torch.cat((specials, stream), dim=2)], 2


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = FakeAggregator()

    def forward(self, frames):
        return self.aggregator(frames)


class MatchedVideoRAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def codec(self, input_dim=8, checkpoint=False):
        return MatchedVideoRAE(input_dim, channels=4, grid=2, size=16,
                               decoder_width=8, decoder_temporal_blocks=2,
                               decoder_resblocks=1, use_checkpoint=checkpoint)

    def test_decoder_initialization_is_paired_across_feature_widths(self):
        torch.manual_seed(78)
        dino = self.codec(8)
        torch.manual_seed(78)
        stream = self.codec(16)
        for key, value in dino.decoder.state_dict().items():
            self.assertTrue(torch.equal(value, stream.decoder.state_dict()[key]), key)
        self.assertEqual(dino.project.weight.shape, (4, 8))
        self.assertEqual(stream.project.weight.shape, (4, 16))

    def test_single_forward_features_share_exact_patch_embedding(self):
        torch.manual_seed(91)
        backbone = FakeBackbone()
        frames = torch.rand(2, 3, 3, 4, 4, requires_grad=True)
        pair = extract_matched_features(backbone, frames)
        self.assertEqual(backbone.aggregator.calls, 1)
        self.assertEqual(pair['dino'].shape, (2, 3, 4, 8))
        self.assertEqual(pair['stream'].shape, (2, 3, 4, 16))
        self.assertTrue(torch.equal(pair['dino'], pair['stream'][..., :8]))
        self.assertFalse(pair['dino'].requires_grad)
        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))
        self.assertEqual(len(backbone.aggregator.patch_embed._forward_hooks), 0)

    def test_hook_is_removed_after_failure(self):
        backbone = FakeBackbone()
        backbone.aggregator.fail = True
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            extract_matched_features(backbone, torch.rand(1, 2, 3, 4, 4))
        self.assertEqual(len(backbone.aggregator.patch_embed._forward_hooks), 0)

    def test_frozen_dino_view_matches_normalized_stream_frontend(self):
        backbone = FakeBackbone()
        frames = torch.rand(1, 3, 3, 4, 4)
        pair = extract_matched_features(backbone, frames)
        dino = FrozenMatchedFeatures(backbone, 'dino').train()
        self.assertFalse(dino.training)
        self.assertFalse(backbone.training)
        self.assertTrue(torch.equal(dino(frames), pair['dino']))
        stream = FrozenMatchedFeatures(backbone, 'stream')
        self.assertTrue(torch.equal(stream(frames), pair['stream']))

    def test_encoded_prefix_is_unchanged_when_future_changes(self):
        torch.manual_seed(18)
        backbone = FakeBackbone()
        frames = torch.rand(1, 5, 3, 4, 4)
        changed = frames.clone()
        changed[:, 3:] = torch.rand_like(changed[:, 3:])
        for name, dim in (('dino', 8), ('stream', 16)):
            source = FrozenMatchedFeatures(backbone, name)
            codec = self.codec(dim)
            full = codec.encode(source(frames))
            altered = codec.encode(source(changed))
            prefix = codec.encode(source(frames[:, :3]))
            torch.testing.assert_close(full[:, :3], altered[:, :3], rtol=0, atol=0)
            torch.testing.assert_close(full[:, :3], prefix, rtol=0, atol=0)

    def test_real_decoder_forward_backward_and_spatial_interface(self):
        torch.manual_seed(101)
        codec = self.codec(checkpoint=True).train()
        features = torch.randn(1, 3, 9, 8)
        rgb, z, restored, target = codec(features)
        self.assertEqual(rgb.shape, (1, 3, 16, 16, 3))
        self.assertEqual(z.shape, (1, 3, 4, 4))
        self.assertEqual(restored.shape, target.shape)
        (rgb.square().mean()+(restored-target).square().mean()).backward()
        self.assertTrue(codec.project.weight.grad.abs().sum() > 0)
        self.assertTrue(codec.decoder.temporal['0'].attn.in_proj_weight.grad.abs().sum() > 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in codec.parameters()
                            if p.grad is not None))
        codec.eval()
        torch.testing.assert_close(codec.decode(z.detach()), codec.decode(z.detach().reshape(1, 3, 2, 2, 4)))
        self.assertFalse(codec.specification()['rgb_skip'])
        self.assertEqual(codec.specification()['temporal_factor'], 1)

    def test_forward_equals_encode_decode_without_rgb_input(self):
        codec = self.codec().eval()
        features = torch.randn(1, 2, 4, 8)
        rgb, z, _, _ = codec(features)
        torch.testing.assert_close(z, codec.encode(features))
        torch.testing.assert_close(rgb, codec.decode(z))
        with self.assertRaises(TypeError):
            codec(features, torch.rand(1, 2, 3, 16, 16))

    def test_invalid_shapes_are_rejected(self):
        codec = self.codec()
        with self.assertRaisesRegex(ValueError, 'square'):
            codec.encode(torch.randn(1, 2, 5, 8))
        with self.assertRaisesRegex(ValueError, 'input_dim'):
            codec.encode(torch.randn(1, 2, 4, 16))
        with self.assertRaisesRegex(ValueError, 'latent'):
            codec.decode(torch.randn(1, 2, 3, 4))
        with self.assertRaises(ValueError):
            FrozenMatchedFeatures(FakeBackbone(), 'unknown')

    def test_real_sampler_video_decoder_never_reads_true_future(self):
        torch.manual_seed(221)
        codec = self.codec().eval()
        flow = SceneFlow(channels=4, grid=2, width=16, depth=1,
                         head_width=32, head_depth=1, heads=4,
                         text_dim=8, checkpoint_blocks=False)
        # Nonzero head ensures this is not a constant-output leakage test.
        torch.nn.init.normal_(flow.output[-1].weight, std=.02)
        z = codec.encode(torch.randn(1, 9, 4, 8))[0].detach()
        raw = torch.randint(0, 256, (9, 3, 16, 16), dtype=torch.uint8)
        sample = dict(id='stable-case', dataset='synthetic', frame_ids=list(range(9)),
                      raw=raw, target=z, anchor=z[:1])
        row, preview, generated = sample_case(flow, codec, sample,
                    torch.zeros(4), torch.ones(4), 'cpu', history=4, steps=3, seed=17)
        changed = dict(sample, target=z.clone(), raw=raw.clone())
        changed['target'][4:] += torch.randn_like(z[4:])*5
        changed['raw'][4:] = 255-changed['raw'][4:]
        other, other_preview, other_generated = sample_case(flow, codec, changed,
                    torch.zeros(4), torch.ones(4), 'cpu', history=4, steps=3, seed=17)
        torch.testing.assert_close(generated, other_generated, rtol=0, atol=0)
        torch.testing.assert_close(preview[:, :, 32:48], other_preview[:, :, 32:48],
                                   rtol=0, atol=0)
        self.assertNotEqual(row['raw_l1'], other['raw_l1'])
        self.assertGreater(float(generated.std()), 0.)
        self.assertEqual(row['decoder_temporal_blocks'], 2)


if __name__ == '__main__':
    unittest.main()
