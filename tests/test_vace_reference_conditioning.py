"""Reference conditioning follows native VACE's latent-prefix protocol.

The CPU sampling test uses the real tiny native backbone and UniPC scheduler;
the explicit codec fixture makes reference ordering and causal decode history
observable without model assets. No NPU or production-quality claim.
"""
import unittest

import torch
from torch.nn import functional as F

from tests.test_vace_edit_pilot import cpu_native_import, tiny_native
from utils.vace_edit_runtime import (
    decode_native, native_forward, pack_context, sample_native,
)


class ObservableCodec:
    """Causal temporal stride four, spatial stride eight, nonzero RGB-zero code."""

    def __init__(self):
        self.encoded_inputs = []
        self.decoded_inputs = []

    @staticmethod
    def encode_one(value):
        # Each channel sees source RGB and has its own nonzero normalization
        # offset: encoding zero RGB must not stand in for a literal zero latent.
        selected = value[:, ::4].permute(1, 0, 2, 3)
        pooled = F.avg_pool2d(selected, 8).permute(1, 0, 2, 3)
        coefficients = value.new_tensor([
            [1 + i / 32, .25 + i / 64, -.125 + i / 128]
            for i in range(16)
        ])
        return torch.einsum('kc,cthw->kthw', coefficients, pooled) + (
            7 + torch.arange(16, device=value.device, dtype=value.dtype) / 16
        )[:, None, None, None]

    def encode(self, values):
        self.encoded_inputs.extend(v.detach().clone() for v in values)
        return [self.encode_one(v) for v in values]

    def decode(self, values):
        self.decoded_inputs.extend(v.detach().clone() for v in values)
        results = []
        for value in values:
            # An untrimmed reference would contaminate ALL future outputs here,
            # even if RGB frames were trimmed after this causal decode.
            history = value[:3].cumsum(dim=1)
            results.append(F.interpolate(history[None], size=(
                4 * (value.shape[1] - 1) + 1,
                value.shape[2] * 8, value.shape[3] * 8,
            ), mode='nearest')[0])
        return results


def inputs():
    frames = torch.linspace(-.9, .9, 3 * 5 * 32 * 48).reshape(3, 5, 32, 48)
    mask = torch.zeros(1, 5, 32, 48)
    for t in range(5):
        mask[:, t, t::3, t::5] = .75
    # Deliberately nonbinary input exercises the official white>0.5 contract.
    mask[:, :, 0, 0] = .5
    return frames, mask


def official_context_without_references(frames, mask):
    binary = (mask > .5).float()
    inactive = ObservableCodec.encode_one(frames * (1 - binary))
    reactive = ObservableCodec.encode_one(frames * binary)
    latent_t = inactive.shape[1]
    # Independent indexing expression for official 8x8 subpixel-mask packing
    # followed by nearest-exact resampling. No production helper is called.
    source_times = (((torch.arange(latent_t) + .5) * frames.shape[1]) /
                    latent_t).long()
    channels = [
        binary[0, source_times, y::8, x::8]
        for y in range(8) for x in range(8)
    ]
    return torch.cat((inactive, reactive, torch.stack(channels)), dim=0)


class VaceReferenceConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_legacy_none_and_empty_keep_exact_official_context(self):
        frames, mask = inputs()
        expected = official_context_without_references(frames, mask)
        for kwargs in ({}, {'reference_images': None}, {'reference_images': []}):
            with self.subTest(kwargs=kwargs):
                codec = ObservableCodec()
                actual = pack_context(codec, frames, mask, **kwargs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(len(codec.encoded_inputs), 2)

    def test_multiple_reference_prefixes_match_official_math_and_keep_order(self):
        frames, mask = inputs()
        refs = [torch.full((3, 1, 32, 48), -.8 + i * .3) for i in range(5)]
        # Five refs exceed the two video latent slots. Prefix allocation must
        # use reference count, not a slice capped by the video mask's length.
        codec = ObservableCodec()
        actual = pack_context(codec, frames, mask, reference_images=refs)
        normal = official_context_without_references(frames, mask)
        self.assertEqual(tuple(actual.shape), (96, 7, 4, 6))
        torch.testing.assert_close(actual[:, 5:], normal, rtol=0, atol=0)
        for i, ref in enumerate(refs):
            torch.testing.assert_close(actual[:16, i:i+1],
                                       ObservableCodec.encode_one(ref), rtol=0, atol=0)
            self.assertEqual(float(actual[16:, i:i+1].abs().sum()), 0.)
            self.assertGreater(float(ObservableCodec.encode_one(torch.zeros_like(ref)).abs().sum()), 0.)
            torch.testing.assert_close(codec.encoded_inputs[2+i], ref, rtol=0, atol=0)
        # The supplied source and mask remain ordinary video slots.
        self.assertGreater(float(actual[32:, 5:].sum()), 0.)

    def test_bad_reference_shape_canvas_and_nonfinite_are_rejected(self):
        frames, mask = inputs()
        invalid = [
            torch.zeros(3, 32, 48), torch.zeros(3, 2, 32, 48),
            torch.zeros(1, 1, 32, 48), torch.zeros(3, 1, 48, 32),
            torch.zeros(3, 1, 32, 32), 'not an image tensor',
        ]
        for value in (float('nan'), float('inf'), -float('inf')):
            ref = torch.zeros(3, 1, 32, 48)
            ref[1, 0, 7, 9] = value
            invalid.append(ref)
        for ref in invalid:
            with self.subTest(shape=getattr(ref, 'shape', type(ref))):
                with self.assertRaises(ValueError):
                    pack_context(ObservableCodec(), frames, mask, [ref])

    def test_decoder_only_receives_video_slots_before_causal_decode(self):
        # Use non-contiguous storage to ensure slicing does not break decode.
        video = torch.arange(16 * 2 * 4 * 6).float().reshape(16, 2, 4, 6)
        prefix = torch.full((16, 3, 4, 6), 10000.)
        latent = torch.cat((prefix, video), dim=1).transpose(2, 3)
        codec = ObservableCodec()
        actual = decode_native(codec, latent, reference_count=3)
        torch.testing.assert_close(codec.decoded_inputs[0], video.transpose(2, 3))
        self.assertEqual(tuple(actual.shape), (3, 5, 48, 32))
        expected = ObservableCodec().decode([video.transpose(2, 3)])[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(codec.decoded_inputs), 1)

    def test_decode_without_reference_preserves_entire_latent(self):
        latent = torch.randn(16, 2, 4, 6, generator=torch.Generator().manual_seed(4))
        codec = ObservableCodec()
        actual = decode_native(codec, latent)
        torch.testing.assert_close(codec.decoded_inputs[0], latent, rtol=0, atol=0)
        torch.testing.assert_close(actual, ObservableCodec().decode([latent])[0], rtol=0, atol=0)

    def test_invalid_reference_count_and_native_latent_are_rejected_before_decode(self):
        latent = torch.zeros(16, 3, 4, 6)
        for count in (-1, 3, 4, .5, 1., True, None):
            with self.subTest(count=count):
                codec = ObservableCodec()
                with self.assertRaises(ValueError):
                    decode_native(codec, latent, count)
                self.assertEqual(codec.decoded_inputs, [])
        for wrong in (torch.zeros(15, 3, 4, 6), torch.zeros(1, 16, 3, 4, 6),
                      torch.zeros(16, 0, 4, 6)):
            with self.subTest(shape=wrong.shape):
                codec = ObservableCodec()
                with self.assertRaises(ValueError):
                    decode_native(codec, wrong)
                self.assertEqual(codec.decoded_inputs, [])

    def test_real_native_forward_and_sampler_retain_reference_slots(self):
        frames, mask = inputs()
        codec = ObservableCodec()
        refs = [frames[:, :1].clone(), frames[:, 4:5].clone()]
        context = pack_context(codec, frames, mask, refs)
        self.assertEqual(tuple(context.shape), (96, 4, 4, 6))
        with cpu_native_import():
            torch.manual_seed(17)
            model = tiny_native().eval().requires_grad_(False)
            latent = torch.randn(16, 4, 4, 6)
            text, negative = torch.randn(5, 12), torch.randn(3, 12)
            timestep = torch.tensor([500.])
            with torch.no_grad():
                expected = model([latent], timestep, [context], [text], seq_len=24)[0]
                actual = native_forward(model, [latent], timestep, [context], [text])[0][0]
            torch.testing.assert_close(actual, expected)
            cfg = dict(sample_steps=3, shift=3., guidance=2., context_scale=1.)
            sampled = sample_native(model, context, text, negative, cfg, seed=39)
            self.assertEqual(tuple(sampled.shape), tuple(latent.shape))
            self.assertTrue(bool(torch.isfinite(sampled).all()))
            decoded = decode_native(codec, sampled, reference_count=2)
            self.assertEqual(tuple(decoded.shape), tuple(frames.shape))
            torch.testing.assert_close(codec.decoded_inputs[-1], sampled[:, 2:])


if __name__ == '__main__':
    unittest.main()
