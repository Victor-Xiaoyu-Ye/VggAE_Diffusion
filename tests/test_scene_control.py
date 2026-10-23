"""Stage56 CPU contracts, not real teacher, Ascend or image-quality validation."""
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist

from utils.scene_control import (ControlledSceneFlow, bind_control, control_features,
                                create_control_model, load_warm, transport_prefix_texture)
from utils.scene_prefix import PrefixFlowNet, prepare_prefix
from utils.window_flow import WindowFlow


def _control_ddp_worker(rank, directory):
    root = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(561 + rank)
        args = dict(channels=4, grid=2, width=24, depth=1, head_width=48,
                    head_depth=1, heads=4, text_dim=8, checkpoint_blocks=True)
        model = create_control_model(args, 'reference')
        with torch.no_grad():
            model.output[-1].weight.normal_(0, .02)
        # No unused-parameter detection: sparse/missing control must still
        # traverse the complete graph, including zero-valued reference input.
        core = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(core.parameters(), lr=1e-3)
        batch = prepare_prefix(torch.randn(1, 9, 4, 4), torch.zeros(4), torch.ones(4))
        points = torch.tensor([[.25, .25]])
        controls = dict(rays=torch.randn(9, 4, 8), motion=torch.randn(9, 4, 8),
            tracks=points[None].expand(9, 1, 2), track_weight=torch.ones(9, 1),
            prefix_tracks=points[None].expand(4, 1, 2), prefix_track_weight=torch.ones(4, 1))
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            sample = {'controls': controls} if rank == 0 else {}
            bound = bind_control(core, batch, sample, 'reference', training=True)
            value = bound(torch.randn_like(batch.target), torch.tensor([.3]), batch.anchor)
            (value - batch.target).square().mean().backward()
            if any(parameter.grad is None for parameter in model.parameters()):
                raise AssertionError('An adapter/core parameter was unused on a sparse-control rank')
            optimizer.step()
        torch.save(model.state_dict(), root/f'rank{rank}.pt')
        dist.barrier()
    finally:
        dist.destroy_process_group()


class SceneControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(56)
        self.args = dict(channels=4, grid=2, width=24, depth=1, head_width=48,
                         head_depth=1, heads=4, text_dim=8, checkpoint_blocks=False)
        self.z = torch.randn(1, 9, 4, 4)
        self.batch = prepare_prefix(self.z, torch.zeros(4), torch.ones(4))
        points = torch.tensor([[.25, .25], [.75, .75]])
        self.controls = {
            'rays': torch.randn(9, 4, 8), 'motion': torch.randn(9, 4, 8),
            'tracks': points[None].expand(9, 2, 2).clone(),
            'track_weight': torch.ones(9, 2),
            'prefix_tracks': points[None].expand(4, 2, 2).clone(),
            'prefix_track_weight': torch.ones(4, 2),
        }
        self.sample = {'target': self.z, 'controls': self.controls}

    def source(self):
        model = create_control_model(self.args, 'baseline')
        # Nontrivial output, attention modulation and camera weights/biases.
        with torch.no_grad():
            for param in model.parameters():
                param.uniform_(-.1, .1)
        return model

    def test_zero_init_matches_trained_baseline_with_nonzero_camera(self):
        source = self.source().eval()
        noise, u = torch.randn_like(self.batch.target), torch.tensor([.4])
        baseline = bind_control(source, self.batch, self.sample, 'baseline')
        expected = baseline(noise, u, self.batch.anchor)
        self.assertGreater(expected.abs().sum().item(), .1)
        for arm in ('motion', 'reference'):
            with self.subTest(arm=arm):
                model = create_control_model(self.args, arm).eval()
                load_warm(model, source.state_dict())
                got = bind_control(model, self.batch, self.sample, arm)(noise, u, self.batch.anchor)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
                self.assertIn('output.1.weight', model.state_dict())
                self.assertEqual(model.args, source.args)

    def test_future_appearance_and_geo_do_not_enter_reference(self):
        expected = control_features(self.batch, self.sample, 'reference')
        changed = self.z.clone()
        changed[:, 4:] = 99999
        alternate = prepare_prefix(changed, torch.zeros(4), torch.ones(4))
        got = control_features(alternate, {'target': changed, 'controls': self.controls,
                                          'rgb': torch.full((9, 3, 518, 518), 999.)}, 'reference')
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        changed[:, :4, :, :2] = -12345  # Geometry isn't an appearance reference.
        alternate = prepare_prefix(changed, torch.zeros(4), torch.ones(4))
        torch.testing.assert_close(control_features(alternate, self.sample, 'reference'),
                                   expected, rtol=0, atol=0)
        self.assertEqual(expected[:, :3, :, 16:].count_nonzero().item(), 0)
        self.assertGreater(expected[:, 3:, :, 16:].abs().sum().item(), 0)

    def test_pure_noise_generation_invariant_to_changed_future_latents(self):
        model = create_control_model(self.args, 'reference').eval()
        load_warm(model, self.source().state_dict())
        with torch.no_grad():
            model.control[-1].weight.uniform_(-.1, .1)
        changed = self.z.clone()
        changed[:, 4:] = -1000
        alternate = prepare_prefix(changed, torch.zeros(4), torch.ones(4))
        noise = torch.randn_like(self.batch.target)
        outputs = []
        for batch in (self.batch, alternate):
            outputs.append(WindowFlow().sample(bind_control(model, batch, self.sample, 'reference'),
                batch.anchor, noise.clone(), steps=3))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_reference_uses_independent_prefix_queries_not_fullwindow_prefix(self):
        expected = control_features(self.batch, self.sample, 'reference')
        controls = {key: value.clone() for key, value in self.controls.items()}
        controls['tracks'][:4] = .5
        controls['track_weight'][:4] = 0
        got = control_features(self.batch, {'controls': controls}, 'reference')
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_latest_visible_prefix_feature_moves_to_requested_token(self):
        observed = torch.zeros(1, 4, 4, 4)
        observed[..., :2] = 99
        observed[0, 1, 0, 2:] = torch.tensor([4., 7.])
        observed[0, 2, 0, 2:] = 888
        src = torch.tensor([.25, .25]).reshape(1, 1, 1, 2).expand(1, 4, 1, 2)
        src_weight = torch.tensor([1., .5, 0., 0.]).reshape(1, 4, 1)
        dst = torch.tensor([.75, .75]).reshape(1, 1, 1, 2)
        out = transport_prefix_texture(observed, src, src_weight, dst, torch.ones(1, 1, 1))
        torch.testing.assert_close(out[0, 0, 3], torch.tensor([4., 7., .5]))
        self.assertEqual(out[0, 0, :3].count_nonzero().item(), 0)

    def test_visibility_and_outside_points_leave_holes(self):
        observed = torch.ones(1, 1, 4, 4)
        src = torch.tensor([[[[.25, .25], [.75, .75]]]])
        dst = torch.tensor([[[[.25, .25], [1.1, .75]]]])
        out = transport_prefix_texture(observed, src, torch.ones(1, 1, 2),
                                       dst, torch.tensor([[[0., 1.]]]))
        self.assertEqual(out.count_nonzero().item(), 0)
        out = transport_prefix_texture(observed, src, torch.zeros(1, 1, 2),
                                       src, torch.ones(1, 1, 2))
        self.assertEqual(out.count_nonzero().item(), 0)

    def test_subpixel_transport_distributes_support_and_handles_collisions(self):
        observed = torch.ones(1, 1, 4, 4)
        src = torch.tensor([[[[.25, .25]]]])
        dst = torch.tensor([[[[.5, .5]]]])
        out = transport_prefix_texture(observed, src, torch.ones(1, 1, 1),
                                       dst, torch.ones(1, 1, 1))
        torch.testing.assert_close(out[0, 0, :, -1], torch.full((4,), .25))
        torch.testing.assert_close(out[0, 0, :, :2], torch.ones(4, 2))

    def test_empty_queries_missing_controls_are_finite_and_ddp_parameters_used(self):
        observed = torch.ones(1, 4, 4, 4)
        out = transport_prefix_texture(observed, torch.empty(1, 4, 0, 2),
                torch.empty(1, 4, 0), torch.empty(1, 5, 0, 2), torch.empty(1, 5, 0))
        self.assertEqual(tuple(out.shape), (1, 5, 4, 3))
        self.assertEqual(out.count_nonzero().item(), 0)
        model = create_control_model(self.args, 'reference')
        load_warm(model, self.source().state_dict())
        pred = bind_control(model, self.batch, {}, 'reference')(
            torch.randn_like(self.batch.target), torch.tensor([.3]), self.batch.anchor)
        pred.square().mean().backward()
        for name, parameter in model.control.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_gradient_reaches_control_stem_then_its_input_layer(self):
        model = create_control_model(self.args, 'reference')
        load_warm(model, self.source().state_dict())
        noise, u = torch.randn_like(self.batch.target), torch.tensor([.3])
        bound = bind_control(model, self.batch, self.sample, 'reference')
        bound(noise, u, self.batch.anchor).square().mean().backward()
        self.assertGreater(model.control[-1].weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model.control[-1].weight.add_(model.control[-1].weight.grad * -100)
        model.zero_grad(set_to_none=True)
        bound(noise, u, self.batch.anchor).square().mean().backward()
        self.assertGreater(model.control[0].weight.grad.abs().sum().item(), 0)

    def test_motion_arm_has_no_reference_and_both_arms_same_parameter_shapes(self):
        motion = control_features(self.batch, self.sample, 'motion')
        reference = control_features(self.batch, self.sample, 'reference')
        torch.testing.assert_close(motion[..., :16], reference[..., :16])
        self.assertEqual(motion[..., 16:].count_nonzero().item(), 0)
        model_b = create_control_model(self.args, 'motion')
        model_c = create_control_model(self.args, 'reference')
        self.assertEqual({k: v.shape for k, v in model_b.state_dict().items()},
                         {k: v.shape for k, v in model_c.state_dict().items()})

    def test_warm_start_rejects_missing_original_or_partial_adapter(self):
        state = self.source().state_dict()
        model = create_control_model(self.args, 'motion')
        invalid = dict(state)
        invalid.pop('input.weight')
        with self.assertRaisesRegex(ValueError, 'missing'):
            load_warm(model, invalid)
        invalid = dict(state)
        invalid['unrelated.weight'] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            load_warm(model, invalid)
        invalid = dict(state)
        invalid['control.0.weight'] = model.control[0].weight
        with self.assertRaisesRegex(ValueError, 'whole control stem'):
            load_warm(model, invalid)

    def test_source_reload_disables_previously_trained_adapter(self):
        source = self.source().eval()
        model = create_control_model(self.args, 'reference').eval()
        load_warm(model, source.state_dict())
        with torch.no_grad():
            for parameter in model.control.parameters():
                parameter.uniform_(-1, 1)
        noise, u = torch.randn_like(self.batch.target), torch.tensor([.4])
        expected = bind_control(source, self.batch, self.sample, 'baseline')(noise, u, self.batch.anchor)
        trained = bind_control(model, self.batch, self.sample, 'reference')(noise, u, self.batch.anchor)
        self.assertFalse(torch.equal(trained, expected))
        load_warm(model, source.state_dict())
        restored = bind_control(model, self.batch, self.sample, 'reference')(noise, u, self.batch.anchor)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)

    def test_bad_track_ids_or_frame_counts_do_not_silently_rebase(self):
        bad = dict(self.controls)
        bad['tracks'] = self.controls['tracks'][:8]
        with self.assertRaisesRegex(ValueError, 'all original frames'):
            control_features(self.batch, {'controls': bad}, 'reference')
        bad = dict(self.controls)
        bad['prefix_tracks'] = self.controls['prefix_tracks'][:, :1]
        with self.assertRaisesRegex(ValueError, 'query IDs'):
            control_features(self.batch, {'controls': bad}, 'reference')
        bad = dict(self.controls)
        bad['motion'] = self.controls['motion'].clone()
        bad['motion'][0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            control_features(self.batch, {'controls': bad}, 'motion')

    def test_legacy_prefix_works_for_no_control_and_checkpointed_blocks(self):
        model = ControlledSceneFlow(**{**self.args, 'checkpoint_blocks': True})
        load_warm(model, self.source().state_dict())
        prediction = PrefixFlowNet(model, self.batch.known)(
            torch.randn_like(self.batch.target), torch.tensor([.2]), self.batch.anchor)
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertIsNotNone(model.control[-1].weight.grad)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_two_rank_sparse_controls_and_checkpointed_model_stay_synchronized(self):
        # Tiny fixture weights only; temporary files are automatically removed.
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(_control_ddp_worker, args=(directory,), nprocs=2, join=True)
            first = torch.load(str(Path(directory)/'rank0.pt'), weights_only=True)
            second = torch.load(str(Path(directory)/'rank1.pt'), weights_only=True)
            for key in first:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
