"""CPU contracts for the future-feature oracle; not a claim of video quality."""
import copy
import unittest

import torch

from models.scene_flow import SceneFlow
from utils.scene_geo_condition import (GEO_MODEL_FAMILY, GeoConditionRuntime,
    bind_geo_condition, create_geo_model, geo_condition_features, load_warm)
from utils.scene_prefix import PrefixFlowNet, prepare_prefix
from utils.window_flow import WindowFlow


class SceneGeoConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(610)
        self.args = dict(channels=256, grid=2, width=24, depth=1,
                         head_width=48, head_depth=1, heads=4,
                         text_dim=8, checkpoint_blocks=False)
        self.mean = torch.linspace(-.3, .3, 256)
        self.std = torch.linspace(.1, 2., 256)
        self.z = torch.randn(1, 9, 4, 256)
        self.sample = dict(target=self.z, anchor=self.z[:, :1].clone(),
                           source_eval_index=7)
        self.batch = prepare_prefix(self.sample, self.mean, self.std)
        self.noise = torch.randn_like(self.batch.target)
        self.u = torch.tensor([.4])

    def source(self):
        model = SceneFlow(**self.args)
        # Verify real warm-start equivalence, including nonzero camera biases,
        # rather than vacuous equality of two zero-initialized output layers.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-.1, .1)
        return model

    def ready_model(self, *, active=True, checkpoint=False):
        args = {**self.args, 'checkpoint_blocks': checkpoint}
        model = create_geo_model(args, 'oracle_geo')
        load_warm(model, self.source().state_dict())
        if active:
            with torch.no_grad():
                model.geo_condition[-1].weight.uniform_(-.2, .2)
                model.geo_condition[-1].bias.fill_(.1)
        return model

    def bound(self, model, sample=None, batch=None, arm='oracle_geo', **kwargs):
        return bind_geo_condition(model, self.batch if batch is None else batch,
            self.sample if sample is None else sample, arm, self.mean, self.std, **kwargs)

    def test_both_arms_have_identical_parameters_and_initial_values(self):
        states = []
        for arm in ('baseline', 'oracle_geo'):
            torch.manual_seed(811)
            model = create_geo_model(self.args, arm)
            states.append(model.state_dict())
        self.assertEqual(set(states[0]), set(states[1]))
        self.assertEqual(sum(x.numel() for x in states[0].values()),
                         sum(x.numel() for x in states[1].values()))
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    def test_zero_adapter_matches_nontrivial_source_exactly(self):
        source = self.source().eval()
        expected = PrefixFlowNet(source, self.batch.known)(self.noise, self.u, self.batch.anchor)
        self.assertGreater(float(expected.abs().sum()), .1)
        for arm in ('baseline', 'oracle_geo'):
            model = create_geo_model(self.args, arm).eval()
            load_warm(model, source.state_dict())
            got = self.bound(model, arm=arm)(self.noise, self.u, self.batch.anchor)
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_original_geo_slice_fixed_stats_and_zero_prefix(self):
        features = geo_condition_features(self.batch, self.sample, 'oracle_geo', self.mean, self.std)
        self.assertEqual(features.shape, (1, 8, 4, 129))
        self.assertEqual(int(features[:, :3].count_nonzero()), 0)
        torch.testing.assert_close(features[:, 3:, :, :128], self.batch.target[..., :128], rtol=0, atol=0)
        self.assertTrue(torch.equal(features[:, 3:, :, -1], torch.ones(1, 5, 4)))
        # Unbatched cache entries preserve the same frame/token indexing.
        unbatched = dict(target=self.z[0])
        got = geo_condition_features(self.batch, unbatched, 'oracle_geo', self.mean, self.std)
        torch.testing.assert_close(got, features, rtol=0, atol=0)

    def test_baseline_does_not_read_any_future_or_donor_values(self):
        model = self.ready_model().eval()
        changed = self.z.clone()
        changed[:, 4:] = float('nan')
        bad_sample = dict(target=changed, donor_geo='not a tensor', raw=object(), depth=object(),
                          geo_condition_variant='invalid but irrelevant to baseline')
        batch = prepare_prefix(bad_sample, self.mean, self.std)
        runtime = GeoConditionRuntime('baseline', self.mean, self.std)
        first = runtime.bind(model, self.batch, self.sample)
        other = runtime.bind(model, batch, bad_sample)
        self.assertEqual(int(other.geo_features.count_nonzero()), 0)
        expected = first(self.noise, self.u, self.batch.anchor)
        got = other(self.noise, self.u, batch.anchor)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        # Compare complete pure-noise integration, not only one denoising call.
        a = WindowFlow().sample(first, self.batch.anchor, self.noise.clone(), steps=3)
        b = WindowFlow().sample(other, batch.anchor, self.noise.clone(), steps=3)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_future_texture_rgb_depth_do_not_enter_oracle(self):
        model = self.ready_model().eval()
        changed = self.z.clone()
        changed[:, 4:, :, 128:] = float('nan')
        sample = dict(target=changed, rgb=object(), raw=object(), depth=object(),
                      controls=dict(rays=object(), motion=object()))
        alternate = prepare_prefix(sample, self.mean, self.std)
        first = self.bound(model)
        other = self.bound(model, sample=sample, batch=alternate)
        torch.testing.assert_close(first.geo_features, other.geo_features, rtol=0, atol=0)
        torch.testing.assert_close(first(self.noise, self.u, self.batch.anchor),
            other(self.noise, self.u, alternate.anchor), rtol=0, atol=0)

    def test_nonzero_adapter_responds_to_changed_future_geo(self):
        model = self.ready_model().eval()
        changed = self.z.clone()
        changed[:, 4:, :, :128] += torch.linspace(-20, 20, 128)
        sample = dict(target=changed)
        alternate = prepare_prefix(sample, self.mean, self.std)
        expected = self.bound(model)(self.noise, self.u, self.batch.anchor)
        got = self.bound(model, sample, alternate)(self.noise, self.u, alternate.anchor)
        self.assertGreater(float((got - expected).abs().max()), 1e-6)

    def test_no_gt_geometry_copy_or_clamp_in_output(self):
        model = self.ready_model().eval()
        with torch.no_grad():
            model.output[-1].weight.zero_()
            model.output[-1].bias.fill_(.125)
        result = self.bound(model)(self.noise, self.u, self.batch.anchor)
        torch.testing.assert_close(result, torch.full_like(result, .125), rtol=0, atol=0)
        self.assertFalse(torch.equal(result[..., :128], self.batch.target[..., :128]))
        self.assertEqual(result.shape[-1], 256)

    def test_zero_and_prefix_injection_stay_zero_with_learned_biases(self):
        model = self.ready_model().eval()
        zero = self.bound(model, variant='zero')
        self.assertEqual(int(zero.geo_features.count_nonzero()), 0)
        self.assertEqual(int(model.condition_delta(zero.geo_features).count_nonzero()), 0)
        correct = self.bound(model)
        delta = model.condition_delta(correct.geo_features)
        self.assertEqual(int(delta[:, :3].count_nonzero()), 0)
        self.assertGreater(int(delta[:, 3:].count_nonzero()), 0)
        baseline = self.bound(model, arm='baseline')
        torch.testing.assert_close(zero(self.noise, self.u, self.batch.anchor),
            baseline(self.noise, self.u, self.batch.anchor), rtol=0, atol=0)

    def test_shuffle_uses_only_explicit_raw_donor_geo(self):
        donor = torch.randn(5, 4, 128)
        sample = dict(target=object(), donor_geo=donor, donor_case_id='another-case')
        got = geo_condition_features(self.batch, sample, 'oracle_geo', self.mean, self.std, variant='shuffle')
        torch.testing.assert_close(got[:, 3:, :, :128],
            ((donor - self.mean[:128]) / self.std[:128])[None])
        self.assertTrue(torch.equal(got[:, 3:, :, -1], torch.ones(1, 5, 4)))
        for invalid in ({}, {'donor_geo': torch.randn(5, 4, 256)}):
            with self.assertRaises(ValueError):
                geo_condition_features(self.batch, invalid, 'oracle_geo', self.mean, self.std, variant='shuffle')

    def test_gradients_reach_adapter_and_all_parameters_are_used(self):
        for arm in ('baseline', 'oracle_geo'):
            model = self.ready_model(active=False, checkpoint=True).train()
            bound = self.bound(model, arm=arm, training=True)
            loss = (bound(self.noise, self.u, self.batch.anchor) - self.batch.target).square().mean()
            loss.backward()
            for name, param in model.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
            grad = model.geo_condition[-1].weight.grad
            if arm == 'baseline':
                self.assertEqual(int(grad.count_nonzero()), 0)
            else:
                self.assertGreater(float(grad.abs().sum()), 0)
                with torch.no_grad():
                    model.geo_condition[-1].weight.add_(grad * -100)
                model.zero_grad(set_to_none=True)
                (bound(self.noise, self.u, self.batch.anchor) - self.batch.target).square().mean().backward()
                self.assertGreater(float(model.geo_condition[0].weight.grad.abs().sum()), 0)

    def test_input_tensors_are_not_modified_and_features_detach(self):
        donor = torch.randn(5, 4, 128, requires_grad=True)
        sample = dict(self.sample, donor_geo=donor)
        before = [x.detach().clone() for x in (self.z, self.mean, self.std, donor,
                                               self.batch.anchor, self.batch.known, self.batch.target)]
        for variant in ('correct', 'zero', 'shuffle'):
            features = geo_condition_features(self.batch, sample, 'oracle_geo', self.mean, self.std, variant=variant)
            self.assertFalse(features.requires_grad)
        for old, now in zip(before, (self.z, self.mean, self.std, donor,
                                     self.batch.anchor, self.batch.known, self.batch.target)):
            torch.testing.assert_close(old, now, rtol=0, atol=0)

    def test_strict_warm_keys_and_reset_for_source_replay(self):
        source = self.source().eval()
        model = self.ready_model().eval()
        state = source.state_dict()
        for changed in ('missing', 'unexpected', 'partial', 'shape'):
            invalid = dict(state)
            if changed == 'missing':
                invalid.pop('input.weight')
            elif changed == 'unexpected':
                invalid['control.0.weight'] = torch.zeros(1)
            elif changed == 'partial':
                invalid['geo_condition.0.weight'] = model.geo_condition[0].weight
            else:
                invalid['input.weight'] = torch.zeros(1)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                load_warm(model, invalid)
        load_warm(model, state)
        expected = PrefixFlowNet(source, self.batch.known)(self.noise, self.u, self.batch.anchor)
        torch.testing.assert_close(self.bound(model)(self.noise, self.u, self.batch.anchor), expected, rtol=0, atol=0)
        full = copy.deepcopy(model.state_dict())
        full['geo_condition.2.bias'].fill_(.2)
        load_warm(model, full)
        torch.testing.assert_close(model.geo_condition[-1].bias, full['geo_condition.2.bias'])

    def test_runtime_checkpoint_family_arm_and_stats_are_verified(self):
        runtime = GeoConditionRuntime('oracle_geo', self.mean, self.std)
        source = dict(model=self.source().state_dict())
        runtime.validate_checkpoint_metadata(source, resume=False)
        with self.assertRaisesRegex(ValueError, 'resume'):
            runtime.validate_checkpoint_metadata(source, resume=True)
        model = runtime.make_model(self.args)
        saved = dict(model=model.state_dict(), **runtime.checkpoint_metadata())
        runtime.validate_checkpoint_metadata(saved, resume=True)
        runtime.validate_checkpoint_metadata(saved, resume=False)
        self.assertEqual(saved['model_family'], GEO_MODEL_FAMILY)
        self.assertFalse(saved['deployable_future_condition'])
        self.assertTrue(saved['future_geo_may_contain_appearance'])
        for key, value in (('model_family', 'ControlledSceneFlow'), ('geo_schema', 'bad'),
                           ('experiment_arm', 'baseline'), ('history', 1),
                           ('geo_condition_stats_sha256', 'bad')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.validate_checkpoint_metadata({**saved, key: value}, resume=True)
        with self.assertRaises(ValueError):
            runtime.validate_checkpoint_metadata(dict(model=model.state_dict()), resume=False)

    def test_runtime_groups_metrics_annotations_and_eval_override(self):
        runtime = GeoConditionRuntime('oracle_geo', self.mean, self.std)
        model = self.ready_model()
        groups = runtime.optimizer_groups(model)
        self.assertEqual([x['lr_scale'] for x in groups], [1., 10.])
        self.assertEqual({id(p) for group in groups for p in group['params']},
                         {id(p) for p in model.parameters()})
        sample = dict(self.sample, geo_condition_variant='zero')
        bound = runtime.bind(model, self.batch, sample)
        metrics = runtime.training_metrics(model, bound, self.noise)
        self.assertEqual(metrics['geo_condition_injection_rms'], 0.)
        self.assertEqual(metrics['geo_condition_presence_fraction'], 0.)
        row = dict(supervision='old')
        runtime.annotate(row, sample)
        self.assertEqual(row['geo_condition_variant'], 'zero')
        self.assertFalse(row['future_control_supplied'])
        with self.assertRaisesRegex(ValueError, 'Training requires'):
            runtime.bind(model, self.batch, sample, training=True)
        self.assertGreater(int(runtime.bind(model, self.batch, self.sample, training=True).geo_features.count_nonzero()), 0)


if __name__ == '__main__':
    unittest.main()
