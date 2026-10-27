"""Small synthetic teacher tests; these do not validate real StreamVGGT/NPU."""
import copy
import json
import unittest

import torch
from torch import nn

from utils.scene_teacher_probe import PROBE_MODULES, probe_teacher, tensor_summary


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)

    def forward(self, x):
        return self.norm(x)


class Tracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.fmap_norm = nn.LayerNorm(4)
        self.ffeat_norm = nn.LayerNorm(4)
        self.ffeat_updater = nn.Sequential(nn.Linear(4, 4), nn.GELU())
        self.vis_predictor = nn.Sequential(nn.Linear(4, 1))
        self.conf_predictor = nn.Sequential(nn.Linear(4, 1))
        self.fail = False
        nn.init.zeros_(self.vis_predictor[0].weight)
        nn.init.zeros_(self.conf_predictor[0].weight)
        nn.init.constant_(self.vis_predictor[0].bias, 3.)
        nn.init.constant_(self.conf_predictor[0].bias, 3.)

    def forward(self, x):
        x = self.fmap_norm(x)
        for i in range(3):
            x = x + self.ffeat_updater(self.ffeat_norm(x))
            if self.fail and i == 1:
                raise RuntimeError('synthetic tracker failure')
        return self.vis_predictor(x).sigmoid(), self.conf_predictor(x).sigmoid()


class TrackHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Extractor()
        self.tracker = Tracker()

    def forward(self, x):
        return self.tracker(self.feature_extractor(x))


class FakeTeacher:
    def __init__(self):
        self.model = nn.Module()
        self.model.track_head = TrackHead()
        self.model.eval()
        self.precision = 'synthetic_float32'
        self.metadata = dict(backend='synthetic', coverage={'matched': 1})

    def infer(self, frames, reference_frame=0):
        t = len(frames)
        x = torch.arange(t*2*4, dtype=torch.float32).reshape(t, 2, 4)/10
        vis, conf = self.model.track_head(x)
        points = torch.tensor([[2., 3.], [4., 5.]])
        return dict(tracks=points[None].repeat(t, 1, 1), query_points=points,
            visibility=vis[..., 0], track_confidence=conf[..., 0], image_size_hw=[14, 14],
            reference_frame=reference_frame, teacher=self.metadata)


def hook_counts(model):
    return [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in model.modules()]


class TeacherForwardProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.teacher = FakeTeacher()
        self.frames = torch.zeros(4, 3, 14, 14)

    def test_complete_iteration_events_logits_and_unchanged_weights(self):
        before = copy.deepcopy(self.teacher.model.state_dict())
        hooks = hook_counts(self.teacher.model)
        result = probe_teacher(self.teacher, self.frames, 'case001', 'repeat_static')
        report = result['report']
        self.assertEqual(report['status'], 'completed')
        self.assertTrue(report['input_frames_identical'])
        self.assertEqual(report['control_evidence_status'], 'nonzero_support_requires_quality_review')
        updates = [e for e in report['events'] if e['module'].endswith('ffeat_updater')]
        self.assertEqual([e['refinement_iteration'] for e in updates], [1, 2, 3])
        logits = [e for e in report['events'] if e['module'].endswith('conf_predictor')][0]['output'][0]
        self.assertAlmostEqual(logits['mean'], 3.)
        self.assertGreater(logits['sigmoid_of_sampled_logits']['minimum'], .95)
        self.assertEqual(hook_counts(self.teacher.model), hooks)
        self.assertEqual(set(report['hook_calls']), set(PROBE_MODULES))
        for key, value in before.items():
            torch.testing.assert_close(value, self.teacher.model.state_dict()[key], rtol=0, atol=0)
        json.dumps(report, allow_nan=False)

    def test_failure_cleanup_then_clean_second_call_and_existing_hook_survives(self):
        calls = []
        external = self.teacher.model.track_head.register_forward_pre_hook(lambda *_: calls.append('external'))
        try:
            hooks = hook_counts(self.teacher.model)
            self.teacher.model.track_head.tracker.fail = True
            failed = probe_teacher(self.teacher, self.frames, 'x', 'real')
            self.assertEqual(failed['report']['status'], 'failed')
            self.assertIn('synthetic tracker failure', failed['report']['error']['message'])
            self.assertIsNone(failed['outputs'])
            self.assertEqual(hook_counts(self.teacher.model), hooks)
            self.teacher.model.track_head.tracker.fail = False
            second = probe_teacher(self.teacher, self.frames, 'x', 'repeat_static')
            self.assertEqual(second['report']['status'], 'completed')
            self.assertEqual(second['report']['tracking_passes'], 1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(hook_counts(self.teacher.model), hooks)
        finally:
            external.remove()

    def test_zero_support_is_not_reported_as_working_controls(self):
        nn.init.constant_(self.teacher.model.track_head.tracker.conf_predictor[0].bias, -22.)
        result = probe_teacher(self.teacher, self.frames, 'bad', 'real')
        self.assertEqual(result['report']['status'], 'completed')
        self.assertEqual(result['report']['control_evidence_status'], 'zero_track_support')
        self.assertEqual(result['report']['context']['positive_weight_count'], 0)

    def test_inside_module_failure_records_entered_phase_and_input_kind(self):
        module = self.teacher.model.track_head.tracker.ffeat_updater
        original = module.forward
        hooks = hook_counts(self.teacher.model)
        self.frames[1].fill_(1)
        def fail(_x):
            raise RuntimeError('inside updater')
        module.forward = fail
        try:
            result = probe_teacher(self.teacher, self.frames, 'moving', 'real')
            report = result['report']
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['kind'], 'real')
            self.assertFalse(report['input_frames_identical'])
            event = report['events'][-1]
            self.assertTrue(event['module'].endswith('ffeat_updater'))
            self.assertEqual(event['phase'], 'entered_without_return')
            self.assertEqual(event['refinement_iteration'], 1)
            self.assertEqual(hook_counts(self.teacher.model), hooks)
        finally:
            module.forward = original

    def test_bounded_noncontiguous_sampling_and_nonfinite_json(self):
        x = torch.arange(120, dtype=torch.float32).reshape(10, 12).T
        report = tensor_summary(x, max_samples=7)
        indices = torch.arange(7)*(x.numel()-1)//6
        expected = x.reshape(-1)[indices]
        self.assertFalse(report['exact'])
        self.assertEqual(report['sample_count'], 7)
        self.assertAlmostEqual(report['mean'], float(expected.double().mean()))
        report = tensor_summary(torch.tensor([float('nan'), float('inf')]))
        self.assertIsNone(report['mean'])
        self.assertEqual(report['nonfinite_sample_count'], 2)
        json.dumps(report, allow_nan=False)

    def test_missing_module_is_reported_without_running_partial_probe(self):
        del self.teacher.model.track_head.tracker.conf_predictor
        hooks = hook_counts(self.teacher.model)
        result = probe_teacher(self.teacher, self.frames, 'missing', 'real')
        self.assertEqual(result['report']['status'], 'failed')
        self.assertIn('track_head.tracker.conf_predictor', result['report']['missing_modules'])
        self.assertEqual(hook_counts(self.teacher.model), hooks)


if __name__ == '__main__':
    unittest.main()
