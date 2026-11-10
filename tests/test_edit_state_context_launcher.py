"""Execute the real Stage68 wrapper/Stage66 lifecycle with cluster stubs."""
import json
import unittest

from tests import test_edit_state_backtracking_launcher as fixtures
from tests.test_scene_matched_launcher import BASH


@unittest.skipUnless(BASH, 'Bash required')
class ContextLauncherTests(unittest.TestCase):
    def run_fixture(self, **kwargs):
        return fixtures.EditStateBacktrackingLauncherTests().run_fixture(
            entry='68_refine_vace_edit_state_context.sh',
            config_name='edit_state_context_sync_v1.json',
            namespace_default='edit_state_context_sync_v1', **kwargs)

    def test_full_priority_run_has_real_fixed_and_sync_controls(self):
        result, calls, ports, cfg, args, _ = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        arms = ['native','rgb_base','rae_base','rae_joint','rae_sync','rgb_joint','rgb_sync']
        phases = [c for c in calls if c not in ('publish','guard-start','guard-stop')]
        self.assertEqual(phases, ['asset','asset','prepare', *['render:'+a for a in arms], 'summarize'])
        self.assertEqual(cfg['arms'], arms)
        self.assertEqual(ports, ['30902','30903','30920','30921','30922','30924','30926','30923','30925','30930'])
        self.assertIn('--nnodes=3', result.stdout)
        self.assertIn('--nproc_per_node=8', result.stdout)
        self.assertTrue(any('/mirror/edit_state_context_sync_v1' in v for v in args))
        self.assertTrue(any('/output/vace_edit_state_v1' in v for v in args))
        self.assertEqual(calls.count('guard-start'), 1)
        self.assertEqual(calls.count('guard-stop'), 1)
        self.assertEqual(cfg['solver'], json.loads((fixtures.REPO/'configs/edit_state_feedback_bt_v2.json').read_text())['solver'])

    def test_smoke_uses_full_selected_weights_and_recovers_smoke_config(self):
        result, calls, _, cfg, args, second = self.run_fixture(stage='smoke',resume='render')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(second.returncode, 0, second.stdout+second.stderr)
        self.assertEqual((cfg['cases'],cfg['sample_steps']), (3,6))
        self.assertEqual(calls.count('prepare'), 1)
        self.assertEqual(calls.count('render:rae_sync'), 2)
        self.assertTrue(any('/edit_state_context_sync_v1_smoke' in v for v in args))
        self.assertFalse(any('vace_edit_state_v1_smoke' in v for v in args))

    def test_failed_arm_does_not_prevent_sync_control_or_summary(self):
        result, calls, _, _, _, _ = self.run_fixture(extra={'STUB_FAIL_ARM':'rae_joint'})
        self.assertEqual(result.returncode, 17, result.stdout+result.stderr)
        self.assertIn('render:rae_sync',calls)
        self.assertIn('render:rgb_sync',calls)
        self.assertIn('summarize',calls)
        self.assertIn('guard-stop',calls)

    def test_partial_resume_restores_manifest_from_mirror(self):
        result, calls, _, cfg, _, _ = self.run_fixture(stage='render',remote='smoke',
            extra={'FEEDBACK_NAMESPACE':'edit_state_context_sync_v1_smoke','FEEDBACK_ARMS':'rae_sync'})
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(cfg['sample_steps'],6)
        self.assertIn('render:rae_sync',calls)
        self.assertNotIn('prepare',calls)


if __name__ == '__main__':
    unittest.main()
