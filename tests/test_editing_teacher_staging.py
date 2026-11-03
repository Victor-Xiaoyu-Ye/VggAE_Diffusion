"""Exercise the real teacher shell, fallback helper and file transfer paths."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.test_scene_matched_launcher import BASH, shell

REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(BASH, 'Bash required')
class EditingTeacherStagingTests(unittest.TestCase):
    def stage(self, mode):
        with tempfile.TemporaryDirectory(prefix='editing teacher ', dir=REPO.parent) as directory:
            root = Path(directory)
            (root/'scripts/lib').mkdir(parents=True)
            for name in ('scripts/stage_editing_teacher.sh', 'scripts/lib/modelarts.sh'):
                shutil.copyfile(REPO/name, root/name)
            # Only the root configuration is substituted. Staging, fallback
            # and moxing_transfer's local copy run their production code.
            shell(root/'scripts/spatialvid_config.sh', '''
PROJECT="${TEST_REPO}"
PYTHON_BIN="${TEST_PYTHON}"
SCALE_REMOTE_ROOT="${TEST_ROOT}/primary"
SCALE_MIRROR_ROOT="${TEST_ROOT}/mirror"
STREAMVGGT_CKPT="${TEST_ROOT}/stream.pt"
''')
            (root/'stream.pt').write_bytes(b'existing stream checkpoint')
            relative = 'scene_rae_c256_15day_v1/production/ae/best_joint.pt'
            selected = root/'mirror'/relative
            selected.parent.mkdir(parents=True)
            selected.write_bytes(b'selected production checkpoint')
            # An obsolete short path must never substitute a different AE.
            wrong = root/'mirror/scene_rae_c256_15day_v1/ae/best_joint.pt'
            wrong.parent.mkdir(parents=True)
            wrong.write_bytes(b'wrong checkpoint')
            target = root/'cache/teacher.pt'
            env = {k:v for k,v in os.environ.items()
                   if not k.startswith(('EDIT_', 'STREAMVGGT_', 'TEST_'))}
            env.update(TEST_REPO=REPO.as_posix(), TEST_ROOT=root.as_posix(),
                       TEST_PYTHON=Path(sys.executable).as_posix(), EDIT_RAE=target.as_posix())
            expected = selected.read_bytes()
            if mode == 'override':
                custom = root/'custom.pt'; custom.write_bytes(b'explicit checkpoint')
                env['EDIT_RAE_URL'] = custom.as_posix()
                expected = custom.read_bytes()
            elif mode == 'local':
                target.parent.mkdir(parents=True); target.write_bytes(b'cached selected checkpoint')
                expected = target.read_bytes()
            result = subprocess.run([BASH, (root/'scripts/stage_editing_teacher.sh').as_posix()],
                                    cwd=REPO.parent, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(target.read_bytes(), expected)
            return result.stdout

    def test_missing_current_output_falls_back_to_production_mirror(self):
        output = self.stage('default')
        self.assertIn('/production/ae/best_joint.pt', output)
        self.assertIn('Staged production Scene C256 AE from', output)

    def test_explicit_checkpoint_override_is_preserved(self):
        output = self.stage('override')
        self.assertIn('/custom.pt', output)
        self.assertNotIn('/mirror/', output)

    def test_existing_local_checkpoint_is_reused(self):
        self.assertNotIn('Staging', self.stage('local'))


if __name__ == '__main__':
    unittest.main()
