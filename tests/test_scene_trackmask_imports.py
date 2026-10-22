"""Cluster-like package collisions, tested in fresh offline Python processes."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from utils.scene_trackmask_vendor import validate_stream_vendor_sources


REPO = Path(__file__).resolve().parents[1]


class StreamVendorImportTests(unittest.TestCase):
    def run_collision(self, preload):
        temporary_root = REPO.parent if os.name == 'nt' else None
        with tempfile.TemporaryDirectory(prefix='stream import ', dir=temporary_root) as tmp:
            root = Path(tmp)
            foreign = root / 'foreign'
            package = foreign / 'third_party'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text("MARKER = 'foreign-installed-package'\n", encoding='utf8')
            unrelated = root / 'unrelated cwd'
            unrelated.mkdir()
            env = os.environ.copy()
            # Even with our repository first, a regular third_party elsewhere
            # wins over the repository's implicit namespace directory.
            env['PYTHONPATH'] = os.pathsep.join((str(REPO), str(foreign), env.get('PYTHONPATH', '')))
            env['HF_HUB_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = '1'
            script = '''
import importlib.util
from pathlib import Path
import sys
assert 'foreign' in str(importlib.util.find_spec('third_party').origin)
if PRELOAD:
    import third_party
    original = third_party
    assert original.MARKER == 'foreign-installed-package'
from utils.scene_trackmask_teacher import import_streamvggt_teacher
from utils.scene_trackmask_stream import CameraHead, DPTHead, TrackHead
model, pose_decoder, provenance = import_streamvggt_teacher()
assert model.__name__ == 'StreamVGGTTrackModel'
assert callable(pose_decoder)
for head in (CameraHead, DPTHead, TrackHead):
    assert head.__module__.startswith('_vggae_stream_trackmask_vendor.')
    location = Path(sys.modules[head.__module__].__file__).resolve()
    assert location.is_relative_to(Path(REPO) / 'third_party' / 'streamvggt_trackmask')
if PRELOAD:
    assert sys.modules['third_party'] is original
import third_party
assert third_party.MARKER == 'foreign-installed-package'
assert len(provenance['code_sha256']) == 64
print('offline full-head imports passed; foreign package preserved')
'''.replace('PRELOAD', repr(preload)).replace('REPO', repr(str(REPO)))
            result = subprocess.run([sys.executable, '-c', script], cwd=unrelated,
                                    env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('full-head imports passed', result.stdout)

    def test_installed_regular_third_party_cannot_shadow_bundled_heads(self):
        self.run_collision(preload=False)

    def test_preloaded_foreign_third_party_is_preserved_and_cannot_shadow_heads(self):
        self.run_collision(preload=True)

    def test_incomplete_code_upload_reports_source_files_instead_of_requesting_weights(self):
        temporary_root = REPO.parent if os.name == 'nt' else None
        with tempfile.TemporaryDirectory(prefix='stream missing ', dir=temporary_root) as tmp:
            with self.assertRaises(FileNotFoundError) as error:
                validate_stream_vendor_sources(tmp)
            message = str(error.exception)
            self.assertIn('heads/camera_head.py', message)
            self.assertIn('complete repository code', message)
            self.assertIn('no new model-weight download', message)


if __name__ == '__main__':
    unittest.main()
