"""Real fresh-process imports with a discoverable fixture accelerator backend.

These verify CPU isolation, not the unobserved native core-dump stack on 910B.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class SummaryCPURuntimeTests(unittest.TestCase):
    def run_fixture(self, code, *, cpu_only=False, autoload='0'):
        with tempfile.TemporaryDirectory(prefix='summary-cpu-', dir=REPO.parent) as temp:
            root = Path(temp)
            backend = root/'torch_npu'
            backend.mkdir()
            (backend/'__init__.py').write_text('''
import os
from pathlib import Path
Path(os.environ['TEST_BACKEND_MARKER']).write_text('imported')
def _autoload():
    Path(os.environ['TEST_AUTOLOAD_MARKER']).write_text('autoloaded')
''', encoding='utf8')
            metadata = root/'vggae_fixture_backend-0.0.0.dist-info'
            metadata.mkdir()
            (metadata/'METADATA').write_text(
                'Metadata-Version: 2.1\nName: vggae-fixture-backend\nVersion: 0.0.0\n',
                encoding='utf8')
            (metadata/'entry_points.txt').write_text(
                '[torch.backends]\nvggae_fixture_backend = torch_npu:_autoload\n',
                encoding='utf8')
            marker, auto_marker = root/'imported', root/'autoloaded'
            env = {k:v for k,v in os.environ.items() if k not in
                   ('VGGAE_CPU_ONLY', 'TORCH_DEVICE_BACKEND_AUTOLOAD')}
            env.update(TORCH_DEVICE_BACKEND_AUTOLOAD=autoload,
                PYTHONPATH=os.pathsep.join([str(root), str(REPO), env.get('PYTHONPATH','')]),
                TEST_BACKEND_MARKER=str(marker), TEST_AUTOLOAD_MARKER=str(auto_marker),
                TEST_OUTPUT=str(root/'output'))
            if cpu_only:
                env['VGGAE_CPU_ONLY'] = '1'
            result = subprocess.run([sys.executable, '-c', code], env=env,
                cwd=REPO, capture_output=True, text=True, encoding='utf8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            return marker.exists(), auto_marker.exists()

    def test_real_summary_imports_and_artifacts_never_load_npu(self):
        imported, autoloaded = self.run_fixture('''
import os, sys
from pathlib import Path
import refine_vace_edit_state
import torch
from utils import device
from utils.scene_run import ArtifactStore
root = Path(os.environ['TEST_OUTPUT'])
store = ArtifactStore(root/'local', [str(root/'primary'), str(root/'mirror')])
store.save('result.pt', {'generated':torch.arange(6)}, 'fixture')
value = store.load('result.pt', 'fixture', required=True)
assert value['generated'].device.type == 'cpu'
assert value['generated'].sum().item() == 15
assert 'torch_npu' not in sys.modules
def unexpected_probe(*a, **k):
    raise AssertionError('CPU summary probed an accelerator')
torch.cuda.is_available = unexpected_probe
from types import SimpleNamespace
torch.npu = SimpleNamespace(is_available=unexpected_probe)
assert device.get_device_name() == 'cpu'
assert device.get_device().type == 'cpu'
assert not device.is_available()
device.empty_cache()
device.manual_seed_all(42)
device.configure_backend_compatibility('npu')
''', cpu_only=True)
        self.assertFalse(imported)
        self.assertFalse(autoloaded)

    def test_normal_device_import_still_registers_npu(self):
        imported, autoloaded = self.run_fixture('import utils.device')
        self.assertTrue(imported)
        self.assertFalse(autoloaded)

    def test_fixture_backend_is_actually_discoverable(self):
        imported, autoloaded = self.run_fixture('import torch', autoload='1')
        self.assertTrue(imported)
        self.assertTrue(autoloaded)


if __name__ == '__main__':
    unittest.main()
