"""Load bundled StreamVGGT heads without the generic ``third_party`` package.

Some cluster dependencies install or preload a regular package called
``third_party``. It takes precedence over our namespace directory even when
the repository is first on PYTHONPATH. Use an isolated package whose search
path is the explicit bundled directory; leave foreign packages untouched.
"""
from importlib import import_module, machinery, util
from pathlib import Path
import sys


STREAM_VENDOR_ROOT = Path(__file__).resolve().parents[1] / 'third_party' / 'streamvggt_trackmask'
STREAM_VENDOR_PACKAGE = '_vggae_stream_trackmask_vendor'
REQUIRED_SOURCES = (
    'heads/camera_head.py', 'heads/dpt_head.py', 'heads/head_act.py',
    'heads/track_head.py', 'heads/utils.py',
    'heads/track_modules/__init__.py', 'heads/track_modules/base_track_predictor.py',
    'heads/track_modules/blocks.py', 'heads/track_modules/modules.py',
    'heads/track_modules/utils.py', 'utils/pose_enc.py', 'utils/rotation.py',
)


def validate_stream_vendor_sources(root=None):
    root = Path(root or STREAM_VENDOR_ROOT).resolve()
    missing = [name for name in REQUIRED_SOURCES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f'Bundled StreamVGGT source files are missing from {root}: {missing}. '
            'Update the complete repository code, including third_party/streamvggt_trackmask; '
            'no new model-weight download is required.')
    return root


def import_stream_vendor(name):
    """Import a bundled submodule by filesystem path, independent of cwd.

    The root is a private namespace package. Relative imports inside the pinned
    sources remain intact and never resolve through an installed third_party.
    """
    root = validate_stream_vendor_sources()
    existing = sys.modules.get(STREAM_VENDOR_PACKAGE)
    if existing is not None:
        paths = [Path(path).resolve() for path in getattr(existing, '__path__', [])]
        if paths != [root]:
            raise RuntimeError('The private StreamVGGT vendor package belongs to another checkout; '
                               'use a fresh Python process.')
    else:
        spec = machinery.ModuleSpec(STREAM_VENDOR_PACKAGE, loader=None, is_package=True)
        spec.submodule_search_locations = [str(root)]
        package = util.module_from_spec(spec)
        sys.modules[STREAM_VENDOR_PACKAGE] = package
    return import_module(f'{STREAM_VENDOR_PACKAGE}.{name}')
