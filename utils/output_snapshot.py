"""Snapshot local output trees without exposing atomic-save intermediates.

Checkpoint hardlinks pin rename-on-save inodes. Logs/JSONL are copied rather
than linked: the source can keep appending, the upload view cannot. This is a
per-file stable view, not a transaction spanning every file in the run.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

# Only directories created by this process may be removed by cleanup_snapshot.
_OWNED = set()


def is_temp_basename(name):
    name = name.lower()
    return (name.startswith('.vggae-sync-') or name.endswith(('.tmp', '.part', '.partial'))
            or any(token in name for token in ('.tmp-', '.pending.', '.partial-', '.part-')))


def _checkpoint(path):
    return path.name.startswith('checkpoint') and path.suffix in ('.pt', '.pth')


def cleanup_snapshot(root):
    key = str(Path(root).absolute())
    if key not in _OWNED:
        raise ValueError('refusing to remove an unowned snapshot directory')
    shutil.rmtree(key)
    _OWNED.remove(key)


def _snapshot_file(source, target, link_checkpoints):
    for attempt in range(3):
        try:
            before = source.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f'non-regular output file: {source}')
            if link_checkpoints and _checkpoint(source):
                try:
                    os.link(source, target, follow_symlinks=False)
                    after = target.lstat()
                    if not stat.S_ISREG(after.st_mode):
                        raise ValueError(f'output changed to symlink: {source}')
                    return 'hardlinked'
                except FileNotFoundError:
                    raise
                except OSError:
                    # Cross-device / unavailable hardlinks: pin an open handle.
                    pass
            with source.open('rb') as src, target.open('wb') as dst:
                opened = os.fstat(src.fileno())
                # Capture the initial length, not a growing log's moving EOF.
                remaining = opened.st_size
                while remaining:
                    block = src.read(min(1024*1024, remaining))
                    if not block:
                        raise OSError(f'output truncated during snapshot: {source}')
                    dst.write(block)
                    remaining -= len(block)
            os.utime(target, ns=(opened.st_atime_ns, opened.st_mtime_ns))
            return 'copied'
        except FileNotFoundError:
            if target.exists():
                target.unlink()  # only our partial snapshot file
            if attempt == 2:
                raise
            time.sleep(.02)
    raise RuntimeError('unreachable snapshot retry')


def build_snapshot(source, link_checkpoints=True):
    source = Path(source).absolute()
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f'expected local non-symlink directory: {source}')
    # Same filesystem allows hardlinks. Parent output watchers skip .vggae-sync-*.
    try:
        root = Path(tempfile.mkdtemp(prefix='.vggae-sync-', dir=source.parent))
    except OSError:
        root = Path(tempfile.mkdtemp(prefix='.vggae-sync-'))
    if source == root or source in root.parents:
        shutil.rmtree(root)
        raise ValueError('snapshot must be outside source')
    _OWNED.add(str(root.absolute()))
    stats = {'files': 0, 'hardlinked': 0, 'copied': 0, 'skipped_temp': 0, 'skipped_symlink': 0}
    def walk_error(error):
        raise error
    try:
        for parent, dirs, files in os.walk(source, followlinks=False, onerror=walk_error):
            parent = Path(parent)
            kept = []
            for name in dirs:
                if is_temp_basename(name):
                    stats['skipped_temp'] += 1
                elif (parent/name).is_symlink():
                    stats['skipped_symlink'] += 1
                else:
                    kept.append(name)
            dirs[:] = kept
            destination = root/parent.relative_to(source)
            destination.mkdir(parents=True, exist_ok=True)
            for name in files:
                if is_temp_basename(name):
                    stats['skipped_temp'] += 1
                    continue
                src = parent/name
                if src.is_symlink():
                    stats['skipped_symlink'] += 1
                    continue
                mode = _snapshot_file(src, destination/name, link_checkpoints)
                stats['files'] += 1
                stats[mode] += 1
        return str(root), stats
    except BaseException:
        cleanup_snapshot(root)
        raise
