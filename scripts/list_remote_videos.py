#!/usr/bin/env python3
"""List video files under a (possibly remote OBS) root, one relative path
per line.

Used to build the --available_list for prepare_spatialvid_splits.py when the
dataset mirror is SPARSE (e.g. spatial-vid-hq-oft holds only the 10k subset's
files while sharing the full 360k metadata CSV). Listing the actual tree once
reproduces, on the cluster, what the local isfile check does on the H200 box.
"""

from __future__ import annotations

import argparse
import os
import tempfile


def parse_args():
    p = argparse.ArgumentParser(description='List (remote) video tree')
    p.add_argument('--root', required=True,
                   help='local dir or obs:// / s3:// prefix')
    p.add_argument('--output', required=True,
                   help='output text file, one relative path per line')
    p.add_argument('--suffix', default='.mp4')
    return p.parse_args()


def list_remote(root, suffix):
    import moxing as mox
    entries = mox.file.list_directory(root, recursive=True)
    rels = []
    prefix = root.rstrip('/') + '/'
    for entry in entries:
        name = entry[len(prefix):] if entry.startswith(prefix) else entry
        if name.endswith(suffix):
            rels.append(name)
    return rels


def list_local(root, suffix):
    rels = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(suffix):
                full = os.path.join(dirpath, filename)
                rels.append(os.path.relpath(full, root).replace('\\', '/'))
    return rels


def main():
    args = parse_args()
    if args.root.startswith(('obs://', 's3://')):
        rels = list_remote(args.root, args.suffix)
    else:
        rels = list_local(args.root, args.suffix)
    rels = sorted(set(rels))
    if not rels:
        raise SystemExit(f'no *{args.suffix} files found under {args.root}')
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix='.tmp')
    with os.fdopen(fd, 'w') as handle:
        handle.write('\n'.join(rels) + '\n')
    os.replace(tmp, args.output)
    print(f'{len(rels)} videos -> {args.output}')


if __name__ == '__main__':
    main()
