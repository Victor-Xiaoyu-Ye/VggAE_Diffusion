#!/usr/bin/env python3
"""Copy ModelArts inputs/outputs with MoXing."""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Corporate MoXing/OBS SDK can emit noisy logging rollover errors when several
# copy_parallel workers initialize the same rotating OBS log. Actual failures
# are still raised by copy_file/copy_directory and retried by --watch only.
logging.raiseExceptions = False
for logger_name in ('obs', 'moxing', 'esdk-obs-python'):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from utils.moxing_io import copy_directory, copy_file, is_remote_path
from utils.output_snapshot import build_snapshot, cleanup_snapshot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--directory', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=300)
    return parser.parse_args()


def transfer(args):
    if args.directory:
        if is_remote_path(args.source):
            copy_directory(args.source, args.destination)
        elif os.path.isdir(args.source):
            snapshot, _ = build_snapshot(args.source)
            try:
                copy_directory(snapshot, args.destination)
            finally:
                cleanup_snapshot(snapshot)
        elif not args.watch:
            raise FileNotFoundError(args.source)
    elif os.path.exists(args.source) or is_remote_path(args.source):
        copy_file(args.source, args.destination)
    elif not args.watch:
        raise FileNotFoundError(args.source)


def main():
    args = parse_args()
    if not args.watch:
        transfer(args)
        return
    while True:
        try:
            transfer(args)
        except Exception as exc:
            print(f'[WARN] MoXing sync failed: {exc}', flush=True)
        time.sleep(max(args.interval, 10))


if __name__ == '__main__':
    main()
