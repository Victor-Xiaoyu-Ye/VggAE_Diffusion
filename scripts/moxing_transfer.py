#!/usr/bin/env python3
"""Copy ModelArts inputs/outputs with MoXing."""

import argparse
import os
import time

from utils.moxing_io import copy_directory, copy_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--directory", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    return parser.parse_args()


def transfer(args):
    if args.directory:
        if os.path.isdir(args.source):
            copy_directory(args.source, args.destination)
    elif os.path.exists(args.source) or args.source.startswith(
            ("obs://", "s3://")):
        copy_file(args.source, args.destination)


def main():
    args = parse_args()
    if not args.watch:
        transfer(args)
        return
    while True:
        try:
            transfer(args)
        except Exception as exc:
            print(f"[WARN] MoXing sync failed: {exc}", flush=True)
        time.sleep(max(args.interval, 10))


if __name__ == "__main__":
    main()

