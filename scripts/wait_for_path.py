#!/usr/bin/env python3
"""Wait for a local or OBS artifact produced by another cluster node."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.moxing_io import is_remote_path, remote_exists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        exists = (
            remote_exists(args.path)
            if is_remote_path(args.path) else os.path.isfile(args.path)
        )
        if exists:
            print(f"Ready: {args.path}")
            return
        time.sleep(max(args.interval, 1))
    raise TimeoutError(f"Timed out waiting for: {args.path}")


if __name__ == "__main__":
    main()
