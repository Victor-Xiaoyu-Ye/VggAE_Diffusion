#!/usr/bin/env python3
"""Exit successfully when a local or OBS path exists."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.moxing_io import is_remote_path, remote_exists


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: remote_exists.py PATH")
    path = sys.argv[1]
    exists = remote_exists(path) if is_remote_path(path) else os.path.exists(path)
    raise SystemExit(0 if exists else 1)


if __name__ == "__main__":
    main()
