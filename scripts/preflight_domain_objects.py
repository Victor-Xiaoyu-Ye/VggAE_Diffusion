"""Check selected object existence without downloading media or loading models."""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.moxing_io import is_remote_path, _mox


def check_objects(csv_files, root, exists, workers=8):
    objects = {}
    for filename in csv_files:
        with open(filename, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                rel = row['video path'].replace('\\', '/')
                if rel.startswith('videos/'):
                    rel = rel[len('videos/'):]
                path = root.rstrip('/') + '/' + rel
                if row['id'] in objects and objects[row['id']] != path:
                    raise ValueError('Conflicting paths for the same clip ID')
                objects[row['id']] = path
    def check(item):
        clip, path = item
        for attempt in range(3):
            try:
                if exists(path):
                    return None
                return dict(video_id=clip, path=path, reason='not_found')
            except Exception as exc:
                if attempt == 2:
                    return dict(video_id=clip, path=path, reason='access_error', error=type(exc).__name__)
                time.sleep(.25 * (attempt + 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        failed = [r for r in pool.map(check, objects.items()) if r is not None]
    return dict(checked=len(objects), failed=len(failed), failures=failed,
                media_downloaded=False, passed=not failed, unix_time=time.time())


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--csv',nargs='+',required=True)
    p.add_argument('--video_root',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    exists = _mox().file.exists if is_remote_path(a.video_root) else os.path.isfile
    result=check_objects(a.csv,a.video_root,exists)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"Object preflight: {result['checked']} selected clips, {result['failed']} unavailable; no media downloaded", flush=True)
    if not result['passed']:
        raise SystemExit('Preflight failed before AE/model loading; inspect object_preflight.json. Do not silently replace clips.')
