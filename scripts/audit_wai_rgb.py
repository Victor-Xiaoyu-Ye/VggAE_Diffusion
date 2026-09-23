"""Audit WAI scene JSON against nonempty RGB objects. Never download media/pickle.

Credentials are environment-only. Receipts are resumable metadata, not a claim
that an image decodes or that camera poses/timestamps have been validated.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time

ROOT = 'data/external/personal/g00833899/y50046448/dataset/wai_dataset/'
BUCKET = 'yw-ads-training-gy1'


def summarize(meta, objects):
    frames = meta.get('frames', [])
    valid, missing, bad = [], [], 0
    for frame in frames:
        path = frame.get('image') or frame.get('file_path', '')
        parts = PurePosixPath(path).parts
        safe = (path.startswith('images/') and '..' not in parts
                and path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
        if frame.get('is_bad', False):
            bad += 1
        elif safe and objects.get(path, 0) > 0:
            valid.append(path)
        else:
            missing.append(path)
    streams = {}
    for frame in frames:
        name = Path(frame.get('image') or frame.get('file_path', '')).stem
        side = re.search(r'_(left|right)[_-]', name)
        number = re.search(r'(\d+)$', name)
        if number:
            streams.setdefault(side[1] if side else 'mono', []).append(int(number[1]))
    sequence = {}
    for camera, numbers in streams.items():
        values = sorted(set(numbers))
        sequence[camera] = dict(count=len(numbers), unique=len(values), first=values[0], last=values[-1],
            step_histogram=dict(Counter(b-a for a,b in zip(values,values[1:]))))
    return dict(referenced_frames=len(frames), listed_rgb=len(objects),
                usable_frames=len(set(valid)), missing_rgb=len(missing),
                flagged_bad_frames=bad, missing_examples=missing[:8],
                image_size=[meta.get('h', frames[0].get('h') if frames else None),
                            meta.get('w', frames[0].get('w') if frames else None)],
                camera_convention=meta.get('camera_convention'),
                scale_type=meta.get('scale_type'),
                shared_intrinsics=meta.get('shared_intrinsics'),
                fps=meta.get('fps'), decode_verified=False,
                pose_frames=sum('transform_matrix' in f for f in frames),
                depth_frames=sum('depth' in f for f in frames),
                interpolated_frames=sum(bool(f.get('_is_interpolated')) for f in frames),
                metadata_streams=sequence,
                tiny_rgb_objects=sum(0<size<256 for size in objects.values()),
                rgb_bytes=sum(objects.values()))


def audit(args):
    from obs import ObsClient
    ak, sk = os.environ.pop('VGG_OBS_AK'), os.environ.pop('VGG_OBS_SK')
    local, clients = threading.local(), []

    def client():
        if not hasattr(local, 'client'):
            local.client = ObsClient(access_key_id=ak, secret_access_key=sk,
                server=os.environ.get('VGG_OBS_ENDPOINT', 'http://10.170.30.79:80'),
                path_style=True, timeout=30)
            clients.append(local.client)
        return local.client

    def checked(response):
        if response.status >= 300:
            raise RuntimeError(f'OBS {response.status} {response.errorCode}')
        return response.body

    tasks = []
    for dataset in args.datasets:
        inv = json.loads((args.snapshots / (dataset + '.json')).read_text())
        if inv['truncated']:
            raise ValueError('Parent listing is truncated')
        tasks += [(dataset, item['prefix']) for item in inv['items'] if 'prefix' in item]
    folder = args.snapshots / 'rgb_receipts'
    folder.mkdir(exist_ok=True)

    def probe(task):
        dataset, prefix = task
        ident = prefix.rstrip('/').rsplit('/', 1)[1]
        path = folder / f'{dataset}_{ident}.json'
        if path.exists() and not args.refresh:
            old = json.loads(path.read_text())
            if old.get('root') == prefix and old.get('status') == 'checked':
                return old
        result = dict(dataset=dataset, scene=ident, root=prefix, media_downloaded=False)
        for attempt in range(3):
            try:
                body = checked(client().getObject(BUCKET, prefix + 'scene_meta.json'))
                try:
                    data = body.response.read(8 * 1024 * 1024 + 1)
                finally:
                    body.response.close()
                if len(data) > 8 * 1024 * 1024:
                    raise ValueError('Metadata exceeds bounded read')
                meta = json.loads(data)
                objects, marker = {}, ''
                while True:
                    b = checked(client().listObjects(BUCKET, prefix=prefix+'images/',
                        delimiter='/', marker=marker, max_keys=1000))
                    objects.update({x.key[len(prefix):]: x.size for x in b.contents or []
                                    if x.size > 0})
                    if not b.is_truncated:
                        break
                    if not b.next_marker or b.next_marker == marker:
                        raise ValueError('Pagination stalled')
                    marker = b.next_marker
                result.update(summarize(meta, objects), status='checked',
                              metadata_sha256=hashlib.sha256(data).hexdigest())
                break
            except Exception as exc:
                result.update(status='read_error', error=f'{type(exc).__name__}: {exc}')
                if attempt < 2:
                    time.sleep(attempt + 1)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False), encoding='utf8')
        temp.replace(path)
        return result

    start, results = time.monotonic(), []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(probe, tasks):
                results.append(result)
                if len(results) % 200 == 0 or len(results) == len(tasks):
                    print(json.dumps(dict(completed=len(results), total=len(tasks),
                        errors=sum(x['status'] != 'checked' for x in results),
                        records_per_second=len(results)/max(time.monotonic()-start, 1e-6))), flush=True)
    finally:
        for c in clients:
            c.close()
    report = {}
    for dataset in args.datasets:
        rows = [x for x in results if x['dataset'] == dataset]
        report[dataset] = dict(scenes=len(rows), checked=sum(x['status']=='checked' for x in rows),
            usable_scenes=sum(x.get('usable_frames',0)>=9 for x in rows),
            usable_rgb=sum(x.get('usable_frames',0) for x in rows),
            missing_rgb=sum(x.get('missing_rgb',0) for x in rows),
            scenes_with_missing=sum(x.get('missing_rgb',0)>0 for x in rows),
            flagged_bad_frames=sum(x.get('flagged_bad_frames',0) for x in rows),
            interpolated_frames=sum(x.get('interpolated_frames',0) for x in rows),
            tiny_rgb_objects=sum(x.get('tiny_rgb_objects',0) for x in rows),
            rgb_bytes=sum(x.get('rgb_bytes',0) for x in rows),
            image_sizes=dict(Counter(str(x.get('image_size')) for x in rows)))
    (args.snapshots/args.report).write_text(json.dumps(report,indent=2),encoding='utf8')
    spring=[r for r in results if r['dataset']=='spring']
    if spring and all('pose_frames' in r for r in spring):
        evidence=[dict(scene=r['scene'],frames=r['referenced_frames'],
                       poses=r['pose_frames'],depth=r['depth_frames']) for r in spring]
        (args.snapshots/'spring_split_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--datasets',nargs='+',default=['DL3DV','mvs_synth','scannetppv2'],
                   choices=['DL3DV','mvs_synth','scannetppv2','pointodyssey','spring','dynamicreplica'])
    p.add_argument('--report',default='rgb_summary.json')
    p.add_argument('--refresh',action='store_true',help='Re-read metadata/objects instead of reusing completed receipts')
    audit(p.parse_args())
