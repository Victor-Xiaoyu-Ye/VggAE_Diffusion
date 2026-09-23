"""Read-only OBS layout audit; only LIST and bounded JSON GET, never media GET.

Requires esdk-obs-python and VGG_OBS_AK/VGG_OBS_SK in the environment. Outputs
per-scene receipts for prepare_rae_rebuild_data.py. This is CPU metadata work:
records/s is reported; no NPU token throughput is fabricated.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import threading
import time

ROOT = 'data/external/x00445638/data/train_spatial/open_datasets/'
BUCKET = 'yw-ads-training-gy1'
OUT = None
AK = SK = None
LOCAL = threading.local()
CLIENTS = []

def client():
    if not hasattr(LOCAL, 'client'):
        from obs import ObsClient
        LOCAL.client = ObsClient(access_key_id=AK, secret_access_key=SK,
            server=os.environ.get('VGG_OBS_ENDPOINT', 'http://10.170.30.79:80'), path_style=True, timeout=30)
        CLIENTS.append(LOCAL.client)
    return LOCAL.client

def checked(r):
    if r.status >= 300: raise RuntimeError(f'OBS:{r.status}:{r.errorCode}')
    return r.body

def listing(rel):
    marker = ''; result = {}
    while True:
        body = checked(client().listObjects(BUCKET, prefix=ROOT+rel,
            delimiter='/', marker=marker, max_keys=1000))
        for x in body.contents or []:
            if x.size > 0: result[x.key.rsplit('/',1)[-1]] = x.size
        if not body.is_truncated: return result
        if not body.next_marker or marker == body.next_marker: raise ValueError('pagination stalled')
        marker = body.next_marker

def read_json(rel):
    if not rel.endswith('.json'): raise ValueError('metadata only')
    b = checked(client().getObject(BUCKET, ROOT+rel))
    try: data = b.response.read(2*1024*1024+1)
    finally: b.response.close()
    if len(data)>2*1024*1024: raise ValueError('metadata too large')
    return json.loads(data), hashlib.sha256(data).hexdigest()

def ids_from_listing(name):
    d=json.loads((OUT/(name.replace('/','__')+'.json')).read_text())
    if d['truncated']:
        raise ValueError('Cannot audit from a truncated parent inventory')
    return [x['prefix'].rstrip('/').split('/')[-1] for x in d['items'] if 'prefix' in x]

def probe(task):
    dataset, ident, rel = task
    path=OUT/'probes'/f'{dataset}_{ident}.json'
    if path.exists():
        cached = json.loads(path.read_text(encoding='utf8'))
        if cached['root'] != rel or cached['id'] != ident:
            raise ValueError('Receipt identity mismatch')
        if cached['status'] == 'metadata_pass': return cached
    result=dict(dataset=dataset, id=ident, root=rel, media_downloaded=False,
        decode_verified=False, pose_convention_verified=False)
    try:
        if dataset=='dl3dv':
            rgb=listing(rel+'/dense/rgb/'); cam=listing(rel+'/dense/cam/')
            images={Path(x).stem for x in rgb if x.endswith('.png')}
            cameras={Path(x).stem for x in cam if x.endswith('.npz')}
            result.update(rgb_frames=len(images),camera_frames=len(cameras),
                matched_frames=len(images&cameras), missing_cameras=len(images-cameras),
                orphan_cameras=len(cameras-images), example_rgb=next(iter(rgb),None))
            if len(images)<9 or images!=cameras:
                raise ValueError('RGB/camera frame set mismatch')
        elif dataset=='mvssynth':
            rgb=listing(rel+'/images/'); cam=listing(rel+'/poses/'); depth=listing(rel+'/depths/')
            images={Path(x).stem for x in rgb if x.endswith(('.png','.jpg'))}
            cameras={Path(x).stem for x in cam if x.endswith('.json')}
            depths={Path(x).stem for x in depth if x.endswith(('.exr','.png','.npy'))}
            result.update(rgb_frames=len(images),camera_frames=len(cameras),depth_frames=len(depths),
                matched_frames=len(images&cameras&depths), example_rgb=next(iter(rgb),None))
            if len(images)<9 or images!=cameras or images!=depths:
                raise ValueError('RGB/camera/depth frame set mismatch')
        else:
            ann=f'OmniWorld/annotations/OmniWorld-Game/{ident}/'
            info,sha=read_json(ann+'split_info.json')
            rgb=listing(f'OmniWorld/videos/OmniWorld-Game/{ident}/color/')
            cameras=listing(ann+'camera/'); captions=listing(ann+'text/')
            frames={int(Path(x).stem) for x in rgb if x.endswith('.png')}
            splits=info['split']; wanted={int(i) for split in splits for i in split}
            missing=[i for i in range(len(splits)) if f'split_{i}.json' not in cameras]
            result.update(rgb_frames=len(frames), split_info_sha256=sha,
                split_frames=[len(x) for x in splits], split_count=len(splits),
                referenced_frames=len(wanted), missing_rgb=len(wanted-frames),
                missing_camera_splits=missing, caption_files=sum(x.endswith('.json') for x in captions),
                split_info=info)
            if len(splits)!=info['split_num'] or not splits:
                raise ValueError('split count mismatch')
            if wanted-frames or missing:
                raise ValueError('missing RGB or camera split')
        result['status']='metadata_pass'
    except Exception as e:
        result.update(status='failed',error=f'{type(e).__name__}: {e}')
    path.parent.mkdir(exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(result,ensure_ascii=False),encoding='utf8'); tmp.replace(path)
    return result

def main():
    global OUT, AK, SK
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots',type=Path,required=True)
    parser.add_argument('--dl3dv-per-batch',type=int,default=4,help='0 means all; default samples 4 per batch')
    args=parser.parse_args()
    if args.dl3dv_per_batch < 0: parser.error('sample count must be nonnegative')
    OUT=args.snapshots
    AK,SK=os.environ.pop('VGG_OBS_AK'),os.environ.pop('VGG_OBS_SK')
    tasks=[]
    for k in range(1,8):
        rel=f'DL3DV_v1/DL3DV-ALL-480P-NEW/processed_dl3dv_ours/{k}K'
        ids=ids_from_listing(rel)
        ids=sorted(ids,key=lambda x:hashlib.sha256(('probe-v1:'+x).encode()).hexdigest())
        if args.dl3dv_per_batch: ids=ids[:args.dl3dv_per_batch]
        tasks.extend(('dl3dv',i,rel+'/'+i) for i in ids)
    tasks.extend(('mvssynth',i,'MVS-Synth/GTAV_1080/'+i)
        for i in ids_from_listing('MVS-Synth/GTAV_1080') if len(i)==4 and i.isdigit())
    tasks.extend(('omniworld',i,'OmniWorld/videos/OmniWorld-Game/'+i)
        for i in ids_from_listing('OmniWorld/videos/OmniWorld-Game'))
    start=time.monotonic(); results=[]
    try:
        with cf.ThreadPoolExecutor(max_workers=12) as pool:
            for result in pool.map(probe,tasks):
                results.append(result)
                if len(results)%100==0 or len(results)==len(tasks):
                    print(json.dumps(dict(completed=len(results),total=len(tasks),failed=sum(x['status']=='failed' for x in results),records_per_second=len(results)/max(time.monotonic()-start,1e-6))),flush=True)
    finally:
        for c in CLIENTS:c.close()
    report={k:{'checked':sum(x['dataset']==k for x in results),'passed':sum(x['dataset']==k and x['status']=='metadata_pass' for x in results)} for k in ('dl3dv','mvssynth','omniworld')}
    (OUT/'probe_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
    if any(x['status']=='failed' for x in results):
        raise SystemExit('Metadata audit has failures; review receipts before preparing a cohort')

if __name__=='__main__': main()
