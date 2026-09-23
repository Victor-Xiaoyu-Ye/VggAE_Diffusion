"""Combine audited RGB versions by parent ID, preserving prior held-out splits.

The output is a small metadata manifest, not media. Prefer the complete old
DL3DV RGB source for overlapping scenes; WAI fills additional scenes and adds
ScanNet++. Never pair WAI intrinsics with pixels from another processing version.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def prepare(previous, wai, output):
    prior = []
    for split in ('train', 'val', 'test'):
        prior += [json.loads(line) for line in (previous/(split+'_candidates.jsonl')).read_text(encoding='utf8').splitlines()]
    records = {}
    for row in prior:
        dataset = row['dataset']
        if dataset not in ('dl3dv', 'spatialvid', 'omniworld'):
            continue
        kind = {'dl3dv':'frames', 'spatialvid':'video', 'omniworld':'omni'}[dataset]
        item = dict(id=row['id'], dataset=dataset, split=row['split'], kind=kind, root=row['rgb'])
        if dataset == 'spatialvid':
            item.update(fps=row['fps'], frames=row['frames'], text_id=row['parent_id'])
        if dataset == 'omniworld':
            item.update(fps=row['fps'], split_info=row['split_info'])
        records[item['id']] = item
    # MVS keeps the existing parent partition even though its processing version changes.
    old_splits = {r['id']: r['split'] for r in prior}
    spring_evidence=json.loads((wai/'spring_split_evidence.json').read_text())
    spring_test={r['scene'] for r in spring_evidence if not r['poses'] and not r['depth']}
    spring_val={'0013','0023','0037'}  # Existing MapAnything held-out scene IDs.
    rejected = Counter()
    for path in sorted((wai/'rgb_receipts').glob('*.json')):
        r = json.loads(path.read_text(encoding='utf8'))
        ds = {'DL3DV':'dl3dv', 'mvs_synth':'mvssynth', 'scannetppv2':'scannetppv2',
              'pointodyssey':'pointodyssey','spring':'spring','dynamicreplica':'dynamicreplica'}[r['dataset']]
        if ds=='dynamicreplica':
            rejected['dynamicreplica/original_split_not_mapped']+=1
            continue
        parent = r['scene'].split('_', 1)[-1] if ds == 'dl3dv' else r['scene']
        key = ds+':'+parent
        if r.get('usable_frames', 0) < 9:
            rejected[ds+'/'+r['status']] += 1
            continue
        if key in records:
            continue  # same physical parent, not an additional training example
        if r.get('image_size') and min(r['image_size']) < 224:
            rejected[ds+'/short_side_under_224'] += 1
            continue  # Dataset selection, not a downstream quality/phase gate.
        v = int(hashlib.sha256(('scene-v1:'+key).encode()).hexdigest()[:8], 16)%100
        split = old_splits.get(key, 'val' if v < 2 else 'test' if v < 4 else 'train')
        if ds=='pointodyssey':
            split=parent.split('-',1)[0]
            if split not in ('train','val','test'):
                raise ValueError('Unknown PointOdyssey original split')
        if ds=='spring':
            split='test' if parent in spring_test else 'val' if parent in spring_val else 'train'
        records[key] = dict(id=key, dataset=ds, kind='wai', split=split,
                           root='obs://yw-ads-training-gy1/'+r['root'],
                           filter_listed_rgb=bool(r.get('missing_rgb')))
        if ds in ('pointodyssey','spring'):
            records[key]['video_sequence']=True
    rows = sorted(records.values(), key=lambda r:r['id'])
    result = dict(schema='scene-rgb-cohort-v1', seed='scene-rgb-20260923-v1',
        media_decode_verified=False, physical_scene_disjoint_verified=False,
        split_scope='Internal parent-level generation split; not an official benchmark protocol',
        scannet_official_split_verified=False,
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [*(previous/(s+'_candidates.jsonl') for s in ('train','val','test')),
                                wai/'rgb_summary.json',wai/'dynamic_rgb_summary.json',wai/'spring_split_evidence.json']},
        historical_split_preserved=True, records=rows,
        counts=dict(Counter(r['dataset']+'/'+r['split'] for r in rows)),
        skipped_wai=dict(rejected))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
    print(json.dumps(dict(counts=result['counts'], skipped_wai=dict(rejected), bytes=output.stat().st_size)))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('previous','wai','output'):
        p.add_argument('--'+k, required=True, type=Path)
    a = p.parse_args()
    prepare(a.previous, a.wai, a.output)
