"""Freeze metadata-only VGGT-RAE cohorts; no model loading or media downloads.

Consumes OBS listing snapshots and per-scene receipts from
audit_rae_rebuild_metadata.py. The output deliberately stays NOT training-ready:
object names cannot verify image decoding, camera conventions or motion quality.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ranked(values, seed):
    return sorted(values, key=lambda x: (hashlib.sha256((seed + ':' + x).encode()).hexdigest(), x))


def group_partition(ids, seed, val_count, test_count, reserved_test=()):
    """Partition whole parent groups, preserving externally reserved test groups."""
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate parent IDs; do not split dataset copies independently')
    reserved = set(reserved_test) & set(ids)
    remaining = ranked(set(ids) - reserved, seed)
    if len(remaining) < val_count + test_count + 1:
        raise ValueError('Not enough independent parent groups for requested split')
    val = set(remaining[:val_count])
    test = reserved | set(remaining[val_count:val_count + test_count])
    return {i: 'val' if i in val else 'test' if i in test else 'train' for i in ids}


def direct_ids(snapshot, root_suffix, pattern):
    content = read_json(snapshot)
    if content['truncated']:
        raise ValueError(f'Truncated inventory: {snapshot}')
    result = []
    for item in content['items']:
        if 'prefix' not in item:
            continue
        prefix = item['prefix']
        name = prefix.rstrip('/').rsplit('/', 1)[-1]
        if not prefix.endswith('/' + root_suffix.rstrip('/') + '/' + name + '/'):
            raise ValueError(f'Unexpected prefix in {snapshot}')
        if re.fullmatch(pattern, name):
            result.append(name)
    if len(result) != len(set(result)):
        raise ValueError('Duplicate scene directory')
    return sorted(result)


def safe_relative(value):
    value = value.replace('\\', '/')
    if not value or value.startswith('/') or ':' in value or any(x in ('', '.', '..') for x in value.split('/')):
        raise ValueError('Unsafe relative path')
    return value


def spatial_stratum(row, policy):
    """Metadata screen only: never label a camera-moving scene as true dynamics."""
    norm = lambda x: str(x).strip().lower()
    if norm(row['sceneType']) not in policy['scene_types']:
        return None
    if norm(row['timeOfDay']) not in policy['time_of_day'] or norm(row['brightness']) not in policy['brightness']:
        return None
    try:
        names = ['fps', 'num frames', 'aesthetic score', 'ocr score', 'motion score', 'dynamicRatio', 'trajTurns']
        v = {k: float(row[k]) for k in names}
        if not all(math.isfinite(x) for x in v.values()):
            return None
        size = [int(x) for x in row['resolution'].lower().split('x')]
        if len(size) != 2 or min(size) < policy['short_side_min']:
            return None
        if v['fps'] < policy['fps_min'] or v['num frames'] / v['fps'] < policy['duration_min_seconds']:
            return None
        if v['aesthetic score'] < policy['aesthetic_min'] or not 0 <= v['ocr score'] <= policy['ocr_max']:
            return None
        if not policy['motion_score_range'][0] <= v['motion score'] <= policy['motion_score_range'][1]:
            return None
        if not policy['dynamic_ratio_range'][0] <= v['dynamicRatio'] <= policy['dynamic_ratio_range'][1]:
            return None
        if not 0 <= v['trajTurns'] <= policy['max_turns']:
            return None
        return 'higher_dynamic_proxy' if v['dynamicRatio'] >= policy['dynamic_high_boundary'] else 'moderate_dynamic_proxy'
    except (ValueError, ZeroDivisionError, KeyError):
        return None


def prepare(snapshots, metadata, inventory, historical, policy_path, output):
    start = time.monotonic()
    snapshots, output = Path(snapshots), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Select a fresh output directory; frozen selections cannot be overwritten')
    policy, history = read_json(policy_path), read_json(historical)
    inputs = {}

    def remember(path):
        path = Path(path)
        # Stable names, not machine-dependent absolute paths, enter the signature.
        name = path.relative_to(snapshots).as_posix() if path.is_relative_to(snapshots) else path.name
        if name in inputs and inputs[name] != digest(path):
            raise ValueError('Conflicting input names')
        inputs[name] = digest(path)
        return path

    for path in (metadata, inventory, historical, policy_path):
        remember(path)
    if inputs[Path(metadata).name] != history['metadata_sha256']:
        raise ValueError('HQ metadata no longer matches the historical reserved IDs')
    if inputs[Path(inventory).name] != history['availability']['inventory_sha256']:
        raise ValueError('HQ object inventory changed; review/re-freeze the policy')

    def listing(rel, pattern):
        path = remember(snapshots / (rel.replace('/', '__') + '.json'))
        return direct_ids(path, rel, pattern)

    excluded, records = [], []

    def probe(dataset, ident, expected_root):
        path = snapshots / 'probes' / f'{dataset}_{ident}.json'
        if not path.exists():
            if policy['require_complete_layout_audit']:
                raise ValueError(f'Missing metadata receipt: {dataset}:{ident}')
            return {'status': 'pending'}
        receipt = read_json(remember(path))
        if (receipt['dataset'], receipt['id'], receipt['root']) != (dataset, ident, expected_root):
            raise ValueError('Metadata receipt identity mismatch')
        return receipt

    def base(dataset, ident, split, role, rgb, camera, status='metadata_pass'):
        return dict(id=f'{dataset}:{ident}', dataset=dataset, parent_id=ident,
                    group_id=f'{dataset}:{ident}', split=split, role=role,
                    rgb=rgb, camera=camera, metadata_status=status, training_ready=False,
                    runtime_checks=['decode_rgb', 'camera_convention_and_scale', 'frame_alignment', 'quality'],
                    physical_scene_disjoint_verified=False)

    # DL3DV: complementary RGB/camera and depth trees, matched by hash + batch.
    rgb_root = 'DL3DV_v1/DL3DV-ALL-480P-NEW/processed_dl3dv_ours'
    depth_root = 'DL3DV_v1/processed_dl3dv_ours'
    valid = {}
    with remember(snapshots / 'DL3DV-valid.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['hash'] in valid:
                raise ValueError('Duplicate official DL3DV ID')
            valid[row['hash']] = row['batch']
    stored_test = {x.rsplit('/', 1)[-1] for x in read_json(remember(snapshots / 'dl3dv_stored_test_index.json'))}
    dl = {}
    for k in range(1, 8):
        rel = f'{rgb_root}/{k}K'
        depth_ids = set(listing(f'{depth_root}/{k}K', '[0-9a-f]{64}'))
        for ident in listing(rel, '[0-9a-f]{64}'):
            if ident in dl:
                raise ValueError('DL3DV parent appears in multiple batches')
            scene = rel + '/' + ident
            receipt = probe('dl3dv', ident, scene)
            if valid.get(ident) != f'{k}K' or receipt['status'] != 'metadata_pass':
                excluded.append(dict(dataset='dl3dv', id=ident, reason='official_valid_list_or_layout', status=receipt['status']))
                continue
            dl[ident] = (scene, f'{depth_root}/{k}K/{ident}' if ident in depth_ids else None, receipt)
    partition = group_partition(dl, policy['seed'] + ':dl3dv', **{
        'val_count': policy['dl3dv']['val_scenes'], 'test_count': policy['dl3dv']['test_scenes'], 'reserved_test': stored_test})
    root = policy['open_datasets_root']
    for ident, (scene, depth, receipt) in dl.items():
        row = base('dl3dv', ident, partition[ident], 'static_multiview_candidate', root + scene + '/dense/rgb/', root + scene + '/dense/cam/')
        row.update(depth_sidecar=root + depth + '/dense/' if depth else None,
                   rgb_frames=receipt['rgb_frames'], camera_frames=receipt['camera_frames'],
                   text_status='missing_not_t2i_ready', static_scene_verified=False,
                   pose_convention='unknown_processed_npz', metric_scale_verified=False)
        records.append(row)

    # MVS-Synth: do not include Scene01 etc (Virtual KITTI layout) under this name.
    rel = 'MVS-Synth/GTAV_1080'
    mvs = {}
    for ident in listing(rel, '[0-9]{4}'):
        receipt = probe('mvssynth', ident, rel + '/' + ident)
        if receipt['status'] != 'metadata_pass':
            excluded.append(dict(dataset='mvssynth', id=ident, reason='layout', status=receipt['status']))
        else:
            mvs[ident] = receipt
    partition = group_partition(mvs, policy['seed'] + ':mvssynth', policy['mvssynth']['val_scenes'], policy['mvssynth']['test_scenes'])
    for ident, receipt in mvs.items():
        scene = root + rel + '/' + ident
        row = base('mvssynth', ident, partition[ident], 'static_multiview_candidate', scene + '/images/', scene + '/poses/')
        row.update(depth_sidecar=scene + '/depths/', rgb_frames=receipt['rgb_frames'],
                   text_status='missing_not_t2i_ready', pose_convention='unverified_extrinsic_json', metric_scale_verified=False)
        records.append(row)

    # OmniWorld: all subclips in one UID stay in one partition, including the
    # non-benchmark subclips of any UID that contains an official benchmark clip.
    rel = 'OmniWorld/videos/OmniWorld-Game'
    omni_ids = set(listing(rel, '[0-9a-f]{12}'))
    ann_ids = set(listing('OmniWorld/annotations/OmniWorld-Game', '[0-9a-f]{12}'))
    with remember(snapshots / 'omniworld_game_metadata.csv').open(encoding='utf-8-sig', newline='') as f:
        omni_meta = {}
        for row in csv.DictReader(f):
            if row['UID'] in omni_meta:
                raise ValueError('Duplicate OmniWorld UID')
            omni_meta[row['UID']] = row
    omni, official_test = {}, set()
    for ident in sorted(omni_ids):
        receipt = probe('omniworld', ident, rel + '/' + ident)
        if ident not in ann_ids or ident not in omni_meta or receipt['status'] != 'metadata_pass':
            excluded.append(dict(dataset='omniworld', id=ident, reason='metadata_or_layout'))
            continue
        meta = omni_meta[ident]
        if int(meta['Split Num']) != receipt['split_count']:
            raise ValueError('Official and OBS OmniWorld split counts disagree')
        test_indices = [int(x) for x in meta['Test Split Index'].split(',') if x.strip()]
        if any(not 0 <= x < receipt['split_count'] for x in test_indices):
            raise ValueError('Invalid official benchmark index')
        if test_indices:
            official_test.add(ident)
        omni[ident] = (receipt, meta, test_indices)
    if not policy['omniworld']['reserve_entire_uid_with_official_test']:
        raise ValueError('Frame-level benchmark exclusions are not supported')
    partition = group_partition(omni, policy['seed'] + ':omniworld', policy['omniworld']['val_scenes'], 0, official_test)
    for ident, (receipt, meta, test_indices) in omni.items():
        ann = root + 'OmniWorld/annotations/OmniWorld-Game/' + ident
        row = base('omniworld', ident, partition[ident], 'dynamic_game_candidate', root + rel + '/' + ident + '/color/', ann + '/camera/')
        row.update(split_info=ann + '/split_info.json', split_info_sha256=receipt['split_info_sha256'],
                   split_frames=receipt['split_frames'], rgb_frames=receipt['rgb_frames'],
                   text_root=ann + '/text/', text_status='coverage_requires_window_join' if receipt['caption_files'] else 'missing_not_t2i_ready',
                   depth_sidecar=ann + '/depth/', dynamic_mask_root=ann + '/gdino_mask/',
                   official_test_split_indices=test_indices, fps=float(meta['FPS']),
                   metric_scale_metadata=float(meta['Metric Scale']), metric_scale_verified=False,
                   pose_convention='unverified_camera_json', game_identity='unknown', object_motion_verified=False)
        records.append(row)

    # SpatialVID: clip-level metadata is not a reliable YouTube source grouping.
    inv = read_json(inventory)
    if inv['truncated']:
        raise ValueError('HQ inventory is truncated')
    available = {x['key']: x['size'] for x in inv['items'] if 'key' in x and x['size'] > 0}
    spatial_root = policy['spatialvid_root']
    key_root = spatial_root.split('/', 3)[3]
    reserve = {x for k, v in history['splits'].items() if k.startswith(('eval_', 'test_')) for x in v}
    old_train = set(history['splits']['train_mixed_2048']) | set(history['splits']['train_single_2048'])
    sp = policy['spatialvid']
    rows, eligible, strata, bad = {}, {}, {}, Counter()
    with Path(metadata).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            ident = row['id']
            if ident in rows:
                raise ValueError('Duplicate SpatialVID clip ID')
            rows[ident] = row
            try:
                path = safe_relative(row['video path'])
                ann = safe_relative(row['annotation path'])
                if not path.startswith('videos/') or not ann.startswith('annotations/'):
                    raise ValueError('Unexpected SpatialVID layout')
                if not path.endswith('/' + ident + '.mp4'):
                    raise ValueError('Clip ID/path disagreement')
            except ValueError:
                bad['invalid_path'] += 1
                continue
            if key_root + 'videos/SpatialVID/' + path not in available:
                bad['unavailable'] += 1
                continue
            if ident in reserve:
                continue
            stratum = spatial_stratum(row, sp)
            if stratum:
                eligible[ident], strata[ident] = row, stratum
            else:
                bad['outside_policy'] += 1
    if not reserve <= rows.keys():
        raise ValueError('Historical heldout rows missing')
    # New validation/test clips were not in the prior small-domain training set.
    heldout_pool = set(eligible) - old_train
    held = ranked(heldout_pool, policy['seed'] + ':spatial-heldout')
    nval, ntest = sp['val_clips'], sp['test_clips']
    if len(held) < nval + ntest:
        raise ValueError('Insufficient SpatialVID heldout candidates')
    val, test = set(held[:nval]), set(held[nval:nval + ntest])
    train_pool = set(eligible) - val - test
    targets = {'higher_dynamic_proxy': round(sp['train_max'] * sp['dynamic_high_quota'])}
    targets['moderate_dynamic_proxy'] = sp['train_max'] - targets['higher_dynamic_proxy']
    train = set()
    actual_quota = {}
    for stratum, maximum in targets.items():
        chosen = ranked([i for i in train_pool if strata[i] == stratum], policy['seed'] + ':' + stratum)[:maximum]
        train.update(chosen)
        actual_quota[stratum] = {'requested': maximum, 'selected': len(chosen)}
    # Only existing TRAIN IDs replay; never train on the familiar evaluation set.
    replay_pool = set(history['splits']['train_mixed_2048']) - train - val - test - reserve
    replay = []
    for ident in ranked(replay_pool, policy['seed'] + ':replay'):
        row = rows[ident]
        key = key_root + 'videos/SpatialVID/' + safe_relative(row['video path'])
        if key in available:
            replay.append(ident)
        if len(replay) == sp['replay_train_max']:
            break
    for ident in sorted(train | val | test | set(replay)):
        item = rows[ident]
        split = 'val' if ident in val else 'test' if ident in test else 'train'
        row = base('spatialvid', ident, split, 'appearance_replay' if ident in replay else 'real_street_video_candidate',
                   spatial_root + 'videos/SpatialVID/' + safe_relative(item['video path']),
                   spatial_root + 'annotations/SpatialVID/' + safe_relative(item['annotation path']), 'video_listed_annotation_pending')
        row.update(scene_type=item['sceneType'], dynamic_ratio_proxy=float(item['dynamicRatio']),
                   motion_stratum=strata.get(ident, 'historical_replay'), frames=int(item['num frames']), fps=float(item['fps']),
                   text_status='caption_join_pending', object_motion_verified=False,
                   source_grouping='clip_id_only_original_source_unknown', metric_scale_verified=False)
        records.append(row)

    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate selected IDs across roles')
    split_sets = {s: {r['group_id'] for r in records if r['split'] == s} for s in ('train', 'val', 'test')}
    if any(split_sets[a] & split_sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Parent group leakage')
    output.mkdir(parents=True, exist_ok=True)
    files = {}

    def write(name, values):
        target = output / name
        temp = target.with_suffix(target.suffix + '.tmp')
        with temp.open('w', encoding='utf8', newline='\n') as f:
            for value in values:
                f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
        temp.replace(target)
        files[name] = digest(target)

    records.sort(key=lambda r: r['id'])
    for split in ('train', 'val', 'test'):
        write(split + '_candidates.jsonl', [r for r in records if r['split'] == split])
    write('excluded.jsonl', sorted(excluded, key=lambda r: (r['dataset'], r['id'])))
    write('historical_regression_ids.jsonl', [dict(dataset='spatialvid', id=i, split=k, training_allowed=False)
        for k, ids in sorted(history['splits'].items()) if k.startswith(('eval_', 'test_')) for i in ids])
    report = dict(schema='rae-rebuild-selection-v1', status='metadata_only_not_training_ready',
        policy=policy, inputs_sha256=inputs, files_sha256=files,
        counts=dict(sorted(Counter(r['dataset'] + '/' + r['split'] for r in records).items())),
        roles=dict(Counter(r['role'] for r in records)), exclusions=len(excluded),
        spatial_screen_counts=dict(bad), spatial_eligible=len(eligible), spatial_train_quota=actual_quota,
        historical_reserved=len(reserve), old_train_replay=len(replay),
        official_omni_test_parent_count=len(official_test),
        parent_id_disjoint=True, physical_scene_disjoint_verified=False,
        media_downloaded=False, rgb_decode_verified=False, camera_conventions_verified=False,
        text_ready=False, cohort_cannot_be_passed_to_legacy_r7_trainer=True)
    # Final receipt is written last; a partial directory has no completed receipt.
    final = output / 'selection.json'
    temporary = final.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf8')
    temporary.replace(final)
    print(json.dumps(dict(counts=report['counts'], exclusions=len(excluded),
        spatial_train_quota=actual_quota, status=report['status'], records_per_second=len(records)/max(time.monotonic()-start,1e-6)), ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshots', 'metadata', 'inventory', 'historical', 'policy', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    prepare(a.snapshots, a.metadata, a.inventory, a.historical, a.policy, a.output)
