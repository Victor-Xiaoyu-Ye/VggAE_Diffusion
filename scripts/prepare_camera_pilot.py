"""Join existing R7 windows to exact-frame camera annotations; no RGB decoding."""
import argparse
from collections import Counter, OrderedDict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from data.latent_shard_dataset import LatentShardDataset, read_shard_manifest
from scripts.window_run_io import atomic, child
from utils.camera_control import (build_window_camera_controls, pack_window_camera_features,
                                  parse_indexes_text)
from utils.camera_training import camera_key
from utils.moxing_io import read_bytes


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def prepare(metadata, train_manifest, eval_manifest, annotation_root, output_dir,
            train_shards=96, seed=42, maxgap_seconds=.25, expected_eval_windows=128,
            progress_dir=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'inputs.json').exists():
        raise ValueError('camera inputs are committed; reuse them instead of preparing again')
    if train_shards < 1 or not math.isfinite(maxgap_seconds) or maxgap_seconds <= 0:
        raise ValueError('train_shards and maxgap_seconds must be positive')
    started = time.monotonic()
    audit = dict(schema='r7-camera-pilot-audit-v1', status='running', seed=seed,
                 maxgap_seconds=maxgap_seconds, failures=[], shard_counts=[],
                 DI_throughput=0., throughput_scope='CPU metadata/annotation processing; no NPU work',
                 source_identity={}, counts={}, motion_tags={})
    identity = audit['source_identity']
    controls_by_key, fps_by_key = {}, {}
    split_keys = {'train': set(), 'eval': set()}
    split_ids = {'train': set(), 'eval': set()}
    counts = {split: dict(attempted=0, valid=0, invalid=0) for split in split_keys}
    tags = {split: Counter() for split in split_keys}
    annotation_cache = OrderedDict()
    source_hashes = {}

    def progress(status='running'):
        elapsed = time.monotonic() - started
        attempted = sum(c['attempted'] for c in counts.values())
        audit.update(status=status, elapsed_seconds=elapsed,
                     windows_per_second_cpu=attempted / max(elapsed, 1e-9), counts=counts,
                     motion_tags={split: dict(v) for split, v in tags.items()})
        atomic(out / 'camera_audit.json', audit)
        if progress_dir:
            # The launcher's watched node logs persist this ledger even when
            # preparation fails before immutable inputs can be committed.
            atomic(Path(progress_dir) / 'camera_audit.json', audit)
        print('[camera prepare] ' + json.dumps(dict(status=status, counts=counts,
              elapsed_seconds=elapsed, windows_per_second_cpu=audit['windows_per_second_cpu'],
              DI_throughput=0., phase='cpu_annotations')), flush=True)
        print('DI_throughput: 0.000 tokens/s/npu (CPU annotation preparation; no active NPU)', flush=True)

    def annotations(vid, row):
        if vid in annotation_cache:
            value = annotation_cache.pop(vid)
            annotation_cache[vid] = value
        else:
            try:
                groups = re.findall(r'(?:^|/)(group_[0-9]+)(?=/|$)', row['video path'].replace('\\', '/'))
                if len(groups) != 1 or not re.fullmatch(r'[A-Za-z0-9_-]+', vid):
                    raise ValueError('ambiguous annotation group or invalid video ID')
                root = child(annotation_root, groups[0] + '/' + vid)
                blobs = {name: read_bytes(child(root, name))
                         for name in ('poses.npy', 'intrinsics.npy', 'indexes.txt')}
                source_hashes[vid] = dict(root=root, sha256={n: sha256(b) for n, b in blobs.items()})
                poses = np.load(io.BytesIO(blobs['poses.npy']), allow_pickle=False)
                intrinsics = np.load(io.BytesIO(blobs['intrinsics.npy']), allow_pickle=False)
                indexes = parse_indexes_text(blobs['indexes.txt'].decode('utf-8-sig'), len(poses))
                value = (poses, intrinsics, indexes)
            except Exception as exc:
                if isinstance(exc, MemoryError):
                    raise
                value = ValueError(f'{type(exc).__name__}: {exc}')
            annotation_cache[vid] = value
            if len(annotation_cache) > 512:
                annotation_cache.popitem(last=False)
        if isinstance(value, Exception):
            raise value
        return value

    try:
        metadata_bytes = read_bytes(metadata)
        identity.update(metadata=metadata, metadata_sha256=sha256(metadata_bytes),
                        train_manifest=train_manifest, train_manifest_sha256=sha256(read_bytes(train_manifest)),
                        eval_manifest=eval_manifest, eval_manifest_sha256=sha256(read_bytes(eval_manifest)),
                        annotation_root=annotation_root, source_pose='opencv_w2c_tx_ty_tz_qx_qy_qz_qw')
        rows = {}
        for row in csv.DictReader(io.StringIO(metadata_bytes.decode('utf-8-sig'))):
            vid = row['id'].strip()
            if not vid or vid in rows:
                raise ValueError('metadata contains empty or duplicate video IDs')
            rows[vid] = row
        all_train = read_shard_manifest(train_manifest)
        eval_shards = read_shard_manifest(eval_manifest)
        if len(set(all_train)) != len(all_train) or len(set(eval_shards)) != len(eval_shards):
            raise ValueError('duplicate shard references in input manifests')
        if len(all_train) < train_shards:
            raise ValueError('training manifest has fewer shards than requested')
        selected = sorted(random.Random(seed).sample(all_train, train_shards))
        if set(selected) & set(eval_shards):
            raise ValueError('training and evaluation shard references overlap')
        identity.update(selected_train_shards=selected, eval_shards=eval_shards,
                        selection='random.Random(seed).sample(all training shards, requested count), sorted')
        reader = LatentShardDataset.__new__(LatentShardDataset)
        progress()
        for split, shards in (('train', selected), ('eval', eval_shards)):
            for shard in shards:
                before = dict(counts[split])
                for sample in reader._iter_shard(shard):
                    counts[split]['attempted'] += 1
                    vid = str(sample.get('video_id', ''))
                    split_ids[split].add(vid)
                    try:
                        row = rows[vid]
                        fps = float(row['fps'])
                        if not math.isfinite(fps) or fps <= 0:
                            raise ValueError('invalid metadata FPS')
                        gap = math.floor(fps * maxgap_seconds)
                        if gap < 1:
                            raise ValueError('FPS too low for the configured interpolation limit')
                        key = camera_key(sample)
                        poses, intrinsics, indexes = annotations(vid, row)
                        control = build_window_camera_controls(poses, intrinsics, indexes,
                            sample['frame_indices'], max_interpolation_gap_frames=gap)
                        # Validate nine-frame shape/packing before counting success.
                        pack_window_camera_features(control, fps, 1.)
                        if key in controls_by_key:
                            if (fps_by_key[key] != fps or not np.array_equal(
                                    controls_by_key[key]['relative_c2w'], control['relative_c2w'])):
                                raise ValueError('same window identity has inconsistent camera annotations')
                        controls_by_key[key], fps_by_key[key] = control, fps
                        split_keys[split].add(key)
                        counts[split]['valid'] += 1
                        for tag in filter(None, re.split(r'[;,|]', row.get('motionTags', ''))):
                            tags[split][tag.strip()] += 1
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                        counts[split]['invalid'] += 1
                        frames = sample.get('frame_indices')
                        audit['failures'].append(dict(split=split, shard=shard, video_id=vid,
                            frame_indices=None if frames is None else np.asarray(frames).tolist(),
                            error=f'{type(exc).__name__}: {exc}'))
                    if sum(c['attempted'] for c in counts.values()) % 64 == 0:
                        progress()
                audit['shard_counts'].append(dict(split=split, shard=shard,
                    **{k: counts[split][k] - before[k] for k in before}))
                progress()
        audit['annotation_sources'] = source_hashes
        if split_ids['train'] & split_ids['eval']:
            raise ValueError('training/evaluation video IDs overlap, including rejected windows')
        if counts['train']['valid'] < .9 * counts['train']['attempted'] or not split_keys['train']:
            raise ValueError('fewer than 90% of selected training windows have valid camera annotations')
        if (counts['eval']['attempted'] != expected_eval_windows or counts['eval']['invalid'] or
                len(split_keys['eval']) != expected_eval_windows or len(split_ids['eval']) != expected_eval_windows):
            raise ValueError(f'all {expected_eval_windows} reviewed evaluation videos/windows must remain valid')
        if any(s['valid'] == 0 for s in audit['shard_counts'] if s['split'] == 'train'):
            raise ValueError('a selected training shard has no valid windows; cannot assign safe DDP partitions')
        nonzero = [float(np.linalg.norm(controls_by_key[k]['relative_c2w'][-1, :3, 3]))
                   for k in sorted(split_keys['train'])]
        nonzero = [x for x in nonzero if x > 0]
        scale = max(1e-3, statistics.median(nonzero) if nonzero else 1e-3)
        features = {k: torch.from_numpy(pack_window_camera_features(c, fps_by_key[k], scale))
                    for k, c in sorted(controls_by_key.items())}
        identity['annotation_sources_sha256'] = sha256(
            json.dumps(source_hashes, sort_keys=True, separators=(',', ':')).encode('utf-8'))
        bank = dict(schema='r7-camera-pilot-bank-v1', features=features,
                    train_keys=sorted(split_keys['train']), eval_keys=sorted(split_keys['eval']),
                    translation_scale=scale, source_identity=identity)
        temp = out / 'camera_bank.pt.tmp'
        torch.save(bank, temp)
        temp.replace(out / 'camera_bank.pt')
        for name, shards in (('train_manifest.txt', selected), ('eval_manifest.txt', eval_shards)):
            tmp = out / (name + '.tmp')
            tmp.write_text('\n'.join(shards) + '\n', encoding='utf-8')
            tmp.replace(out / name)
        audit.update(translation_scale=scale,
            translation_scale_contract='median nonzero endpoint displacement of unique valid TRAIN windows; floor 1e-3; held fixed for eval/inference; not metric calibration',
            unique_windows={s: len(k) for s, k in split_keys.items()},
            video_id_disjoint=True, original_tar_references_preserved=True,
            output_sha256={n: sha256((out / n).read_bytes())
                           for n in ('camera_bank.pt', 'train_manifest.txt', 'eval_manifest.txt')})
        progress('complete')
        return audit
    except Exception as exc:
        audit.update(error=f'{type(exc).__name__}: {exc}', annotation_sources=source_hashes)
        progress('failed')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'train_manifest', 'eval_manifest', 'annotation_root', 'output_dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--train_shards', type=int, default=96)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--maxgap_seconds', type=float, default=.25)
    p.add_argument('--progress_dir', default=None)
    prepare(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
