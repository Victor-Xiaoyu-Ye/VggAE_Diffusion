"""Paper evaluation inputs. Future edited RGB has a separate evaluation API.

Byte-group exclusion and conservative visual screening are reproducible, but
do not certify absence of common parent scenes when the dataset has no IDs.
"""
import copy
from pathlib import Path

import numpy as np
import torch

from utils.editing_data import digest, file_hash, spatial_plan, apply_spatial


def candidate_rows(bank, count=128, smoke=False, seed=42):
    manifest = bank.get('manifest', {})
    rows = manifest.get('records', bank['records'])
    exposed = {r['group'] for r in rows if r.get('split') in ('train', 'dev')}
    groups = {}
    wanted = 'dev' if smoke else 'test'
    for row in rows:
        if row.get('split') != wanted or (not smoke and row['group'] in exposed):
            continue
        groups.setdefault(row['group'], []).append(row)
    chosen = []
    for group in sorted(groups, key=lambda g: digest(['paper-cohort', seed, g])):
        row = min(groups[group], key=lambda r: digest(['paper-prompt', seed, r['id']]))
        # The source-only manifest deliberately has no target path.
        chosen.append({k: copy.deepcopy(row[k]) for k in
                       ('id', 'source', 'group', 'instruction', 'split')})
    # Screen the reserved pool BEFORE limiting, never refill by output quality.
    return chosen, dict(requested=count, candidates=len(chosen),
                        source_groups=len(groups), exposed_groups=len(exposed))


def load_source(root, row, cfg):
    """Decode only source video at physical timestamps, preserving its FOV."""
    import imageio.v2 as imageio
    from utils.editing_data import safe_path
    name = safe_path(row['source'])
    path = Path(root).joinpath(*name.parts)
    if file_hash(path) != row['group']:
        raise ValueError('Source bytes differ from the frozen dataset group')
    reader = imageio.get_reader(str(path), 'ffmpeg')
    try:
        meta = reader.get_meta_data()
        rate, count = float(meta['fps']), reader.count_frames()
        ids = [round(i*rate/cfg['fps']) for i in range(cfg['frames'])]
        if rate <= 0 or len(set(ids)) != len(ids) or max(ids) >= count:
            raise ValueError('Source too short/low-fps for declared physical window')
        frames = torch.from_numpy(np.stack([reader.get_data(i)[..., :3] for i in ids])).permute(0, 3, 1, 2)
        width, height = meta['size']
        if list(frames.shape[-2:]) != [height, width]:
            raise ValueError('Source rotation/geometry requires explicit alignment')
        spatial = spatial_plan(height, width, cfg['height']*cfg['width'], cfg['max_side'])
        return dict(source=apply_spatial(frames, spatial), spatial=spatial,
                    timing=dict(indices=ids, original_fps=rate, original_count=count,
                                fps=cfg['fps']), source_sha256=row['group'])
    finally:
        reader.close()


def source_fingerprint(source, rect):
    """Three source-time pHashes plus low-texture flags; never uses edits."""
    import cv2
    y, x, h, w = rect
    frames = torch.as_tensor(source).detach().cpu()
    if frames.dtype != torch.uint8:
        frames = (frames.float().clamp(0, 1)*255).round().byte()
    hashes, texture = [], []
    for t in (0, (len(frames)-1)//2, len(frames)-1):
        rgb = frames[t, :, y:y+h, x:x+w].permute(1, 2, 0).numpy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(gray)[:8, :8].ravel()[1:]
        bits = dct > np.median(dct)
        number = sum(int(v) << i for i, v in enumerate(bits))
        hashes.append(format(number, '016x'))
        texture.append(float(gray.std()))
    return dict(phash=hashes, texture_std=texture, method='three_time_dct63_v1')


def near_duplicate(a, b, threshold=4):
    if min(a['texture_std'] + b['texture_std']) < 2:
        return False  # Uniform images need manual review, not automatic removal.
    distances = [(int(x, 16)^int(y, 16)).bit_count() for x, y in zip(a['phash'], b['phash'])]
    return len(distances) == 3 and max(distances) <= threshold


def freeze_cohort(candidates, source_reports, seen_reports, count, seed=42, smoke=False):
    available = {r['id']: r for r in source_reports if r['status'] == 'completed'}
    seen = [r for r in seen_reports if r['status'] == 'completed']
    selected, excluded = [], []
    for row in candidates:
        evidence = available.get(row['id'])
        if evidence is None:
            excluded.append(dict(id=row['id'], reason='source_decode_or_cache_failure'))
            continue
        matches = [] if smoke else [s['id'] for s in seen if
                   row['group'] == s['group'] or near_duplicate(evidence['fingerprint'], s['fingerprint'])]
        if matches:
            excluded.append(dict(id=row['id'], reason='exposed_group_or_visual_near_duplicate', matches=matches))
            continue
        duplicate = next((r['id'] for r in selected if
                          near_duplicate(evidence['fingerprint'], available[r['id']]['fingerprint'])), None)
        if duplicate:
            excluded.append(dict(id=row['id'], reason='within_cohort_visual_near_duplicate', matches=[duplicate]))
            continue
        if len(selected) < count:
            selected.append(dict(row, source_file=evidence['file'], fingerprint=evidence['fingerprint']))
    ordered = sorted(selected, key=lambda r: digest(['paper-ablation-subset', seed, r['group']]))
    subset = [r['id'] for r in ordered[:min(32, len(ordered))]]
    value = dict(records=selected, ablation_ids=subset, requested=count, selected=len(selected),
        excluded=excluded, exposed_screen_requested=len(seen_reports), exposed_screen_completed=len(seen),
        exposed_screen_failures=[r for r in seen_reports if r['status'] != 'completed'],
        protocol='development_smoke' if smoke else 'reserved_byte_groups_with_source_visual_screen',
        caveat='No parent-scene identifiers; visual screening is not a scene-disjoint guarantee. '
               'Missing screen inputs are reported; outputs are never used to select cases.')
    value['identity'] = digest(value)
    return value


def load_evaluation_labels(root, row, bank, source_payload):
    """Only the evaluator calls this. It never returns labels to sampling."""
    from utils.editing_data import load_pair
    original = next(r for r in bank['manifest']['records'] if r['id'] == row['id'])
    source, target, timing = load_pair(root, original, bank['config'])
    if timing['spatial'] != source_payload['spatial'] or not torch.equal(source, source_payload['source']):
        raise ValueError('Evaluation target/source times or geometry differ from inference')
    return target


class TrainingCacheView:
    """Reuse exact Stage65 examples read-only, write only the new SFT run.

    This translates the three explicit cache namespaces, never arbitrary
    missing files, so a model checkpoint cannot silently come from an old run.
    """
    def __init__(self, output, source, old_identity, identity):
        self.output, self.source = output, source
        self.old_identity, self.identity = old_identity, identity
        self.local, self.roots = output.local, output.roots

    def load(self, name, identity=None, required=False):
        if name.startswith(('states/', 'native/', 'supervision/')):
            if identity != self.identity:
                raise ValueError('New training/cache contract identity differs')
            return self.source.load(name, self.old_identity, required)
        return self.output.load(name, identity, required)

    def __getattr__(self, name):
        return getattr(self.output, name)
