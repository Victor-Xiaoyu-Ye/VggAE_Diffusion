"""VACE gray-to-color inputs; RGB targets are an evaluation-only API.

This is the public V2V/gray task at our frozen 17-frame window, not a
text-replacement benchmark. Do not treat depth, layout or inpainting as RGB.
"""
import copy
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import torch

from utils.editing_data import digest, file_hash, safe_path, apply_spatial, rae_canvas
from utils.editing_paper_data import load_source, near_duplicate

ARMS = ('native', 'sft', 'rgb_full', 'rae_full')
INDEX_FILES = ('indices/real.txt', 'indices/synthetic.txt', 'indices/composition.txt')


def ensure_file(root, name, remote=None):
    """Stage only named files from the existing OBS reference; never HF."""
    relative = safe_path(name)
    path = Path(root).joinpath(*relative.parts)
    if path.is_file() and path.stat().st_size:
        return path
    if not remote:
        raise FileNotFoundError(str(path))
    from utils.moxing_io import copy_file
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    try:
        copy_file(remote.rstrip('/')+'/'+str(relative), str(temporary))
        if not temporary.stat().st_size:
            raise ValueError('Empty benchmark object: '+name)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def audit_index(root, remote=None):
    rows, hashes = [], {}
    for name in INDEX_FILES:
        path = ensure_file(root, name, remote)
        hashes[name] = file_hash(path)
        for line in path.read_text(encoding='utf8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
    counts = Counter((r.get('source'), r.get('task'), r.get('subtask')) for r in rows)
    candidates, labels = [], {}
    for row in rows:
        if (row.get('task'), row.get('subtask')) != ('V2V', 'gray'):
            continue
        if row.get('src_ref_images') or not all(row.get(k) for k in
                ('src_video', 'src_mask', 'raw_video', 'en_prompt')):
            raise ValueError('Incomplete official gray task: '+str(row.get('sample_id')))
        key = 'vace-'+str(row['source'])+'-'+str(row['sample_id'])
        if key in labels:
            raise ValueError('Repeated public case ID: '+key)
        for field in ('src_video', 'src_mask', 'raw_video'):
            safe_path(row[field])
        candidates.append(dict(id=key, sample_id=str(row['sample_id']),
            domain=row['source'], source=row['src_video'], mask=row['src_mask'],
            instruction=row['en_prompt'].strip(), prompt_kind='official_output_description',
            task='gray_to_color', split='public_test'))
        labels[key] = dict(target=row['raw_video'], semantics='original_color_video',
                          source=row['src_video'])
    report = dict(index_sha256=hashes, total_records=len(rows),
        asset_groups={'vace-'+str(r['source'])+'-'+str(r['sample_id']):digest(r['raw_video'])
                      for r in rows if r.get('raw_video')},
        tasks=[dict(domain=d, task=t, subtask=s, count=n)
               for (d,t,s),n in sorted(counts.items())],
        eligible=len(candidates), eligible_domains=dict(Counter(r['domain'] for r in candidates)),
        protocol='official V2V/gray sources and captions; custom 17-frame evaluation',
        not_claimed=['arbitrary instruction editing', 'official leaderboard reproduction'])
    return sorted(candidates, key=lambda r:r['id']), labels, report


def exposed_benchmark_ids(bank):
    """Include all old benchmark development inputs, even failed ones."""
    rows = list(bank.get('records', []))
    rows += bank.get('manifest', {}).get('benchmark', [])
    result = set()
    for row in rows:
        if row.get('split') == 'benchmark_dev' or row.get('id', '').startswith('benchmark-'):
            key = row['id']
            if key.startswith('benchmark-'):
                result.add('vace-'+key[len('benchmark-'):])
    return result


def decode_indices(path, indices):
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(path), 'ffmpeg')
    try:
        meta = reader.get_meta_data()
        count = reader.count_frames()
        if not indices or min(indices)<0 or max(indices)>=count:
            raise ValueError('Video cannot supply the fixed physical window')
        frames = torch.from_numpy(np.stack([reader.get_data(i)[..., :3]
                                            for i in indices])).permute(0,3,1,2)
        return frames, dict(fps=float(meta['fps']), count=count, size=list(meta['size']))
    finally:
        reader.close()


def public_source(root, row, geometry, remote=None):
    source = ensure_file(root, row['source'], remote)
    mask = ensure_file(root, row['mask'], remote)
    group = file_hash(source)
    payload = load_source(root, dict(row, group=group), geometry)
    masks, meta = decode_indices(mask, payload['timing']['indices'])
    timing = payload['timing']
    if abs(meta['fps']-timing['original_fps']) > .05 or meta['count'] != timing['original_count']:
        raise ValueError('Official source/mask times disagree')
    if list(masks.shape[-2:]) != payload['spatial']['original']:
        raise ValueError('Official source/mask geometry differs')
    # Gray uses an all-white generation mask. Do not silently reinterpret
    # inpainting or an inverted mask as full-frame colorization.
    if float(masks.float().mean()) < 254 or int(masks.min()) < 250:
        raise ValueError('Gray task requires the official all-white mask')
    mask_canvas = apply_spatial(masks, payload['spatial'])[:, :1]
    chroma = payload['source'].float().amax(1)-payload['source'].float().amin(1)
    if float(chroma.mean()) > 1:
        raise ValueError('V2V/gray input is not grayscale')
    payload.update(native_mask=mask_canvas, mask_sha256=file_hash(mask), group=group,
                   source_only=True, protocol='official_gray_input')
    return payload


def freeze_public(candidates, reports, exposed, old_benchmark_ids, count, smoke=False, asset_groups=None):
    good = {r['id']:r for r in reports if r['status']=='completed'}
    seen = [r for r in exposed if r.get('status')=='completed' and r.get('fingerprint')]
    rows, excluded = [], []
    asset_groups = asset_groups or {}
    old_assets = {asset_groups[key] for key in old_benchmark_ids if key in asset_groups}
    # Alternate domains, with source-only hash ordering, before limiting.
    buckets = {}
    for r in candidates:
        buckets.setdefault(r['domain'], []).append(r)
    for values in buckets.values():
        values.sort(key=lambda r:digest(['public-gray-v1', r['id']]))
    ordered = [buckets[d][i] for i in range(max(map(len,buckets.values()),default=0))
               for d in sorted(buckets) if i<len(buckets[d])]
    for row in ordered:
        report = good.get(row['id'])
        if row['id'] in old_benchmark_ids:
            reason = 'previous_benchmark_development'
        elif asset_groups.get(row['id']) in old_assets:
            reason = 'same_original_asset_as_previous_benchmark_input'
        elif report is None:
            reason = 'source_decode_failure'
        elif any(report['group']==s.get('group') or
                 near_duplicate(report['fingerprint'], s['fingerprint']) for s in seen):
            reason = 'previous_source_or_visual_duplicate'
        elif any(report['group']==s['group'] or
                 near_duplicate(report['fingerprint'], s['fingerprint']) for s in rows):
            reason = 'within_cohort_duplicate'
        else:
            reason = None
        if reason:
            excluded.append(dict(id=row['id'],reason=reason))
        elif len(rows)<count:
            rows.append(dict(row, group=report['group'], source_file=report['file'],
                             fingerprint=report['fingerprint']))
    result = dict(records=rows, selected=len(rows), requested=count, excluded=excluded,
        candidate_count=len(candidates), domains=dict(Counter(r['domain'] for r in rows)),
        exposed_screen_count=len(seen), exposed_screen_failures=[
            r for r in exposed if r.get('status')!='completed'],
        protocol='development_smoke' if smoke else 'source_selected_public_gray_colorization',
        caveat='Unseen relative to recorded adaptation/development/evaluation sources; '
               'foundation-model pretraining overlap is unknown. No output-based selection.')
    result['identity'] = digest(result)
    return result


def read_target(root, label, payload, geometry, remote=None):
    """Only evaluation calls this function; identical FOV and source timestamps."""
    from utils.editing_data import paired_indices
    import imageio.v2 as imageio
    path = ensure_file(root, label['target'], remote)
    reader = imageio.get_reader(str(path), 'ffmpeg')
    try:
        meta = reader.get_meta_data()
        count = reader.count_frames()
        timing = payload['timing']
        ids, target_ids = paired_indices(timing['original_count'], count,
            timing['original_fps'], meta['fps'], geometry['frames'], geometry['fps'])
        if ids != timing['indices'] or list(meta['size'])[::-1] != payload['spatial']['original']:
            raise ValueError('Target/source timing or FOV differs; no independent crop allowed')
        frames = torch.from_numpy(np.stack([reader.get_data(i)[..., :3] for i in target_ids])).permute(0,3,1,2)
        if list(frames.shape[-2:]) != payload['spatial']['original']:
            raise ValueError('Target decoded geometry changed')
        return apply_spatial(frames, payload['spatial']), dict(
            sha256=file_hash(path), indices=target_ids, original_fps=meta['fps'],
            semantics=label['semantics'])
    finally:
        reader.close()


def shared_reference(source, anchor, support, spatial, mask, cfg):
    """Same production RGB proposal, independent of RAE preparation success."""
    from utils.editing_video_transport import propagate_reference
    from utils.editing_joint_state import build_source_edges
    from utils.vace_state_data import build_reference
    indices = list(range(0,len(source),2))
    square, rect = rae_canvas(source[indices], spatial['valid_rect'])
    ar, _ = rae_canvas(anchor[None], spatial['valid_rect'])
    sup, _ = rae_canvas(support[None], spatial['valid_rect'])
    flow = propagate_reference(square, ar[0], sup[0], cfg['transport'])
    flow['proposal'][0] = ar[0]
    u8 = lambda v:(v.detach().cpu().clamp(0,1)*255).round().byte()
    inputs = dict(source=u8(square), proposal=u8(flow['proposal']), anchor=u8(ar[0]),
        support=sup[0].half(), confidence=flow['confidence'].cpu().half(),
        masks=flow['masks'].cpu().half(), observed_indices=indices, valid_rect=rect,
        source_spatial=copy.deepcopy(spatial), source_frames=len(source),
        flow_metadata=flow['metadata'], protocol='source_plus_model_anchor')
    ref = build_reference(inputs, source, anchor, cfg)
    # The public gray task's input mask, identical in all four native contexts.
    ref['masks'] = mask.float()/255
    full, _ = rae_canvas(source, spatial['valid_rect'])
    options = cfg['transport']
    inputs['edges'] = build_source_edges(full[indices], options)
    flows, confidence = [], []
    for t in range(len(full)-1):
        edge = propagate_reference(full[t:t+2], full[t], torch.ones_like(full[t,:1]),
                                   dict(options,flow_direct_anchor=False))
        flows.append(edge['flows_backward'][0].half())
        confidence.append(edge['confidence'][1].half())
    inputs.update(full_flows_backward=torch.stack(flows),
                  full_edge_confidence=torch.stack(confidence))
    return inputs, ref


def color_metrics(source, target, generated):
    """Paired chroma error plus structural diagnostics; no semantic success label."""
    import cv2
    from utils.editing_paper_metrics import rgb_array, ssim
    src, truth, pred = map(rgb_array, (source,target,generated))
    lab = lambda x:np.stack([cv2.cvtColor(np.ascontiguousarray(f),cv2.COLOR_RGB2LAB) for f in x])
    a,b = lab(truth),lab(pred)
    # Mean-channel grayscale is explicit; paired target luminance is separate.
    gray = lambda x:np.repeat(x.mean(-1,keepdims=True),3,axis=-1)
    return dict(target_lab_ab_l1=float(np.abs(a[...,1:]-b[...,1:]).mean()),
        target_lab_l_l1=float(np.abs(a[...,:1]-b[...,:1]).mean()),
        source_gray_ssim=ssim(gray(pred),gray(src)),
        source_gray_l1=float(np.abs(gray(pred)-gray(src)).mean()))
