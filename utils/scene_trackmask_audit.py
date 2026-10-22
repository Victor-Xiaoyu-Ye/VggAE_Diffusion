"""Read-only media, point-support metrics and plots for the VGGT mask audit.

Inputs are labelled, lossy stage53/54 previews, NOT original RGB tensors.
Sparse tracks support small disks, not a semantic segmentation of the scene.
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.window_run_io import child
from utils.moxing_io import copy_file
from utils.native_audit_io import recover_json, safe_relative, sha256


INPUT_NOTE = ('Lossy MP4 preview columns with the 32-pixel label strip removed; '
              'pixel metrics are not directly comparable to original-tensor evaluation.')
MASK_NOTE = ('Camera-compensated sparse track neighborhoods, NOT semantic segmentation '
             'or human motion ground truth. Full-window teacher uses future RGB; '
             'diagnostic only, never a generation condition.')


def read_case(source_local, source_roots, index, step=6000, seed=101, weights='ema'):
    """Recover only JSON/MP4 verified by the source receipt, never latent/weights."""
    prefix = f'eval/step{step:07d}/{weights}/case{index:03d}_seed{seed}'
    receipt = recover_json(prefix+'.done.json', source_local, source_roots, required=True)
    if receipt.get('status') != 'completed':
        raise ValueError('source case was not completed')
    payloads = {}
    for suffix in ('.json', '.mp4'):
        name = safe_relative(prefix+suffix)
        expected = receipt['files'][name]
        target = Path(source_local)/name
        def valid(path):
            return (path.is_file() and path.stat().st_size == expected['size']
                    and sha256(path) == expected['sha256'])
        if not valid(target):
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name+f'.{os.getpid()}.partial')
            errors = []
            for root in source_roots:
                try:
                    copy_file(child(root, name), str(partial))
                    if not valid(partial):
                        raise ValueError('source content checksum mismatch')
                    os.replace(partial, target)
                    break
                except Exception as exc:
                    errors.append(repr(exc))
                    partial.unlink(missing_ok=True)
            else:
                raise RuntimeError(f'Cannot recover verified {name}: {errors}')
        payloads[suffix] = target
    row = json.loads(payloads['.json'].read_text(encoding='utf8'))
    if row.get('case_index') != index or row.get('seed') != seed:
        raise ValueError('source case/seed metadata mismatch')
    if row.get('weights') != weights or row.get('step') != step:
        raise ValueError('source checkpoint metadata mismatch')
    return payloads['.mp4'], row, dict(source_contract=receipt['contract_hash'],
        video_sha256=receipt['files'][prefix+'.mp4']['sha256'],
        metadata_sha256=receipt['files'][prefix+'.json']['sha256'])


def split_preview(frames, metadata, header=32):
    """No resize: keep exact pixel alignment between all four preview columns."""
    frames = np.asarray(frames)
    columns = metadata.get('columns')
    if columns != ['RAW', 'AE', 'GENERATED', 'COPY']:
        raise ValueError('expected ordinary RAW/AE/GENERATED/COPY preview, not an oracle')
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError('preview must be uint8 [T,H,W,3]')
    if frames.shape[2] % 4 or header < 0 or frames.shape[1] <= header:
        raise ValueError('invalid preview header/column layout')
    total = metadata['history_frames']+metadata['future_frames']
    if len(frames) != total or len(metadata['frame_ids']) != total:
        raise ValueError('frame count and source frame IDs must agree')
    width = frames.shape[2]//4
    if frames.shape[1]-header != width:
        raise ValueError('expected square stage54 tiles without codec resizing')
    return {label: frames[:, header:, i*width:(i+1)*width].copy()
            for i, label in enumerate(columns)}


def decode_preview(path, metadata):
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    try:
        info = next(reader)
        w, h = info['size']
        frames = [np.frombuffer(data, np.uint8).reshape(h, w, 3).copy() for data in reader]
    finally:
        reader.close()
    return split_preview(np.asarray(frames), metadata)


def point_support(tracks, labels, height, width, radius):
    """Nearest classified point within radius; all uncovered pixels stay unknown.

    A nearby unknown point takes precedence over more distant classified points.
    We intentionally do not fill all image pixels from nearest sparse tracks.
    """
    tracks, labels = np.asarray(tracks), np.asarray(labels)
    if tracks.shape[:2] != labels.shape or tracks.shape[-1] != 2:
        raise ValueError('track/label shape mismatch')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('support radius must be positive')
    result = np.full((len(tracks), height, width), -1, dtype=np.int8)
    nearest = np.full(result.shape, np.inf, dtype=np.float32)
    for t in range(len(tracks)):
        for point, label in zip(tracks[t], labels[t]):
            if not np.isfinite(point).all():
                continue
            x, y = point
            left, right = max(0, int(np.ceil(x-radius))), min(width, int(np.floor(x+radius))+1)
            top, bottom = max(0, int(np.ceil(y-radius))), min(height, int(np.floor(y+radius))+1)
            if left >= right or top >= bottom:
                continue
            yy, xx = np.ogrid[top:bottom, left:right]
            distance = (xx-x)**2+(yy-y)**2
            previous = nearest[t, top:bottom, left:right]
            current = result[t, top:bottom, left:right]
            tied = (distance <= radius**2) & np.isclose(distance, previous, atol=1e-6, rtol=0)
            conflict = tied & (current != label)
            selected = (distance <= radius**2) & (distance < previous)
            previous[selected] = distance[selected]
            current[selected] = label
            current[conflict] = -1
    return result


def region_metrics(columns, support, history):
    """Paired future-only RGB and local edge proxies; empty regions are null."""
    raw = columns['RAW'].astype(np.float32)/255
    if support.shape != raw.shape[:-1] or not 0 < history < len(raw):
        raise ValueError('invalid support or observed-prefix length')
    errors = {key: np.abs(value.astype(np.float32)/255-raw).mean(-1)
              for key, value in columns.items() if key != 'RAW'}
    rows = []
    for t in range(history, len(raw)):
        regions = {}
        for label, state in [('dynamic', 1), ('static', 0), ('unknown', -1)]:
            mask = support[t] == state
            count = int(mask.sum())
            regions[label] = dict(pixels=count, fraction=float(mask.mean()),
                rgb_l1={key: float(value[t][mask].mean()) if count else None
                        for key, value in errors.items()})
        rows.append(dict(frame_index=t, regions=regions))
    summary = {}
    for label, state in [('dynamic', 1), ('static', 0), ('unknown', -1)]:
        mask = support[history:] == state
        count = int(mask.sum())
        summary[label] = dict(pixels=count, fraction=float(mask.mean()),
            rgb_l1={key: float(value[history:][mask].mean()) if count else None
                    for key, value in errors.items()})
    return dict(summary=summary, per_future_frame=rows,
        scope='Identical GT-teacher support for AE/GENERATED/COPY, future frames only.',
        caveat=INPUT_NOTE+' '+MASK_NOTE+' A different plausible future can have larger paired error.')


def _tile(frame, title, size=259):
    image = Image.fromarray(frame).resize((size, size), Image.Resampling.LANCZOS)
    tile = Image.new('RGB', (size, size+32), '#161616')
    tile.paste(image, (0, 32))
    ImageDraw.Draw(tile).text((5, 7), title, fill='white')
    return tile


def input_contact_sheet(columns, history, path):
    indices = list(range(history-1, len(columns['RAW'])))
    labels = ('RAW', 'AE', 'GENERATED')
    canvas = Image.new('RGB', (len(indices)*259, len(labels)*291+38), '#161616')
    ImageDraw.Draw(canvas).text((7, 10), 'LOSSY PREVIEW | last observed, then future | no mask inference yet', fill='white')
    for row, key in enumerate(labels):
        for col, t in enumerate(indices):
            canvas.paste(_tile(columns[key][t], f'{key} F{t}'), (col*259, 38+row*291))
    canvas.save(path, quality=93)


def mask_overlay(raw, tracks, result, *, naive=False):
    colors = {-1: (230, 190, 50), 0: (50, 210, 255), 1: (255, 50, 65)}
    frames = []
    reference = result['diagnostics'].get('reference_frame', 0)
    base = result['diagnostics'].get('config', {}).get('base_threshold_px', 2.)
    naive_threshold = max(1., base*max(raw.shape[1:3])/518.)
    for t, frame in enumerate(raw):
        image = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(image)
        for n, point in enumerate(tracks[t]):
            if not np.isfinite(point).all():
                continue
            x, y = map(float, point)
            if not (0 <= x < image.width and 0 <= y < image.height):
                continue
            label = int(result['labels'][t, n])
            if naive:
                displacement = np.linalg.norm(point-tracks[reference, n])
                # Raw displacement must not borrow depth/global-camera filters
                # from the compensated method: that would hide its false positives.
                reasons = result.get('reason_bits')
                usable = (reasons is None or (int(reasons[t, n]) & (1 | 2)) == 0)
                usable = usable and t != reference and np.isfinite(displacement)
                label = int(displacement > naive_threshold) if usable else -1
            color = colors[label]
            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
            if not naive and result['valid'][t, n]:
                rigid = result['rigid_tracks'][t, n]
                if np.isfinite(rigid).all():
                    draw.line((float(rigid[0]), float(rigid[1]), x, y), fill=color, width=1)
        frames.append(np.asarray(image))
    return np.stack(frames)


def render_mask_review(columns, full, prefix, output, history, fps=8.):
    """Full future-assisted and separately recomputed observed-prefix views."""
    import imageio.v2 as imageio
    output = Path(output)
    raw = columns['RAW']
    naive = mask_overlay(raw, full['tracks'], full, naive=True)
    compensated = mask_overlay(raw, full['tracks'], full)
    names = ('RAW', 'DISPLACEMENT', 'RIGID RESIDUAL', 'AE', 'GENERATED')
    arrays = (raw, naive, compensated, columns['AE'], columns['GENERATED'])
    frames = []
    for t in range(len(raw)):
        image = Image.new('RGB', (259*5, 291+35), '#161616')
        ImageDraw.Draw(image).text((5, 8), 'FULL-WINDOW GT TEACHER | red: dynamic candidate, cyan: static, yellow: unknown', fill='white')
        for i, (label, values) in enumerate(zip(names, arrays)):
            image.paste(_tile(values[t], f'{label} F{t}'), (i*259, 35))
        frames.append(np.asarray(image))
    # Width must be even for yuv420p; pad instead of rescaling image coordinates.
    movie = np.pad(np.stack(frames), ((0, 0), (0, 0), (0, 1), (0, 0)))
    video_path = output/'mask_review.mp4'
    imageio.mimwrite(str(video_path), movie, fps=fps, macro_block_size=1,
                    codec='libx264', quality=8, pixelformat='yuv420p')
    selected = [history-1, history, len(raw)-1]
    sheet = Image.new('RGB', (movie.shape[2], len(selected)*movie.shape[1]), '#161616')
    for i, t in enumerate(selected):
        sheet.paste(Image.fromarray(movie[t]), (0, i*movie.shape[1]))
    sheet.save(output/'mask_review.jpg', quality=94)
    observed = mask_overlay(raw[:history], prefix['tracks'], prefix)
    prefix_sheet = Image.new('RGB', (history*259, 291+35), '#161616')
    ImageDraw.Draw(prefix_sheet).text((5, 8), 'OBSERVED PREFIX ONLY | separate teacher forward; no future location prediction', fill='white')
    for t in range(history):
        prefix_sheet.paste(_tile(observed[t], f'PREFIX F{t}'), (259*t, 35))
    prefix_sheet.save(output/'observed_prefix.jpg', quality=94)
    return ['mask_review.mp4', 'mask_review.jpg', 'observed_prefix.jpg']
