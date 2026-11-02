"""Señorita archive/metadata pairing. No torch, HF login, or network at import."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def safe_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe data member: ' + name)
    return p


def unpack_archive(source, output):
    """Only ordinary MP4/JSON files; never extract links or trust tar paths."""
    source, output = Path(source), Path(output)
    marker = output / (source.name + '.complete.json')
    signature = dict(size=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns)
    if marker.is_file() and json.loads(marker.read_text()) == signature:
        return
    output.mkdir(parents=True, exist_ok=True)
    count, started = 0, time.monotonic()
    with tarfile.open(source, 'r|gz') as archive:
        for member in archive:
            rel = safe_path(member.name)
            if not member.isfile() or rel.suffix not in ('.mp4', '.json'):
                continue
            dest = output.joinpath(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_file() and dest.stat().st_size == member.size:
                continue
            temp = dest.with_name(dest.name + '.partial')
            with archive.extractfile(member) as src, temp.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if temp.stat().st_size != member.size:
                raise IOError('Incomplete tar member: ' + member.name)
            os.replace(temp, dest)
            count += 1
            if count % 200 == 0:
                print(f'[Senorita unpack] {source.name}: {count} files, {time.monotonic()-started:.0f}s', flush=True)
    marker.write_text(json.dumps(signature))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def make_manifest(root, cfg):
    """Pair by metadata path suffix, group identical source bytes before split.

    Official tar: ./4/hash_org.mp4; metadata: local_style_transfer/4/hash_org.mp4.
    Distinct encodings of the same parent are NOT detectable by byte hashing;
    this is an internal development split, not a certified benchmark split.
    """
    root = Path(root)
    metadata = sorted(root.rglob('updated_data_local_style_transfer.json'))
    if not metadata:
        raise FileNotFoundError('Need official updated_data_local_style_transfer.json under ' + str(root))
    if len(metadata) != 1:
        raise ValueError('Multiple local-style metadata files; provide an unambiguous pilot root')
    index = {}
    for p in sorted(root.rglob('*.mp4')):
        key = '/'.join(p.parts[-2:])
        if key in index and file_hash(p) != file_hash(index[key]):
            raise ValueError('Ambiguous package/video suffix: ' + key)
        index[key] = p
    values = json.loads(metadata[0].read_text(encoding='utf8'))
    if not isinstance(values, list):
        raise ValueError('Expected official JSON array')
    pairs, missing, duplicate = [], 0, 0
    seen, hashes = set(), {}
    for entry in values:
        paths = [safe_path(entry[k]).parts for k in ('source_video_path', 'target_video_path')]
        source, target = [index.get('/'.join(p[-2:])) for p in paths]
        if source is None or target is None:
            missing += 1
            continue
        prompt = (entry.get('enhanced_instruction') or entry.get('instruction') or '').strip()
        if not prompt:
            continue
        source_hash = hashes.setdefault(str(source), None)
        if source_hash is None:
            source_hash = hashes[str(source)] = file_hash(source)
        key = digest([source_hash, file_hash(target), prompt])
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        pairs.append(dict(id=key[:24], source=str(source.relative_to(root)).replace('\\', '/'),
            target=str(target.relative_to(root)).replace('\\', '/'), group=source_hash,
            instruction=prompt, prompt_kind='edit_instruction'))
    # Reserve whole source groups, not individual edit variants.
    groups = {}
    for row in pairs:
        groups.setdefault(row['group'], []).append(row)
    order = sorted(groups, key=lambda k: digest([cfg['seed'], k]))
    selected = {'train': [], 'dev': [], 'test': []}
    for group in order:
        split = ('test' if len(selected['test']) < cfg['test_pairs'] else
                 'dev' if len(selected['dev']) < cfg['dev_pairs'] else 'train')
        if split == 'train' and len(selected[split]) >= cfg['train_pairs']:
            continue
        selected[split].extend(dict(row, split=split) for row in groups[group])
    return dict(schema='senorita-pairs-v1', records=sum(selected.values(), []),
        counts={k: len(v) for k,v in selected.items()}, available_pairs=len(pairs),
        available_groups=len(groups), metadata_unavailable_pairs=missing, duplicates=duplicate,
        metadata_sha256=file_hash(metadata[0]),
        split_caveat='Source byte-hash groups; re-encoded duplicates/parent-scene overlap not certified. Internal dev only.')


def paired_indices(source_count, target_count, source_fps, target_fps, frames, fps):
    if min(source_fps, target_fps, fps) <= 0:
        raise ValueError('Invalid FPS')
    # Do not stretch unrelated durations or duplicate short clips into a freeze.
    if abs(source_count/source_fps-target_count/target_fps) > 2/max(source_fps,target_fps):
        raise ValueError('Source/target durations disagree')
    ids = [[round(i*rate/fps) for i in range(frames)] for rate in (source_fps,target_fps)]
    if any(len(set(x)) != frames or max(x) >= n for x,n in zip(ids,(source_count,target_count))):
        raise ValueError('Clip too short or FPS too low for the declared window')
    return ids


def spatial_plan(height,width,max_area,max_side=768):
    """Keep the full FOV, never enlarge RGB, pad minimally for native patching."""
    if min(height,width)<=0 or max_area<256 or max_side<16:
        raise ValueError('Invalid video geometry')
    scale=min(1.,math.sqrt(max_area/(height*width)),max_side/max(height,width))
    while True:
        h,w=max(1,round(height*scale)),max(1,round(width*scale))
        canvas_h,canvas_w=(h+15)//16*16,(w+15)//16*16
        if canvas_h*canvas_w<=max_area and max(canvas_h,canvas_w)<=max_side: break
        scale*=.99
    top,left=(canvas_h-h)//2,(canvas_w-w)//2
    return dict(original=[height,width],resized=[h,w],canvas=[canvas_h,canvas_w],
        valid_rect=[top,left,h,w],scale=scale,policy='fit_area_pad16_no_upscale_v1')


def apply_spatial(x,plan):
    import torch.nn.functional as F
    h,w=plan['resized']; ch,cw=plan['canvas']; top,left,_,_=plan['valid_rect']
    x=F.interpolate(x.float(),size=(h,w),mode='bilinear',align_corners=False)
    return F.pad(x,(left,cw-w-left,top,ch-h-top)).round().clamp(0,255).byte()


def rae_canvas(frames,valid_rect,size=518):
    """Production AE wants 518 square: letterbox, do not squeeze a moving body."""
    import torch.nn.functional as F
    top,left,h,w=valid_rect
    frames=frames[:,:,top:top+h,left:left+w]
    scale=size/max(h,w); rh,rw=round(h*scale),round(w*scale)
    x=F.interpolate(frames.float(),size=(rh,rw),mode='bilinear',align_corners=False)
    y,x0=(size-rh)//2,(size-rw)//2
    return F.pad(x,(x0,size-rw-x0,y,size-rh-y)),[y,x0,rh,rw]


def load_pair(root, row, cfg):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import imageio.v2 as imageio
    readers = []
    try:
        for k in ('source','target'):
            readers.append(imageio.get_reader(str(Path(root)/row[k]), 'ffmpeg'))
        counts = [r.count_frames() for r in readers]
        rates = [r.get_meta_data()['fps'] for r in readers]
        output_fps=cfg['benchmark_fps'] if row.get('split')=='benchmark_dev' else cfg['fps']
        indices = paired_indices(*counts, *rates, cfg['frames'], output_fps)
        sw,sh=readers[0].get_meta_data()['size']
        tw,th=readers[1].get_meta_data()['size']
        if (sh,sw)!=(th,tw):
            raise ValueError('Paired frame sizes differ; explicit alignment required, not independent crops')
        plan=spatial_plan(sh,sw,cfg['height']*cfg['width'],cfg['max_side'])
        values = []
        for reader, ids in zip(readers, indices):
            x = torch.from_numpy(np.stack([reader.get_data(i)[...,:3] for i in ids])).permute(0,3,1,2)
            if list(x.shape[-2:])!=plan['original']:
                raise ValueError('Decoded size differs from reported size (possibly rotation); explicit alignment required')
            values.append(apply_spatial(x,plan))
        return *values, dict(indices=indices, original_counts=counts, original_fps=rates,
                             output_fps=output_fps,spatial=plan)
    finally:
        for reader in readers:
            reader.close()
