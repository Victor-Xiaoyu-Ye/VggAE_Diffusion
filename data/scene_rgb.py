"""RGB-only WAI/SpatialVID data. No point clouds, depths or rendered RGB targets.

Sampling is a pure function of the requested cursor, so prefetched-but-unused
items do not change resume order. Bad examples return an explicit receipt.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
from collections import OrderedDict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.moxing_io import read_text, stage_remote_file


def seed_for(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def camera_rays(frames, meta, indices, grid):
    """OpenCV c2w -> reference-relative, unit-baseline rays.

    WAI schema defines transform_matrix as c2w. Unknown/non-pinhole cameras
    yield an absent condition, not fabricated stationary camera supervision.
    """
    out = np.zeros((len(indices), grid*grid, 8), dtype=np.float32)
    if meta.get('camera_convention') != 'opencv' or meta.get('camera_model') != 'PINHOLE':
        return torch.from_numpy(out)
    try:
        poses = np.asarray([frames[i]['transform_matrix'] for i in indices], np.float64)
        if poses.shape != (len(indices), 4, 4) or not np.isfinite(poses).all():
            return torch.from_numpy(out)
        if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-4):
            return torch.from_numpy(out)
        rot = poses[:, :3, :3]
        if not np.allclose(rot @ rot.transpose(0, 2, 1), np.eye(3), atol=.01):
            return torch.from_numpy(out)
        if not np.allclose(np.linalg.det(rot), 1., atol=.01):
            return torch.from_numpy(out)
        poses = np.linalg.inv(poses[0])[None] @ poses
        scale = max(float(np.linalg.norm(poses[:, :3, 3], axis=-1).max()), 1e-6)
        yy, xx = np.meshgrid((np.arange(grid)+.5)/grid, (np.arange(grid)+.5)/grid, indexing='ij')
        for j, i in enumerate(indices):
            item = meta if meta.get('shared_intrinsics') else frames[i]
            h, w, fx, fy, cx, cy = [float(item[k]) for k in ('h','w','fl_x','fl_y','cx','cy')]
            if min(h, w, fx, fy) <= 0:
                return torch.zeros_like(torch.from_numpy(out))
            side = min(h, w)
            x = (xx*side+(w-side)/2-cx)/fx
            y = (yy*side+(h-side)/2-cy)/fy
            ray = np.stack((x, y, np.ones_like(x)), -1).reshape(-1, 3)
            ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
            ray = ray @ poses[j, :3, :3].T
            origin = poses[j, :3, 3]/scale
            out[j, :, :3] = ray
            out[j, :, 3:6] = np.cross(origin, ray)
            # COLMAP and metric sources cannot share an absolute-distance code.
            # Only indicate whether translation exists; preserve relative rays.
            out[j, :, 6] = float(scale > 1e-6)
            out[j, :, 7] = 1
        if not np.isfinite(out).all():
            out.fill(0)
    except (KeyError, ValueError, np.linalg.LinAlgError):
        out.fill(0)
    return torch.from_numpy(out)


def resize_rgb(image, size):
    # One deterministic center crop, shared by RGB and camera intrinsics.
    image = image.convert('RGB')
    w, h = image.size
    side = min(w, h)
    left, top = (w-side)/2, (h-side)/2
    image = image.resize((size, size), Image.Resampling.BICUBIC,
                         box=(left, top, left+side, top+side))
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()/255


def temporal_stream(frames, rng):
    """Keep one camera and contiguous source indices; never interleave stereo."""
    streams = {}
    for frame in frames:
        if frame.get('_is_interpolated', False):
            continue
        name = PurePosixPath(frame.get('image') or frame.get('file_path','')).stem
        number = re.search(r'(\d+)$',name)
        side = re.search(r'_(left|right)[_-]',name)
        if number is None:
            raise ValueError('Video source has no frame index')
        streams.setdefault(side[1] if side else 'mono',[]).append((int(number[1]),frame))
    if not streams:
        raise ValueError('No original video frames')
    stream=sorted(streams[rng.choice(sorted(streams))],key=lambda x:x[0])
    segments=[[]]
    previous=None
    for index,frame in stream:
        if previous is not None and index != previous+1:
            segments.append([])
        segments[-1].append(frame)
        previous=index
    return max(segments,key=len)


class SceneRGB(Dataset):
    def __init__(self, cohort, split='train', size=518, views=9, grid=18,
                 cache_dir='', weights=None, finite=False, windows=4):
        self.cohort = json.loads(Path(cohort).read_text(encoding='utf8')) if isinstance(cohort, (str, Path)) else cohort
        self.rows = [r for r in self.cohort['records'] if r['split'] == split]
        self.rows.sort(key=lambda r: seed_for(self.cohort['seed'], r['id']))
        if not self.rows:
            raise ValueError('No RGB sources in split '+split)
        self.by_source = {}
        for row in self.rows:
            self.by_source.setdefault(row['dataset'], []).append(row)
        self.sources = sorted(self.by_source)
        weights = weights or dict(dl3dv=.40, scannetppv2=.15, mvssynth=.05, spatialvid=.40)
        self.weights = [weights.get(k, .05) for k in self.sources]
        self.size, self.views, self.grid = size, views, grid
        self.cache_dir = cache_dir or os.environ.get('MOX_VIDEO_CACHE_DIR', '/cache/yexiaoyu/vggae_runtime/cache/scene_rgb')
        self.metadata = OrderedDict()
        self.finite, self.windows = finite, windows

    def __len__(self):
        return len(self.rows)*self.windows if self.finite else 100_000_000

    def row_at(self, index):
        rng = random.Random(seed_for(self.cohort['seed'], index))
        if self.finite:
            return self.rows[index//self.windows], index % self.windows, rng
        source = rng.choices(self.sources, self.weights)[0]
        return rng.choice(self.by_source[source]), rng.randrange(1_000_000), rng

    def local(self, uri):
        return stage_remote_file(uri, self.cache_dir,
            max_cache_bytes=int(float(os.environ.get('MOX_VIDEO_CACHE_GB', '200'))*1024**3), retries=2)

    def load(self, index):
        row, window, rng = self.row_at(index)
        if row['kind'] == 'wai':
            root = row['root'].rstrip('/')+'/'
            if root not in self.metadata:
                meta = json.loads(read_text(root+'scene_meta.json'))
                if row.get('filter_listed_rgb'):
                    if root.startswith('obs://'):
                        import moxing
                        names = moxing.file.list_directory(root+'images/', recursive=False)
                    else:
                        names = os.listdir(root+'images')
                    names = {n.rsplit('/', 1)[-1] for n in names}
                    meta['frames'] = [f for f in meta['frames'] if
                        (f.get('image') or f.get('file_path','')).rsplit('/',1)[-1] in names]
                self.metadata[root] = meta
                if len(self.metadata) > 32:
                    self.metadata.popitem(last=False)
            meta = self.metadata[root]
            all_frames = meta['frames']
            frames = [f for f in all_frames if not f.get('is_bad', False)]
            if row.get('video_sequence'):
                frames = temporal_stream(frames,rng)
            stride = rng.randint(1, 2)
            stride = min(stride, max(1, (len(frames)-1)//max(self.views-1,1)))
            span = (self.views-1)*stride+1
            if len(frames) < span:
                raise ValueError('Insufficient distinct RGB views')
            start = rng.randrange(len(frames)-span+1)
            indices = [start+i*stride for i in range(self.views)]
            images, frame_ids, dimensions_match = [], [], True
            for i in indices:
                path = frames[i].get('image') or frames[i].get('file_path', '')
                if not path.startswith('images/') or '..' in PurePosixPath(path).parts:
                    raise ValueError('Not an original RGB image path')
                with Image.open(self.local(root+path)) as image:
                    calibration = meta if meta.get('shared_intrinsics') else frames[i]
                    expected = (calibration.get('w'), calibration.get('h'))
                    dimensions_match = dimensions_match and image.size == expected
                    images.append(resize_rgb(image, self.size))
                frame_ids.append(path)
            rgb = torch.stack(images)
            rays = (camera_rays(frames, meta, indices, self.grid) if dimensions_match else
                    torch.zeros(self.views, self.grid*self.grid, 8))
            temporal_valid = bool(row.get('video_sequence'))
        elif row['kind'] == 'frames':
            root = row['root'].rstrip('/')+'/'
            if root not in self.metadata:
                if root.startswith('obs://'):
                    import moxing
                    names = moxing.file.list_directory(root, recursive=False)
                else:
                    names = os.listdir(root)
                self.metadata[root] = sorted(n.rsplit('/', 1)[-1] for n in names
                    if n.lower().endswith(('.png', '.jpg', '.jpeg')))
                if len(self.metadata) > 32:
                    self.metadata.popitem(last=False)
            names = self.metadata[root]
            if len(names) < self.views:
                raise ValueError('Insufficient distinct RGB views')
            start = rng.randrange(len(names)-self.views+1)
            frame_ids = names[start:start+self.views]
            images = []
            for name in frame_ids:
                with Image.open(self.local(root+name)) as image:
                    images.append(resize_rgb(image, self.size))
            rgb = torch.stack(images)
            rays = torch.zeros(self.views, self.grid*self.grid, 8)
            temporal_valid = False
        elif row['kind'] == 'omni':
            if row['split_info'] not in self.metadata:
                self.metadata[row['split_info']] = json.loads(read_text(row['split_info']))['split']
                if len(self.metadata) > 32:
                    self.metadata.popitem(last=False)
            parts = [x for x in self.metadata[row['split_info']] if len(x) >= self.views]
            if not parts:
                raise ValueError('No sufficiently long OmniWorld subclip')
            part = rng.choice(parts)
            span = min(len(part), max(self.views, round(row['fps'])+1))
            start = rng.randrange(len(part)-span+1)
            indices = np.linspace(start, start+span-1, self.views).round().astype(int)
            frame_ids = [int(part[i]) for i in indices]
            images = []
            for i in frame_ids:
                with Image.open(self.local(row['root'].rstrip('/')+f'/{i:06d}.png')) as image:
                    images.append(resize_rgb(image, self.size))
            rgb = torch.stack(images)
            rays = torch.zeros(self.views, self.grid*self.grid, 8)
            temporal_valid = True
        else:
            # Only the requested one-second clip is decoded; no point/depth input.
            import cv2
            local = self.local(row['root'])
            cap = cv2.VideoCapture(local)
            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or row['fps']
                span = min(count, max(self.views, round(fps)+1))
                if count < self.views:
                    raise ValueError('Video has insufficient frames')
                start = rng.randrange(count-span+1)
                indices = np.linspace(start, start+span-1, self.views).round().astype(int).tolist()
                images = []
                for i in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError('Video frame cannot be decoded: '+str(i))
                    images.append(resize_rgb(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), self.size))
                rgb = torch.stack(images)
            finally:
                cap.release()
            frame_ids = indices
            rays = torch.zeros(self.views, self.grid*self.grid, 8)
            temporal_valid = True
        return dict(frames=rgb, rays=rays, temporal_valid=temporal_valid,
            id=row['id'], dataset=row['dataset'], window=window, frame_ids=frame_ids,
            cursor=index, error=None)

    def __getitem__(self, index):
        try:
            return self.load(index)
        except Exception as exc:
            row, _, _ = self.row_at(index)
            return dict(error=f'{type(exc).__name__}: {exc}', cursor=index,
                        id=row['id'], dataset=row['dataset'])


def singleton(items):
    return items[0]


class CursorSampler:
    def __init__(self, length, rank=0, world=1, cursor=0):
        self.length, self.rank, self.world, self.cursor = length, rank, world, cursor

    def __iter__(self):
        return iter(range(self.rank+self.cursor*self.world, self.length, self.world))

    def __len__(self):
        return max(0, (self.length-self.rank-1)//self.world+1-self.cursor)
