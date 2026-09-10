"""Audited infrastructure for the new window trainer, no probe-trainer imports."""
import contextlib
import hashlib
import io
import json
import shutil
import time
from collections import OrderedDict
from pathlib import Path
import torch

from data.latent_shard_dataset import latent_collate_fn
from utils.moxing_io import open_file, read_text, join_remote, is_remote_path


def load_artifact(path):
    with open_file(path, 'rb') as stream:
        return torch.load(io.BytesIO(stream.read()), map_location='cpu', weights_only=False)


def digest(value):
    def encode(x):
        if torch.is_tensor(x):
            return dict(dtype=str(x.dtype), shape=list(x.shape), value=x.tolist())
        raise TypeError(type(x))
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=encode).encode()).hexdigest()


def validate_statistics(stats):
    from utils.r7_representation import R7_CONTRACT_SCHEMA
    rep = stats.get('representation', {})
    cfg = rep.get('config', {})
    factor = cfg.get('temporal_factor')
    channels = cfg.get('geo_latent_dim', 0)+cfg.get('tex_latent_dim', 0)
    if (stats.get('normalization_version') != 2 or rep.get('schema') != R7_CONTRACT_SCHEMA
            or factor not in (1, 2) or cfg.get('seq_len') != 9 or channels != 192
            or cfg.get('latent_grid') != 18 or rep.get('layout', {}).get('anchor_chunks') != [1, 8//factor]):
        raise ValueError('expected signed R7 absolute t1/t2 C192 seq9 statistics')
    if set(('streamvggt', 'source_dual_ae', 'r7'))-set(rep.get('signatures', {})):
        raise ValueError('cache lacks representation signatures')
    for name, length in (('cond', 1), ('target', 8//factor)):
        group = stats[name]
        if tuple(group['mean'].shape) != (length, channels) or group['std'].shape != group['mean'].shape:
            raise ValueError('invalid frame/channel statistics shape')
        if not torch.isfinite(group['mean']).all() or not torch.isfinite(group['std']).all() or (group['std'] <= 0).any():
            raise ValueError('invalid normalization values')
    return cfg


def normalize(x, group):
    return (x.float()-group['mean'][None, :, None])/group['std'][None, :, None]


def inverse(x, group):
    return x.float()*group['std'][None, :, None]+group['mean'][None, :, None]


def collate(batch):
    result = latent_collate_fn(batch)
    if all('rgb' in x for x in batch):
        result['rgb'] = torch.stack([x['rgb'] for x in batch])
    if all('anchor_relative_l2' in x for x in batch):
        result['anchor_relative_l2'] = [x['anchor_relative_l2'] for x in batch]
    return result


def validate_batch(batch, future, grid=18, channels=192):
    c, y = batch['cond'], batch['target']
    if c.shape[1:] != (1, grid*grid, channels) or y.shape[1:] != (future, grid*grid, channels):
        raise ValueError('cache tensor shape mismatch')
    if not torch.isfinite(c).all() or not torch.isfinite(y).all():
        raise ValueError('cache contains nonfinite latents')
    if not all(batch['video_id']) or batch['video_id'] != batch['requested_video_id']:
        raise ValueError('missing/replaced source video identity')
    return c, y


class CaptionBank:
    """Strict reader of existing UMT5 sidecars; never falls back for missing IDs."""
    def __init__(self, root):
        def path(name):
            return join_remote(root, name) if is_remote_path(root) else str(Path(root)/name)
        success = json.loads(read_text(path('_SUCCESS')))
        if success.get('schema') != 'wan-umt5xxl-text-embeddings-v1':
            raise ValueError('unsupported caption sidecar')
        index = json.loads(read_text(path('index.json')))
        self.empty = load_artifact(path('empty_prompt.pt')).float()
        self.values, self.path, self.cache = index, path, OrderedDict()
        sha = hashlib.sha256()
        def check(key, value):
            if value.ndim != 2 or value.shape[-1] != 4096 or not 0 < value.shape[0] <= int(success['text_len']):
                raise ValueError('invalid UMT5 tensor')
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite caption embedding')
            sha.update(key.encode()); sha.update(value.float().contiguous().numpy().tobytes())
        check('', self.empty)
        # Scan for an exact content signature without retaining the entire 10K
        # embedding bank in every rank's RAM. At runtime retain at most 2 shards.
        for name in sorted(set(index.values())):
            part = load_artifact(path(name))
            keys = sorted(k for k, shard in index.items() if shard == name)
            if set(keys)-set(part):
                raise ValueError('incomplete caption shards')
            for key in keys: check(key, part[key])
            del part
        self.signature = sha.hexdigest()

    def get(self, key):
        name = self.values[key]
        if name not in self.cache:
            self.cache[name] = load_artifact(self.path(name))
            if len(self.cache) > 2: self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name][key]

    def batch(self, ids, device, dropout=0.):
        missing = set(ids)-set(self.values)
        if missing:
            raise ValueError(f'missing caption IDs: {sorted(missing)[:4]}')
        drops = torch.rand(len(ids), device=device) < dropout if dropout else [False]*len(ids)
        values = [self.empty if bool(drops[i]) else self.get(key) for i, key in enumerate(ids)]
        n = max(x.shape[0] for x in values)
        data = torch.zeros(len(ids), n, 4096, device=device)
        valid = torch.zeros(len(ids), n, device=device, dtype=torch.bool)
        for i, value in enumerate(values):
            data[i, :len(value)] = value.to(device); valid[i, :len(value)] = True
        return data, valid


@contextlib.contextmanager
def use_ema(model, ema):
    backup = {k:v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema.state_dict(), strict=True)
    try:
        yield
    finally:
        model.load_state_dict(backup, strict=True)


def resume_contract(args, identity, world):
    # Eval settings are immutable too, so checkpoint selection remains comparable.
    mutable = {'resume', 'output_dir', 'stop_after_steps', 'manifest', 'eval_manifest',
               'stats', 'eval_stats', 'r7_ckpt', 'text_dir', 'cpu_test'}
    if not getattr(args, 'memorize_clips', 0):
        mutable.add('memorize_clips')
    if getattr(args, 'time_distribution', 'logit_normal_0_1') == 'logit_normal_0_1':
        mutable.add('time_distribution')  # Preserve historical full-resume identities.
    # Disabled auxiliary defaults preserve pre-auxiliary full-resume contracts.
    if not getattr(args, 'aux_layer', 0) and not getattr(args, 'aux_weight', 0):
        mutable |= {'aux_layer', 'aux_weight'}
    return dict(args={k:v for k, v in vars(args).items() if k not in mutable},
                identity=identity, world_size=world)


def validate_resume(saved, expected):
    if saved.get('schema') != 'r7-window-trainer-v1' or saved.get('contract') != expected:
        raise ValueError('resume model/objective/data/statistics/world-size mismatch')


def reconcile_history(directory, step):
    """Keep measured history but archive rows beyond the resumed checkpoint."""
    for path in Path(directory).glob('*.jsonl'):
        lines = path.read_text(encoding='utf-8').splitlines()
        kept = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # Crash may have left one partial tail; original is archived.
            if row.get('step', -1) <= step: kept.append(line)
        if kept != lines:
            archive = Path(directory)/'history_before_resume'/str(time.time_ns())/path.name
            archive.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,archive)
            temp = path.with_suffix('.jsonl.tmp')
            temp.write_text(''.join(x+'\n' for x in kept),encoding='utf-8')
            temp.replace(path)
