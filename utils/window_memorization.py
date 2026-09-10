"""Training-only fixed subset and balanced deterministic distributed replay."""
import hashlib
import random
import torch
from torch.utils.data import IterableDataset
from data.latent_shard_dataset import LatentShardDataset


def select_training_subset(manifest, count, seed):
    samples, seen = [], set()
    for sample in LatentShardDataset(manifest, 0, seed, False, 0, 1):
        key = sample.get('video_id')
        if not key:
            raise ValueError('missing subset video ID')
        if key in seen:
            continue
        seen.add(key); samples.append(sample)
        if len(samples) == count:
            return samples
    raise ValueError(f'only {len(samples)} distinct training videos, requested {count}')


def subset_identity(samples):
    sha = hashlib.sha256()
    ids = []
    for sample in samples:
        key = sample['video_id']; ids.append(key)
        sha.update(key.encode()); sha.update(b'\0')
        for name in ('cond', 'target'):
            tensor = sample[name].detach().cpu().contiguous()
            sha.update(str((name, tensor.shape, tensor.dtype)).encode())
            sha.update(tensor.view(torch.uint8).numpy().tobytes())
    return dict(schema='window-memory-v1', video_ids=ids, sha256=sha.hexdigest(),
                source='train manifest only', selection='first distinct IDs in seeded shard order',
                schedule='global shuffled stream strided by rank; no epoch-tail drop')


class MemorizationDataset(IterableDataset):
    def __init__(self, samples, seed, rank, world):
        super().__init__()
        if not samples or not 0 <= rank < world:
            raise ValueError('invalid memory dataset/rank')
        self.samples, self.seed, self.rank, self.world = samples, seed, rank, world

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError('memorization replay requires num_workers=0')
        cursor, epoch = 0, 0
        while True:
            order = list(range(len(self.samples)))
            random.Random(self.seed + epoch).shuffle(order)
            for index in order:
                if cursor % self.world == self.rank:
                    yield self.samples[index]
                cursor += 1
            epoch += 1
