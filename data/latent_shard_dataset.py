import io
import os
import random
import tarfile

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from utils.moxing_io import (
    is_remote_path,
    join_remote,
    open_file,
    read_text,
)


def read_shard_manifest(manifest_path):
    remote_manifest = is_remote_path(manifest_path)
    base_dir = (
        manifest_path.rsplit("/", 1)[0]
        if remote_manifest else os.path.dirname(os.path.abspath(manifest_path))
    )
    shards = []
    for line in read_text(manifest_path).splitlines():
        path = line.strip()
        if not path or path.startswith("#"):
            continue
        if not is_remote_path(path) and not os.path.isabs(path):
            path = (
                join_remote(base_dir, path)
                if remote_manifest else os.path.join(base_dir, path)
            )
        shards.append(path)
    if not shards:
        raise ValueError(f"No latent shards found in {manifest_path}")
    return shards


class LatentShardDataset(IterableDataset):
    """Stream compact latent samples from tar shards.

    Each tar member must be a torch-serialized ``.pt`` dictionary containing
    ``target`` [S,N,D], ``cond`` [1,N,D], and optional caption/video metadata.
    Shards are divided across DDP ranks and DataLoader workers without overlap.
    """

    def __init__(self, manifest_path, shuffle_buffer=256, seed=42, repeat=True,
                 rank=None, world_size=None):
        super().__init__()
        self.shards = read_shard_manifest(manifest_path)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.repeat = repeat
        self.rank = rank
        self.world_size = world_size

    def _consumer_info(self):
        rank = self.rank
        world_size = self.world_size
        if rank is None:
            rank = (
                dist.get_rank()
                if dist.is_available() and dist.is_initialized() else 0)
        if world_size is None:
            world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized() else 1)
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        consumer_id = rank * num_workers + worker_id
        num_consumers = world_size * num_workers
        return consumer_id, num_consumers

    def _iter_shard(self, path):
        remote = is_remote_path(path)
        source = open_file(path, "rb")
        tar_mode = "r|*" if remote else "r:*"
        with source, tarfile.open(fileobj=source, mode=tar_mode) as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".pt"):
                    continue
                fileobj = archive.extractfile(member)
                if fileobj is None:
                    continue
                sample = torch.load(
                    io.BytesIO(fileobj.read()), map_location="cpu",
                    weights_only=False)
                if "target" not in sample or "cond" not in sample:
                    raise KeyError(
                        f"{path}:{member.name} must contain target and cond")
                yield sample

    @staticmethod
    def _shuffle_stream(stream, buffer_size, rng):
        if buffer_size <= 1:
            yield from stream
            return

        buffer = []
        for sample in stream:
            if len(buffer) < buffer_size:
                buffer.append(sample)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = sample
        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self):
        consumer_id, num_consumers = self._consumer_info()
        shards = self.shards[consumer_id::num_consumers]
        if not shards:
            raise RuntimeError(
                f"Consumer {consumer_id}/{num_consumers} received no shards. "
                "Reduce DataLoader workers or create more cache shards.")

        epoch = 0
        while True:
            rng = random.Random(self.seed + consumer_id + epoch * num_consumers)
            ordered_shards = list(shards)
            rng.shuffle(ordered_shards)

            def sample_stream():
                for shard in ordered_shards:
                    yield from self._iter_shard(shard)

            yield from self._shuffle_stream(
                sample_stream(), self.shuffle_buffer, rng)
            epoch += 1
            if not self.repeat:
                break


def latent_collate_fn(batch):
    target = torch.stack([sample["target"] for sample in batch])
    cond = torch.stack([sample["cond"] for sample in batch])
    result = {
        "target": target,
        "cond": cond,
        "caption": [sample.get("caption", "") for sample in batch],
        "video_id": [sample.get("video_id", "") for sample in batch],
    }
    if all("i0_rgb" in sample for sample in batch):
        result["i0_rgb"] = torch.stack(
            [sample["i0_rgb"] for sample in batch])
    return result
