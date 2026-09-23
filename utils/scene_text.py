"""Node-shared compact text cache for a small cohort inside the full-HQ bank.

Download an original shard at most once per node/cache identity, retain only
selected video IDs, and release the large source shard. No text re-encoding.
"""
import hashlib
import os
from pathlib import Path
import torch
from utils.fullhq_text import ShardedCaptionBank
from utils.moxing_io import _acquire_lock
from utils.training import atomic_torch_save


class SceneCaptionBank(ShardedCaptionBank):
    def __init__(self, root, ids, local):
        self.ids=set(ids)
        self.local=Path(local)/hashlib.sha256('\n'.join(sorted(self.ids)).encode()).hexdigest()[:16]
        self.local.mkdir(parents=True,exist_ok=True)
        super().__init__(root,max_shards=2)

    def _load(self,name):
        if name=='empty_prompt.pt':
            return super()._load(name)
        key=hashlib.sha256((self.root+name+self.manifest['files'][name]).encode()).hexdigest()
        path=self.local/(key+'.pt')
        def read():
            try:
                return torch.load(path,map_location='cpu',weights_only=False)
            except (OSError,RuntimeError,EOFError):
                return None
        value=read()
        if value is not None:
            return value
        lock=str(path)+'.lock'
        if not _acquire_lock(lock,timeout=1800,stale_seconds=3600):
            raise TimeoutError('Waiting for node-shared caption subset '+name)
        try:
            value=read()
            if value is not None:
                return value
            full=super()._load(name)  # Original SHA256 is checked before deserialization.
            values=full.get('embeddings',full)
            captions=full.get('captions',{})
            # Clone small tensors so torch.save cannot retain source shard storage.
            value=dict(embeddings={k:v.clone() for k,v in values.items() if k in self.ids},
                       captions={k:v for k,v in captions.items() if k in self.ids})
            atomic_torch_save(value,str(path))
            return value
        finally:
            Path(lock).unlink(missing_ok=True)
