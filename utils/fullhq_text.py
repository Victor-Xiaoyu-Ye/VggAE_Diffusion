"""Content-checked lazy text shards for full-HQ TI2V; no whole-bank scan."""
import hashlib
import io
import json
from collections import OrderedDict

import torch
from scripts.window_run_io import child
from utils.moxing_io import read_bytes
from utils.window_training import CaptionBank


def checked_bytes(root, name, expected):
    data = read_bytes(child(root, name))
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('text artifact checksum mismatch: '+name)
    return data


class ShardedCaptionBank(CaptionBank):
    def __init__(self, root, max_shards=8):
        success = json.loads(read_bytes(child(root, '_SUCCESS')))
        if success['schema'] != 'fullhq-text-v1':
            raise ValueError('full-HQ requires versioned text manifest')
        self.manifest = json.loads(checked_bytes(root, 'index.json', success['index_sha256']))
        self.values = self.manifest['index']
        self.root, self.max_shards = root, max_shards
        self.cache = OrderedDict()
        self.caption_cache = {}
        self.empty = self._load('empty_prompt.pt').float()
        self._check(self.empty)
        self.signature = success['index_sha256']

    def _check(self, value):
        if value.ndim != 2 or value.shape[1] != 4096 or not 0 < len(value) <= self.manifest['text_len'] or not torch.isfinite(value).all():
            raise ValueError('invalid text tensor')

    def _load(self, name):
        data = checked_bytes(self.root, name, self.manifest['files'][name])
        return torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)

    def get(self, key):
        name = self.values[key]
        if name is None:
            return self.empty
        if name not in self.cache:
            part = self._load(name)
            if 'embeddings' in part:
                self.caption_cache[name] = part['captions']
                part = part['embeddings']
            for value in part.values(): self._check(value)
            self.cache[name] = part
            if len(self.cache) > self.max_shards:
                evicted,_=self.cache.popitem(last=False)
                self.caption_cache.pop(evicted,None)
        self.cache.move_to_end(name)
        return self.cache[name][key]

    def caption(self,key):
        if self.values[key] is None:return ''
        self.get(key)
        return self.caption_cache.get(self.values[key],{}).get(key,'')


class TextCFG:
    """Text guidance while retaining the same image condition in both calls."""
    def __init__(self, model, empty, scale):
        self.model, self.empty, self.scale = model, empty, float(scale)

    def __call__(self, noisy, u, anchor, text, text_valid):
        conditional = self.model(noisy, u, anchor, text, text_valid)
        empty = self.empty.to(noisy.device).unsqueeze(0).expand(len(noisy), -1, -1)
        valid = torch.ones(empty.shape[:2], dtype=torch.bool, device=noisy.device)
        uncond = self.model(noisy, u, anchor, empty, valid)
        return uncond.float()+self.scale*(conditional.float()-uncond.float())
