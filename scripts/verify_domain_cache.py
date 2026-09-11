"""Require complete frozen cohort caches before domain training."""
import argparse
import io
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from utils.moxing_io import read_bytes


def validate_stats(stats, expected_sha, count, windows, training=False):
    cfg = stats['config']
    if (stats['num_samples'] != count or stats.get('num_failed', 0) != 0
            or cfg.get('csv_sha256') != expected_sha
            or cfg.get('clips_per_video') != windows or cfg.get('decode_retries') != 0
            or not cfg.get('independent_anchor') or cfg.get('window_ae_norm') != 'legacy'
            or (not training and not cfg.get('store_rgb'))):
        raise ValueError('Incomplete or incompatible domain cache')
    if training and stats['num_shards'] < 48:
        raise ValueError('Need at least 48 train shards')


def verify(root, manifest, arm):
    records = json.loads(Path(manifest).read_text())['files']
    for suffix, name, count, windows in [('train',f'train_{arm}_2048',8192,4),
            ('eval','eval_street_128',128,1), ('other/eval','eval_other_128',128,1)]:
        stats = torch.load(io.BytesIO(read_bytes(root.rstrip('/')+'/'+suffix+'/stats.pt')), map_location='cpu', weights_only=False)
        validate_stats(stats, records[name]['sha256'], count, windows, suffix == 'train')
        print(f'PASS {suffix}: {count} samples, frozen CSV identity, no failures/replacements')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--domain_manifest',required=True)
    p.add_argument('--arm',choices=['single','mixed'],required=True)
    a=p.parse_args();verify(a.root,a.domain_manifest,a.arm)
