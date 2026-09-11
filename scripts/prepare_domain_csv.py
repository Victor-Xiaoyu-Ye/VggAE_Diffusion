"""Materialize frozen CSV-only domain cohorts; no video or annotation downloads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def materialize(metadata, selection, output):
    metadata, selection, output = map(Path, (metadata, selection, output))
    plan = json.loads(selection.read_text(encoding='utf-8'))
    source_hash = hashlib.sha256(metadata.read_bytes()).hexdigest()
    if source_hash != plan['metadata_sha256']:
        raise ValueError('Metadata hash differs from frozen domain selection')
    splits = plan['splits']
    sets = {k: set(v) for k, v in splits.items()}
    for k, ids in splits.items():
        if len(ids) != len(sets[k]):
            raise ValueError(f'Duplicate ID in {k}')
    for a in sets:
        for b in sets:
            if a >= b or (a.startswith('train_') and b.startswith('train_')):
                continue
            if sets[a] & sets[b]:
                raise ValueError(f'Clip leakage between {a} and {b}')
    wanted = set.union(*sets.values())
    selected = {}
    with metadata.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            if row['id'] in wanted:
                if row['id'] in selected:
                    raise ValueError('Duplicate ID in source metadata')
                selected[row['id']] = row
    if set(selected) != wanted:
        raise ValueError('Selected IDs absent from metadata')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'metadata_sha256': source_hash, 'selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
                'source_disjoint_verified': False, 'files': {}}
    for name, ids in splits.items():
        target = output / (name + '.csv')
        temporary = target.with_suffix('.csv.tmp')
        with temporary.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected[i] for i in ids)
        temporary.replace(target)
        manifest['files'][name] = {'rows': len(ids), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    (output / 'domain_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(materialize(a.metadata, a.selection, a.output), indent=2))
