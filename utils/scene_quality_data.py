"""Strict, deterministic street-only views of the frozen Scene latent cache.

This module does not coordinate workers or copy OBS objects. Source shards may
be compacted by a caller using ``select_street_samples``. Normalization always
comes from the original cache; filtering never fits new statistics.
"""
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import random


SELECTION_SCHEMA = 'scene-street-ids-v1'
DATASETS = frozenset(('dl3dv', 'scannetppv2', 'mvssynth', 'omniworld',
                      'pointodyssey', 'spring', 'spatialvid'))
DEFAULT_SELECTION = Path(__file__).resolve().parents[1] / 'configs/scene_street_ids_v1.json'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf8')


def _cohort(value):
    if isinstance(value, (str, Path)):
        raw = Path(value).read_bytes()
        return json.loads(raw), _sha(raw)
    return value, None


def _allowed(selection):
    values = selection['allowed_ids'] if isinstance(selection, dict) else selection
    ids = frozenset(values)
    if not ids or any(not isinstance(v, str) or not v.startswith('spatialvid:') for v in ids):
        raise ValueError('Street selection needs nonempty SpatialVID IDs')
    return ids


def load_street_selection(cohort, selection_path=None):
    """Load a versioned ID manifest, or derive it from original role JSONL.

    A dictionary cohort is accepted and checked using its canonical hash. The
    packaged manifest also pins the exact original cohort and selection bytes.
    Returned values are JSON serializable; ``allowed_ids`` are sorted full IDs.
    """
    cohort, cohort_file_sha = _cohort(cohort)
    cohort_sha = _sha(_canonical(cohort))
    path = Path(selection_path) if selection_path is not None else DEFAULT_SELECTION
    raw = path.read_bytes()
    if path.suffix == '.jsonl':
        records = [json.loads(line) for line in raw.decode('utf8').splitlines() if line.strip()]
        eligible = [r['id'] for r in records if r.get('dataset') == 'spatialvid'
                    and r.get('split') == 'train' and r.get('role') == 'real_street_video_candidate']
        value = dict(schema=SELECTION_SCHEMA, allowed_ids=sorted(eligible), count=len(eligible),
                     cohort_canonical_sha256=cohort_sha, cohort_sha256=cohort_file_sha,
                     source_selection_sha256=_sha(raw),
                     source_selection_name=path.name,
                     rule='dataset=spatialvid; split=train; role=real_street_video_candidate')
    else:
        value = json.loads(raw)
        if value.get('schema') != SELECTION_SCHEMA:
            raise ValueError('Unknown street selection schema')
        if value.get('cohort_canonical_sha256') != cohort_sha:
            raise ValueError('Street selection belongs to another cohort')
        if cohort_file_sha is not None and value.get('cohort_sha256') != cohort_file_sha:
            raise ValueError('Street selection cohort file SHA mismatch')
    ids = _allowed(value)
    if len(ids) != len(value['allowed_ids']) or value.get('count') != len(ids):
        raise ValueError('Duplicate IDs or wrong street selection count')
    if value.get('allowed_ids_sha256') not in (None, _sha(_canonical(sorted(ids)))):
        raise ValueError('Street selection ID hash mismatch')
    inputs = cohort.get('input_sha256', {})
    source_name = value.get('source_selection_name')
    if source_name in inputs and value.get('source_selection_sha256') != inputs[source_name]:
        raise ValueError('Street selection source differs from cohort provenance')
    cohort_train = {r['id'] for r in cohort['records']
                    if r.get('dataset') == 'spatialvid' and r.get('split') == 'train'}
    forbidden = {r['id'] for r in cohort['records'] if r.get('split') != 'train'}
    if not ids <= cohort_train or ids & forbidden:
        raise ValueError('Street selection contains non-train or unknown IDs')
    seen_ids = set(ids)
    for split in ('val', 'test'):
        field = split + '_ids'
        if field not in value:
            continue
        heldout = _allowed(value[field])
        permitted = {r['id'] for r in cohort['records']
                     if r.get('dataset') == 'spatialvid' and r.get('split') == split}
        if (len(heldout) != len(value[field]) or not heldout <= permitted
                or seen_ids & heldout):
            raise ValueError(f'Street {split} IDs disagree with the original split')
        if value.get(field + '_sha256') not in (None, _sha(_canonical(sorted(heldout)))):
            raise ValueError(f'Street {split} ID hash mismatch')
        source_name = value.get('source_' + split + '_selection_name')
        if source_name in inputs and value.get('source_' + split + '_selection_sha256') != inputs[source_name]:
            raise ValueError(f'Street {split} source differs from cohort provenance')
        seen_ids.update(heldout)
    value = copy.deepcopy(value)
    value['allowed_ids'] = sorted(ids)
    value['selection_sha256'] = _sha(raw)
    value['allowed_ids_sha256'] = _sha(_canonical(value['allowed_ids']))
    return value


def select_street_samples(samples, selection):
    """Filter by BOTH dataset and the audited role-derived ID allowlist.

    Known auxiliary domains and replay IDs are excluded. Unknown/malformed
    dataset tags fail explicitly, allowing the caller to record a bad shard.
    The returned items are the original objects (no tensor copies).
    """
    allowed = _allowed(selection)
    selected = []
    for sample in samples:
        if not isinstance(sample, dict) or sample.get('dataset') not in DATASETS:
            raise ValueError('Unknown or absent sample dataset')
        ident = sample.get('id')
        if not isinstance(ident, str) or not ident.startswith(sample['dataset'] + ':'):
            raise ValueError('Sample ID and dataset disagree')
        if sample['dataset'] == 'spatialvid' and ident in allowed:
            selected.append(sample)
    return selected


def filter_street_manifest(manifest, selection=None, indexes=None):
    """Return a candidate manifest without modifying source indices/statistics.

    ``indexes`` optionally maps shard filename to allowed payload indices; an
    omitted filename then selects no items. Original ``sources`` and ``count``
    stay intact so indices address the actual payload. ``indices`` selects the
    items and ``selected_count`` reports their total. Without per-shard ``ids``
    or an external exact index, replay is excluded at payload validation, not
    guessed from the source-only manifest. Compact shards should already have
    only allowed IDs. Unknown dataset tags are always rejected.
    """
    result = copy.deepcopy(manifest)
    result['shards'] = []
    allowed = _allowed(selection) if selection is not None else None
    total = 0
    seen = set()
    for original in manifest['shards']:
        name, sources = original['file'], original['sources']
        if name in seen:
            raise ValueError('Duplicate shard filename')
        seen.add(name)
        if any(s not in DATASETS for s in sources):
            raise ValueError('Unknown dataset in shard manifest')
        if original.get('count', len(sources)) != len(sources):
            raise ValueError('Shard count and source index disagree')
        candidates = list(original.get('indices', range(len(sources))))
        if indexes is not None:
            requested = set(indexes.get(name, []))
            if any(type(i) is not int or i < 0 or i >= len(sources) for i in requested):
                raise ValueError('Invalid requested shard index')
            candidates = [i for i in candidates if i in requested]
        if len(candidates) != len(set(candidates)) or any(
                type(i) is not int or i < 0 or i >= len(sources) for i in candidates):
            raise ValueError('Invalid shard indices')
        if 'ids' in original and len(original['ids']) != len(sources):
            raise ValueError('Shard IDs and sources disagree')
        selected = [i for i in candidates if sources[i] == 'spatialvid'
                    and (allowed is None or 'ids' not in original or original['ids'][i] in allowed)]
        if selected:
            shard = copy.deepcopy(original)
            shard['indices'] = selected
            result['shards'].append(shard)
            total += len(selected)
    result['selected_count'] = total
    result['selection_scope'] = 'spatialvid payload indices; ID allowlist required at load'
    return result


class FilteredLatentSamples:
    """Uniform street-only sampling, pure cursor RNG, bounded bad-shard retry.

    No source-weight defaults exist. The bank expects compact street shards or
    an exact ``indices`` view. Every selected payload ID is checked. Persist
    ``state_dict`` alongside the training cursor to retain the bad-shard set on
    resume; a repaired shard is not silently reintroduced into that run.
    """
    def __init__(self, store, manifest, selection, seed=42, on_bad_shard=None,
                 max_attempts=32, max_cached_shards=2):
        if max_attempts < 1 or max_cached_shards < 1:
            raise ValueError('Retry and cache bounds must be positive')
        self.store, self.manifest = store, manifest
        self.allowed_ids = _allowed(selection)
        self.seed, self.max_attempts = int(seed), int(max_attempts)
        self.max_cached_shards, self.on_bad_shard = int(max_cached_shards), on_bad_shard
        self.cache, self.bad_shards = OrderedDict(), {}
        self.shards, self.entries = {}, []
        filtered = filter_street_manifest(manifest, selection)
        for shard in filtered['shards']:
            self.shards[shard['file']] = shard
            self.entries.extend((shard['file'], i) for i in shard['indices'])
        if not self.entries:
            raise ValueError('Street latent manifest has no candidate samples')
        self.identity = _sha(_canonical(dict(cache=manifest['identity'], entries=self.entries,
                                           ids=sorted(self.allowed_ids), seed=self.seed)))

    def __len__(self):
        return len(self.entries)

    def _load(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        payload = self.store.load(name, self.manifest['identity'], required=True)
        samples = payload['samples']
        shard = self.shards[name]
        if not isinstance(samples, (list, tuple)) or len(samples) != len(shard['sources']):
            raise ValueError('Shard payload count differs from manifest')
        for sample, source in zip(samples, shard['sources']):
            if not isinstance(sample, dict) or sample.get('dataset') != source or source not in DATASETS:
                raise ValueError('Shard payload dataset differs from manifest')
        for i in shard['indices']:
            if samples[i]['dataset'] != 'spatialvid' or samples[i].get('id') not in self.allowed_ids:
                raise ValueError('Non-allowlisted sample in street shard')
            if 'ids' in shard and samples[i]['id'] != shard['ids'][i]:
                raise ValueError('Shard payload ID differs from manifest')
        self.cache[name] = samples
        if len(self.cache) > self.max_cached_shards:
            self.cache.popitem(last=False)
        return samples

    def get(self, cursor):
        if type(cursor) is not int or cursor < 0:
            raise ValueError('Cursor must be a nonnegative integer')
        digest = hashlib.sha256(f'{self.seed}:{cursor}'.encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        for attempt in range(self.max_attempts):
            name, index = rng.choice(self.entries)
            if name in self.bad_shards:
                continue
            try:
                return self._load(name)[index]
            except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError, EOFError) as exc:
                error = f'{type(exc).__name__}: {exc}'
                self.bad_shards[name] = error
                if self.on_bad_shard is not None:
                    self.on_bad_shard(dict(file=name, error=error, cursor=cursor,
                                           attempt=attempt, identity=self.manifest['identity']))
        raise RuntimeError(f'No valid street latent after {self.max_attempts} attempts '
                           f'at cursor {cursor}; bad shards={len(self.bad_shards)}')

    def state_dict(self):
        return dict(schema='scene-street-bank-v1', identity=self.identity,
                    max_attempts=self.max_attempts, bad_shards=dict(self.bad_shards))

    def load_state_dict(self, state):
        if state.get('schema') != 'scene-street-bank-v1' or state.get('identity') != self.identity:
            raise ValueError('Street bank state belongs to another selection/cache/seed')
        if state.get('max_attempts') != self.max_attempts:
            raise ValueError('Street bank retry policy changed on resume')
        bad = state.get('bad_shards', {})
        if not isinstance(bad, dict) or not set(bad) <= set(self.shards):
            raise ValueError('Invalid bad-shard state')
        self.bad_shards = dict(bad)
        self.cache.clear()
