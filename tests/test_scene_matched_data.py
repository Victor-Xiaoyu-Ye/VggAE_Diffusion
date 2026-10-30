"""CPU contracts for paired bank, fixed fit/dev probes, resumable dual outputs."""
import argparse
import json
import os
from pathlib import Path
import socket

import pytest
import torch

import prepare_scene_matched as matched
from utils.scene_run import ArtifactStore


def config():
    return dict(train_windows=2, dev_cases=1, train_eval_cases=1,
                views=3, history=1, size=14, grid=1, sample_steps=2,
                eval_seeds=[101, 202])


def sample(ident, value=80):
    target = torch.full((3, 1, 4), .1)
    return dict(id=ident, dataset='spatialvid', frame_ids=[0, 2, 4],
                target=target, anchor=target[:1].clone(),
                raw=torch.full((3, 3, 518, 518), value, dtype=torch.uint8))


def fixtures(tmp_path):
    source = ArtifactStore(tmp_path/'source', [])
    generator = ArtifactStore(tmp_path/'generator', [])
    codec = ArtifactStore(tmp_path/'codec', [])
    weights = generator.save('train/weights_step0031069.pt',
        dict(step=31069, model_args={}, model={}), 'original-generator')
    ae = codec.save('ae.pt', dict(model={}, spec=dict(source_config={}, channels=4)), 'original-ae')
    source.write_json('contract.json', dict(identity='source', source_weights={k: weights[k] for k in ('file','identity','size','sha256')},
        ae={k: ae[k] for k in ('file','identity','size','sha256')}, model={}, mean=[0]*4, std=[1]*4))
    source.save('data/shards/a.pt', dict(samples=[sample('train0'), sample('train1', 100)]), 'source')
    source.save('data/eval.pt', dict(samples=[sample('dev0', 120)]), 'source')
    source.write_json('data/cache_complete.json', dict(identity='source', eval_cases=1,
        shards=[dict(file='data/shards/a.pt', count=2, ids=['train0', 'train1'])]))
    cohort = tmp_path/'cohort.json'
    cohort.write_text(json.dumps(dict(seed=42, records=[dict(id=name, dataset='spatialvid',
        split='train' if name.startswith('train') else 'val', kind='video', root='unused.mp4')
        for name in ('train0','train1','dev0')])), encoding='utf8')
    encoder = tmp_path/'encoder.pt'
    encoder.write_bytes(b'existing-pinned-encoder')
    out = tmp_path/'output'
    roots = [str(tmp_path/'mirror1'), str(tmp_path/'mirror2')]
    store = ArtifactStore(out, roots)
    a = argparse.Namespace(cohort=str(cohort), encoder=str(encoder),
        source_local=str(source.local), source_root=[], output=str(out), root=roots)
    return a, source, store, generator, codec


def patch_features(monkeypatch):
    calls = []
    monkeypatch.setattr(matched, 'code_signature', lambda: 'test-code')
    monkeypatch.setattr(matched, 'make_encoder', lambda *_: object())
    def extract(_encoder, frames):
        calls.append(frames.detach().clone())
        # Deliberately non-normalized to catch accidental preprocessing of
        # frozen targets; LayerNorm belongs to the learned codec.
        return {name: torch.full((1, frames.shape[1], 1, dim), float(frames.mean())*10)
                for name, dim in (('dino',1024),('stream',2048))}
    monkeypatch.setattr('models.matched_video_rae.extract_matched_features', extract)
    return calls


def test_paired_prepare_resume_and_mirror_recovery(tmp_path, monkeypatch):
    a, source, store, _, _ = fixtures(tmp_path)
    calls = patch_features(monkeypatch)
    matched.prepare(a, config(), store, source, torch.device('cpu'), 0, 1)
    bank = store.json('data/bank.json', True)
    assert len(bank['train_eval']) == 1
    assert bank['train_eval'][0]['id'] in ('train0', 'train1')
    assert {v['id'] for v in bank['train']}.isdisjoint(v['id'] for v in bank['dev'])
    assert len(calls) == 6
    for row in bank['train']+bank['dev']:
        value = store.load(row['file'], bank['identity'], True)
        assert value['dino'].shape == (3, 1, 1024)
        assert value['stream'].shape == (3, 1, 2048)
        assert value['dino'].dtype == torch.bfloat16
        assert ('source_sample' in value) == row['source_audit']
        rgb = value['raw'].float().div(255).unsqueeze(0)
        assert any(torch.equal(rgb, image) for image in calls)
        for root in a.root:
            assert (Path(root)/row['file']).is_file()
    missing = store.local/bank['train'][0]['file']
    missing.unlink()
    # A completed resume must never instantiate the multi-billion encoder.
    monkeypatch.setattr(matched, 'make_encoder', lambda *_: pytest.fail('Encoder loaded on completed resume'))
    matched.prepare(a, config(), store, source, torch.device('cpu'), 0, 1)
    assert missing.is_file()
    assert len(calls) == 6


def test_shared_data_skip_does_not_hide_output_failure(tmp_path, monkeypatch):
    a, source, store, _, _ = fixtures(tmp_path)
    patch_features(monkeypatch)
    samples = [sample('train0'), sample('train1')]
    samples[1]['raw'] = samples[1]['raw'][:, :, :10, :10]
    source.save('data/shards/a.pt', dict(samples=samples), 'source')
    matched.prepare(a, config(), store, source, torch.device('cpu'), 0, 1)
    bank = store.json('data/bank.json', True)
    assert len(bank['train']) == 1 and len(bank['bad']) == 1
    assert bank['bad'][0]['expected_id'] == 'train1'
    # The same sample is absent for both representation arms, not separate masks.
    assert bank['train'][0]['file'] == 'data/train/000000.pt'
    other = ArtifactStore(tmp_path/'failing-output', [])
    def fail_save(*_args, **_kwargs):
        raise RuntimeError('durable output unavailable')
    monkeypatch.setattr(other, 'save', fail_save)
    with pytest.raises(RuntimeError, match='durable output unavailable'):
        matched.prepare(a, config(), other, source, torch.device('cpu'), 0, 1)
    assert other.json('data/bank.json') is None


def test_fixed_unique_train_probes_and_overlap_rejected():
    cfg = dict(config(), train_windows=3, train_eval_cases=2)
    manifest = dict(eval_cases=1, shards=[dict(file='x', count=3, ids=['a','a','b'])])
    jobs = matched.job_references(manifest, cfg)
    assert {j['expected_id'] for j in jobs if j['split']=='train' and j['source_audit']} == {'a','b'}
    assert sum(j['source_audit'] for j in jobs if j['split']=='train') == 2
    plan = dict(identity='id', schema='test', limits=matched.limits(cfg),
                source_contract=dict(identity='source'), feature_dims=dict(dino=1024,stream=2048))
    rows = [dict(j, id='a') for j in jobs]
    with pytest.raises(ValueError, match='leakage'):
        matched.bank_manifest(plan, [dict(records=rows, bad=[])])


def test_completed_bank_keeps_skips_when_rgb_becomes_readable(tmp_path, monkeypatch):
    a, source, store, _, _ = fixtures(tmp_path)
    patch_features(monkeypatch)
    original = matched.source_raw
    def temporarily_missing(sample, *args):
        if sample['id']=='train1':
            raise OSError('temporarily unavailable video')
        return original(sample,*args)
    monkeypatch.setattr(matched,'source_raw',temporarily_missing)
    matched.prepare(a,config(),store,source,torch.device('cpu'),0,1)
    before=store.json('data/bank.json',True)
    assert len(before['bad'])==1
    monkeypatch.setattr(matched,'source_raw',original)
    monkeypatch.setattr(matched,'make_encoder',lambda *_: pytest.fail('Immutable bank was expanded'))
    matched.prepare(a,config(),store,source,torch.device('cpu'),0,1)
    assert before==store.json('data/bank.json',True)


def test_prefix_replacement_prevents_future_condition_leakage():
    full = {name: torch.arange(12.).reshape(1,3,2,2) for name in ('dino','stream')}
    independent = {name: torch.ones(1,1,2,2) for name in ('dino','stream')}
    values, report = matched.prefix_safe_features(full, independent, 1)
    for name in ('dino','stream'):
        assert torch.equal(values[name][:,:1], independent[name])
        assert torch.equal(values[name][:,1:], full[name][:,1:])
        assert report[name]['relative_rms'] > 0
        assert report[name]['independent_prefix_used']


def test_encoder_or_code_change_refuses_stale_resume(tmp_path, monkeypatch):
    a, source, store, _, _ = fixtures(tmp_path)
    patch_features(monkeypatch)
    matched.prepare(a, config(), store, source, torch.device('cpu'), 0, 1)
    Path(a.encoder).write_bytes(b'different-encoder')
    with pytest.raises(ValueError, match='fresh matched namespace'):
        matched.prepare(a, config(), store, source, torch.device('cpu'), 0, 1)


def test_original_audit_uses_saved518_inputs_and_sealed_resume(tmp_path, monkeypatch):
    a, source, bank_store, generator, codec = fixtures(tmp_path)
    patch_features(monkeypatch)
    matched.prepare(a, config(), bank_store, source, torch.device('cpu'), 0, 1)
    a.bank_local, a.bank_root = str(bank_store.local), bank_store.roots
    a.generator_local, a.generator_root = str(generator.local), []
    a.codec_local, a.codec_root = str(codec.local), []
    audit_store = ArtifactStore(tmp_path/'audit', [str(tmp_path/'audit-mirror')])
    class EmptyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    monkeypatch.setattr('models.scene_flow.SceneFlow', EmptyModel)
    monkeypatch.setattr('models.scene_rae.SceneRAE', EmptyModel)
    calls = []
    def fake_sample(_model, _rae, item, _mean, _std, _device, history, steps, seed):
        calls.append((item['id'], history, seed))
        assert item['raw'].shape == (3,3,518,518)
        assert item['target'].shape == (3,1,4)
        metrics = dict(raw_l1=.2, ae_l1=.1, copy_l1=.3, future_geo_mse=.5,
            future_texture_mse=.6, generated_motion_l1=.2, gt_motion_l1=.2,
            generated_laplacian_abs=.3, nominal_fps=8.)
        return metrics, torch.zeros(3,2,8,3), torch.zeros(2,1,4)
    monkeypatch.setattr('utils.scene_prefix_eval.sample_case', fake_sample)
    def fake_video(path, _frames, fps):
        assert _frames.shape[1] == 34  # new label strip; original metric RGB is untouched
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'test-video')
    monkeypatch.setattr(matched, 'video', fake_video)
    matched.audit(a, config(), audit_store, source, torch.device('cpu'), 0, 1)
    summary = audit_store.json('summary.json', True)
    assert summary['status'] == 'completed' and summary['completed'] == 4
    assert summary['train_eval']['cases_seeds'] == 2
    expected_train = bank_store.json('data/bank.json', True)['train_eval'][0]['id']
    assert {c[0] for c in calls} == {expected_train,'dev0'}
    assert {c[1] for c in calls} == {1}
    assert {c[2] for c in calls} == {101,202}
    matched.audit(a, config(), audit_store, source, torch.device('cpu'), 0, 1)
    assert len(calls) == 4


def _distributed_prepare_worker(rank, args, source_path, cfg, port):
    import torch.distributed as dist
    import models.matched_video_rae as codec_module
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank), GROUP_RANK='0',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=2)
    matched.code_signature = lambda: 'distributed-test-code'
    matched.make_encoder = lambda *_: object()
    def extract(_encoder, frames):
        return {name: torch.ones(1, frames.shape[1], 1, dim)
                for name, dim in (('dino',1024),('stream',2048))}
    codec_module.extract_matched_features = extract
    a = argparse.Namespace(**args)
    try:
        matched.prepare(a, cfg, ArtifactStore(a.output, a.root),
            ArtifactStore(source_path, []), torch.device('cpu'), rank, 2)
    finally:
        dist.destroy_process_group()


def test_two_rank_preparation_commits_one_shared_bank(tmp_path):
    a, source, store, _, _ = fixtures(tmp_path)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(_distributed_prepare_worker,
        args=(vars(a), str(source.local), config(), port), nprocs=2, join=True)
    bank = store.json('data/bank.json', True)
    assert len(bank['train']) == 2 and len(bank['dev']) == 1
    assert len({row['file'] for row in bank['train']+bank['dev']}) == 3
    assert store.json('prepare/rank000.json', True)['completed'] == 2
    assert store.json('prepare/rank001.json', True)['completed'] == 1
    for row in bank['train']+bank['dev']:
        for root in a.root:
            assert (Path(root)/row['file']).is_file()
