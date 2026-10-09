"""CPU inference checks for the saved SceneFlow online/legacy-EMA audit."""
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import audit_scene_flow_weights as audit
from audit_scene_flow_weights import audit_case, jobs_for_rank, profile_checkpoints, select_cases
from models.scene_flow import SceneFlow
from utils.scene_run import ArtifactStore, digest
from utils.window_flow import WindowFlow


class FixtureRAE(nn.Module):
    grid = 2
    channels = 4

    def decode(self, latent):
        return latent[..., :3].sigmoid()


def fixture_video(path, frames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'fixture-rgb\n' + frames.numpy().tobytes())


def source_hashes(source):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (source.local, Path(source.roots[0]))
        for path in folder.rglob('*') if path.is_file()}


def distributed_audit_worker(rank, root, arguments):
    import torch.distributed as dist
    torch.set_num_threads(1)
    os.environ.update(LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE='2', GROUP_RANK='0')
    root = Path(root)
    dist.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    source_loads, staged_reads = [], []
    native_load, native_staged = ArtifactStore.load, audit.load_staged
    def record_source_load(store, name, *args, **kwargs):
        if '_inputs' in store.local.parts:
            source_loads.append(name)
        return native_load(store, name, *args, **kwargs)
    def record_staged(path):
        staged_reads.append(Path(path).relative_to(Path(arguments['output'])/'_inputs/node0').as_posix())
        return native_staged(path)
    try:
        with patch.object(audit, 'SceneRAE', lambda *args: FixtureRAE()), \
             patch.object(audit, 'video', fixture_video), \
             patch.object(ArtifactStore, 'load', record_source_load), \
             patch.object(audit, 'load_staged', record_staged):
            report = audit.run(SimpleNamespace(**arguments), torch.device('cpu'), rank, 2)
        (root/f'worker{rank}.json').write_text(json.dumps(
            dict(report=report, source_loads=source_loads, staged_reads=staged_reads)))
    finally:
        dist.destroy_process_group()


class SceneFlowAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        torch.manual_seed(17)
        model = SceneFlow(channels=4, grid=2, width=24, depth=1,
            head_width=48, head_depth=1, heads=4, text_dim=8,
            checkpoint_blocks=False).eval()
        # Nonzero gates and output make this an actual conditional denoiser,
        # instead of the trivial zero output of an untrained DiT initializer.
        for parameter in model.parameters():
            nn.init.normal_(parameter, std=.1)
        anchor = torch.randn(1, 4, 4)
        target = torch.cat((anchor, torch.randn(2, 4, 4)), 0)
        rays = torch.randn(3, 4, 8)
        rays[..., 7] = 1
        sample = dict(id='fixture:scene', dataset='fixture', anchor=anchor,
            target=target, rays=rays, text=torch.randn(2, 8), has_caption=True,
            raw=torch.randint(0, 256, (3, 3, 2, 2), dtype=torch.uint8))
        return model, FixtureRAE(), sample, torch.tensor([.1, -.2, .3, 0.]), torch.tensor([.5, 1., 2., .7])

    def prepare_source(self, root):
        model, rae, sample, mean, std = self.fixture()
        source = ArtifactStore(root/'source', [str(root/'source_mirror')])
        pipeline_id, ae_id, cache_id = 'fixture-pipeline', 'fixture-ae', 'fixture-cache'
        source.write_json('contract.json', dict(identity=pipeline_id))
        ae_receipt = source.save('ae/best_joint.pt',
            dict(spec=dict(source_config={}, channels=4), model=rae.state_dict()), ae_id)
        source.write_json('ae/complete.json', dict(identity=ae_id, selected='ae/best_joint.pt'))
        source.write_json('cache/complete.json', dict(identity=cache_id, pipeline=pipeline_id,
            ae=ae_receipt['sha256'], mean=mean.tolist(), std=std.tolist()))
        source.save('cache/eval.pt', dict(samples=[sample]), cache_id)
        model_args = model.args
        flow_id = digest(dict(pipeline=pipeline_id, cache=cache_id, stage='video', model=model_args))
        source.write_json('video/model.json', dict(model=model_args, identity=flow_id))
        source.save('video/checkpoint_final.pt', dict(model=model.state_dict(),
            ema={name: value.bfloat16() for name, value in model.state_dict().items()},
            model_args=model_args, step=137104), flow_id)
        args = SimpleNamespace(source_local=str(source.local), source_root=source.roots,
            output=str(root/'audit'), root=[str(root/'audit_primary'), str(root/'audit_mirror')],
            profile='smoke', sample_steps=8, sample_method='euler', cases_per_domain=1, max_cases=0)
        return source, args, flow_id

    def legacy_sample(self, model, sample, index, stage, mean, std, steps=8):
        anchor = (sample['anchor'].unsqueeze(0).float() - mean) / std
        text = sample['text'].unsqueeze(0).float()
        valid = torch.ones(text.shape[:2], dtype=torch.bool)
        image_task = stage == 'image'
        frames = 1 if image_task else sample['target'].shape[0] - 1
        rays = None if image_task else sample['rays'][1:].unsqueeze(0)
        present = torch.tensor([not image_task])
        generator = torch.Generator().manual_seed(101 + index)
        noise = torch.randn((1, frames, 4, 4), generator=generator)
        def net(z, u, c, t, v):
            return model(z, u, c, t, v, ref_present=present, rays=rays)
        return WindowFlow().sample(net, anchor, noise, steps=steps,
            dtype=torch.float32, text=text, text_valid=valid)

    def test_selection_keeps_original_indices_for_seeds(self):
        samples = [dict(id=str(i), dataset=source)
                   for i, source in enumerate(('a', 'a', 'b', 'c', 'b', 'a'))]
        selected = select_cases(samples, per_domain=1)
        self.assertEqual(len(selected), 3)
        self.assertEqual({sample['dataset'] for _, sample in selected}, {'a', 'b', 'c'})
        for index, sample in selected:
            self.assertIs(sample, samples[index])
        self.assertEqual(len(select_cases(samples, per_domain=2, max_cases=3)), 3)
        self.assertEqual(select_cases([], per_domain=2), [])

    def test_jobs_cover_each_case_and_weight_source_once(self):
        selected = [(2, dict(id='two')), (7, dict(id='seven')), (11, dict(id='eleven'))]
        for world in (1, 2, 4, 8):
            with self.subTest(world=world):
                partitions = [jobs_for_rank(selected, rank, world) for rank in range(world)]
                jobs = [job for partition in partitions for job in partition]
                self.assertEqual(len(jobs), 6)
                self.assertEqual({(index, source) for index, _, source in jobs},
                    {(index, source) for index, _ in selected for source in ('online', 'legacy_ema')})
                self.assertLessEqual(max(map(len, partitions)) - min(map(len, partitions)), 1)
                for index, sample, _ in jobs:
                    self.assertIs(sample, dict(selected)[index])

    def test_profiles_keep_smoke_small_and_include_early_middle_final(self):
        smoke = profile_checkpoints('smoke')
        self.assertEqual(len(smoke), 1)
        self.assertEqual(smoke[0]['stage'], 'video')
        self.assertTrue(smoke[0]['file'].endswith('checkpoint_final.pt'))
        standard = profile_checkpoints('standard')
        self.assertEqual(len(standard), 4)
        self.assertEqual(sum(item['stage'] == 'image' for item in standard), 1)
        files = {item['file'] for item in standard}
        self.assertIn('video/weights_step0009185.pt', files)
        self.assertIn('video/weights_step0071609.pt', files)
        self.assertIn('video/checkpoint_final.pt', files)
        self.assertIn('image/checkpoint_final.pt', files)

    def test_online_and_legacy_weights_use_the_original_noise_and_sampling_contract(self):
        model, rae, sample, mean, std = self.fixture()
        online = copy.deepcopy(model.state_dict())
        legacy_ema = {name: value.bfloat16() for name, value in online.items()}
        for state in (online, legacy_ema):
            model.load_state_dict(state)
            for stage in ('image', 'video'):
                with self.subTest(stage=stage, dtype=state['input.weight'].dtype):
                    expected = self.legacy_sample(model, sample, 7, stage, mean, std)
                    first = audit_case(model, rae, sample, 7, stage, torch.device('cpu'), mean, std, steps=8)
                    second = audit_case(model, rae, sample, 7, stage, torch.device('cpu'), mean, std, steps=8)
                    torch.testing.assert_close(first[2], expected, rtol=0, atol=0)
                    torch.testing.assert_close(second[2], first[2], rtol=0, atol=0)
                    torch.testing.assert_close(second[1], first[1], rtol=0, atol=0)
                    self.assertEqual(first[0]['seed'], 108)
                    self.assertTrue(first[3])

    def test_image_generation_has_no_reference_image_leak(self):
        model, rae, sample, mean, std = self.fixture()
        changed = dict(sample, anchor=sample['anchor'] * 100 + 50)
        first = audit_case(model, rae, sample, 3, 'image', torch.device('cpu'), mean, std, steps=8)
        second = audit_case(model, rae, changed, 3, 'image', torch.device('cpu'), mean, std, steps=8)
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        torch.testing.assert_close(first[1][..., -2:, :], second[1][..., -2:, :], rtol=0, atol=0)
        self.assertEqual(tuple(first[1].shape), (1, 2, 6, 3))

    def test_video_anchor_is_reconstruction_and_scores_only_future(self):
        model, rae, sample, mean, std = self.fixture()
        metrics, preview, normalized, _ = audit_case(
            model, rae, sample, 4, 'video', torch.device('cpu'), mean, std, steps=8)
        self.assertEqual(tuple(preview.shape), (3, 2, 6, 3))
        generated = preview[:, :, 4:, :]
        expected_anchor = rae.decode(sample['anchor'].reshape(1, 1, 2, 2, 4))[0, 0]
        torch.testing.assert_close(generated[0], expected_anchor)
        future = rae.decode((normalized * std + mean).reshape(1, 2, 2, 2, 4))[0]
        torch.testing.assert_close(generated[1:], future)
        raw = sample['raw'].float().permute(0, 2, 3, 1) / 255
        self.assertAlmostEqual(metrics['raw_l1'], (future - raw[1:]).abs().mean().item(), places=6)
        self.assertAlmostEqual(metrics['copy_l1'], (raw[:1] - raw[1:]).abs().mean().item(), places=6)
        self.assertEqual(metrics['metric_scope'], 'future frames only')
        self.assertTrue(metrics['camera_present'])

    def test_saved_artifact_run_resume_mirror_recovery_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, args, flow_id = self.prepare_source(root)
            plan = audit.build_plan(source, args)
            self.assertEqual(plan['checkpoints'][0]['receipt']['identity'], flow_id)
            before = source_hashes(source)
            with patch.object(audit, 'SceneRAE', lambda *args: FixtureRAE()), \
                 patch.object(audit, 'video', fixture_video), \
                 patch.dict(os.environ, dict(LOCAL_RANK='0', LOCAL_WORLD_SIZE='1', GROUP_RANK='0')):
                with patch.object(audit, 'audit_case', wraps=audit_case) as generate:
                    first = audit.run(args, torch.device('cpu'), 0, 1)
                    self.assertEqual(generate.call_count, 2)
                self.assertEqual(first['status'], 'completed')
                self.assertEqual(first['completed'], 2)
                self.assertEqual({row['weight_source'] for row in first['samples']}, {'online', 'legacy_ema'})
                self.assertEqual(source_hashes(source), before)
                staged = Path(args.output)/'_inputs/node0'
                self.assertFalse((staged/'video/checkpoint_final.pt').exists())
                self.assertFalse((staged/'video/checkpoint_final.pt.json').exists())
                self.assertTrue((staged/'cache/eval.pt').is_file())
                self.assertTrue((staged/'ae/best_joint.pt').is_file())
                self.assertTrue((source.local/'video/checkpoint_final.pt').is_file())
                with patch.object(audit, 'audit_case', side_effect=AssertionError('must resume committed cases')):
                    second = audit.run(args, torch.device('cpu'), 0, 1)
                self.assertEqual(second['status'], 'completed')
                self.assertEqual(second['samples'], first['samples'])
                relative = first['samples'][0]['path'] + '/preview.mp4'
                local_preview = Path(args.output)/relative
                expected_bytes = local_preview.read_bytes()
                local_preview.write_bytes(b'corrupt-local')
                (Path(args.root[0])/relative).write_bytes(b'corrupt-primary')
                with patch.object(audit, 'audit_case', side_effect=AssertionError('must recover mirror')):
                    third = audit.run(args, torch.device('cpu'), 0, 1)
                self.assertEqual(third['status'], 'completed')
                self.assertEqual(local_preview.read_bytes(), expected_bytes)
                self.assertEqual((Path(args.root[0])/relative).read_bytes(), expected_bytes)
                self.assertEqual((Path(args.root[1])/relative).read_bytes(), expected_bytes)
                self.assertEqual(source_hashes(source), before)
                complete = json.loads((Path(args.output)/'complete.json').read_text())
                self.assertEqual(complete['completed'], 2)
                self.assertEqual(complete['errors'], 0)
            # A checkpoint copied from another pipeline must not enter this
            # audit just because its model dimensions happen to match.
            foreign = source.load('video/checkpoint_final.pt', flow_id, required=True)
            source.save('video/checkpoint_final.pt', foreign, 'another-pipeline-flow')
            with self.assertRaisesRegex(ValueError, 'Source flow belongs to another pipeline/cache'):
                audit.build_plan(source, args)

    def test_two_rank_gloo_audit_shares_staging_and_gathers_distinct_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, args, _ = self.prepare_source(root)
            before = source_hashes(source)
            torch.multiprocessing.spawn(distributed_audit_worker,
                args=(str(root), vars(args)), nprocs=2, join=True)
            workers = [json.loads((root/f'worker{rank}.json').read_text()) for rank in range(2)]
            expected_assets = {'cache/eval.pt', 'ae/best_joint.pt', 'video/checkpoint_final.pt'}
            self.assertEqual(set(workers[0]['source_loads']), expected_assets)
            self.assertEqual(len(workers[0]['source_loads']), 3)
            self.assertEqual(workers[1]['source_loads'], [])
            for worker in workers:
                self.assertEqual(set(worker['staged_reads']), expected_assets)
                self.assertEqual(worker['report']['status'], 'completed')
                self.assertEqual(worker['report']['completed'], 2)
                self.assertEqual(worker['report']['errors'], 0)
                rows = worker['report']['samples']
                self.assertEqual(len({row['path'] for row in rows}), 2)
                self.assertEqual({(row['rank'], row['weight_source']) for row in rows},
                                 {(0, 'online'), (1, 'legacy_ema')})
            self.assertEqual(source_hashes(source), before)
            self.assertFalse((Path(args.output)/'_inputs/node0/video/checkpoint_final.pt').exists())
            complete = json.loads((Path(args.output)/'complete.json').read_text())
            self.assertEqual(complete['completed'], 2)


if __name__ == '__main__':
    unittest.main()
