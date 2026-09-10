import itertools
import unittest
import tempfile
from pathlib import Path
import torch
from utils.window_memorization import MemorizationDataset, subset_identity
from utils.window_subspace import repairs, removed_component, fit_basis


class MemorySubspaceTests(unittest.TestCase):
    def test_balanced_global_stream_and_replay(self):
        samples = [{'video_id': str(i)} for i in range(64)]
        global_stream = list(itertools.islice(MemorizationDataset(samples, 42, 0, 1), 48*64))
        for rank in range(48):
            stream = list(itertools.islice(MemorizationDataset(samples, 42, rank, 48), 64))
            self.assertEqual(stream, global_stream[rank::48])
        for start in range(0, len(global_stream), 64):
            self.assertEqual(len({x['video_id'] for x in global_stream[start:start+64]}), 64)

    def test_fingerprint_catches_tensor_change(self):
        samples = [dict(video_id='a', cond=torch.zeros(1, 4, 6), target=torch.ones(2, 4, 6))]
        before = subset_identity(samples)
        samples[0]['target'][0, 0, 0] += 1
        self.assertNotEqual(before['sha256'], subset_identity(samples)['sha256'])

    def test_projection_partition_and_matched_residual(self):
        torch.manual_seed(9)
        y, e = torch.randn(1, 4, 324, 40), torch.randn(1, 4, 324, 40)
        basis = torch.eye(40)
        parts = [removed_component(e, 'frequency_'+band, basis) for band in ('low','mid','high')]
        torch.testing.assert_close(sum(parts), e, atol=2e-6, rtol=2e-6)
        outputs = dict(repairs(y, y+e, basis))
        torch.testing.assert_close(outputs['target'], y)
        for arm in ('frequency_low','frequency_mid','frequency_high','pca8','pca32'):
            torch.testing.assert_close((outputs[arm]-y).square().mean((2,3)),
                (outputs[arm+'_matched_residual']-y).square().mean((2,3)))
        torch.testing.assert_close(outputs['pca8'][..., :8], y[..., :8])
        torch.testing.assert_close(outputs['pca8'][..., 8:], (y+e)[..., 8:])

    def test_training_covariance_basis(self):
        stats = dict(mean=torch.zeros(2, 4), std=torch.ones(2, 4))
        samples = [dict(target=torch.randn(2, 12, 4)) for _ in range(4)]
        basis, values = fit_basis(samples, stats)
        torch.testing.assert_close(basis.T@basis, torch.eye(4), atol=1e-6, rtol=1e-6)
        self.assertTrue(bool((values[:-1] >= values[1:]).all()))

    def test_oracle_diagnostic_records_all_arms(self):
        from utils.window_subspace import audit_sample, ARMS
        y = torch.randn(1, 2, 4, 40)
        cr = torch.randn(1, 1, 4, 40)
        stats = dict(mean=torch.zeros(2, 40), std=torch.ones(2, 40))
        def decode(c, z):
            return torch.cat((c, z), 1).reshape(1, 3, 2, 2, 40)[..., :3].sigmoid()
        class Flow:
            def sample(self, *args, **kwargs): return y+.2
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rows = audit_sample(model=None, flow=Flow(), c=cr, y=y, cr=cr,
                ae=decode(cr, y), raw=decode(cr, y), stats=stats, decode=decode,
                sync=lambda: None, dtype=torch.float32, seeds=[42], index=0,
                video_id='x', out=out, rank=0, previews=0, status=lambda *a, **k: None,
                basis=torch.eye(40))
            self.assertEqual({r['arm'] for r in rows}, set(ARMS))
            self.assertEqual(len(rows), 12)
            self.assertEqual(next(r for r in rows if r['arm']=='target')['rgb_l1_vs_ae'], 0.)
            self.assertEqual(len(list((out/'latents').glob('*.pt'))), 1)


if __name__ == '__main__': unittest.main()
