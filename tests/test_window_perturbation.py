import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.window_perturbation import matched_random, audit_sample


class PerturbationTests(unittest.TestCase):
    def test_slot_rms_zero_and_rng(self):
        error = torch.randn(2, 4, 9, 3)
        error[:, 1] = 0
        state = torch.random.get_rng_state().clone()
        result = matched_random(error, 19)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.testing.assert_close(result.square().mean((2,3)), error.square().mean((2,3)))
        self.assertTrue(torch.equal(result, matched_random(error, 19)))
        self.assertTrue(torch.isfinite(result).all())

    def test_native_decode_anchor_and_endpoints(self):
        y = torch.randn(1, 2, 4, 3)
        c = torch.randn(1, 1, 4, 3)
        cr = c*2+4
        stats = dict(mean=torch.full((2,3), 2.), std=torch.full((2,3), .3))
        from utils.window_training import inverse
        def decode(anchor, future):
            self.assertTrue(torch.equal(anchor, cr))
            return torch.cat([anchor, future], 1).reshape(1, 3, 2, 2, 3)
        ae = decode(cr, inverse(y, stats))
        generated = y + .2
        class Flow:
            def sample(self, *args, **kwargs): return generated.clone()
        with tempfile.TemporaryDirectory() as tmp, patch('utils.window_perturbation.save_video_preview'):
            rows = audit_sample(model=None, flow=Flow(), c=c, y=y, cr=cr, ae=ae,
                raw=ae, stats=stats, decode=decode, sync=lambda:None, dtype=torch.float32,
                seeds=[42], index=0, video_id='test', out=Path(tmp), rank=0,
                previews=0, status=lambda *a, **kw:None)
            self.assertEqual(len(rows), 12)
            for r in rows:
                if r['alpha'] == 0:
                    self.assertEqual(r['rgb_mse_vs_ae'], 0)
                    self.assertIsNone(r['rgb_rms_over_latent_rms'])
            endpoint = next(r for r in rows if r['arm']=='generated_a1')
            self.assertAlmostEqual(endpoint['rgb_l1_vs_ae'], .06, places=5)
            saved = torch.load(Path(tmp)/'latents/eval_clip0_seed42.pt', weights_only=False)
            torch.testing.assert_close(saved['generated_normalized'], generated)
            for alpha in (.05, .1, .25, .5, 1.):
                pair = [r for r in rows if r['alpha']==alpha]
                self.assertAlmostEqual(pair[0]['latent_mse'], pair[1]['latent_mse'], places=6)


if __name__ == '__main__': unittest.main()
