"""EMA precision and checkpoint recovery regressions on real PyTorch tensors."""
import unittest
import warnings

import torch
from torch import nn

from utils.training import EMA


class EMATests(unittest.TestCase):
    def model(self, value=1.):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(value)
        return model

    def test_mature_ema_tracks_updates_below_bfloat16_resolution(self):
        model = self.model()
        ema = EMA(model, decay=.9995, dtype=torch.float32, warmup=True)
        ema.num_updates = 20000
        with torch.no_grad():
            model.weight.fill_(1.5)
        for _ in range(10000):
            ema.update(model)
        expected = 1.5 - .5 * .9995**10000
        self.assertAlmostEqual(ema.shadow['weight'].item(), expected, delta=3e-4)
        self.assertEqual(ema.shadow['weight'].dtype, torch.float32)

    def test_warmup_tracks_slow_online_drift(self):
        model = self.model()
        ema = EMA(model, decay=.9995, dtype=torch.float32, warmup=True)
        expected = 1.
        for step in range(1, 10001):
            value = 1. + .5 * step / 10000
            with torch.no_grad():
                model.weight.fill_(value)
            decay = min(.9995, (1 + step) / (10 + step))
            expected = decay * expected + (1 - decay) * value
            ema.update(model)
        self.assertAlmostEqual(ema.shadow['weight'].item(), expected, delta=3e-4)

    def test_low_precision_resume_preserves_values_and_records_history_limit(self):
        model = self.model()
        saved = {'weight': torch.tensor([[1.125]], dtype=torch.bfloat16)}
        ema = EMA(model, decay=.9995, dtype=torch.float32, warmup=True)
        with self.assertWarnsRegex(RuntimeWarning, 'cannot recover lost EMA history'):
            ema.load_state_dict(saved)
        # A legacy checkpoint has no dtype metadata. Keep its update counter.
        ema.load_metadata(dict(num_updates=20000, decay=.9995, warmup=True))
        self.assertEqual(ema.shadow['weight'].dtype, torch.float32)
        self.assertEqual(ema.shadow['weight'].item(), 1.125)
        self.assertEqual(ema.num_updates, 20000)
        self.assertEqual(ema.metadata()['dtype'], 'torch.float32')
        migration = ema.metadata()['precision_migrations']
        self.assertEqual(migration, [dict(source_dtype='torch.bfloat16',
            target_dtype='torch.float32', history_recovered=False)])
        with torch.no_grad():
            model.weight.fill_(1.5)
        ema.update(model)
        self.assertGreater(ema.shadow['weight'].item(), 1.125)
        # The subsequent FP32 checkpoint retains provenance without warning anew.
        restored = EMA(model, decay=.9995, dtype=torch.float32, warmup=True)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            restored.load_state_dict(ema.state_dict())
            restored.load_metadata(ema.metadata())
        self.assertFalse(captured)
        self.assertEqual(restored.metadata()['precision_migrations'], migration)

    def test_restored_shadow_never_aliases_snapshot(self):
        for configured_dtype in (None, torch.float32):
            with self.subTest(dtype=configured_dtype):
                saved = {'weight': torch.tensor([[1.]], dtype=torch.float32)}
                model = self.model(2.)
                ema = EMA(model, decay=.5, dtype=configured_dtype)
                ema.load_state_dict(saved)
                saved['weight'].fill_(3.)
                self.assertEqual(ema.shadow['weight'].item(), 1.)
                ema.update(model)
                self.assertEqual(ema.shadow['weight'].item(), 1.5)
                self.assertEqual(saved['weight'].item(), 3.)

    def test_unspecified_dtype_and_absent_metadata_remain_compatible(self):
        model = self.model()
        ema = EMA(model, dtype=None)
        saved = {'weight': torch.tensor([[.75]], dtype=torch.bfloat16)}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            ema.load_state_dict(saved)
            ema.load_metadata(None)
        self.assertFalse(captured)
        self.assertEqual(ema.shadow['weight'].dtype, torch.bfloat16)
        self.assertEqual(ema.metadata()['shadow_dtypes'], ['torch.bfloat16'])
        self.assertEqual(ema.num_updates, 0)


if __name__ == '__main__':
    unittest.main()
