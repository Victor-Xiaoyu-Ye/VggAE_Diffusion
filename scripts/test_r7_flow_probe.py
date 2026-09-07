#!/usr/bin/env python3
"""Formula/static tests without torch, with optional CPU tensor regression tests."""
import ast
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def formula_checks():
    rng = random.Random(5)
    for t in (0., .001, .1, .5, .9, .999, 1.):
        for _ in range(10):
            data, noise = rng.gauss(0, 1), rng.gauss(0, 1)
            x = (1-t)*noise+t*data
            d = t*t+(1-t)**2
            f = ((1-t)*data-t*noise)/math.sqrt(d)
            clean = t/d*x+(1-t)/math.sqrt(d)*f
            v = (2*t-1)/d*x+f/math.sqrt(d)
            assert abs(clean-data) < 1e-12
            assert abs(v-(data-noise)) < 1e-12
    for name in ('train_r7_flow_probe.py', 'models/r7_flow_probe.py',
                 'utils/latent_generation_metrics.py'):
        ast.parse((ROOT/name).read_text(encoding='utf-8'))
    trainer = (ROOT/'train_r7_flow_probe.py').read_text(encoding='utf-8')
    assert 'warmup=True' in trainer
    assert "'train-memory'" in trainer and "'held-out'" in trainer
    assert 'complete_prefix(c, z,' in trainer
    for shell, target in (
        ('scripts/scale/25_run_r7_flow_probe.sh', 'train_r7_flow_probe.py'),
        ('scripts/scale/24_replay_r7_wan_denoising.sh', 'evaluate_r7_wan_denoising.py')):
        tree = ast.parse((ROOT/target).read_text(encoding='utf-8'))
        flags = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' and node.args
                    and isinstance(node.args[0], ast.Constant)):
                flags.add(node.args[0].value)
        # Required arguments constructed by the small for-loop in each CLI.
        flags.update('--'+v for v in ('csv','eval_csv','video_root','encoder_ckpt','r7_ckpt','output_dir',
                                      'checkpoint','wan_ckpt_dir','manifest','text_embedding_dir'))
        text = (ROOT/shell).read_text(encoding='utf-8')
        import re
        invoked = set(re.findall(r'--[a-z][a-z0-9_]*', text)) - {'--directory','--resume','--no'}
        assert not invoked-flags, (shell, invoked-flags)
    print('PASS: flow endpoint/algebra, static and shell/CLI contracts')


def tensor_checks():
    try:
        import torch
    except ImportError:
        print('SKIP: torch unavailable; model/gradient/sampler/normalization tests NOT run')
        return
    from models.r7_flow_probe import (R7FlowProbe, coefficients, network_target,
        clean_prediction, velocity_prediction, sample_flow)
    from utils.latent_generation_metrics import (fit_statistics, normalize, inverse,
        generation_metrics, complete_prefix)
    from utils.training import EMA
    from train_causal_video_diffusion import ema_weights
    from evaluate_r7_wan_denoising import integrate, csv_values
    assert csv_values('model,ema', str) == ['model', 'ema']
    torch.manual_seed(12)
    y, noise = torch.randn(2, 2, 4, 6), torch.randn(2, 2, 4, 6)
    c = torch.randn(2, 1, 4, 6)
    class Oracle(torch.nn.Module):
        def forward(self, noisy, t, cond=None, text_emb=None):
            return y
    for start in (0., .3, .9):
        sampled = integrate(Oracle(), c, noise, None, None, 10, 3., 1.,
                            torch.float32, start, y if start else None)
        assert torch.allclose(sampled, y, atol=1e-5)
    stats = fit_statistics(y)
    assert torch.allclose(inverse(normalize(y, stats), stats), y, atol=1e-6)
    old_mean = stats['mean'].clone()
    normalize(y+100, stats)
    assert torch.equal(old_mean, stats['mean'])
    constats = fit_statistics(c)
    for tval in (0., .5, .99, 1.):
        t = torch.full((2,), tval)
        x = (1-tval)*noise+tval*y
        f = network_target(y, noise, t, 'preconditioned')
        assert torch.allclose(clean_prediction(f, x, t, 'preconditioned'), y, atol=1e-6)
        assert torch.allclose(velocity_prediction(f, x, t, 'preconditioned'), y-noise, atol=1e-6)
    full = complete_prefix(c, y, 9)
    assert full.shape == (2, 9, 4, 6)
    assert torch.equal(full[:, 3:], y[:, -1:].expand(-1, 6, -1, -1))
    for kind in ('plain_x0', 'preconditioned'):
        model = R7FlowProbe(latent_dim=6, num_tokens=4, future_frames=2,
                            hidden_dim=24, depth=1, num_heads=3, prediction=kind)
        t = torch.tensor([.2, .8]); te = t[:, None, None, None]
        x = (1-te)*noise+te*y
        f = model(x, t, c)
        loss = (f-network_target(y, noise, t, kind)).square().mean()
        loss.backward()
        assert model.head.weight.grad is not None
        a = sample_flow(model, c, noise, 4)
        b = sample_flow(model, c, noise, 4)
        assert torch.isfinite(a).all() and torch.equal(a, b)
        ema = EMA(model, .9999, warmup=True)
        snapshot = {k:v.clone() for k,v in model.state_dict().items()}
        ema.update(model)
        with ema_weights(model, ema):
            pass
        assert all(torch.equal(snapshot[k], v) for k,v in model.state_dict().items())
    # A shared position mean can yield perfect raw motion without any content.
    yc = torch.randn(2, 2, 4, 6)*.001+100
    cc = torch.zeros_like(c)
    ys, cs = fit_statistics(yc), fit_statistics(cc)
    pred = ys['mean'][None, :, None].expand_as(yc)
    metrics = generation_metrics(pred, yc, cc, ys, cs)
    assert metrics['mean_only/chunk1/motion_cosine'] > .999
    stationary = generation_metrics(cc.expand(-1, 2, -1, -1),
        cc.expand(-1, 2, -1, -1), cc, fit_statistics(cc.expand(-1, 2, -1, -1)), cs)
    assert stationary['raw/chunk1/motion_cosine'] is None
    print('PASS: CPU tensor, model gradient, deterministic sampler, EMA and metric regressions')


if __name__ == '__main__':
    formula_checks()
    tensor_checks()
