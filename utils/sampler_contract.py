"""Data-time sampler identities; these are plumbing tests, not generation gates."""
import torch

from models.r7_flow_probe import sample_flow
from utils.latent_generation_metrics import inverse, normalize


class ConstantClean(torch.nn.Module):
    prediction = 'plain_x0'

    def __init__(self, clean):
        super().__init__()
        self.register_buffer('clean', clean)

    def forward(self, noisy, time, anchor, text=None, text_mask=None):
        return self.clean.expand_as(noisy)


class DeterministicClean(torch.nn.Module):
    """The historical predictor consumes/returns RAW latents, not flow statistics."""
    prediction = 'plain_x0'

    def __init__(self, predictor, anchor_stats, target_stats):
        super().__init__()
        self.predictor = predictor
        self.anchor_stats, self.target_stats = anchor_stats, target_stats

    def forward(self, noisy, time, anchor, text=None, text_mask=None):
        # Historical deterministic evaluation was FP32; preserve that contract
        # even when called inside the flow sampler's AMP context.
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            raw_anchor = inverse(anchor, self.anchor_stats)
            raw_target = self.predictor(raw_anchor[:, 0], 1)[:, None]
            return normalize(raw_target, self.target_stats)


@torch.no_grad()
def check_endpoint(model, anchor, reference, stats, decode, seeds, steps,
                   dtype=torch.float32, alpha=1., progress=None):
    """Compare real sampler endpoints and decoded RGB to the direct reference."""
    expected_raw = inverse(reference, stats)
    expected_rgb = decode(expected_raw)
    rows = []
    last_rgb = None
    for seed in seeds:
        noise = torch.randn(reference.shape, generator=torch.Generator('cpu')
                            .manual_seed(seed)).to(reference.device)
        for count in steps:
            if progress:
                progress(f'sampling:seed={seed}:steps={count}:grid={alpha}')
            sampled = sample_flow(model, anchor, noise, steps=count,
                                  dtype=dtype, grid_alpha=alpha)
            last_rgb = decode(inverse(sampled, stats))
            latent_error = float((sampled-reference).abs().max())
            rgb_error = float((last_rgb-expected_rgb).abs().max())
            passed = (bool(torch.isfinite(sampled).all())
                      and bool(torch.isfinite(last_rgb).all())
                      and latent_error <= 1e-4 and rgb_error <= 2e-3)
            rows.append(dict(seed=seed, steps=count, grid_alpha=alpha,
                             normalized_max_abs=latent_error,
                             rgb_max_abs=rgb_error, passed=passed))
    if not rows:
        raise ValueError('nonempty seeds and sampling steps required')
    return rows, last_rgb
